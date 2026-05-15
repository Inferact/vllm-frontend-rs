use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use llm_multimodal::{
    AsyncMultiModalTracker, FieldLayout, ImageProcessorRegistry, MediaConnector,
    MediaConnectorConfig, MediaContentPart, Modality, ModelMetadata, ModelRegistry,
    ModelSpecificValue, PreProcessorConfig, PreprocessedImages, PromptReplacement, TokenResolver,
    TrackedMedia,
};
use vllm_engine_core_client::protocol::multimodal::{
    MultiModalBatchedField, MultiModalFeatureSpec, MultiModalFeatures, MultiModalField,
    MultiModalFieldElem, MultiModalFlatField, MultiModalKwargsItem, MultiModalSharedField,
    MultiModalSlice, NestedTensorValue, PlaceholderRange, SliceSpec,
};
use vllm_engine_core_client::protocol::tensor_wire::WireTensor;
use vllm_text::Prompt;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};

use crate::error::{Error, Result};
use crate::renderer::RenderedPrompt;
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};

#[derive(Debug, Clone)]
pub struct MultimodalModelInfo {
    pub model_id: String,
    pub model_type: Option<String>,
    pub config: serde_json::Value,
    pub preprocessor_config: PreProcessorConfig,
}

impl MultimodalModelInfo {
    pub fn from_paths(
        model_id: String,
        model_type: Option<String>,
        config_path: Option<&Path>,
        preprocessor_config_path: Option<&Path>,
    ) -> Result<Self> {
        let config = match config_path {
            Some(path) => {
                let text = fs::read_to_string(path).map_err(|error| {
                    Error::Multimodal(format!("failed to read config.json: {error}"))
                })?;
                serde_json::from_str(&text).map_err(|error| {
                    Error::Multimodal(format!("failed to parse config.json: {error}"))
                })?
            }
            None => serde_json::Value::Object(Default::default()),
        };
        let preprocessor_config = match preprocessor_config_path {
            Some(path) => {
                let text = fs::read_to_string(path).map_err(|error| {
                    Error::Multimodal(format!("failed to read preprocessor_config.json: {error}"))
                })?;
                PreProcessorConfig::from_json(&text).map_err(|error| {
                    Error::Multimodal(format!("failed to parse preprocessor_config.json: {error}"))
                })?
            }
            None => PreProcessorConfig::default(),
        };

        Ok(Self {
            model_id,
            model_type,
            config,
            preprocessor_config,
        })
    }
}

pub(crate) fn placeholder_token(
    info: &MultimodalModelInfo,
    tokenizer: &dyn Tokenizer,
) -> Result<String> {
    with_model_spec(info, tokenizer, |spec, metadata| {
        spec.placeholder_token(metadata)
            .map_err(|error| Error::Multimodal(error.to_string()))
    })
}

pub(crate) async fn finalize_rendered_prompt(
    request: &ChatRequest,
    rendered: RenderedPrompt,
    tokenizer: DynTokenizer,
    info: Option<&MultimodalModelInfo>,
) -> Result<(Prompt, Option<MultiModalFeatures>)> {
    if !request.has_multimodal() {
        return Ok((rendered.prompt, None));
    }
    let info = info.ok_or(Error::UnsupportedMultimodalRenderer)?;
    let Prompt::Text(prompt) = rendered.prompt else {
        return Err(Error::Multimodal(
            "multimodal chat renderer must return a text prompt before expansion".to_string(),
        ));
    };
    let media_parts = extract_media_parts(request)?;

    let mut prompt_token_ids = tokenizer
        .encode(&prompt, request.add_special_tokens)
        .map_err(|error| Error::Multimodal(error.to_string()))?;
    let prepared =
        prepare_multimodal(info, &*tokenizer, media_parts, &mut prompt_token_ids).await?;

    Ok((Prompt::TokenIds(prompt_token_ids), Some(prepared)))
}

async fn prepare_multimodal(
    info: &MultimodalModelInfo,
    tokenizer: &dyn Tokenizer,
    media_parts: Vec<MediaContentPart>,
    prompt_token_ids: &mut Vec<u32>,
) -> Result<MultiModalFeatures> {
    if media_parts.is_empty() {
        return Ok(Vec::new());
    }

    let connector = Arc::new(
        MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
            .map_err(|error| Error::Multimodal(error.to_string()))?,
    );
    let mut tracker = AsyncMultiModalTracker::new(connector);
    for part in &media_parts {
        tracker
            .push_part(part.clone())
            .map_err(|error| Error::Multimodal(error.to_string()))?;
    }
    let tracker_output =
        tracker.finalize().await.map_err(|error| Error::Multimodal(error.to_string()))?;

    let images = tracker_output.data.get(&Modality::Image).cloned().unwrap_or_default();
    if images.len() != media_parts.len() {
        return Err(Error::Multimodal(format!(
            "expected {} fetched images, got {}",
            media_parts.len(),
            images.len()
        )));
    }
    let image_frames = images
        .into_iter()
        .map(|media| match media {
            TrackedMedia::Image(frame) => Ok(frame),
            _ => Err(Error::UnsupportedMultimodalContent("non-image")),
        })
        .collect::<Result<Vec<_>>>()?;

    let preprocessed = preprocess_images(info, &image_frames).await?;
    let replacements = with_model_spec(info, tokenizer, |spec, metadata| {
        spec.prompt_replacements(metadata, &preprocessed)
            .map_err(|error| Error::Multimodal(error.to_string()))
    })?;
    if replacements.len() != media_parts.len() {
        return Err(Error::Multimodal(format!(
            "expected {} prompt replacements, got {}",
            media_parts.len(),
            replacements.len()
        )));
    }

    let placeholder_token = with_model_spec(info, tokenizer, |spec, metadata| {
        spec.placeholder_token(metadata)
            .map_err(|error| Error::Multimodal(error.to_string()))
    })?;
    let ranges = expand_prompt_tokens(
        prompt_token_ids,
        tokenizer,
        &placeholder_token,
        &replacements,
    )?;
    let field_layouts = with_model_spec(info, tokenizer, |spec, _| Ok(spec.field_layouts()))?;
    let keep_on_cpu_keys = with_model_spec(info, tokenizer, |spec, _| Ok(spec.keep_on_cpu_keys()))?;

    build_features(
        &preprocessed,
        &image_frames,
        &tracker_output.uuids,
        &ranges,
        &field_layouts,
        &keep_on_cpu_keys,
    )
}

fn extract_media_parts(request: &ChatRequest) -> Result<Vec<MediaContentPart>> {
    let mut parts = Vec::new();
    for message in &request.messages {
        let content = match message {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        extract_media_parts_from_content(content, &mut parts)?;
    }
    Ok(parts)
}

fn extract_media_parts_from_content(
    content: &ChatContent,
    out: &mut Vec<MediaContentPart>,
) -> Result<()> {
    let ChatContent::Parts(parts) = content else {
        return Ok(());
    };
    for part in parts {
        match part {
            ChatContentPart::Text { .. } => {}
            ChatContentPart::ImageUrl {
                image_url,
                detail,
                uuid,
            } => out.push(MediaContentPart::ImageUrl {
                url: image_url.clone(),
                detail: *detail,
                uuid: uuid.clone(),
            }),
        }
    }
    Ok(())
}

async fn preprocess_images(
    info: &MultimodalModelInfo,
    image_frames: &[Arc<llm_multimodal::ImageFrame>],
) -> Result<PreprocessedImages> {
    let model_id = info.model_id.clone();
    let model_type = info.model_type.clone();
    let config = info.preprocessor_config.clone();
    let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

    tokio::task::spawn_blocking(move || {
        let registry = ImageProcessorRegistry::with_defaults();
        let processor = registry.find(&model_id, model_type.as_deref()).ok_or_else(|| {
            Error::Multimodal(format!("no image processor registered for `{model_id}`"))
        })?;
        processor
            .preprocess(&images, &config)
            .map_err(|error| Error::Multimodal(error.to_string()))
    })
    .await
    .map_err(|error| Error::Multimodal(format!("image preprocessing task failed: {error}")))?
}

fn expand_prompt_tokens(
    prompt_token_ids: &mut Vec<u32>,
    tokenizer: &dyn Tokenizer,
    placeholder_token: &str,
    replacements: &[PromptReplacement],
) -> Result<Vec<PlaceholderRange>> {
    let mut cursor = 0;
    let marker = marker_tokens(placeholder_token, tokenizer)?;
    let mut ranges = Vec::with_capacity(replacements.len());
    for replacement in replacements {
        let offset = find_subsequence(prompt_token_ids, &marker, cursor).ok_or_else(|| {
            Error::Multimodal(format!(
                "placeholder token `{placeholder_token}` was not found in tokenized prompt"
            ))
        })?;
        let replacement_tokens =
            replacement.tokens.iter().map(|token| *token as u32).collect::<Vec<_>>();
        let replacement_len = replacement_tokens.len();
        prompt_token_ids.splice(offset..offset + marker.len(), replacement_tokens);
        ranges.push(PlaceholderRange {
            offset,
            length: replacement_len,
            is_embed: None,
        });
        cursor = offset + replacement_len;
    }
    Ok(ranges)
}

fn marker_tokens(placeholder_token: &str, tokenizer: &dyn Tokenizer) -> Result<Vec<u32>> {
    if let Some(token_id) = tokenizer.token_to_id(placeholder_token) {
        return Ok(vec![token_id]);
    }
    let ids = tokenizer
        .encode(placeholder_token, false)
        .map_err(|error| Error::Multimodal(error.to_string()))?;
    if ids.is_empty() {
        return Err(Error::Multimodal(format!(
            "placeholder token `{placeholder_token}` encoded to no tokens"
        )));
    }
    Ok(ids)
}

fn find_subsequence(haystack: &[u32], needle: &[u32], start: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (start..=haystack.len() - needle.len())
        .find(|&index| &haystack[index..index + needle.len()] == needle)
}

fn build_features(
    preprocessed: &PreprocessedImages,
    image_frames: &[Arc<llm_multimodal::ImageFrame>],
    uuids: &llm_multimodal::MultiModalUUIDs,
    ranges: &[PlaceholderRange],
    field_layouts: &HashMap<String, FieldLayout>,
    keep_on_cpu_keys: &[String],
) -> Result<MultiModalFeatures> {
    let keep_on_cpu_keys = keep_on_cpu_keys.iter().cloned().collect::<HashSet<_>>();
    let tensors = collect_tensors(preprocessed);
    let image_uuids = uuids.get(&Modality::Image);

    let mut features = Vec::with_capacity(image_frames.len());
    for (index, frame) in image_frames.iter().enumerate() {
        let mut data = MultiModalKwargsItem::new();
        for (key, tensor) in &tensors {
            let keep_on_cpu = keep_on_cpu_keys.contains(key);
            let (value, field) = match field_layouts.get(key) {
                Some(FieldLayout::Batched) => (
                    slice_batched_tensor_value(tensor, index)?,
                    MultiModalField::Batched(MultiModalBatchedField { keep_on_cpu }),
                ),
                Some(FieldLayout::Flat { sizes_key }) => {
                    let (start, end) = flat_range_for_index(&tensors, sizes_key, index)?;
                    (
                        slice_flat_tensor_value(tensor, start, end)?,
                        MultiModalField::Flat(MultiModalFlatField {
                            slices: vec![MultiModalSlice::Slice(SliceSpec {
                                start: Some(0),
                                stop: Some((end - start) as isize),
                                step: None,
                            })],
                            dim: 0,
                            keep_on_cpu,
                        }),
                    )
                }
                None => (
                    full_tensor_value(tensor)?,
                    MultiModalField::Shared(MultiModalSharedField {
                        batch_size: image_frames.len(),
                        keep_on_cpu,
                    }),
                ),
            };
            data.insert(
                key.clone(),
                MultiModalFieldElem {
                    data: Some(value),
                    field,
                },
            );
        }

        let uuid = image_uuids.and_then(|values| values.get(index)).and_then(|value| value.clone());
        let hash = frame.hash.clone();
        features.push(MultiModalFeatureSpec {
            data: Some(data),
            modality: "image".to_string(),
            identifier: uuid.unwrap_or_else(|| hash.clone()),
            mm_position: ranges[index].clone(),
            mm_hash: Some(hash),
        });
    }

    Ok(features)
}

#[derive(Clone)]
enum TensorValue {
    F32 { data: Vec<f32>, shape: Vec<usize> },
    I64 { data: Vec<i64>, shape: Vec<usize> },
    U32 { data: Vec<u32>, shape: Vec<usize> },
    Scalar(NestedTensorValue),
}

fn collect_tensors(preprocessed: &PreprocessedImages) -> HashMap<String, TensorValue> {
    let mut tensors = HashMap::from([(
        "pixel_values".to_string(),
        TensorValue::F32 {
            data: preprocessed.pixel_values.iter().copied().collect(),
            shape: preprocessed.pixel_values.shape().to_vec(),
        },
    )]);
    for (key, value) in &preprocessed.model_specific {
        tensors.insert(key.clone(), tensor_value_from_model_specific(value));
    }
    tensors
}

fn tensor_value_from_model_specific(value: &ModelSpecificValue) -> TensorValue {
    match value {
        ModelSpecificValue::Tensor { data, shape } => TensorValue::F32 {
            data: data.clone(),
            shape: shape.clone(),
        },
        ModelSpecificValue::IntTensor { data, shape } => TensorValue::I64 {
            data: data.clone(),
            shape: shape.clone(),
        },
        ModelSpecificValue::UintTensor { data, shape } => TensorValue::U32 {
            data: data.clone(),
            shape: shape.clone(),
        },
        ModelSpecificValue::Int(value) => TensorValue::Scalar(NestedTensorValue::Int(*value)),
        ModelSpecificValue::Float(value) => TensorValue::Scalar(NestedTensorValue::Float(*value)),
        ModelSpecificValue::IntVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values.iter().map(|value| NestedTensorValue::Int(*value)).collect(),
        )),
        ModelSpecificValue::UintVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values.iter().map(|value| NestedTensorValue::Int(*value as i64)).collect(),
        )),
        ModelSpecificValue::FloatVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values.iter().map(|value| NestedTensorValue::Float(*value as f64)).collect(),
        )),
        ModelSpecificValue::TupleVec(values) => TensorValue::Scalar(NestedTensorValue::List(
            values
                .iter()
                .map(|(height, width)| {
                    NestedTensorValue::List(vec![
                        NestedTensorValue::Int(*height as i64),
                        NestedTensorValue::Int(*width as i64),
                    ])
                })
                .collect(),
        )),
        ModelSpecificValue::Bool(value) => {
            TensorValue::Scalar(NestedTensorValue::Int(i64::from(*value)))
        }
    }
}

fn slice_batched_tensor_value(tensor: &TensorValue, index: usize) -> Result<NestedTensorValue> {
    match tensor {
        TensorValue::F32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_f32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::I64 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_i64(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::U32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_u32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::Scalar(value) => Ok(value.clone()),
    }
}

fn slice_flat_tensor_value(
    tensor: &TensorValue,
    start: usize,
    end: usize,
) -> Result<NestedTensorValue> {
    match tensor {
        TensorValue::F32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_f32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::I64 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_i64(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::U32 { data, shape } => {
            let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
            Ok(NestedTensorValue::Tensor(
                WireTensor::from_u32(shape, data).map_err(Error::Multimodal)?,
            ))
        }
        TensorValue::Scalar(value) => Ok(value.clone()),
    }
}

fn full_tensor_value(tensor: &TensorValue) -> Result<NestedTensorValue> {
    match tensor {
        TensorValue::F32 { data, shape } => Ok(NestedTensorValue::Tensor(
            WireTensor::from_f32(shape.clone(), data.clone()).map_err(Error::Multimodal)?,
        )),
        TensorValue::I64 { data, shape } => Ok(NestedTensorValue::Tensor(
            WireTensor::from_i64(shape.clone(), data.clone()).map_err(Error::Multimodal)?,
        )),
        TensorValue::U32 { data, shape } => Ok(NestedTensorValue::Tensor(
            WireTensor::from_u32(shape.clone(), data.clone()).map_err(Error::Multimodal)?,
        )),
        TensorValue::Scalar(value) => Ok(value.clone()),
    }
}

fn flat_range_for_index(
    tensors: &HashMap<String, TensorValue>,
    sizes_key: &str,
    index: usize,
) -> Result<(usize, usize)> {
    let sizes = tensors.get(sizes_key).ok_or_else(|| {
        Error::Multimodal(format!("flat tensor sizes key `{sizes_key}` is missing"))
    })?;
    let sizes = tensor_as_usize_vec(sizes)?;
    let size = *sizes.get(index).ok_or_else(|| {
        Error::Multimodal(format!(
            "flat tensor sizes key `{sizes_key}` has no entry for image {index}"
        ))
    })?;
    let start = sizes[..index].iter().sum::<usize>();
    Ok((start, start + size))
}

fn tensor_as_usize_vec(tensor: &TensorValue) -> Result<Vec<usize>> {
    match tensor {
        TensorValue::I64 { data, .. } => data
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .map_err(|_| Error::Multimodal(format!("negative flat tensor size `{value}`")))
            })
            .collect(),
        TensorValue::U32 { data, .. } => Ok(data.iter().map(|value| *value as usize).collect()),
        _ => Err(Error::Multimodal(
            "flat tensor sizes must be int64 or uint32".to_string(),
        )),
    }
}

fn slice_first_axis_range<T: Clone>(
    shape: &[usize],
    data: &[T],
    start: usize,
    end: usize,
    drop_axis: bool,
) -> Result<(Vec<usize>, Vec<T>)> {
    let first_dim = *shape
        .first()
        .ok_or_else(|| Error::Multimodal("tensor has no first dimension".to_string()))?;
    if start > end || end > first_dim {
        return Err(Error::Multimodal(format!(
            "invalid tensor slice {start}..{end} for first dimension {first_dim}"
        )));
    }
    let stride = shape[1..].iter().product::<usize>();
    let data_start = start * stride;
    let data_end = end * stride;
    let out_shape = if drop_axis {
        shape[1..].to_vec()
    } else {
        let mut shape = shape.to_vec();
        shape[0] = end - start;
        shape
    };
    Ok((out_shape, data[data_start..data_end].to_vec()))
}

fn with_model_spec<R>(
    info: &MultimodalModelInfo,
    tokenizer: &dyn Tokenizer,
    f: impl FnOnce(&dyn llm_multimodal::ModelProcessorSpec, &ModelMetadata<'_>) -> Result<R>,
) -> Result<R> {
    let resolver = TokenizerResolver { tokenizer };
    let metadata = ModelMetadata {
        model_id: &info.model_id,
        tokenizer: &resolver,
        config: &info.config,
    };
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).ok_or_else(|| {
        Error::Multimodal(format!("unsupported multimodal model `{}`", info.model_id))
    })?;
    f(spec, &metadata)
}

struct TokenizerResolver<'a> {
    tokenizer: &'a dyn Tokenizer,
}

impl TokenResolver for TokenizerResolver<'_> {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_text::tokenizer::{IncrementalDecoder, Tokenizer, TokenizerError};

    use super::*;

    struct TestTokenizer;

    impl Tokenizer for TestTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> std::result::Result<Vec<u32>, TokenizerError> {
            Ok(match text {
                "<image>" => vec![999],
                text => text.bytes().map(u32::from).collect(),
            })
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> std::result::Result<String, TokenizerError> {
            Ok(String::new())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<image>" => Some(999),
                "<|image_pad|>" => Some(151655),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            match id {
                999 => Some("<image>".to_string()),
                151655 => Some("<|image_pad|>".to_string()),
                _ => None,
            }
        }

        fn create_decode_stream(
            &self,
            _prompt_token_ids: &[u32],
            _skip_special_tokens: bool,
            _min_bytes_to_buffer: usize,
        ) -> Box<dyn IncrementalDecoder + '_> {
            unreachable!("not used")
        }
    }

    fn qwen_info() -> MultimodalModelInfo {
        MultimodalModelInfo {
            model_id: "qwen2-vl-test".to_string(),
            model_type: Some("qwen2_vl".to_string()),
            config: serde_json::json!({
                "model_type": "qwen2_vl",
                "vision_token_id": 151655
            }),
            preprocessor_config: PreProcessorConfig::default(),
        }
    }

    #[test]
    fn resolves_qwen_placeholder_token() {
        let placeholder = placeholder_token(&qwen_info(), &TestTokenizer).unwrap();

        assert_eq!(placeholder, "<image>");
    }

    #[test]
    fn expand_prompt_tokens_replaces_placeholder_marker() {
        let mut prompt_token_ids = vec![1, 999, 2];
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<image>",
            vec![151655, 151655, 151655],
        )];

        let ranges = expand_prompt_tokens(
            &mut prompt_token_ids,
            &TestTokenizer,
            "<image>",
            &replacements,
        )
        .unwrap();

        assert_eq!(prompt_token_ids, vec![1, 151655, 151655, 151655, 2]);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 3);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_placeholder_missing() {
        let mut prompt_token_ids = vec![1, 2, 3];
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<image>",
            vec![151655],
        )];

        let error = expand_prompt_tokens(
            &mut prompt_token_ids,
            &TestTokenizer,
            "<image>",
            &replacements,
        )
        .unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[tokio::test]
    async fn finalizes_qwen_image_data_url_into_token_ids_and_features() {
        let tokenizer: DynTokenizer = Arc::new(TestTokenizer);
        let request = ChatRequest {
            messages: vec![ChatMessage::user(vec![ChatContentPart::ImageUrl {
                image_url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=".to_string(),
                detail: None,
                uuid: Some("image-1".to_string()),
            }])],
            ..ChatRequest::for_test()
        };
        let rendered = RenderedPrompt {
            prompt: Prompt::Text("<image>".to_string()),
        };

        let (prompt, features) =
            finalize_rendered_prompt(&request, rendered, tokenizer, Some(&qwen_info()))
                .await
                .unwrap();

        let token_ids = prompt.into_token_ids().unwrap();
        assert!(!token_ids.is_empty());
        assert!(token_ids.iter().all(|id| *id == 151655));

        let features = features.unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].identifier, "image-1");
        assert_eq!(features[0].mm_position.offset, 0);
        assert_eq!(features[0].mm_position.length, token_ids.len());
        let data = features[0].data.as_ref().unwrap();
        assert!(data.contains_key("pixel_values"));
        assert!(data.contains_key("image_grid_thw"));
    }
}
