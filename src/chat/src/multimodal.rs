//! Chat-layer multimodal image preparation.
//!
//! This module owns the narrow image-only multimodal path for chat requests:
//! it extracts image parts from structured chat messages, fetches and
//! preprocesses them through `llm-multimodal`, expands rendered prompt
//! placeholders after tokenization, and builds the engine-facing
//! `MultiModalFeatures` payload.
//!
//! Raw media stays above `vllm-text`; this module lowers it into token IDs and
//! opaque tensor payloads before the request is handed to text generation.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use llm_multimodal::{
    AsyncMultiModalTracker, FieldLayout, ImagePreProcessor, ImageProcessorRegistry, MediaConnector,
    MediaConnectorConfig, MediaContentPart, Modality, ModelMetadata, ModelProcessorSpec,
    ModelRegistry, ModelSpecificValue, PreProcessorConfig, PreprocessedImages, PromptReplacement,
    TokenResolver, TrackedMedia,
};
use tracing::warn;
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

/// Resolved multimodal support for one loaded model.
///
/// A value of this type means the model has both a registered prompt/spec
/// handler and an image preprocessor. Unsupported models are represented by
/// `None` at the backend boundary rather than by partially initialized fields.
#[derive(Clone)]
pub struct MultimodalModelInfo {
    /// Model identifier used for registry matching and diagnostics.
    pub model_id: String,
    /// Model type from `config.json`, when available.
    pub model_type: Option<String>,
    /// Raw model config passed to model-specific multimodal specs.
    pub config: serde_json::Value,
    /// Parsed preprocessor config passed to image preprocessors.
    pub preprocessor_config: PreProcessorConfig,
    /// Static model-specific prompt/tensor-layout spec.
    spec: &'static dyn ModelProcessorSpec,
    /// Static model-specific image preprocessor.
    image_processor: &'static dyn ImagePreProcessor,
}

impl MultimodalModelInfo {
    /// Load and resolve multimodal support from model files.
    ///
    /// Returns `Ok(Some(_))` only when both the model spec and image processor
    /// are registered. File read/parse failures are real errors; unsupported
    /// model families are logged and returned as `Ok(None)`.
    pub fn from_paths(
        model_id: String,
        model_type: Option<String>,
        config_path: Option<&Path>,
        preprocessor_config_path: Option<&Path>,
        tokenizer: &dyn Tokenizer,
    ) -> Result<Option<Self>> {
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

        let Some(spec) = lookup_model_spec(&model_id, &config, tokenizer) else {
            warn!(
                model_id = %model_id,
                model_type = ?model_type,
                "multimodal model spec is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };
        let image_processor = lookup_image_processor(&model_id, model_type.as_deref());
        let Some(image_processor) = image_processor else {
            warn!(
                model_id = %model_id,
                model_type = ?model_type,
                "image processor is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };

        Ok(Some(Self {
            model_id,
            model_type,
            config,
            preprocessor_config,
            spec,
            image_processor,
        }))
    }

    /// Build the borrowed metadata object required by `llm-multimodal` specs.
    fn metadata<'a>(&'a self, resolver: &'a TokenizerResolver<'a>) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: &self.model_id,
            tokenizer: resolver,
            config: &self.config,
        }
    }
}

/// Return the template-visible placeholder token for this model.
///
/// The HF renderer uses this token while flattening image content in string
/// content format.
pub(crate) fn placeholder_token(
    info: &MultimodalModelInfo,
    tokenizer: &dyn Tokenizer,
) -> Result<String> {
    let resolver = TokenizerResolver { tokenizer };
    let metadata = info.metadata(&resolver);
    info.spec
        .placeholder_token(&metadata)
        .map_err(|error| Error::Multimodal(error.to_string()))
}

/// Finalize a rendered chat prompt into text-generation input.
///
/// Text-only requests pass through unchanged as `Prompt::Text`. Multimodal
/// requests are tokenized in chat, their image placeholders are expanded, and
/// preprocessed image features are attached for engine-core transport.
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

// ============================================================================
// Pipeline orchestration
// ============================================================================

/// Model-specific plan derived after image preprocessing.
///
/// The prompt replacements depend on preprocessed image metadata, while field
/// layouts and CPU placement hints come from the static model spec.
struct MultimodalSpecPlan {
    /// Per-image token sequences that replace the rendered placeholder marker.
    replacements: Vec<PromptReplacement>,
    /// Mapping from tensor key to per-item slicing semantics.
    field_layouts: HashMap<String, FieldLayout>,
    /// Tensor keys that should remain on CPU in the backend.
    keep_on_cpu_keys: Vec<String>,
}

/// Run media fetch, image preprocessing, prompt expansion, and feature build.
///
/// `prompt_token_ids` is mutated in place because placeholder expansion changes
/// both the final prompt and the offsets recorded in `PlaceholderRange`.
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
    let resolver = TokenizerResolver { tokenizer };
    let metadata = info.metadata(&resolver);
    let replacements = info
        .spec
        .prompt_replacements(&metadata, &preprocessed)
        .map_err(|error| Error::Multimodal(error.to_string()))?;
    let spec_plan = MultimodalSpecPlan {
        replacements,
        field_layouts: info.spec.field_layouts(),
        keep_on_cpu_keys: info.spec.keep_on_cpu_keys(),
    };
    if spec_plan.replacements.len() != media_parts.len() {
        return Err(Error::Multimodal(format!(
            "expected {} prompt replacements, got {}",
            media_parts.len(),
            spec_plan.replacements.len()
        )));
    }

    let ranges = expand_prompt_tokens(prompt_token_ids, tokenizer, &spec_plan.replacements)?;

    build_features(
        &preprocessed,
        &image_frames,
        &tracker_output.uuids,
        &ranges,
        &spec_plan.field_layouts,
        &spec_plan.keep_on_cpu_keys,
    )
}

// ============================================================================
// Stage 1: extract media parts from the chat request
// ============================================================================

/// Extract image media parts from chat messages in message/content order.
///
/// Assistant history is skipped because generated assistant blocks are already
/// represented as text for prompt rendering in this crate.
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

/// Append media parts found in one structured chat content value.
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

// ============================================================================
// Stage 2: image preprocessing
// ============================================================================

/// Preprocess fetched image frames with the model's resolved image processor.
///
/// The processor work is CPU-heavy relative to request wiring, so it runs in a
/// blocking task and returns owned tensors ready for wire conversion.
async fn preprocess_images(
    info: &MultimodalModelInfo,
    image_frames: &[Arc<llm_multimodal::ImageFrame>],
) -> Result<PreprocessedImages> {
    let config = info.preprocessor_config.clone();
    let processor = info.image_processor;
    let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

    tokio::task::spawn_blocking(move || {
        processor
            .preprocess(&images, &config)
            .map_err(|error| Error::Multimodal(error.to_string()))
    })
    .await
    .map_err(|error| Error::Multimodal(format!("image preprocessing task failed: {error}")))?
}

// ============================================================================
// Stage 3: expand placeholder tokens into the tokenized prompt
// ============================================================================

/// Replace rendered placeholder markers with model-specific replacement tokens.
///
/// Replacements are consumed in order, matching the original media-part order.
/// The returned ranges point into the already-expanded prompt.
fn expand_prompt_tokens(
    prompt_token_ids: &mut Vec<u32>,
    tokenizer: &dyn Tokenizer,
    replacements: &[PromptReplacement],
) -> Result<Vec<PlaceholderRange>> {
    let mut cursor = 0;
    let mut ranges = Vec::with_capacity(replacements.len());
    for replacement in replacements {
        if replacement.modality != Modality::Image {
            return Err(Error::Multimodal(format!(
                "unsupported prompt replacement modality `{}`",
                replacement.modality
            )));
        }
        let marker = marker_tokens(&replacement.placeholder_token, tokenizer)?;
        let offset = find_subsequence(prompt_token_ids, &marker, cursor).ok_or_else(|| {
            Error::Multimodal(format!(
                "placeholder token `{}` was not found in tokenized prompt",
                replacement.placeholder_token
            ))
        })?;
        let replacement_tokens = replacement
            .tokens
            .iter()
            .map(|token| {
                u32::try_from(*token).map_err(|_| {
                    Error::Multimodal(format!("negative replacement token id `{token}`"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if replacement_tokens.is_empty() {
            return Err(Error::Multimodal(format!(
                "placeholder token `{}` expanded to no tokens",
                replacement.placeholder_token
            )));
        }
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

/// Resolve the tokenized marker sequence for a placeholder string.
///
/// Most supported models use a single special token, but this accepts a
/// multi-token marker if the tokenizer does not expose the string as one ID.
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

/// Find `needle` in `haystack`, starting at `start`.
///
/// This is intentionally order-preserving rather than a global replace: each
/// image consumes the next placeholder occurrence.
fn find_subsequence(haystack: &[u32], needle: &[u32], start: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (start..=haystack.len() - needle.len())
        .find(|&index| &haystack[index..index + needle.len()] == needle)
}

// ============================================================================
// Stage 4: build MultiModalFeatures from preprocessed tensors
// ============================================================================

/// Convert preprocessed image tensors into engine-core multimodal features.
///
/// One `MultiModalFeatureSpec` is produced per image. Tensor fields are sliced
/// according to the model spec's field layout declarations.
fn build_features(
    preprocessed: &PreprocessedImages,
    image_frames: &[Arc<llm_multimodal::ImageFrame>],
    uuids: &llm_multimodal::MultiModalUUIDs,
    ranges: &[PlaceholderRange],
    field_layouts: &HashMap<String, FieldLayout>,
    keep_on_cpu_keys: &[String],
) -> Result<MultiModalFeatures> {
    if image_frames.len() != ranges.len() {
        return Err(Error::Multimodal(format!(
            "expected {} placeholder ranges, got {}",
            image_frames.len(),
            ranges.len()
        )));
    }

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

// ============================================================================
// Tensor abstraction & slicing
// ============================================================================

/// Owned tensor/scalar representation used while splitting fields per image.
#[derive(Clone)]
enum TensorValue {
    /// Float tensor with row-major flat data and shape.
    F32 { data: Vec<f32>, shape: Vec<usize> },
    /// Signed integer tensor with row-major flat data and shape.
    I64 { data: Vec<i64>, shape: Vec<usize> },
    /// Unsigned integer tensor with row-major flat data and shape.
    U32 { data: Vec<u32>, shape: Vec<usize> },
    /// Non-tensor nested value that is shared or copied as-is.
    Scalar(NestedTensorValue),
}

/// Collect `pixel_values` and model-specific outputs into one tensor map.
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

/// Convert one `llm-multimodal` model-specific value into wire-builder input.
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

/// Extract one image from a batched tensor field.
///
/// Batched fields use their first axis as image index and drop that axis in the
/// per-feature value, matching vLLM's batched-field semantics.
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

/// Extract one image's variable-length range from a flat tensor field.
///
/// Flat fields keep the first axis as the sliced length for this image.
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

/// Convert a tensor or scalar value without per-image slicing.
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

/// Compute the first-axis range for one image in a flat tensor.
///
/// `sizes_key` names a companion tensor whose entries are cumulative slice
/// sizes per image.
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

/// Read a tensor value as per-image sizes for flat slicing.
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

/// Slice a flat row-major tensor along its first axis.
///
/// The function validates shape/data consistency before slicing so malformed
/// preprocessor outputs become explicit multimodal errors instead of panics.
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
    let expected_len = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            Error::Multimodal(format!("tensor shape {shape:?} has too many elements"))
        })
    })?;
    if expected_len != data.len() {
        return Err(Error::Multimodal(format!(
            "tensor shape {shape:?} expects {expected_len} elements, got {}",
            data.len()
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

// ============================================================================
// Model spec & registry lookup
// ============================================================================

/// Resolve a static model processor spec for one loaded model.
fn lookup_model_spec(
    model_id: &str,
    config: &serde_json::Value,
    tokenizer: &dyn Tokenizer,
) -> Option<&'static dyn ModelProcessorSpec> {
    let resolver = TokenizerResolver { tokenizer };
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &resolver,
        config,
    };
    model_registry().lookup(&metadata)
}

/// Return the process-wide model spec registry.
fn model_registry() -> &'static ModelRegistry {
    static REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);
    &REGISTRY
}

/// Resolve a static image preprocessor for one loaded model.
fn lookup_image_processor(
    model_id: &str,
    model_type: Option<&str>,
) -> Option<&'static dyn ImagePreProcessor> {
    image_processor_registry().find(model_id, model_type)
}

/// Return the process-wide image preprocessor registry.
fn image_processor_registry() -> &'static ImageProcessorRegistry {
    static REGISTRY: LazyLock<ImageProcessorRegistry> =
        LazyLock::new(ImageProcessorRegistry::with_defaults);
    &REGISTRY
}

/// Adapter from the frontend tokenizer trait to `llm-multimodal`.
struct TokenizerResolver<'a> {
    /// Tokenizer used for placeholder/string token lookup.
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

// ============================================================================
// Tests
// ============================================================================

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
        let model_id = "qwen2-vl-test".to_string();
        let config = serde_json::json!({
            "model_type": "qwen2_vl",
            "vision_token_id": 151655
        });
        let spec =
            lookup_model_spec(&model_id, &config, &TestTokenizer).expect("qwen spec should match");
        let image_processor = lookup_image_processor(&model_id, Some("qwen2_vl"))
            .expect("qwen image processor should match");

        MultimodalModelInfo {
            model_id,
            model_type: Some("qwen2_vl".to_string()),
            config,
            preprocessor_config: PreProcessorConfig::default(),
            spec,
            image_processor,
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

        let ranges =
            expand_prompt_tokens(&mut prompt_token_ids, &TestTokenizer, &replacements).unwrap();

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

        let error =
            expand_prompt_tokens(&mut prompt_token_ids, &TestTokenizer, &replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_tokens_uses_each_replacement_placeholder() {
        struct MultiMarkerTokenizer;

        impl Tokenizer for MultiMarkerTokenizer {
            fn encode(
                &self,
                text: &str,
                _add_special_tokens: bool,
            ) -> std::result::Result<Vec<u32>, TokenizerError> {
                Ok(match text {
                    "<image_a>" => vec![900],
                    "<image_b>" => vec![901],
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
                    "<image_a>" => Some(900),
                    "<image_b>" => Some(901),
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

        let mut prompt_token_ids = vec![1, 900, 2, 901, 3];
        let replacements = vec![
            PromptReplacement::sequence(Modality::Image, "<image_a>", vec![151655, 151655]),
            PromptReplacement::sequence(Modality::Image, "<image_b>", vec![151656]),
        ];

        let ranges =
            expand_prompt_tokens(&mut prompt_token_ids, &MultiMarkerTokenizer, &replacements)
                .unwrap();

        assert_eq!(prompt_token_ids, vec![1, 151655, 151655, 2, 151656, 3]);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 2);
        assert_eq!(ranges[1].offset, 4);
        assert_eq!(ranges[1].length, 1);
    }

    #[test]
    fn expand_prompt_tokens_rejects_negative_replacement_token() {
        let mut prompt_token_ids = vec![999];
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<image>",
            vec![-1],
        )];

        let error =
            expand_prompt_tokens(&mut prompt_token_ids, &TestTokenizer, &replacements).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("negative replacement token id"))
        );
    }

    #[test]
    fn slice_first_axis_range_errors_on_shape_data_mismatch() {
        let error = slice_first_axis_range(&[2, 2], &[1.0_f32, 2.0, 3.0], 0, 1, true).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expects 4 elements"))
        );
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
