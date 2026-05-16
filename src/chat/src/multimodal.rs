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

use itertools::izip;
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
#[derive(Clone)]
pub struct MultimodalModelInfo {
    context: MultimodalModelContext,
    spec: ResolvedMultimodalSpec,
    image_processor: ResolvedImageProcessor,
    media_connector: Arc<MediaConnector>,
}

/// Model metadata and tokenizer access shared by all multimodal specs.
#[derive(Clone)]
struct MultimodalModelContext {
    model_id: String,
    model_type: Option<String>,
    config: serde_json::Value,
    tokenizer: TokenizerResolver,
}

impl MultimodalModelContext {
    fn metadata(&self) -> ModelMetadata<'_> {
        ModelMetadata {
            model_id: &self.model_id,
            tokenizer: &self.tokenizer,
            config: &self.config,
        }
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.0.as_ref()
    }

    /// Resolve a static model processor spec for one loaded model.
    fn resolve_model_spec(&self) -> Option<&'static dyn ModelProcessorSpec> {
        static REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);
        REGISTRY.lookup(&self.metadata())
    }

    /// Resolve a static image preprocessor for one loaded model.
    fn resolve_image_processor(&self) -> Option<&'static dyn ImagePreProcessor> {
        static REGISTRY: LazyLock<ImageProcessorRegistry> =
            LazyLock::new(ImageProcessorRegistry::new);
        REGISTRY.find(&self.model_id, self.model_type.as_deref())
    }
}

/// Static model-specific prompt and tensor-layout behavior.
#[derive(Clone)]
struct ResolvedMultimodalSpec {
    raw: &'static dyn ModelProcessorSpec,
    placeholder_token: String,
    placeholder_marker_token_id: u32,
    field_layouts: HashMap<String, FieldLayout>,
    keep_on_cpu_keys: HashSet<String>,
}

impl ResolvedMultimodalSpec {
    fn new(raw: &'static dyn ModelProcessorSpec, context: &MultimodalModelContext) -> Result<Self> {
        let metadata = context.metadata();
        let placeholder_token = raw
            .placeholder_token(&metadata)
            .map_err(|error| Error::Multimodal(error.to_string()))?;
        // This is the rendered prompt marker, so resolve it from the token
        // string itself. Do not use `ModelProcessorSpec::placeholder_token_id()`:
        // for specs such as Qwen2-VL and Llama 4 that ID is the replacement
        // vision/pad/patch token, not necessarily the token ID of
        // `placeholder_token`.
        let placeholder_marker_token_id =
            context.tokenizer().token_to_id(&placeholder_token).ok_or_else(|| {
                Error::Multimodal(format!(
                    "placeholder token `{placeholder_token}` is not in the tokenizer vocabulary"
                ))
            })?;

        Ok(Self {
            raw,
            placeholder_token,
            placeholder_marker_token_id,
            field_layouts: raw.field_layouts(),
            keep_on_cpu_keys: raw.keep_on_cpu_keys().into_iter().collect(),
        })
    }

    fn prompt_replacements(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedImages,
    ) -> Result<Vec<PromptReplacement>> {
        self.raw
            .prompt_replacements(&context.metadata(), preprocessed)
            .map_err(|error| Error::Multimodal(error.to_string()))
    }
}

/// Static image preprocessor plus its loaded config.
#[derive(Clone)]
struct ResolvedImageProcessor {
    raw: &'static dyn ImagePreProcessor,
    config: PreProcessorConfig,
}

/// Request-scoped fetched media, kept together with tracker UUID metadata.
struct FetchedImageMedia {
    frames: Vec<Arc<llm_multimodal::ImageFrame>>,
    uuids: Vec<Option<String>>,
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
        tokenizer: DynTokenizer,
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

        let context = MultimodalModelContext {
            model_id,
            model_type,
            config,
            tokenizer: TokenizerResolver(tokenizer),
        };

        let Some(spec) = context.resolve_model_spec() else {
            warn!(
                model_id = %context.model_id,
                model_type = ?context.model_type,
                "multimodal model spec is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };
        let spec = ResolvedMultimodalSpec::new(spec, &context)?;

        let Some(image_processor) = context.resolve_image_processor() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "image processor is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };

        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
                .map_err(|error| Error::Multimodal(error.to_string()))?,
        );

        Ok(Some(Self {
            context,
            spec,
            image_processor: ResolvedImageProcessor {
                raw: image_processor,
                config: preprocessor_config,
            },
            media_connector,
        }))
    }

    /// Return the template-visible placeholder token for this model.
    ///
    /// The HF renderer uses this token while flattening image content in string
    /// content format.
    pub(crate) fn placeholder_token(&self) -> &str {
        &self.spec.placeholder_token
    }
}

/// Finalize a rendered chat prompt into text-generation input.
///
/// Text-only requests pass through unchanged as `Prompt::Text`. Multimodal
/// requests are tokenized in chat, their image placeholders are expanded, and
/// preprocessed image features are attached for engine-core transport.
pub(crate) async fn finalize_rendered_prompt(
    request: &ChatRequest,
    rendered: RenderedPrompt,
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

    let mut prompt_token_ids = info
        .context
        .tokenizer()
        .encode(&prompt, request.add_special_tokens)
        .map_err(|error| Error::Multimodal(error.to_string()))?;
    let prepared = info.prepare_multimodal(media_parts, &mut prompt_token_ids).await?;

    Ok((Prompt::TokenIds(prompt_token_ids), Some(prepared)))
}

/// Extract image media parts from chat messages in message/content order.
///
/// Assistant history is skipped because generated assistant blocks are already
/// represented as text for prompt rendering in this crate.
fn extract_media_parts(request: &ChatRequest) -> Result<Vec<MediaContentPart>> {
    let mut all_parts = Vec::new();
    for message in &request.messages {
        let content = match message {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        let ChatContent::Parts(parts) = content else {
            continue;
        };
        for part in parts {
            match part {
                ChatContentPart::Text { .. } => {}
                ChatContentPart::ImageUrl {
                    image_url,
                    detail,
                    uuid,
                } => all_parts.push(MediaContentPart::ImageUrl {
                    url: image_url.clone(),
                    detail: *detail,
                    uuid: uuid.clone(),
                }),
            }
        }
    }
    Ok(all_parts)
}

impl MultimodalModelInfo {
    /// Run media fetch, image preprocessing, prompt expansion, and feature
    /// build.
    ///
    /// `prompt_token_ids` is mutated in place because placeholder expansion
    /// changes both the final prompt and the offsets recorded in
    /// `PlaceholderRange`.
    async fn prepare_multimodal(
        &self,
        media_parts: Vec<MediaContentPart>,
        prompt_token_ids: &mut Vec<u32>,
    ) -> Result<MultiModalFeatures> {
        if media_parts.is_empty() {
            return Ok(Vec::new());
        }

        let fetched = self.fetch_images(&media_parts).await?;
        let preprocessed = self.preprocess_images(&fetched.frames).await?;
        let replacements = self.spec.prompt_replacements(&self.context, &preprocessed)?;
        let ranges = self.expand_prompt_tokens(prompt_token_ids, &replacements)?;

        let features = self.build_features(preprocessed, fetched, ranges)?;
        if features.len() != media_parts.len() {
            return Err(Error::Multimodal(format!(
                "number of built multimodal features {} does not match number of media parts {}",
                features.len(),
                media_parts.len()
            )));
        }
        Ok(features)
    }

    /// Fetch all image parts and preserve their request-order UUID metadata.
    async fn fetch_images(&self, media_parts: &[MediaContentPart]) -> Result<FetchedImageMedia> {
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&self.media_connector));
        for part in media_parts {
            tracker
                .push_part(part.clone())
                .map_err(|error| Error::Multimodal(error.to_string()))?;
        }

        let tracker_output =
            tracker.finalize().await.map_err(|error| Error::Multimodal(error.to_string()))?;
        let images = tracker_output.data.get(&Modality::Image).cloned().unwrap_or_default();
        let uuids = tracker_output.uuids.get(&Modality::Image).cloned().unwrap_or_default();

        let frames = images
            .into_iter()
            .map(|media| match media {
                TrackedMedia::Image(frame) => Ok(frame),
                _ => Err(Error::UnsupportedMultimodalContent("non-image")),
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(FetchedImageMedia { frames, uuids })
    }

    /// Preprocess fetched image frames with the model's resolved image
    /// processor.
    ///
    /// The processor work is CPU-heavy relative to request wiring, so it runs
    /// in a blocking task and returns owned tensors ready for wire
    /// conversion.
    async fn preprocess_images(
        &self,
        image_frames: &[Arc<llm_multimodal::ImageFrame>],
    ) -> Result<PreprocessedImages> {
        let config = self.image_processor.config.clone();
        let processor = self.image_processor.raw;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        tokio::task::spawn_blocking(move || {
            processor
                .preprocess(&images, &config)
                .map_err(|error| Error::Multimodal(error.to_string()))
        })
        .await
        .map_err(|error| Error::Multimodal(format!("image preprocessing task failed: {error}")))?
    }

    /// Replace rendered placeholder markers with model-specific replacement
    /// tokens.
    ///
    /// Replacements are consumed in order, matching the original media-part
    /// order. The returned ranges point into the already-expanded prompt.
    fn expand_prompt_tokens(
        &self,
        prompt_token_ids: &mut Vec<u32>,
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
            let offset = find_next_token(
                prompt_token_ids,
                self.spec.placeholder_marker_token_id,
                cursor,
            )
            .ok_or_else(|| {
                Error::Multimodal(format!(
                    "placeholder token `{}` was not found in tokenized prompt",
                    self.spec.placeholder_token
                ))
            })?;
            let replacement_tokens =
                replacement.tokens.iter().map(|&token| token as u32).collect::<Vec<_>>();
            if replacement_tokens.is_empty() {
                return Err(Error::Multimodal(format!(
                    "placeholder token `{}` expanded to no tokens",
                    self.spec.placeholder_token
                )));
            }
            let replacement_len = replacement_tokens.len();
            prompt_token_ids.splice(offset..offset + 1, replacement_tokens);
            ranges.push(PlaceholderRange {
                offset,
                length: replacement_len,
                is_embed: None,
            });
            cursor = offset + replacement_len;
        }
        Ok(ranges)
    }

    /// Convert preprocessed image tensors into engine-core multimodal features.
    ///
    /// One `MultiModalFeatureSpec` is produced per image. Tensor fields are
    /// sliced according to the model spec's field layout declarations.
    fn build_features(
        &self,
        preprocessed: PreprocessedImages,
        images: FetchedImageMedia,
        ranges: Vec<PlaceholderRange>,
    ) -> Result<MultiModalFeatures> {
        let len = images.frames.len();
        let tensors = collect_tensors(&preprocessed);

        let mut features = Vec::with_capacity(images.frames.len());
        for (index, (frame, uuid, range)) in izip!(images.frames, images.uuids, ranges).enumerate()
        {
            let mut data = MultiModalKwargsItem::new();
            for (key, tensor) in &tensors {
                let keep_on_cpu = self.spec.keep_on_cpu_keys.contains(key);
                let (value, field) = match self.spec.field_layouts.get(key) {
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
                            batch_size: len,
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

            let hash = frame.hash.clone();
            features.push(MultiModalFeatureSpec {
                data: Some(data),
                modality: "image".to_string(),
                identifier: uuid.unwrap_or_else(|| hash.clone()),
                mm_position: range,
                mm_hash: Some(hash),
            });
        }

        Ok(features)
    }
}

/// Find `needle` in `haystack`, starting at `start`.
///
/// This is intentionally order-preserving rather than a global replace: each
/// image consumes the next placeholder occurrence.
fn find_next_token(haystack: &[u32], needle: u32, start: usize) -> Option<usize> {
    haystack
        .get(start..)?
        .iter()
        .position(|token| *token == needle)
        .map(|offset| start + offset)
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

/// Adapter from the frontend tokenizer trait to `llm-multimodal`.
#[derive(Clone)]
struct TokenizerResolver(DynTokenizer);

impl TokenResolver for TokenizerResolver {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
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
        let context = MultimodalModelContext {
            model_id,
            model_type: Some("qwen2_vl".to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(TestTokenizer)),
        };
        let spec = context.resolve_model_spec().expect("qwen spec should match");
        let spec = ResolvedMultimodalSpec::new(spec, &context).unwrap();
        let raw_image_processor =
            context.resolve_image_processor().expect("qwen image processor should match");
        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default()).unwrap(),
        );

        MultimodalModelInfo {
            context,
            spec,
            image_processor: ResolvedImageProcessor {
                raw: raw_image_processor,
                config: PreProcessorConfig::default(),
            },
            media_connector,
        }
    }

    #[test]
    fn resolves_qwen_placeholder_token() {
        let info = qwen_info();
        let placeholder = info.placeholder_token();

        assert_eq!(placeholder, "<image>");
    }

    #[test]
    fn expand_prompt_tokens_replaces_placeholder_marker() {
        let info = qwen_info();
        let mut prompt_token_ids = vec![1, 999, 2];
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<image>",
            vec![151655, 151655, 151655],
        )];

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, &replacements).unwrap();

        assert_eq!(prompt_token_ids, vec![1, 151655, 151655, 151655, 2]);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 3);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_placeholder_missing() {
        let info = qwen_info();
        let mut prompt_token_ids = vec![1, 2, 3];
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<image>",
            vec![151655],
        )];

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, &replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_tokens_uses_cached_model_placeholder() {
        let info = qwen_info();
        let mut prompt_token_ids = vec![1, 999, 2, 999, 3];
        let replacements = vec![
            PromptReplacement::sequence(Modality::Image, "<image>", vec![151655, 151655]),
            PromptReplacement::sequence(Modality::Image, "<image>", vec![151656]),
        ];

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, &replacements).unwrap();

        assert_eq!(prompt_token_ids, vec![1, 151655, 151655, 2, 151656, 3]);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 2);
        assert_eq!(ranges[1].offset, 4);
        assert_eq!(ranges[1].length, 1);
    }

    #[test]
    fn expand_prompt_tokens_rejects_negative_replacement_token() {
        let info = qwen_info();
        let mut prompt_token_ids = vec![999];
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<image>",
            vec![-1],
        )];

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, &replacements).unwrap_err();

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
        let info = qwen_info();
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
            finalize_rendered_prompt(&request, rendered, Some(&info)).await.unwrap();

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
