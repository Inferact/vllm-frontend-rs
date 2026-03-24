use std::sync::Arc;

use crate::error::Result;

/// Tokenizer/model-derived hints used to enrich text-generation requests before they are lowered
/// into engine-core.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingHints {
    pub primary_eos_token_id: Option<u32>,
    pub extra_eos_token_ids: std::collections::BTreeSet<u32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<i32>,
    pub default_min_p: Option<f32>,
    pub default_repetition_penalty: Option<f32>,
    pub default_max_tokens: Option<u32>,
    /// Model context window size (`max_position_embeddings` from `config.json`).
    pub max_model_len: Option<u32>,
}

/// Stateful incremental decoder that emits text chunks one token at a time.
pub trait IncrementalDecoder: Send {
    /// Push one generated token and return the newly decoded text chunk, if any.
    ///
    /// Returns `Ok(None)` when the token does not yet produce a stable text fragment (e.g. in the
    /// middle of a multi-byte UTF-8 sequence).
    fn step(&mut self, token_id: u32) -> Result<Option<String>>;

    /// Flush any remaining buffered text that has not yet been emitted.
    ///
    /// Called after the final generated token to force out incomplete fragments.
    fn flush(&mut self) -> Result<Option<String>>;
}

/// Minimal text-processing backend needed by `vllm-text`.
pub trait TextBackend: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Create a stateful incremental decoder primed with the given prompt tokens.
    ///
    /// The prompt tokens provide left context for the first generated token; the decoder does not
    /// re-emit prompt text.
    ///
    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream {
            backend: self,
            skip_special_tokens,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
        })
    }

    /// Return the backend model ID when available.
    fn model_id(&self) -> Option<&str> {
        None
    }

    /// Return tokenizer/model-derived hints used to enrich southbound sampling parameters.
    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints::default())
    }
}

/// Shared trait-object form of [`TextBackend`].
pub type DynTextBackend = Arc<dyn TextBackend>;

/// [`IncrementalDecoder`] built on [`TextBackend::decode()`] with prefix-diffing.
///
/// This is the same sliding-window algorithm used by `tokenizers::DecodeStream` and
/// `fastokens::DecodeStream`.
struct DecodeStream<'a, B: TextBackend + ?Sized> {
    backend: &'a B,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl<B: TextBackend + ?Sized> IncrementalDecoder for DecodeStream<'_, B> {
    fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let new_prefix = self.backend.decode(&self.ids, self.skip_special_tokens)?;
            if !new_prefix.ends_with('\u{FFFD}') {
                self.prefix = new_prefix;
                self.prefix_index = self.ids.len();
            }
        }

        self.ids.push(token_id);
        let string = self.backend.decode(&self.ids, self.skip_special_tokens)?;
        if string.len() > self.prefix.len() && !string.ends_with('\u{FFFD}') {
            let new_text = string[self.prefix.len()..].to_string();
            let new_prefix_index = self.ids.len() - self.prefix_index;
            self.ids = self.ids.drain(self.prefix_index..).collect();
            self.prefix = self.backend.decode(&self.ids, self.skip_special_tokens)?;
            self.prefix_index = new_prefix_index;
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }

    fn flush(&mut self) -> Result<Option<String>> {
        if self.ids.is_empty() {
            return Ok(None);
        }
        let text = self.backend.decode(&self.ids, self.skip_special_tokens)?;
        let remaining = &text[self.prefix.len()..];
        self.ids.clear();
        self.prefix.clear();
        if remaining.is_empty() { Ok(None) } else { Ok(Some(remaining.to_string())) }
    }
}
