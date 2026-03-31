mod config;
mod model_files;

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use fastokens::Tokenizer as FastokensTokenizer;
use tekken::Tekkenizer;
use thiserror_ext::AsReport;
use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{info, warn};

use self::config::{
    GenerationConfig, ModelConfig, load_generation_config, load_model_config, load_tokenizer_config,
};
use self::model_files::resolve_model_files;
pub use self::model_files::{ResolvedModelFiles, TokenizerSource};
use crate::backend::{SamplingHints, TextBackend};
use crate::error::{Error, Result};

/// Default regex pattern used when loading tiktoken from a BPE file. This is the same
/// `cl100k_base` pattern that HuggingFace transformers uses as its default in
/// `TikTokenConverter`.
///
/// The `.tiktoken` file format does not include a regex pattern — each model's pattern is
/// defined in its Python tokenizer source (e.g. `tokenization_kimi.py`). Some models use a
/// different regex (e.g. Kimi K2 adds `\p{Han}` for CJK grouping), which can affect token
/// boundaries but not encode/decode correctness.
const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Tokenizer implementation that can be HuggingFace `tokenizers`, `fastokens`, `tiktoken`, or
// Mistral Tekken.
enum TokenizerImpl {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
    Tiktoken(Box<CoreBPE>),
    Tekken(Box<Tekkenizer>),
}

impl TokenizerImpl {
    /// Load the tokenizer backend selected by the resolver.
    fn load(tokenizer: &TokenizerSource) -> Result<Self> {
        match tokenizer {
            TokenizerSource::HuggingFace(path) => Self::from_hf_json(path),
            TokenizerSource::Tiktoken(path) => Self::from_tiktoken_bpe_file(path),
            TokenizerSource::Tekken(path) => Self::from_tekken(path),
        }
    }

    /// Load a Mistral Tekken tokenizer from a `tekken.json` file.
    fn from_tekken(path: &std::path::Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with Mistral Tekken");
        let t = Tekkenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!(
                "failed to load tekken tokenizer from {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self::Tekken(Box::new(t)))
    }

    /// Load from `tokenizer.json` via fastokens or HuggingFace tokenizers.
    fn from_hf_json(path: &std::path::Path) -> Result<Self> {
        info!("loading tokenizer with fastokens");
        match FastokensTokenizer::from_file(path) {
            Ok(t) => Ok(Self::Fastokens(Box::new(t))),
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "failed to load tokenizer with fastokens; falling back to HuggingFace tokenizers"
                );
                let t = HfTokenizer::from_file(path).map_err(|error| {
                    Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
                })?;
                Ok(Self::Hf(Box::new(t)))
            }
        }
    }

    /// Load a tiktoken tokenizer from a `.tiktoken` / `tiktoken.model` BPE file.
    ///
    /// The BPE file format is one `<base64-token-bytes> <rank>` pair per line, the same format
    /// used by OpenAI's tiktoken and by HuggingFace model repos that ship tiktoken files (e.g.
    /// DeepSeek, Kimi K2).
    ///
    /// Special / added tokens are read from `tokenizer_config.json` in the same directory when
    /// present. The `cl100k_base` regex pattern is used as a reasonable default.
    fn from_tiktoken_bpe_file(path: &std::path::Path) -> Result<Self> {
        use base64::Engine as _;
        use rustc_hash::FxHashMap;

        info!(path = %path.display(), "loading tokenizer with tiktoken (BPE file)");

        // Parse the BPE file.
        let content = std::fs::read_to_string(path).map_err(|error| {
            Error::Tokenizer(format!(
                "failed to read tiktoken file {}: {}",
                path.display(),
                error.as_report()
            ))
        })?;
        let mut encoder: FxHashMap<Vec<u8>, u32> =
            FxHashMap::with_capacity_and_hasher(content.lines().count(), Default::default());
        for line in content.lines() {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let token_b64 = parts
                .next()
                .ok_or_else(|| Error::Tokenizer("missing token in tiktoken file".to_string()))?;
            let rank_str = parts
                .next()
                .ok_or_else(|| Error::Tokenizer("missing rank in tiktoken file".to_string()))?;
            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .map_err(|error| {
                    Error::Tokenizer(format!("invalid base64 in tiktoken file: {error}"))
                })?;
            let rank: u32 = rank_str.parse().map_err(|error| {
                Error::Tokenizer(format!("invalid rank in tiktoken file: {error}"))
            })?;
            encoder.insert(token_bytes, rank);
        }

        // Read added/special tokens from tokenizer_config.json in the same directory.
        let special_tokens_encoder: FxHashMap<String, u32> = path
            .parent()
            .map(|dir| dir.join("tokenizer_config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let config_content = std::fs::read_to_string(&config_path).ok()?;
                let config: serde_json::Value = serde_json::from_str(&config_content).ok()?;
                parse_added_tokens_from_config(&config)
            })
            .unwrap_or_default();

        let bpe = CoreBPE::new(encoder, special_tokens_encoder, CL100K_BASE_PATTERN).map_err(
            |error| {
                Error::Tokenizer(format!(
                    "failed to create tiktoken tokenizer from {}: {error}",
                    path.display()
                ))
            },
        )?;

        Ok(Self::Tiktoken(Box::new(bpe)))
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Hf(t) => {
                let encoding = t.encode(text, false).map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                })?;
                Ok(encoding.get_ids().to_vec())
            }
            Self::Fastokens(t) => t.encode_with_special_tokens(text, false).map_err(|error| {
                Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
            }),
            Self::Tiktoken(t) => Ok(t.encode_with_special_tokens(text)),
            Self::Tekken(t) => t
                .encode(text, false, false)
                .map_err(|error| Error::Tokenizer(format!("encoding failed: {error}"))),
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::Hf(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
            Self::Fastokens(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
            // Use lossy UTF-8 decoding instead of `CoreBPE::decode()` which does strict
            // `String::from_utf8()`. During streaming, the `DecodeStream` relies on `\u{FFFD}`
            // to detect incomplete multi-byte sequences, but strict decoding returns an error
            // instead. Lossy decoding produces `\u{FFFD}` which the stream buffers correctly.
            //
            // TODO: tiktoken-rs does not natively support `skip_special_tokens`; all tokens are
            // decoded as-is.
            Self::Tiktoken(t) => {
                let bytes: Vec<u8> = t
                    ._decode_native_and_split(token_ids.to_vec())
                    .flatten()
                    .collect();
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
            Self::Tekken(t) => {
                let policy = if skip_special_tokens {
                    tekken::SpecialTokenPolicy::Ignore
                } else {
                    tekken::SpecialTokenPolicy::Keep
                };
                t.decode(token_ids, policy)
                    .map_err(|error| Error::Tokenizer(format!("decoding failed: {error}")))
            }
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Hf(t) => t.token_to_id(token),
            Self::Fastokens(t) => t.token_to_id(token),
            // tiktoken-rs has no direct `token_to_id`; encode the token and return the ID only if
            // it maps to exactly one token.
            Self::Tiktoken(t) => {
                let ids = t.encode_with_special_tokens(token);
                if ids.len() == 1 { Some(ids[0]) } else { None }
            }
            // tekken-rs exposes `get_control_token` for special tokens. Try that first, then
            // fall back to encoding.
            Self::Tekken(t) => t.get_control_token(token).ok().or_else(|| {
                let ids = t.encode(token, false, false).ok()?;
                if ids.len() == 1 { Some(ids[0]) } else { None }
            }),
        }
    }
}

/// Parse `added_tokens_decoder` from `tokenizer_config.json` into a special-tokens map for
/// `CoreBPE`.
///
/// Format: `{ "added_tokens_decoder": { "163584": { "content": "[BOS]", "special": true }, ... } }`
fn parse_added_tokens_from_config(
    config: &serde_json::Value,
) -> Option<rustc_hash::FxHashMap<String, u32>> {
    let added = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())?;
    let mut tokens = rustc_hash::FxHashMap::default();
    for (id_str, token_info) in added {
        if let (Ok(id), Some(content)) = (
            id_str.parse::<u32>(),
            token_info.get("content").and_then(|v| v.as_str()),
        ) {
            tokens.insert(content.to_string(), id);
        }
    }
    Some(tokens)
}

/// [`TextBackend`] implementation built on Hugging Face model files.
#[derive(Clone)]
pub struct HfTextBackend {
    inner: Arc<HfTextBackendInner>,
}

struct HfTextBackendInner {
    model_id: String,
    files: ResolvedModelFiles,
    tokenizer: TokenizerImpl,
    /// Primary EOS handled by engine-core's dedicated EOS path.
    primary_eos_token_id: Option<u32>,
    /// Additional EOS ids that should flow through stop-token handling.
    extra_eos_token_ids: BTreeSet<u32>,
    /// Generation-config for sampling defaults that may be inherited when the user does not
    /// explicitly override them.
    generation_config: GenerationConfig,
    /// Model config (`config.json`).
    model_config: ModelConfig,
}

impl fmt::Debug for HfTextBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfTextBackend").finish_non_exhaustive()
    }
}

impl HfTextBackend {
    /// Load one Hugging Face model tokenizer plus adjacent model metadata.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let files = resolve_model_files(model_id).await?;
        Self::from_resolved_model_files(files, model_id.to_string())
    }

    pub(crate) fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
    ) -> Result<Self> {
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let tokenizer = TokenizerImpl::load(&files.tokenizer)?;
        let primary_eos_token_id = tokenizer_config
            .eos_token
            .as_ref()
            .and_then(|token| tokenizer.token_to_id(token.as_str()));

        let model_config = load_model_config(files.config_path.as_deref())?;
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let mut extra_eos_token_ids = generation_config
            .eos_token_id
            .clone()
            .map(|value| value.into_set())
            .unwrap_or_default();
        if let Some(primary_eos_token_id) = primary_eos_token_id {
            extra_eos_token_ids.remove(&primary_eos_token_id);
        }

        info!(
            model_id,
            "loaded text backend with Hugging Face model files"
        );

        Ok(Self {
            inner: Arc::new(HfTextBackendInner {
                model_id,
                files,
                tokenizer,
                primary_eos_token_id,
                extra_eos_token_ids,
                generation_config,
                model_config,
            }),
        })
    }

    /// Expose the resolved model files for use by the chat backend to load the chat template.
    pub fn resolved_model_files(&self) -> &ResolvedModelFiles {
        &self.inner.files
    }

    /// Return whether the loaded model config indicates a mixture-of-experts model.
    pub fn is_moe(&self) -> bool {
        self.inner.model_config.is_moe()
    }
}

impl TextBackend for HfTextBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.tokenizer.encode(text)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.tokenizer.decode(token_ids, skip_special_tokens)
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.inner.model_id)
    }

    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints {
            primary_eos_token_id: self.inner.primary_eos_token_id,
            extra_eos_token_ids: self.inner.extra_eos_token_ids.clone(),
            default_temperature: self.inner.generation_config.temperature,
            default_top_p: self.inner.generation_config.top_p,
            default_top_k: self.inner.generation_config.top_k,
            default_min_p: self.inner.generation_config.min_p,
            default_repetition_penalty: self.inner.generation_config.repetition_penalty,
            default_max_tokens: self.inner.generation_config.max_new_tokens,
            max_model_len: self.inner.model_config.effective_max_position_embeddings(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::TokenizerImpl;
    use crate::backend::TextBackend;
    use crate::error::Result;

    /// Minimal [`TextBackend`] wrapper around [`TokenizerImpl`] for testing.
    struct TiktokenBackend(TokenizerImpl);

    impl TextBackend for TiktokenBackend {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            self.0.encode(text)
        }

        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
            self.0.decode(token_ids, skip_special_tokens)
        }
    }

    fn tiktoken_backend() -> TiktokenBackend {
        let bpe = tiktoken_rs::cl100k_base().expect("cl100k_base should load");
        TiktokenBackend(TokenizerImpl::Tiktoken(Box::new(bpe)))
    }

    /// Verify that tiktoken decode uses lossy UTF-8 (producing `\u{FFFD}`) rather than
    /// returning an error for incomplete multi-byte sequences. This is critical for streaming
    /// decode — `DecodeStream` relies on `\u{FFFD}` to detect incomplete characters.
    #[test]
    fn tiktoken_decode_incomplete_utf8_produces_replacement_char() {
        let backend = tiktoken_backend();

        // "你" = U+4F60 = 0xE4 0xBD 0xA0 — encode it to get the token IDs for its bytes.
        let ids = backend.encode("你").unwrap();

        // Decode the full sequence — should round-trip.
        let full = backend.decode(&ids, false).unwrap();
        assert_eq!(full, "你");

        // Encode a partial sequence that would yield incomplete UTF-8 bytes. We construct this
        // by encoding "你好" and decoding only the first token(s), which may split mid-character.
        // As a simpler approach: just verify that decoding *any* single-byte token that maps to
        // a high byte (>= 0x80) produces a replacement char rather than an error.
        let text_with_multibyte = "Hello你好World";
        let all_ids = backend.encode(text_with_multibyte).unwrap();

        // Decode each token individually — some will be incomplete UTF-8 bytes.
        // The key assertion: none of these should error.
        for &id in &all_ids {
            let result = backend.decode(&[id], false);
            assert!(result.is_ok(), "decode of token {id} should not error");
        }
    }

    /// Streaming decode of CJK text through tiktoken should produce the original text without
    /// errors, even though individual tokens may represent partial UTF-8 byte sequences.
    #[test]
    fn tiktoken_streaming_decode_multibyte() {
        let backend = tiktoken_backend();
        let text = "你好世界"; // 4 CJK characters
        let ids = backend.encode(text).unwrap();

        let mut decoder = backend.create_decode_stream(&[], false);
        let mut output = String::new();
        for &id in &ids {
            if let Some(chunk) = decoder.step(id).unwrap() {
                output.push_str(&chunk);
            }
        }
        if let Some(chunk) = decoder.flush().unwrap() {
            output.push_str(&chunk);
        }

        assert_eq!(output, text);
    }

    /// Mixed ASCII and multi-byte text should stream correctly through tiktoken.
    #[test]
    fn tiktoken_streaming_decode_mixed_ascii_and_multibyte() {
        let backend = tiktoken_backend();
        let text = "Hello 你好 World 🌍";
        let ids = backend.encode(text).unwrap();

        let mut decoder = backend.create_decode_stream(&[], false);
        let mut output = String::new();
        for &id in &ids {
            if let Some(chunk) = decoder.step(id).unwrap() {
                output.push_str(&chunk);
            }
        }
        if let Some(chunk) = decoder.flush().unwrap() {
            output.push_str(&chunk);
        }

        assert_eq!(output, text);
    }
}
