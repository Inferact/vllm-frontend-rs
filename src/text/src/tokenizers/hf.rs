use std::path::Path;
use std::sync::Arc;

use fastokens::Tokenizer as FastokensTokenizer;
use thiserror_ext::AsReport as _;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{info, warn};

use crate::Error;
use crate::error::Result;
use crate::tokenizers::Tokenizer;

/// Tokenizer from `tokenizer.json` in HuggingFace format.
enum Backend {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
}

pub struct HuggingFaceTokenizer {
    backend: Backend,
    special_token_ids: Arc<[u32]>,
}

impl HuggingFaceTokenizer {
    fn from_hf_backend(tokenizer: HfTokenizer) -> Self {
        let special_token_ids = collect_special_token_ids(&tokenizer);
        Self {
            backend: Backend::Hf(Box::new(tokenizer)),
            special_token_ids,
        }
    }

    fn from_fastokens_backend(
        tokenizer: FastokensTokenizer,
        special_token_ids: Arc<[u32]>,
    ) -> Self {
        Self {
            backend: Backend::Fastokens(Box::new(tokenizer)),
            special_token_ids,
        }
    }

    /// Load from `tokenizer.json` with `fastokens`.
    ///
    /// This always loads HuggingFace's `tokenizers` first so we can inspect the
    /// complete added-token metadata, then optionally upgrades the execution backend
    /// to `fastokens` for better performance.
    // TODO: once `fastokens` supports added-token metadata, we can simplify this by loading
    // directly with `fastokens` and only falling back to `tokenizers` if loading fails (e.g. due to
    // unsupported tokenizer features).
    pub fn new_fastokens(path: &Path) -> Result<Self> {
        info!(
            path = %path.display(),
            "loading tokenizer metadata with huggingface tokenizers"
        );
        let hf_tokenizer = HfTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;
        let special_token_ids = collect_special_token_ids(&hf_tokenizer);

        info!(path = %path.display(), "loading tokenizer with fastokens");
        let t = FastokensTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;
        Ok(Self::from_fastokens_backend(t, special_token_ids))
    }

    /// Load from `tokenizer.json` with Hugging Face `tokenizers`.
    pub fn new_hf(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with huggingface tokenizers");
        let t = HfTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;
        Ok(Self::from_hf_backend(t))
    }

    /// Load from `tokenizer.json` via fastokens or HuggingFace tokenizers.
    pub fn new(path: &Path) -> Result<Self> {
        info!(
            path = %path.display(),
            "loading tokenizer metadata with huggingface tokenizers"
        );
        let hf_tokenizer = HfTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;

        match FastokensTokenizer::from_file(path) {
            Ok(tokenizer) => {
                info!(path = %path.display(), "upgrading tokenizer backend to fastokens");
                let special_token_ids = collect_special_token_ids(&hf_tokenizer);
                Ok(Self::from_fastokens_backend(tokenizer, special_token_ids))
            }
            Err(error) => {
                warn!(
                    path = %path.display(),
                    error = %error.as_report(),
                    "failed to load tokenizer with fastokens; falling back to HuggingFace tokenizers"
                );
                Ok(Self::from_hf_backend(hf_tokenizer))
            }
        }
    }
}

fn collect_special_token_ids(tokenizer: &HfTokenizer) -> Arc<[u32]> {
    let mut ids: Vec<u32> = tokenizer
        .get_added_tokens_decoder()
        .iter()
        .filter(|(_id, token)| token.special)
        .map(|(id, _token)| *id)
        .collect();

    ids.sort_unstable();
    ids.dedup();
    Arc::from(ids)
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match &self.backend {
            Backend::Hf(t) => {
                let encoding = t.encode(text, add_special_tokens).map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                })?;
                Ok(encoding.get_ids().to_vec())
            }
            Backend::Fastokens(t) => t
                .encode_with_special_tokens(text, add_special_tokens)
                .map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                }),
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match &self.backend {
            Backend::Hf(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
            Backend::Fastokens(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.backend {
            Backend::Hf(t) => t.token_to_id(token),
            Backend::Fastokens(t) => t.token_to_id(token),
        }
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.special_token_ids.binary_search(&token_id).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

    use super::{HuggingFaceTokenizer, Tokenizer};

    #[test]
    fn hf_constructor_resolves_added_token_ids() {
        let model = WordLevel::builder()
            .vocab(
                [("<unk>".to_string(), 0u32), ("hello".to_string(), 1u32)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("<unk>".to_string())
            .build()
            .expect("build wordlevel tokenizer");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");

        let wrapper = HuggingFaceTokenizer::new_hf(&path).expect("load hf wrapper");
        assert_eq!(wrapper.token_to_id("<|im_end|>"), Some(2));
    }

    #[test]
    fn new_preserves_special_ids_when_fastokens_is_used() {
        let model = WordLevel::builder()
            .vocab(
                [("<unk>".to_string(), 0u32), ("hello".to_string(), 1u32)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("<unk>".to_string())
            .build()
            .expect("build wordlevel tokenizer");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");

        let wrapper = HuggingFaceTokenizer::new(&path).expect("load wrapper with metadata");
        assert!(wrapper.is_special_id(2));
    }
}
