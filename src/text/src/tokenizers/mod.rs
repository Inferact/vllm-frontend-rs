use crate::error::Result;

mod hf;
mod tekken;
mod tiktoken;

pub use hf::HuggingFaceTokenizer;
pub use tekken::TekkenTokenizer;
pub use tiktoken::TiktokenTokenizer;

pub trait Tokenizer {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Convert one token string into a token ID, returning `None` if the token is not in the
    /// tokenizer vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;
}

pub type DynTokenizer = Box<dyn Tokenizer + Send + Sync>;
