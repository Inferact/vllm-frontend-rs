mod cohere_cmd;
mod deepseek_r1;
mod delimited;
mod kimi;
mod qwen3;

use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;
use vllm_text::tokenizers::Tokenizer;

pub use self::cohere_cmd::CohereCmdReasoningParser;
pub use self::deepseek_r1::DeepSeekR1ReasoningParser;
pub(crate) use self::delimited::DelimitedReasoningParser;
pub use self::kimi::KimiReasoningParser;
pub use self::qwen3::Qwen3ReasoningParser;

/// Standard `<think>...</think>` parser used by several model families.
pub type BaseReasoningParser = Qwen3ReasoningParser;
/// DeepSeek V3.1 currently shares the standard `<think>...</think>` parser.
pub type DeepSeekV31ReasoningParser = Qwen3ReasoningParser;
/// GLM45 currently shares the standard `<think>...</think>` parser.
pub type Glm45ReasoningParser = Qwen3ReasoningParser;
/// Kimi K2.5 currently shares the standard `<think>...</think>` parser.
pub type KimiK25ReasoningParser = Qwen3ReasoningParser;
/// Kimi thinking mode currently shares the standard `<think>...</think>` parser.
pub type KimiThinkingReasoningParser = Qwen3ReasoningParser;
/// MiniMax currently shares the standard `<think>...</think>` parser.
pub type MiniMaxReasoningParser = Qwen3ReasoningParser;
/// Nano V3 currently shares the standard `<think>...</think>` parser.
pub type NanoV3ReasoningParser = Qwen3ReasoningParser;
/// Qwen thinking mode currently shares the standard `<think>...</think>` parser.
pub type Qwen3ThinkingReasoningParser = Qwen3ReasoningParser;
/// Step3 currently shares the standard `<think>...</think>` parser.
pub type Step3ReasoningParser = Qwen3ReasoningParser;

/// Result alias for reasoning parser operations.
pub type Result<T> = std::result::Result<T, ReasoningError>;

/// One parsed streaming delta split into reasoning and visible content.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: Option<String>,
    pub content: Option<String>,
}

impl ReasoningDelta {
    /// Return true when this delta carries neither reasoning nor content text.
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }

    /// Append text to the reasoning portion, creating it on first use.
    pub(crate) fn push_reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.reasoning {
            Some(existing) => existing.push_str(text),
            None => self.reasoning = Some(text.to_string()),
        }
    }

    /// Append text to the visible content portion, creating it on first use.
    pub(crate) fn push_content(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.content {
            Some(existing) => existing.push_str(text),
            None => self.content = Some(text.to_string()),
        }
    }
}

/// Incremental parser that splits decoded text deltas into reasoning and content.
pub trait ReasoningParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tokenizer: &dyn Tokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static;

    /// Initialize parser state from prompt token IDs before output deltas arrive.
    fn initialize(&mut self, _prompt_token_ids: &[u32]) -> Result<()> {
        Ok(())
    }

    /// Feed one decoded text delta into the parser.
    fn push(&mut self, delta: &str) -> Result<ReasoningDelta>;

    /// Flush any buffered partial delimiter state at end of stream.
    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(ReasoningDelta::default())
    }
}

/// Errors produced while creating or running reasoning parsers.
#[derive(Debug, Error)]
pub enum ReasoningError {
    #[error(
        "reasoning parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    UnknownParser {
        name: String,
        available_names: Vec<String>,
    },
    #[error("reasoning parsing is not available for model `{model_id}`")]
    UnknownModel { model_id: String },
    #[error("tokenizer is missing reasoning delimiter token `{token}`")]
    MissingToken { token: String },
}

/// Format the available-parser suffix used in user-facing error messages.
fn available_parser_hint(available_names: &[String]) -> String {
    if available_names.is_empty() {
        String::new()
    } else {
        format!(" (choose from: {})", available_names.join(", "))
    }
}

/// Constructor signature for one registered reasoning parser implementation.
type ParserCreator = Arc<dyn Fn(&dyn Tokenizer) -> Result<Box<dyn ReasoningParser>> + Send + Sync>;

/// Registry and model matcher for reasoning stream parsers.
#[derive(Clone, Default)]
pub struct ReasoningParserFactory {
    creators: HashMap<String, ParserCreator>,
    patterns: Vec<(String, String)>,
}

impl ReasoningParserFactory {
    /// Create the default registry with built-in parser names and model mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();
        factory.register_parser_type::<BaseReasoningParser>("base");
        factory.register_parser_type::<CohereCmdReasoningParser>("cohere_cmd");
        factory.register_parser_type::<DeepSeekR1ReasoningParser>("deepseek_r1");
        factory.register_parser_type::<DeepSeekV31ReasoningParser>("deepseek_v31");
        factory.register_parser_type::<Glm45ReasoningParser>("glm45");
        factory.register_parser_type::<KimiReasoningParser>("kimi");
        factory.register_parser_type::<KimiK25ReasoningParser>("kimi_k25");
        factory.register_parser_type::<KimiThinkingReasoningParser>("kimi_thinking");
        factory.register_parser_type::<MiniMaxReasoningParser>("minimax");
        factory.register_parser_type::<NanoV3ReasoningParser>("nano_v3");
        factory.register_parser_type::<Qwen3ReasoningParser>("qwen3");
        factory.register_parser_type::<Qwen3ThinkingReasoningParser>("qwen3_thinking");
        factory.register_parser_type::<Step3ReasoningParser>("step3");

        factory.register_pattern("deepseek-r1", "deepseek_r1");
        factory.register_pattern("deepseek-v3.1", "deepseek_v31");
        factory.register_pattern("deepseek-v3-1", "deepseek_v31");
        factory.register_pattern("qwen3-thinking", "qwen3_thinking");
        factory.register_pattern("qwen-thinking", "qwen3_thinking");
        factory.register_pattern("qwen3", "qwen3");
        factory.register_pattern("qwen", "qwen3");
        factory.register_pattern("glm45", "glm45");
        factory.register_pattern("glm47", "glm45");
        factory.register_pattern("kimi-k2-thinking", "kimi_thinking");
        factory.register_pattern("kimi-k2.5", "kimi_k25");
        factory.register_pattern("kimi", "kimi");
        factory.register_pattern("step3", "step3");
        factory.register_pattern("minimax", "minimax");
        factory.register_pattern("minimax-m2", "minimax");
        factory.register_pattern("mm-m2", "minimax");
        factory.register_pattern("cohere", "cohere_cmd");
        factory.register_pattern("command", "cohere_cmd");
        factory.register_pattern("nano", "nano_v3");
        factory.register_pattern("nemotron", "nano_v3");

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser_type<T>(&mut self, name: &str)
    where
        T: ReasoningParser + 'static,
    {
        self.creators.insert(name.to_string(), Arc::new(T::create));
    }

    /// Add a case-insensitive substring match from model ID to parser name.
    pub fn register_pattern(&mut self, pattern: &str, parser_name: &str) {
        self.patterns
            .push((pattern.to_string(), parser_name.to_string()));
    }

    /// Return the first registered parser name matching the given model ID.
    pub fn find_parser_for_model(&self, model_id: &str) -> Option<String> {
        let model_lower = model_id.to_lowercase();
        self.patterns
            .iter()
            .find(|(pattern, _)| model_lower.contains(&pattern.to_lowercase()))
            .map(|(_, parser_name)| parser_name.clone())
    }

    /// Return true if the exact parser name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.creators.contains_key(name)
    }

    /// Return all registered parser names sorted for stable display.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<_> = self.creators.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<Box<dyn ReasoningParser>> {
        let creator = self
            .creators
            .get(name)
            .ok_or_else(|| ReasoningError::UnknownParser {
                name: name.to_string(),
                available_names: self.list(),
            })?;
        creator(tokenizer)
    }

    /// Resolve a parser from model ID and then construct it.
    pub fn create_for_model(
        &self,
        model_id: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<Box<dyn ReasoningParser>> {
        let parser_name =
            self.find_parser_for_model(model_id)
                .ok_or_else(|| ReasoningError::UnknownModel {
                    model_id: model_id.to_string(),
                })?;
        self.create(&parser_name, tokenizer)
    }
}

#[cfg(test)]
mod tests;
