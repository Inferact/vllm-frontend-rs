use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;
use vllm_text::tokenizers::Tokenizer;

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
    fn push_reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.reasoning {
            Some(existing) => existing.push_str(text),
            None => self.reasoning = Some(text.to_string()),
        }
    }

    /// Append text to the visible content portion, creating it on first use.
    fn push_content(&mut self, text: &str) {
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

/// Factory for model-aware reasoning stream parsers.
pub trait ReasoningStreamParserFactory: Send + Sync {
    /// Return true if a parser with this exact registered name exists.
    fn contains(&self, name: &str) -> bool;

    /// Return all registered parser names in display-ready form.
    fn list(&self) -> Vec<String>;

    /// Create a parser by exact registered name.
    fn create(&self, name: &str, tokenizer: &dyn Tokenizer) -> Result<Box<dyn ReasoningParser>>;

    /// Resolve and create a parser from a model identifier.
    fn create_for_model(
        &self,
        model_id: &str,
        tokenizer: &dyn Tokenizer,
    ) -> Result<Box<dyn ReasoningParser>>;
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

/// Generic streaming parser for reasoning delimited by explicit start/end tags.
pub struct DelimitedReasoningParser {
    current_in_reasoning: bool,
    buffer: String,
    start_token: String,
    end_token: String,
    start_token_id: u32,
    end_token_id: u32,
}

impl DelimitedReasoningParser {
    /// Create a delimited parser with deferred prompt-based initialization.
    pub fn new(
        tokenizer: &dyn Tokenizer,
        start_token: &'static str,
        end_token: &'static str,
    ) -> Result<Self> {
        let start_token_id =
            tokenizer
                .token_to_id(start_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: start_token.to_string(),
                })?;
        let end_token_id =
            tokenizer
                .token_to_id(end_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: end_token.to_string(),
                })?;

        Ok(Self {
            current_in_reasoning: false,
            buffer: String::new(),
            start_token: start_token.to_string(),
            end_token: end_token.to_string(),
            start_token_id,
            end_token_id,
        })
    }

    /// Parse text that is known not to end with a partial delimiter suffix.
    fn parse_stable_text(&mut self, mut stable: &str) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();

        while !stable.is_empty() {
            if self.current_in_reasoning {
                if let Some(end_idx) = stable.find(&self.end_token) {
                    delta.push_reasoning(&stable[..end_idx]);
                    stable = &stable[end_idx + self.end_token.len()..];
                    self.current_in_reasoning = false;
                } else {
                    delta.push_reasoning(stable);
                    break;
                }
            } else if let Some(start_idx) = stable.find(&self.start_token) {
                delta.push_content(&stable[..start_idx]);
                stable = &stable[start_idx + self.start_token.len()..];
                self.current_in_reasoning = true;
            } else {
                delta.push_content(stable);
                break;
            }
        }

        delta
    }

    /// Return the longest trailing suffix that could still complete a delimiter.
    fn partial_suffix_len(&self, text: &str) -> usize {
        let mut best = 0;
        for idx in text.char_indices().map(|(idx, _)| idx).skip(1) {
            let suffix = &text[idx..];
            if self.start_token.starts_with(suffix) && self.start_token != suffix {
                best = best.max(text.len() - idx);
            }
            if self.end_token.starts_with(suffix) && self.end_token != suffix {
                best = best.max(text.len() - idx);
            }
        }

        if self.start_token.starts_with(text) && self.start_token != text {
            best = best.max(text.len());
        }
        if self.end_token.starts_with(text) && self.end_token != text {
            best = best.max(text.len());
        }

        best
    }
}

impl ReasoningParser for DelimitedReasoningParser {
    /// Initialize parser state from the last relevant reasoning boundary in the prompt.
    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.current_in_reasoning =
            last_reasoning_boundary(prompt_token_ids, self.start_token_id, self.end_token_id);
        Ok(())
    }

    /// Buffer the delta, hold any trailing partial delimiter, and parse the stable prefix.
    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        self.buffer.push_str(delta);

        let partial_suffix_len = self.partial_suffix_len(&self.buffer);
        let stable_len = self.buffer.len() - partial_suffix_len;
        let stable_text = self.buffer[..stable_len].to_string();
        let pending_suffix = self.buffer[stable_len..].to_string();
        self.buffer = pending_suffix;

        Ok(self.parse_stable_text(&stable_text))
    }

    /// Flush the remaining buffered text at end of stream.
    fn finish(&mut self) -> Result<ReasoningDelta> {
        let stable_text = std::mem::take(&mut self.buffer);
        Ok(self.parse_stable_text(&stable_text))
    }
}

/// Thin Qwen-family wrapper around the generic delimited parser.
pub struct Qwen3ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl Qwen3ReasoningParser {
    /// Create a Qwen reasoning parser using `<think>...</think>` delimiters.
    pub fn new(tokenizer: &dyn Tokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>")?,
        })
    }
}

impl ReasoningParser for Qwen3ReasoningParser {
    /// Initialize the wrapped delimited parser from prompt token IDs.
    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)
    }

    /// Forward one delta to the wrapped delimited parser.
    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        self.inner.push(delta)
    }

    /// Flush the wrapped delimited parser.
    fn finish(&mut self) -> Result<ReasoningDelta> {
        self.inner.finish()
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
        factory.register_parser("base", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("cohere_cmd", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer,
                "<|START_THINKING|>",
                "<|END_THINKING|>",
            )?))
        });
        factory.register_parser("deepseek_r1", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("deepseek_v31", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("glm45", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("kimi", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer,
                "◁think▷",
                "◁/think▷",
            )?))
        });
        factory.register_parser("kimi_k25", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("kimi_thinking", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("minimax", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("nano_v3", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });
        factory.register_parser("qwen3", |tokenizer| {
            Ok(Box::new(Qwen3ReasoningParser::new(tokenizer)?))
        });
        factory.register_parser("qwen3_thinking", |tokenizer| {
            Ok(Box::new(Qwen3ReasoningParser::new(tokenizer)?))
        });
        factory.register_parser("step3", |tokenizer| {
            Ok(Box::new(DelimitedReasoningParser::new(
                tokenizer, "<think>", "</think>",
            )?))
        });

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

    /// Register a parser constructor under an exact name.
    pub fn register_parser<F>(&mut self, name: &str, creator: F)
    where
        F: Fn(&dyn Tokenizer) -> Result<Box<dyn ReasoningParser>> + Send + Sync + 'static,
    {
        self.creators.insert(name.to_string(), Arc::new(creator));
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
}

impl ReasoningStreamParserFactory for ReasoningParserFactory {
    /// Return true if the exact parser name is registered.
    fn contains(&self, name: &str) -> bool {
        self.creators.contains_key(name)
    }

    /// Return all registered parser names sorted for stable display.
    fn list(&self) -> Vec<String> {
        let mut names: Vec<_> = self.creators.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// Construct a parser from an exact name.
    fn create(&self, name: &str, tokenizer: &dyn Tokenizer) -> Result<Box<dyn ReasoningParser>> {
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
    fn create_for_model(
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

/// Determine whether the prompt currently ends inside or outside a reasoning span.
fn last_reasoning_boundary(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
) -> bool {
    for token_id in prompt_token_ids.iter().rev() {
        if *token_id == start_token_id {
            return true;
        }
        if *token_id == end_token_id {
            return false;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use vllm_text::tokenizers::Tokenizer;

    use super::{
        DelimitedReasoningParser, Qwen3ReasoningParser, ReasoningParser, ReasoningParserFactory,
        ReasoningStreamParserFactory,
    };

    struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_text::Result<Vec<u32>> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_text::Result<String> {
            Ok(token_ids
                .iter()
                .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
                .collect())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<think>" => Some(1),
                "</think>" => Some(2),
                "<|START_THINKING|>" => Some(3),
                "<|END_THINKING|>" => Some(4),
                "◁think▷" => Some(5),
                "◁/think▷" => Some(6),
                _ => None,
            }
        }
    }

    #[test]
    fn delimited_content_only_stream() {
        let tokenizer = FakeTokenizer;
        let mut parser = DelimitedReasoningParser::new(&tokenizer, "<think>", "</think>").unwrap();

        assert_eq!(
            parser.push("plain content").unwrap().content.as_deref(),
            Some("plain content")
        );
    }

    #[test]
    fn delimited_single_chunk_with_reasoning_and_content() {
        let tokenizer = FakeTokenizer;
        let mut parser = DelimitedReasoningParser::new(&tokenizer, "<think>", "</think>").unwrap();

        let delta = parser.push("<think>reason</think>answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn delimited_partial_tokens_across_chunks() {
        let tokenizer = FakeTokenizer;
        let mut parser = DelimitedReasoningParser::new(&tokenizer, "<think>", "</think>").unwrap();

        assert!(parser.push("<thi").unwrap().is_empty());
        let delta = parser.push("nk>reason</think>answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn delimited_finish_flushes_buffer() {
        let tokenizer = FakeTokenizer;
        let mut parser = DelimitedReasoningParser::new(&tokenizer, "<think>", "</think>").unwrap();
        parser.initialize(&[1]).unwrap();

        assert!(parser.push("unfinished</thi").unwrap().reasoning.is_some());
        let final_delta = parser.finish().unwrap();
        assert_eq!(final_delta.reasoning.as_deref(), Some("</thi"));
    }

    #[test]
    fn qwen3_without_prompt_markers_expects_start_token() {
        let tokenizer = FakeTokenizer;
        let mut parser = Qwen3ReasoningParser::new(&tokenizer).unwrap();

        let delta = parser.push("reason</think>answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("reason</think>answer"));
    }

    #[test]
    fn qwen3_prompt_end_marker_starts_in_content() {
        let tokenizer = FakeTokenizer;
        let mut parser = Qwen3ReasoningParser::new(&tokenizer).unwrap();
        parser.initialize(&[2]).unwrap();

        let delta = parser.push("answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn qwen3_tolerates_old_and_new_formats() {
        let tokenizer = FakeTokenizer;

        let mut old_parser = Qwen3ReasoningParser::new(&tokenizer).unwrap();
        let old = old_parser.push("<think>reason</think>answer").unwrap();
        assert_eq!(old.reasoning.as_deref(), Some("reason"));
        assert_eq!(old.content.as_deref(), Some("answer"));

        let mut new_parser = Qwen3ReasoningParser::new(&tokenizer).unwrap();
        new_parser.initialize(&[1]).unwrap();
        let new = new_parser.push("reason</think>answer").unwrap();
        assert_eq!(new.reasoning.as_deref(), Some("reason"));
        assert_eq!(new.content.as_deref(), Some("answer"));
    }

    #[test]
    fn factory_contains_and_lists_registered_parsers() {
        let factory = ReasoningParserFactory::new();
        assert!(factory.contains("qwen3"));
        assert!(factory.list().contains(&"qwen3".to_string()));
    }

    #[test]
    fn factory_rejects_unknown_parser_names() {
        let tokenizer = FakeTokenizer;
        let factory = ReasoningParserFactory::new();
        let error = match factory.create("missing", &tokenizer) {
            Ok(_) => panic!("expected parser lookup to fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("choose from"));
    }

    #[test]
    fn factory_rejects_unknown_models() {
        let tokenizer = FakeTokenizer;
        let factory = ReasoningParserFactory::new();
        let error = match factory.create_for_model("definitely-unknown-model", &tokenizer) {
            Ok(_) => panic!("expected model lookup to fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("not available for model"));
    }
}
