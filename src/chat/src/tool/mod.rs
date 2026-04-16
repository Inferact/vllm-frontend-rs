//! Streaming tool parsers for chat completions.
//!
//! This module intentionally starts as a local ownership boundary for tool
//! parser registration and selection, without yet taking over the concrete
//! parsing implementation from the external `tool-parser` crate. The goal is
//! to establish the northbound trait and factory shape inside `vllm-chat`
//! first, so later steps can attach adaptor-based implementations and then
//! gradually replace them with native parsers as needed.

mod external;

use async_trait::async_trait;
use openai_protocol::common::Tool as OpenAiTool;
use thiserror::Error;

use crate::parser::{ParserFactory, available_parser_hint};

/// Result alias for tool parser operations.
pub type Result<T> = std::result::Result<T, ToolParserError>;
pub type ParserResult<T> = tool_parser::errors::ParserResult<T>;

pub(crate) use external::ExternalToolParserAdaptor;
pub use tool_parser::types::{StreamingParseResult, ToolCall, ToolCallItem};

/// Incremental parser that extracts tool calls from assistant output.
#[async_trait]
pub trait ToolParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create() -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static;

    /// Parse complete tool calls from final output.
    ///
    /// Returns the remaining plain assistant text plus any parsed tool calls.
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)>;

    /// Parse tool calls incrementally from one assistant text chunk.
    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[OpenAiTool],
    ) -> ParserResult<StreamingParseResult>;

    /// Return tool arguments that were buffered until end-of-stream.
    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        None
    }
}

/// Errors produced while creating tool parsers.
#[derive(Debug, Error)]
pub enum ToolParserError {
    #[error(
        "tool call parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    UnknownParser {
        name: String,
        available_names: Vec<String>,
    },
    #[error("tool parsing is not available for model `{model_id}`")]
    UnknownModel { model_id: String },
}

/// Constructor signature for one registered tool parser implementation.
type ToolParserCreator = fn() -> Result<Box<dyn ToolParser>>;

/// Registry and model matcher for tool parsers.
pub type ToolParserFactory = ParserFactory<ToolParserCreator>;

impl ToolParserFactory {
    /// Create the default registry with built-in parser names and model mappings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ToolParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(&self, name: &str) -> Result<Box<dyn ToolParser>> {
        let creator = self
            .creator(name)
            .ok_or_else(|| ToolParserError::UnknownParser {
                name: name.to_string(),
                available_names: self.list(),
            })?;
        creator()
    }

    /// Resolve a parser from model ID and then construct it.
    pub fn create_for_model(&self, model_id: &str) -> Result<Box<dyn ToolParser>> {
        let name =
            self.resolve_name_for_model(model_id)
                .ok_or_else(|| ToolParserError::UnknownModel {
                    model_id: model_id.to_string(),
                })?;
        self.create(name)
    }
}

#[cfg(test)]
mod tests;
