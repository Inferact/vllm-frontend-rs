use async_trait::async_trait;
use openai_protocol::common::Tool as OpenAiTool;
use tool_parser::types::{StreamingParseResult, ToolCall, ToolCallItem};

use super::{ParserResult, Result};
use crate::ToolParser;

/// Adaptor that exposes the external `tool-parser` trait object through the
/// local [`ToolParser`] interface.
pub(crate) struct ExternalToolParserAdaptor {
    pub(crate) inner: Box<dyn tool_parser::ToolParser>,
}

impl ExternalToolParserAdaptor {
    pub(crate) fn new(inner: Box<dyn tool_parser::ToolParser>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl ToolParser for ExternalToolParserAdaptor {
    fn create() -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        unreachable!("external tool parser adaptor is constructed from an existing parser")
    }

    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.inner.parse_complete(output).await
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[OpenAiTool],
    ) -> ParserResult<StreamingParseResult> {
        self.inner.parse_incremental(chunk, tools).await
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        self.inner.get_unstreamed_tool_args()
    }
}
