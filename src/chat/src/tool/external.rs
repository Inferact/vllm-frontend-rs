use async_trait::async_trait;
use openai_protocol::common::Tool as OpenAiTool;

use super::{Result, ToolCallDelta, ToolParseResult};
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

    async fn parse_complete(&self, output: &str) -> Result<ToolParseResult> {
        self.inner
            .parse_complete(output)
            .await
            .map(|(normal_text, tool_calls)| ToolParseResult {
                normal_text,
                calls: tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(tool_index, tool_call)| ToolCallDelta {
                        tool_index,
                        name: Some(tool_call.function.name),
                        arguments: tool_call.function.arguments,
                    })
                    .collect(),
            })
            .map_err(Into::into)
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[OpenAiTool],
    ) -> Result<ToolParseResult> {
        self.inner
            .parse_incremental(chunk, tools)
            .await
            .map(convert_parse_result)
            .map_err(Into::into)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallDelta>> {
        self.inner
            .get_unstreamed_tool_args()
            .map(|items| items.into_iter().map(convert_tool_call_item).collect())
    }
}

fn convert_tool_call_item(item: tool_parser::types::ToolCallItem) -> ToolCallDelta {
    ToolCallDelta {
        tool_index: item.tool_index,
        name: item.name,
        arguments: item.parameters,
    }
}

fn convert_parse_result(result: tool_parser::types::StreamingParseResult) -> ToolParseResult {
    ToolParseResult {
        normal_text: result.normal_text,
        calls: result
            .calls
            .into_iter()
            .map(convert_tool_call_item)
            .collect(),
    }
}
