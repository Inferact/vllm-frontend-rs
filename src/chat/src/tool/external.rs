use async_trait::async_trait;
use openai_protocol::common::Tool as OpenAiTool;

use super::{Result, ToolCallDelta, ToolParseResult};
use crate::ToolParser;
use crate::request::ChatTool;

/// Adaptor that exposes the external `tool-parser` trait object through the
/// local [`ToolParser`] interface.
pub(crate) struct ExternalToolParserAdaptor {
    pub(crate) inner: Box<dyn tool_parser::ToolParser>,
    tools: Vec<OpenAiTool>,
}

impl ExternalToolParserAdaptor {
    pub(crate) fn new(inner: Box<dyn tool_parser::ToolParser>, tools: &[ChatTool]) -> Self {
        let tools = tools.iter().map(ChatTool::to_openai_tool).collect();
        Self { inner, tools }
    }
}

#[async_trait]
impl ToolParser for ExternalToolParserAdaptor {
    fn create(_tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        unreachable!("external tool parser adaptor is constructed from an existing parser")
    }

    async fn parse_complete(&self, output: &str) -> Result<ToolParseResult> {
        self.inner
            .parse_complete(output)
            .await
            .map(|(normal_text, tool_calls)| {
                // The external `parse_complete()` path does not receive tools and may therefore
                // return calls with invalid names. Filter them here against the request-scoped tool
                // set captured at parser creation time.
                let calls = tool_calls
                    .into_iter()
                    .filter(|tool_call| {
                        self.tools
                            .iter()
                            .any(|tool| tool.function.name == tool_call.function.name)
                    })
                    .enumerate()
                    .map(|(tool_index, tool_call)| ToolCallDelta {
                        tool_index,
                        name: Some(tool_call.function.name),
                        arguments: tool_call.function.arguments,
                    })
                    .collect();

                ToolParseResult { normal_text, calls }
            })
            .map_err(Into::into)
    }

    async fn parse_incremental(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.inner
            .parse_incremental(chunk, &self.tools)
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
