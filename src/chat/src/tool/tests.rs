use async_trait::async_trait;

use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserFactory};
use crate::request::ChatTool;
use crate::tool::names;

struct FakeToolParser;

#[async_trait]
impl ToolParser for FakeToolParser {
    fn create(_tools: &[ChatTool]) -> super::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    async fn parse_complete(&self, _output: &str) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }

    async fn parse_incremental(&mut self, _chunk: &str) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallDelta>> {
        None
    }
}

#[test]
fn factory_creates_registered_parser_for_model() {
    let mut factory = ToolParserFactory::default();
    factory
        .register_parser::<FakeToolParser>("fake")
        .register_pattern("fake-model", "fake");

    factory.create_for_model("my-fake-model-v1", &[]).unwrap();
}

#[test]
fn factory_new_registers_builtin_json_parser() {
    let factory = ToolParserFactory::new();
    assert!(factory.contains(names::QWEN));
    assert!(factory.list().contains(&names::QWEN.to_string()));
}

#[test]
fn factory_new_resolves_qwen_to_builtin_qwen_parser() {
    let factory = ToolParserFactory::new();
    factory.create_for_model("Qwen/Qwen3-0.6B", &[]).unwrap();
}
