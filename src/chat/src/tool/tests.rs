use async_trait::async_trait;

use super::{
    OpenAiTool, ParserResult, StreamingParseResult, ToolCall, ToolCallItem, ToolParser,
    ToolParserError, ToolParserFactory,
};

struct FakeToolParser;

#[async_trait]
impl ToolParser for FakeToolParser {
    fn create() -> super::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    async fn parse_complete(&self, _output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok((String::new(), Vec::new()))
    }

    async fn parse_incremental(
        &mut self,
        _chunk: &str,
        _tools: &[OpenAiTool],
    ) -> ParserResult<StreamingParseResult> {
        Ok(StreamingParseResult::default())
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        None
    }
}

#[test]
fn factory_starts_empty() {
    let factory = ToolParserFactory::new();
    assert!(factory.list().is_empty());
}

#[test]
fn factory_contains_and_creates_registered_parsers() {
    let mut factory = ToolParserFactory::new();
    factory.register_parser::<FakeToolParser>("fake");

    assert!(factory.contains("fake"));
    assert!(factory.list().contains(&"fake".to_string()));
    factory.create("fake").unwrap();
}

#[test]
fn factory_rejects_unknown_parser_names() {
    let factory = ToolParserFactory::new();
    let error = match factory.create("missing") {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, ToolParserError::UnknownParser { .. }));
}

#[test]
fn factory_rejects_unknown_models() {
    let factory = ToolParserFactory::new();
    let error = match factory.create_for_model("definitely-unknown-model") {
        Ok(_) => panic!("expected model lookup to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, ToolParserError::UnknownModel { .. }));
}

#[test]
fn factory_creates_registered_parser_for_model() {
    let mut factory = ToolParserFactory::new();
    factory
        .register_parser::<FakeToolParser>("fake")
        .register_pattern("fake-model", "fake");

    factory.create_for_model("my-fake-model-v1").unwrap();
}
