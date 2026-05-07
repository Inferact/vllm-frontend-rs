use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::utils::{parse_buffered_event, safe_text_len};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError, parsing_failed};
use crate::request::ChatTool;

const TOOL_CALLS_START: &str = "<｜tool▁calls▁begin｜>";
const TOOL_CALLS_END: &str = "<｜tool▁calls▁end｜>";
const TOOL_CALL_START: &str = "<｜tool▁call▁begin｜>";
const TOOL_CALL_END: &str = "<｜tool▁call▁end｜>";
const TOOL_CALL_SEPARATOR: &str = "<｜tool▁sep｜>";
const V3_JSON_START: &str = "\n```json\n";
const V3_ARGUMENT_END: &str = "\n```<｜tool▁call▁end｜>";

type DeepSeekJsonInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepSeekJsonFormat {
    V3,
    V31,
}

impl DeepSeekJsonFormat {
    /// Return the parser name used in diagnostics.
    const fn parser_name(self) -> &'static str {
        match self {
            Self::V3 => "DeepSeek V3",
            Self::V31 => "DeepSeek V3.1",
        }
    }

    /// Return the marker that closes the raw JSON arguments payload.
    const fn argument_end_marker(self) -> &'static str {
        match self {
            Self::V3 => V3_ARGUMENT_END,
            Self::V31 => TOOL_CALL_END,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepSeekJsonMode {
    Text,
    ToolBlock,
    Header,
    Arguments,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeepSeekJsonEvent {
    Text { len: usize },
    ToolCallsStart,
    ToolCallStart,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallEnd,
    ToolCallsEnd,
    IgnoredRest,
}

/// Tool parser core for DeepSeek JSON-argument tool calls.
struct DeepSeekJsonToolParser {
    buffer: String,
    mode: DeepSeekJsonMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
    format: DeepSeekJsonFormat,
}

impl DeepSeekJsonToolParser {
    /// Create a parser for one DeepSeek JSON-argument format.
    fn new(format: DeepSeekJsonFormat) -> Self {
        Self {
            buffer: String::new(),
            mode: DeepSeekJsonMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
            format,
        }
    }

    /// Apply one parsed DeepSeek JSON event to parser state and output.
    fn apply_event(
        &mut self,
        event: DeepSeekJsonEvent,
        result: &mut ToolParseResult,
    ) -> Result<()> {
        match event {
            DeepSeekJsonEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            DeepSeekJsonEvent::ToolCallsStart => self.mode = DeepSeekJsonMode::ToolBlock,
            DeepSeekJsonEvent::ToolCallStart => self.mode = DeepSeekJsonMode::Header,
            DeepSeekJsonEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = DeepSeekJsonMode::Arguments;
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            DeepSeekJsonEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "{} arguments without an active tool call",
                        self.format.parser_name()
                    ));
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            DeepSeekJsonEvent::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = DeepSeekJsonMode::ToolBlock;
            }
            DeepSeekJsonEvent::ToolCallsEnd => {
                self.active_tool_index = None;
                self.mode = DeepSeekJsonMode::Done;
            }
            DeepSeekJsonEvent::IgnoredRest => {}
        }
        Ok(())
    }

    /// Push one decoded text chunk through the DeepSeek JSON parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_deepseek_json_event(input, self.mode, self.format)
        })? {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        match self.mode {
            DeepSeekJsonMode::Text => result.normal_text.push_str(&self.buffer),
            DeepSeekJsonMode::ToolBlock | DeepSeekJsonMode::Done => {}
            DeepSeekJsonMode::Header | DeepSeekJsonMode::Arguments => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.format.parser_name()
                ));
            }
        }
        self.reset();
        Ok(result)
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = DeepSeekJsonMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
    }
}

/// Tool parser for DeepSeek V3 JSON-fenced tool calls.
///
/// Example tool call content:
///
/// ````text
/// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
/// ```json
/// {"location":"Tokyo"}
/// ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
/// ````
///
/// Arguments are already OpenAI-style JSON text inside the markdown fence, so
/// they are streamed as raw argument deltas without schema conversion or JSON
/// normalization.
pub struct DeepSeekV3ToolParser(DeepSeekJsonToolParser);

impl DeepSeekV3ToolParser {
    /// Create a DeepSeek V3 tool parser.
    fn new(_tools: &[ChatTool]) -> Self {
        Self(DeepSeekJsonToolParser::new(DeepSeekJsonFormat::V3))
    }
}

impl ToolParser for DeepSeekV3ToolParser {
    /// Create a boxed DeepSeek V3 tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the DeepSeek V3 parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.0.push(chunk)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        self.0.finish()
    }
}

/// Tool parser for DeepSeek V3.1 raw JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub struct DeepSeekV31ToolParser(DeepSeekJsonToolParser);

impl DeepSeekV31ToolParser {
    /// Create a DeepSeek V3.1 tool parser.
    fn new(_tools: &[ChatTool]) -> Self {
        Self(DeepSeekJsonToolParser::new(DeepSeekJsonFormat::V31))
    }
}

impl ToolParser for DeepSeekV31ToolParser {
    /// Create a boxed DeepSeek V3.1 tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the DeepSeek V3.1 parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.0.push(chunk)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        self.0.finish()
    }
}

/// Parse a DeepSeek JSON event for the current parser mode.
fn parse_next_deepseek_json_event(
    input: &mut DeepSeekJsonInput<'_>,
    mode: DeepSeekJsonMode,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    match mode {
        DeepSeekJsonMode::Text => parse_text_event(input),
        DeepSeekJsonMode::ToolBlock => parse_tool_block_event(input),
        DeepSeekJsonMode::Header => tool_call_header_event(input, format),
        DeepSeekJsonMode::Arguments => parse_arguments_event(input, format),
        DeepSeekJsonMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode DeepSeek JSON event.
fn parse_text_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse one event inside the DeepSeek tool-calls section.
fn parse_tool_block_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    ws0.void().parse_next(input)?;
    alt((tool_calls_end_event, tool_call_start_event)).parse_next(input)
}

/// Parse one event inside a DeepSeek tool-call arguments payload.
fn parse_arguments_event(
    input: &mut DeepSeekJsonInput<'_>,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    alt((
        |input: &mut DeepSeekJsonInput<'_>| tool_call_end_event(input, format),
        |input: &mut DeepSeekJsonInput<'_>| argument_delta_event(input, format),
    ))
    .parse_next(input)
}

/// Parse a DeepSeek tool-calls start marker.
fn tool_calls_start_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    literal(TOOL_CALLS_START)
        .value(DeepSeekJsonEvent::ToolCallsStart)
        .parse_next(input)
}

/// Parse a DeepSeek tool-calls end marker.
fn tool_calls_end_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    literal(TOOL_CALLS_END).value(DeepSeekJsonEvent::ToolCallsEnd).parse_next(input)
}

/// Parse a DeepSeek tool-call start marker.
fn tool_call_start_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    literal(TOOL_CALL_START)
        .value(DeepSeekJsonEvent::ToolCallStart)
        .parse_next(input)
}

/// Parse a DeepSeek tool-call end marker.
fn tool_call_end_event(
    input: &mut DeepSeekJsonInput<'_>,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    literal(format.argument_end_marker())
        .value(DeepSeekJsonEvent::ToolCallEnd)
        .parse_next(input)
}

/// Parse a DeepSeek tool-call header before the JSON arguments payload.
fn tool_call_header_event(
    input: &mut DeepSeekJsonInput<'_>,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    match format {
        DeepSeekJsonFormat::V3 => v3_tool_call_header_event(input),
        DeepSeekJsonFormat::V31 => v31_tool_call_header_event(input),
    }
}

/// Parse a DeepSeek V3 tool-call header.
fn v3_tool_call_header_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    let name = seq!(
        _: literal("function"),
        _: literal(TOOL_CALL_SEPARATOR),
        take_until(1.., V3_JSON_START),
        _: literal(V3_JSON_START),
    )
    .parse_next(input)?;

    Ok(DeepSeekJsonEvent::ToolCallHeader {
        function_name: name.0.trim().to_string(),
    })
}

/// Parse a DeepSeek V3.1 tool-call header.
fn v31_tool_call_header_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    let (name, _) = (
        take_until(1.., TOOL_CALL_SEPARATOR),
        literal(TOOL_CALL_SEPARATOR),
    )
        .parse_next(input)?;

    Ok(DeepSeekJsonEvent::ToolCallHeader {
        function_name: name.trim().to_string(),
    })
}

/// Parse a DeepSeek raw JSON arguments delta.
fn argument_delta_event(
    input: &mut DeepSeekJsonInput<'_>,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    safe_text_len(input, format.argument_end_marker())
        .map(|len| DeepSeekJsonEvent::Arguments { len })
}

/// Parse a safe text run before the next DeepSeek tool-calls section.
fn safe_text_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    safe_text_len(input, TOOL_CALLS_START).map(|len| DeepSeekJsonEvent::Text { len })
}

/// Parse ignored rest after the DeepSeek tool-calls section ends.
fn ignored_rest_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    rest.value(DeepSeekJsonEvent::IgnoredRest).parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::{
        DeepSeekV3ToolParser, DeepSeekV31ToolParser, TOOL_CALL_END, TOOL_CALL_SEPARATOR,
        TOOL_CALL_START, TOOL_CALLS_END, TOOL_CALLS_START, ToolParser, V3_ARGUMENT_END,
        V3_JSON_START,
    };
    use crate::parser::tool::ToolParseResult;
    use crate::parser::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn v3_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "{TOOL_CALL_START}function{TOOL_CALL_SEPARATOR}{function_name}{V3_JSON_START}{arguments}{V3_ARGUMENT_END}"
        )
    }

    fn v31_tool_call(function_name: &str, arguments: &str) -> String {
        format!("{TOOL_CALL_START}{function_name}{TOOL_CALL_SEPARATOR}{arguments}{TOOL_CALL_END}")
    }

    fn tool_section(tool_calls: &[String]) -> String {
        format!("{TOOL_CALLS_START}{}{TOOL_CALLS_END}", tool_calls.join(""))
    }

    #[test]
    fn deepseek_v3_parse_complete_without_tool_call_keeps_text() {
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v3_parse_complete_extracts_raw_json_arguments() {
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let result = parser
            .parse_complete(&format!(
                "Let me check.\n{} trailing text",
                tool_section(&[v3_tool_call("get_weather", arguments)])
            ))
            .unwrap();

        assert_eq!(result.normal_text, "Let me check.\n");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn deepseek_v3_does_not_validate_or_normalize_arguments() {
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser
            .parse_complete(&tool_section(&[v3_tool_call("get_weather", arguments)]))
            .unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn deepseek_v3_streaming_emits_argument_deltas() {
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());
        let chunks = [
            TOOL_CALLS_START,
            TOOL_CALL_START,
            "function",
            TOOL_CALL_SEPARATOR,
            "get_weather",
            V3_JSON_START,
            "{\"location\":",
            "\"Beijing\"",
            "}",
            V3_ARGUMENT_END,
            TOOL_CALLS_END,
        ];

        let mut result = ToolParseResult::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.push(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            result.append(next);
        }
        result.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            result.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn deepseek_v3_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            tool_section(&[v3_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn deepseek_v3_streaming_extracts_multiple_tool_calls() {
        let input = tool_section(&[
            v3_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            v3_tool_call("add", r#"{"x":1,"y":2}"#),
        ]);
        let chunks = split_by_chars(&input, 7);
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParseResult {
                normal_text: "",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_weather",
                        ),
                        arguments: "{\"location\":\"Shanghai\"}",
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some(
                            "add",
                        ),
                        arguments: "{\"x\":1,\"y\":2}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&result);
    }

    #[test]
    fn deepseek_v3_finish_fails_incomplete_tool_call() {
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());
        parser
            .push(&format!(
                "{TOOL_CALLS_START}{TOOL_CALL_START}function{TOOL_CALL_SEPARATOR}get_weather{V3_JSON_START}{{\"location\""
            ))
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete DeepSeek V3 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn deepseek_v3_malformed_type_fails_fast() {
        let mut parser = DeepSeekV3ToolParser::new(&test_tools());
        let input = format!(
            "{TOOL_CALLS_START}{TOOL_CALL_START}tool{TOOL_CALL_SEPARATOR}get_weather{V3_JSON_START}{{}}"
        );

        let error = parser.push(&input).unwrap_err();

        expect!["tool parser parsing failed: "].assert_eq(&error.to_report_string());
    }

    #[test]
    fn deepseek_v31_parse_complete_without_tool_call_keeps_text() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v31_parse_complete_extracts_raw_json_arguments() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let result = parser
            .parse_complete(&format!(
                "Let me check.{} trailing text",
                tool_section(&[v31_tool_call("get_weather", arguments)])
            ))
            .unwrap();

        assert_eq!(result.normal_text, "Let me check.");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn deepseek_v31_does_not_validate_or_normalize_arguments() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser
            .parse_complete(&tool_section(&[v31_tool_call("get_weather", arguments)]))
            .unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn deepseek_v31_streaming_emits_argument_deltas() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let chunks = [
            TOOL_CALLS_START,
            TOOL_CALL_START,
            "get_weather",
            TOOL_CALL_SEPARATOR,
            "{\"location\":",
            "\"Beijing\"",
            "}",
            TOOL_CALL_END,
            TOOL_CALLS_END,
        ];

        let mut result = ToolParseResult::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.push(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            result.append(next);
        }
        result.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            result.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn deepseek_v31_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            tool_section(&[v31_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn deepseek_v31_streaming_extracts_multiple_tool_calls() {
        let input = tool_section(&[
            v31_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            v31_tool_call("add", r#"{"x":1,"y":2}"#),
        ]);
        let chunks = split_by_chars(&input, 7);
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParseResult {
                normal_text: "",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_weather",
                        ),
                        arguments: "{\"location\":\"Shanghai\"}",
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some(
                            "add",
                        ),
                        arguments: "{\"x\":1,\"y\":2}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&result);
    }

    #[test]
    fn deepseek_v31_streaming_drops_eos_after_complete_tool_calls() {
        let input = format!(
            "{}<｜end▁of▁sentence｜>",
            tool_section(&[v31_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &[&input]);

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn deepseek_v31_finish_fails_incomplete_tool_call() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        parser
            .push(&format!(
                "{TOOL_CALLS_START}{TOOL_CALL_START}get_weather{TOOL_CALL_SEPARATOR}{{\"location\""
            ))
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete DeepSeek V3.1 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn deepseek_v31_malformed_empty_name_fails_fast() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let input = format!("{TOOL_CALLS_START}{TOOL_CALL_START}{TOOL_CALL_SEPARATOR}{{}}");

        let error = parser.push(&input).unwrap_err();

        expect!["tool parser parsing failed: "].assert_eq(&error.to_report_string());
    }
}
