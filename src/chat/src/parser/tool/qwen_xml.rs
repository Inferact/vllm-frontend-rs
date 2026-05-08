use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::literal;

use super::utils::{
    JsonObjectScanState, json_str, parse_buffered_event, safe_text_len, take_json_object,
};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError, parsing_failed};
use crate::request::ChatTool;

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

type QwenXmlInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum QwenXmlMode {
    Text,
    Header,
    Arguments { json_scan: JsonObjectScanState },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QwenXmlEvent {
    Text { len: usize },
    ToolCallStart,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallEnd,
}

/// Tool parser for Qwen XML-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location":"Tokyo"}}
/// </tool_call>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// Note: parallel calls are represented as repeated
/// `<tool_call>...</tool_call>` blocks, not as multiple calls inside one tag.
pub struct Qwen3XmlToolParser {
    buffer: String,
    mode: QwenXmlMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl Qwen3XmlToolParser {
    /// Create a Qwen XML tool parser.
    fn new(_tools: &[ChatTool]) -> Self {
        Self {
            buffer: String::new(),
            mode: QwenXmlMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Apply one parsed Qwen XML event to parser state and output.
    fn apply_event(&mut self, event: QwenXmlEvent, result: &mut ToolParseResult) -> Result<()> {
        match event {
            QwenXmlEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            QwenXmlEvent::ToolCallStart => self.mode = QwenXmlMode::Header,
            QwenXmlEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = QwenXmlMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            QwenXmlEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Qwen XML arguments without an active tool call"
                    ));
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            QwenXmlEvent::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = QwenXmlMode::Text;
            }
        }
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = QwenXmlMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
    }
}

impl ToolParser for Qwen3XmlToolParser {
    /// Create a boxed Qwen XML tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the Qwen XML parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_qwen_xml_event(input, &mut self.mode)
        })? {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        match &self.mode {
            QwenXmlMode::Text => result.normal_text.push_str(&self.buffer),
            QwenXmlMode::Header | QwenXmlMode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete Qwen XML tool call"));
            }
        }
        self.reset();
        Ok(result)
    }
}

/// Parse a Qwen XML event for the current parser mode.
fn parse_next_qwen_xml_event(
    input: &mut QwenXmlInput<'_>,
    mode: &mut QwenXmlMode,
) -> ModalResult<QwenXmlEvent> {
    match mode {
        QwenXmlMode::Text => parse_text_event(input),
        QwenXmlMode::Header => tool_call_header_event(input),
        QwenXmlMode::Arguments { json_scan } => parse_arguments_event(input, json_scan),
    }
}

/// Parse a text-mode Qwen XML event.
fn parse_text_event(input: &mut QwenXmlInput<'_>) -> ModalResult<QwenXmlEvent> {
    alt((tool_call_start_event, safe_text_event)).parse_next(input)
}

/// Parse a Qwen XML tool-call start marker.
fn tool_call_start_event(input: &mut QwenXmlInput<'_>) -> ModalResult<QwenXmlEvent> {
    literal(TOOL_CALL_START).value(QwenXmlEvent::ToolCallStart).parse_next(input)
}

/// Parse a Qwen XML tool-call header before the raw JSON arguments payload.
fn tool_call_header_event(input: &mut QwenXmlInput<'_>) -> ModalResult<QwenXmlEvent> {
    let (function_name,) = seq!(
        _: ws0,
        _: literal("{"),
        _: ws0,
        _: name_field,
        _: ws0,
        _: literal(":"),
        _: ws0,
        json_str,
        _: ws0,
        _: literal(","),
        _: ws0,
        _: arguments_field,
        _: ws0,
        _: literal(":"),
        _: ws0,
    )
    .parse_next(input)?;

    Ok(QwenXmlEvent::ToolCallHeader { function_name })
}

/// Parse a Qwen XML `name` field key.
fn name_field(input: &mut QwenXmlInput<'_>) -> ModalResult<()> {
    json_field(input, "name")
}

/// Parse a Qwen XML `arguments` field key.
fn arguments_field(input: &mut QwenXmlInput<'_>) -> ModalResult<()> {
    json_field(input, "arguments")
}

/// Parse a Qwen XML fixed-envelope field key.
fn json_field(input: &mut QwenXmlInput<'_>, expected: &'static str) -> ModalResult<()> {
    let actual = json_str(input)?;
    if actual == expected {
        Ok(())
    } else {
        Err(qwen_header_error(StrContextValue::Description(
            match expected {
                "name" => "field `name`",
                "arguments" => "field `arguments`",
                _ => "expected field",
            },
        )))
    }
}

/// Parse one event inside a Qwen XML tool-call arguments payload.
fn parse_arguments_event(
    input: &mut QwenXmlInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<QwenXmlEvent> {
    if json_scan.complete() {
        tool_call_end_event(input)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parse a Qwen XML raw JSON arguments delta.
fn argument_delta_event(
    input: &mut QwenXmlInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<QwenXmlEvent> {
    take_json_object(input, json_scan).map(|len| QwenXmlEvent::Arguments { len })
}

/// Parse a Qwen XML tool-call end marker.
fn tool_call_end_event(input: &mut QwenXmlInput<'_>) -> ModalResult<QwenXmlEvent> {
    seq!(
        _: ws0,
        _: literal("}"),
        _: ws0,
        _: literal(TOOL_CALL_END),
    )
    .value(QwenXmlEvent::ToolCallEnd)
    .parse_next(input)
}

/// Parse a safe text run before the next Qwen XML tool call.
fn safe_text_event(input: &mut QwenXmlInput<'_>) -> ModalResult<QwenXmlEvent> {
    safe_text_len(input, TOOL_CALL_START).map(|len| QwenXmlEvent::Text { len })
}

fn qwen_header_error(expected: StrContextValue) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label("Qwen XML tool call header"));
    error.push(StrContext::Expected(expected));
    ErrMode::Cut(error)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::{Qwen3XmlToolParser, ToolParser};
    use crate::parser::tool::ToolParseResult;
    use crate::parser::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": {arguments}}}\n</tool_call>"
        )
    }

    #[test]
    fn qwen_xml_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn qwen_xml_parse_complete_extracts_raw_json_arguments() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let result = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_call("get_weather", arguments)
            ))
            .unwrap();

        assert_eq!(result.normal_text, "Let me check.\n");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_does_not_validate_or_normalize_arguments() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser.parse_complete(&build_tool_call("get_weather", arguments)).unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_streaming_emits_argument_deltas() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let chunks = [
            "<tool_call>",
            "\n{\"name\": \"get_weather\", \"arguments\": ",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}\n</tool_call>",
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
    fn qwen_xml_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn qwen_xml_keeps_end_marker_literal_inside_json_string() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{"text":"literal </tool_call> inside"}"#;
        let result = parser.parse_complete(&build_tool_call("echo", arguments)).unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_decodes_escaped_function_name() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let result = parser
            .parse_complete(r#"<tool_call>{"name":"say_\"hi","arguments":{}}</tool_call>"#)
            .unwrap();

        assert_eq!(result.calls[0].name.as_deref(), Some("say_\"hi"));
    }

    #[test]
    fn qwen_xml_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

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
    fn qwen_xml_finish_fails_incomplete_tool_call() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        parser
            .push(r#"<tool_call>{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Qwen XML tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn qwen_xml_malformed_field_order_fails_fast() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let error = parser
            .push(r#"<tool_call>{"arguments":{},"name":"get_weather"}</tool_call>"#)
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid Qwen XML tool call header
            expected field `name`"#]]
        .assert_eq(&error.to_report_string());
    }
}
