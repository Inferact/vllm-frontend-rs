use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, repeat, terminated};
use winnow::error::ErrMode;
use winnow::prelude::*;
use winnow::stream::{Offset, Partial, Stream};
use winnow::token::{literal, take_until};

use super::parameters::ToolSchemas;
use super::utils::safe_text_len;
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError, parsing_failed};
use crate::request::ChatTool;

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

type QwenCoderInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenCoderMode {
    Text,
    ToolCall,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QwenCoderEvent {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
}

/// Tool parser for Qwen Coder XML-style tool calls.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/bf45e6d0a558da2b8d7b60efb07b4aa394f3b60b/vllm/tool_parsers/qwen3coder_tool_parser.py>
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=location>杭州</parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Streaming strategy: **buffer until one complete tool call closes**
///
/// Qwen Coder arguments are emitted only after a full
/// `<tool_call>...</tool_call>` block is available. Plain assistant text is
/// still emitted incrementally before the next possible tool-call marker.
pub struct Qwen3CoderToolParser {
    buffer: String,
    mode: QwenCoderMode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl Qwen3CoderToolParser {
    /// Create a Qwen Coder tool parser.
    fn new(tools: &[ChatTool]) -> Self {
        Self {
            buffer: String::new(),
            mode: QwenCoderMode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed Qwen Coder event to parser state and output.
    fn apply_event(&mut self, event: QwenCoderEvent, result: &mut ToolParseResult) -> Result<()> {
        match event {
            QwenCoderEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            QwenCoderEvent::ToolCallStart => self.mode = QwenCoderMode::ToolCall,
            QwenCoderEvent::ToolCall { name, raw_params } => {
                self.mode = QwenCoderMode::Text;
                let arguments = self
                    .tool_parameters
                    .convert_params_with_schema(&name, raw_params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                result.calls.push(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
        }
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = QwenCoderMode::Text;
        self.emitted_tool_count = 0;
    }
}

impl ToolParser for Qwen3CoderToolParser {
    /// Create a boxed Qwen Coder tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the Qwen Coder parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        loop {
            let mut input = Partial::new(self.buffer.as_str());
            let checkpoint = input.checkpoint();
            let event = match parse_next_qwen_coder_event(&mut input, self.mode) {
                Ok(event) => event,
                Err(ErrMode::Incomplete(_)) => break,
                Err(error) => {
                    return Err(parsing_failed!(
                        "failed to parse Qwen Coder event: {}",
                        error
                    ));
                }
            };
            let consumed_len = input.offset_from(&checkpoint);
            if consumed_len == 0 {
                break;
            }
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        if !self.buffer.is_empty() {
            if self.mode == QwenCoderMode::ToolCall || self.buffer.starts_with(TOOL_CALL_START) {
                return Err(parsing_failed!("incomplete Qwen Coder tool call"));
            }
            result.normal_text.push_str(&self.buffer);
        }
        self.reset();
        Ok(result)
    }
}

/// Parse a Qwen Coder event for the current parser mode.
fn parse_next_qwen_coder_event(
    input: &mut QwenCoderInput<'_>,
    mode: QwenCoderMode,
) -> ModalResult<QwenCoderEvent> {
    match mode {
        QwenCoderMode::Text => parse_text_event(input),
        QwenCoderMode::ToolCall => tool_call_event(input),
    }
}

/// Parse a text-mode Qwen Coder event.
fn parse_text_event(input: &mut QwenCoderInput<'_>) -> ModalResult<QwenCoderEvent> {
    alt((tool_call_start_event, safe_text_event)).parse_next(input)
}

/// Parse a Qwen Coder tool-call start marker.
fn tool_call_start_event(input: &mut QwenCoderInput<'_>) -> ModalResult<QwenCoderEvent> {
    literal(TOOL_CALL_START)
        .value(QwenCoderEvent::ToolCallStart)
        .parse_next(input)
}

/// Parse a safe text run before the next Qwen Coder marker.
fn safe_text_event(input: &mut QwenCoderInput<'_>) -> ModalResult<QwenCoderEvent> {
    safe_text_len(input, TOOL_CALL_START).map(|len| QwenCoderEvent::Text { len })
}

/// Parse a complete Qwen Coder tool call.
fn tool_call_event(input: &mut QwenCoderInput<'_>) -> ModalResult<QwenCoderEvent> {
    let (_, body, _) =
        (ws0, take_until(0.., TOOL_CALL_END), literal(TOOL_CALL_END)).parse_next(input)?;

    parse_tool_call_body(body)
}

/// Parse a Qwen Coder function block.
fn function_event(input: &mut &str) -> ModalResult<QwenCoderEvent> {
    let (_, name, _, _, raw_params, _) = (
        literal(FUNCTION_START),
        take_until(1.., ">"),
        ">",
        ws0,
        repeat(0.., terminated(parameter, ws0)),
        literal(FUNCTION_END),
    )
        .parse_next(input)?;

    Ok(QwenCoderEvent::ToolCall {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a Qwen Coder parameter block.
fn parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (_, name, _, value, _) = (
        literal(PARAMETER_START),
        take_until(1.., ">"),
        ">",
        take_until(0.., PARAMETER_END),
        literal(PARAMETER_END),
    )
        .parse_next(input)?;

    Ok((
        name.to_string(),
        trim_one_wrapping_newline(value).to_string(),
    ))
}

/// Parse a Qwen Coder tool-call body.
fn parse_tool_call_body(body: &str) -> ModalResult<QwenCoderEvent> {
    let mut input = body;
    delimited(ws0, function_event, (ws0, eof)).parse_next(&mut input)
}

/// Trim a single leading and trailing newline from a parameter value.
fn trim_one_wrapping_newline(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{Qwen3CoderToolParser, ToolParser};
    use crate::request::ChatTool;

    fn test_tools() -> Vec<ChatTool> {
        vec![
            ChatTool {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" },
                        "date": { "type": "string" },
                        "days": { "type": "integer" }
                    }
                }),
                strict: None,
            },
            ChatTool {
                name: "convert".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "whole": { "type": "number" },
                        "flag": { "type": "boolean" },
                        "payload": { "type": "object" },
                        "items": { "type": "array" },
                        "empty": { "type": "string" }
                    }
                }),
                strict: None,
            },
        ]
    }

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<parameter={name}>{value}</parameter>"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("<tool_call>\n<function={function_name}>\n{params}\n</function>\n</tool_call>")
    }

    fn collect_stream(chunks: &[&str], tools: &[ChatTool]) -> crate::parser::tool::ToolParseResult {
        let mut parser = Qwen3CoderToolParser::new(tools);
        let mut result = crate::parser::tool::ToolParseResult::default();
        for chunk in chunks {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        result.coalesce_calls()
    }

    fn split_by_chars(text: &str, chunk_chars: usize) -> Vec<&str> {
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut count = 0;

        for (index, _) in text.char_indices() {
            if count == chunk_chars {
                chunks.push(&text[start..index]);
                start = index;
                count = 0;
            }
            count += 1;
        }

        if start < text.len() {
            chunks.push(&text[start..]);
        }

        chunks
    }

    #[test]
    fn qwen_coder_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn qwen_coder_parse_complete_extracts_single_tool_call() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2026-04-29")],
            ))
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "location": "SF",
                "date": "2026-04-29"
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_preserves_prefix_text() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = format!(
            "Thinking... {}",
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let result = parser.parse_complete(&output).unwrap();

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn qwen_coder_parse_complete_converts_schema_types() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_call(
                "convert",
                &[
                    ("whole", "5.0"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "42"),
                ],
            ))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn qwen_coder_streaming_extracts_single_tool_call() {
        let result = collect_stream(
            &[
                "<tool_call>\n",
                "<function=get_weather>\n",
                "<parameter=location>SF</parameter>\n",
                "</function>\n",
                "</tool_call>",
            ],
            &test_tools(),
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn qwen_coder_streaming_preserves_prefix_text() {
        let result = collect_stream(
            &[
                "Thinking... ",
                "<tool_call>\n",
                "<function=get_weather>\n",
                "<parameter=location>SF</parameter>\n",
                "</function>\n",
                "</tool_call>",
            ],
            &test_tools(),
        );

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn qwen_coder_streaming_without_tool_call_emits_text_incrementally() {
        let result = collect_stream(&["Hello, ", "world!"], &test_tools());

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn qwen_coder_streaming_extracts_multiple_tool_calls_in_order() {
        let text = format!(
            "{}\n{}",
            build_tool_call("get_weather", &[("location", "SF")]),
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let result = collect_stream(&[&text], &test_tools());

        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[1].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[1].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn qwen_coder_streaming_handles_start_token_split_across_chunks() {
        let text = build_tool_call("get_weather", &[("location", "SF")]);
        let chunks = split_by_chars(&text, 3);
        let result = collect_stream(&chunks, &test_tools());

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn qwen_coder_streaming_does_not_emit_incomplete_tool_call() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let result = parser
            .push("<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>")
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert!(result.calls.is_empty());
    }

    #[test]
    fn qwen_coder_finish_fails_incomplete_tool_call() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        parser
            .push("<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>")
            .unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn qwen_coder_malformed_tool_call_fails_fast() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let error = parser.push("<tool_call>\n<bad>\n</tool_call>").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("failed to parse Qwen Coder event")
        );
    }

    #[test]
    fn qwen_coder_missing_parameter_end_fails_fast_after_function_end() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let error = parser
            .push(
                "<tool_call>\n<function=get_weather>\n<parameter=location>SF</function>\n</tool_call>",
            )
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("failed to parse Qwen Coder event")
        );
    }

    #[test]
    fn qwen_coder_parse_function_body_trims_one_wrapping_newline() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nHangzhou\n</parameter>\n</function>\n</tool_call>",
            )
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "Hangzhou" })
        );
    }
}
