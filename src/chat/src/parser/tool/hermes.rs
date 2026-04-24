use serde_json::Value;

use super::utils::partial_prefix_len;
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser};
use crate::request::{ChatRequest, ChatTool};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

/// Tool parser for Hermes-style tool calls.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/hermes_tool_parser.py>
///
/// Handles JSON tool calls wrapped by `<tool_call>...</tool_call>`:
///
/// ```text
/// <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>
/// ```
///
/// Streaming strategy: **parse valid JSON, then diff arguments**
///
/// The parser keeps a chunk buffer and withholds partial start/end tags. It
/// emits the function name as soon as it can safely read it, but only emits
/// argument deltas once the current tool-call JSON is valid. This keeps the
/// implementation robust while still allowing the default `parse_complete()`
/// path to reuse `push()+finish()`.
pub struct HermesToolParser {
    buffer: String,
    next_tool_index: usize,
    active_tool: Option<ActiveHermesTool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveHermesTool {
    tool_index: usize,
    name_sent: bool,
    streamed_arguments: String,
}

impl HermesToolParser {
    fn new(_tools: &[ChatTool]) -> Self {
        Self {
            buffer: String::new(),
            next_tool_index: 0,
            active_tool: None,
        }
    }

    fn begin_tool_call(&mut self) {
        let tool_index = self.next_tool_index;
        self.next_tool_index += 1;
        self.active_tool = Some(ActiveHermesTool {
            tool_index,
            name_sent: false,
            streamed_arguments: String::new(),
        });
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.next_tool_index = 0;
        self.active_tool = None;
    }

    fn process_text_mode(&mut self, result: &mut ToolParseResult) -> bool {
        let Some(start_idx) = self.buffer.find(TOOL_CALL_START) else {
            let keep_len = partial_prefix_len(&self.buffer, TOOL_CALL_START);
            let emit_len = self.buffer.len().saturating_sub(keep_len);
            if emit_len > 0 {
                result.normal_text.push_str(&self.buffer[..emit_len]);
                self.buffer.drain(..emit_len);
                return true;
            }
            return false;
        };

        if start_idx > 0 {
            result.normal_text.push_str(&self.buffer[..start_idx]);
            self.buffer.drain(..start_idx);
            return true;
        }

        self.buffer.drain(..TOOL_CALL_START.len());
        self.begin_tool_call();
        true
    }

    fn process_tool_mode(&mut self, result: &mut ToolParseResult) -> Result<bool> {
        let Some(tool_index) = self.active_tool.as_ref().map(|tool| tool.tool_index) else {
            return Ok(false);
        };

        if let Some(end_idx) = self.buffer.find(TOOL_CALL_END) {
            let body = self.buffer[..end_idx].trim().to_string();
            if self.active_tool.as_ref().is_some_and(|tool| tool.name_sent) {
                self.emit_body_delta(tool_index, &body, result);
            } else if let Some((name, arguments)) = parse_complete_body(&body) {
                self.emit_tool_name(tool_index, name, result);
                self.emit_arguments_diff(tool_index, arguments, result);
            } else {
                result.normal_text.push_str(TOOL_CALL_START);
                result
                    .normal_text
                    .push_str(&self.buffer[..end_idx + TOOL_CALL_END.len()]);
            }

            self.buffer.drain(..end_idx + TOOL_CALL_END.len());
            self.active_tool = None;
            return Ok(true);
        }

        let keep_len = partial_prefix_len(&self.buffer, TOOL_CALL_END);
        let body_len = self.buffer.len().saturating_sub(keep_len);
        if body_len == 0 {
            return Ok(false);
        }

        let body = self.buffer[..body_len].trim().to_string();
        Ok(self.emit_body_delta(tool_index, &body, result))
    }

    fn emit_body_delta(
        &mut self,
        tool_index: usize,
        body: &str,
        result: &mut ToolParseResult,
    ) -> bool {
        let mut progressed = false;

        if self
            .active_tool
            .as_ref()
            .is_some_and(|tool| !tool.name_sent)
            && let Some(name) = extract_name(body)
        {
            self.emit_tool_name(tool_index, name, result);
            progressed = true;
        }

        if let Some((_, arguments)) = parse_complete_body(body)
            && self.emit_arguments_diff(tool_index, arguments, result)
        {
            progressed = true;
        }

        progressed
    }

    fn emit_tool_name(&mut self, tool_index: usize, name: String, result: &mut ToolParseResult) {
        let Some(active_tool) = self.active_tool.as_mut() else {
            return;
        };
        if active_tool.name_sent {
            return;
        }
        active_tool.name_sent = true;
        result.calls.push(ToolCallDelta {
            tool_index,
            name: Some(name),
            arguments: String::new(),
        });
    }

    fn emit_arguments_diff(
        &mut self,
        tool_index: usize,
        arguments: String,
        result: &mut ToolParseResult,
    ) -> bool {
        let Some(active_tool) = self.active_tool.as_mut() else {
            return false;
        };
        let Some(diff) = arguments.strip_prefix(&active_tool.streamed_arguments) else {
            return false;
        };
        if diff.is_empty() {
            return false;
        }
        let diff = diff.to_string();

        active_tool.streamed_arguments = arguments;
        result.calls.push(ToolCallDelta {
            tool_index,
            name: None,
            arguments: diff,
        });
        true
    }
}

impl ToolParser for HermesToolParser {
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn adjust_request(&self, request: &mut ChatRequest) -> Result<()> {
        if request.tool_parsing_enabled() {
            request.decode_options.skip_special_tokens = false;
        }
        Ok(())
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        loop {
            let progressed = if self.active_tool.is_some() {
                self.process_tool_mode(&mut result)?
            } else {
                self.process_text_mode(&mut result)
            };
            if !progressed {
                break;
            }
        }

        Ok(result)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();

        match self.active_tool.as_ref().map(|tool| tool.tool_index) {
            Some(tool_index) if self.active_tool.as_ref().is_some_and(|tool| tool.name_sent) => {
                let keep_len = partial_prefix_len(&self.buffer, TOOL_CALL_END);
                let body_len = self.buffer.len().saturating_sub(keep_len);
                let body = self.buffer[..body_len].trim().to_string();
                self.emit_body_delta(tool_index, &body, &mut result);
            }
            Some(tool_index) => {
                let keep_len = partial_prefix_len(&self.buffer, TOOL_CALL_END);
                let body_len = self.buffer.len().saturating_sub(keep_len);
                let body = self.buffer[..body_len].trim();
                if let Some((name, arguments)) = parse_complete_body(body) {
                    self.emit_tool_name(tool_index, name, &mut result);
                    self.emit_arguments_diff(tool_index, arguments, &mut result);
                } else {
                    result.normal_text.push_str(TOOL_CALL_START);
                    result.normal_text.push_str(&self.buffer);
                }
            }
            None => {
                result.normal_text.push_str(&self.buffer);
            }
        }

        self.reset();
        Ok(result)
    }
}

fn parse_complete_body(body: &str) -> Option<(String, String)> {
    let value = serde_json::from_str::<Value>(body).ok()?;
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?.to_string();
    let arguments = object.get("arguments")?;
    Some((name, arguments.to_string()))
}

fn extract_name(body: &str) -> Option<String> {
    parse_complete_body(body)
        .map(|(name, _)| name)
        .or_else(|| extract_json_string_field(body, "name"))
}

fn extract_json_string_field(body: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\"");
    let key_pos = body.find(&key)?;
    let mut rest = &body[key_pos + key.len()..];
    rest = rest.trim_start();
    rest = rest.strip_prefix(':')?.trim_start();
    if !rest.starts_with('"') {
        return None;
    }

    let mut escaped = false;
    for (index, ch) in rest.char_indices().skip(1) {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return serde_json::from_str::<String>(&rest[..=index]).ok(),
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{HermesToolParser, TOOL_CALL_END, TOOL_CALL_START};
    use crate::parser::tool::{ToolCallDelta, ToolParseResult, ToolParser, ToolParserFactory};
    use crate::request::{ChatRequest, ChatToolChoice};

    fn collect_chunks(chunks: &[&str]) -> ToolParseResult {
        let mut parser = HermesToolParser::new(&[]);
        let mut result = ToolParseResult::default();
        for chunk in chunks {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        result.coalesce_calls()
    }

    fn collect_chars(text: &str) -> ToolParseResult {
        let chunks = text.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        let chunk_refs = chunks.iter().map(String::as_str).collect::<Vec<_>>();
        collect_chunks(&chunk_refs)
    }

    #[test]
    fn hermes_factory_registers_explicit_name() {
        let factory = ToolParserFactory::new();
        assert!(factory.contains("hermes"));
        factory.create("hermes", &[]).unwrap();
    }

    #[test]
    fn hermes_adjust_request_keeps_special_tokens_for_tools() {
        let parser = HermesToolParser::new(&[]);
        let mut request = ChatRequest::for_test();
        request.tool_choice = ChatToolChoice::Auto;
        request.tools.push(crate::request::ChatTool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        });

        parser.adjust_request(&mut request).unwrap();

        assert!(!request.decode_options.skip_special_tokens);
    }

    #[test]
    fn hermes_plain_text_passthrough() {
        let result = collect_chunks(&["This is plain text."]);
        assert_eq!(result.normal_text, "This is plain text.");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn hermes_parse_single_complete_tool_call() {
        let result = collect_chunks(&[
            r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
        ]);

        assert_eq!(result.normal_text, "");
        assert_eq!(
            result.calls,
            vec![ToolCallDelta {
                tool_index: 0,
                name: Some("get_weather".to_string()),
                arguments: r#"{"city":"Paris"}"#.to_string(),
            }]
        );
    }

    #[test]
    fn hermes_parse_content_then_tool_call() {
        let result = collect_chunks(&[
            r#"Sure. <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
        ]);

        assert_eq!(result.normal_text, "Sure. ");
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn hermes_parse_multiple_sequential_tool_calls() {
        let result = collect_chunks(&[
            r#"<tool_call>{"name":"search","arguments":{"q":"cats"}}</tool_call><tool_call>{"name":"search","arguments":{"q":"dogs"}}</tool_call>"#,
        ]);

        assert_eq!(
            result.calls,
            vec![
                ToolCallDelta {
                    tool_index: 0,
                    name: Some("search".to_string()),
                    arguments: r#"{"q":"cats"}"#.to_string(),
                },
                ToolCallDelta {
                    tool_index: 1,
                    name: Some("search".to_string()),
                    arguments: r#"{"q":"dogs"}"#.to_string(),
                },
            ]
        );
    }

    #[test]
    fn hermes_parse_split_start_and_end_tags() {
        let result = collect_chunks(&[
            "Hi <tool",
            "_call>{\"name\":\"final_answer\",\"arguments\":{\"trigger\":true}}</tool",
            "_call>",
        ]);

        assert_eq!(result.normal_text, "Hi ");
        assert_eq!(result.calls[0].name.as_deref(), Some("final_answer"));
        assert_eq!(result.calls[0].arguments, r#"{"trigger":true}"#);
    }

    #[test]
    fn hermes_parse_character_by_character() {
        let result = collect_chars(
            r#"A<tool_call>{"name":"get_current_temperature","arguments":{"location":"San Francisco","unit":"celsius"}}</tool_call>"#,
        );

        assert_eq!(result.normal_text, "A");
        assert_eq!(
            result.calls,
            vec![ToolCallDelta {
                tool_index: 0,
                name: Some("get_current_temperature".to_string()),
                arguments: r#"{"location":"San Francisco","unit":"celsius"}"#.to_string(),
            }]
        );
    }

    #[test]
    fn hermes_incomplete_before_name_falls_back_to_text_on_finish() {
        let result = collect_chunks(&["before ", TOOL_CALL_START, r#"{"na"#]);

        assert_eq!(result.normal_text, r#"before <tool_call>{"na"#);
        assert!(result.calls.is_empty());
    }

    #[test]
    fn hermes_invalid_complete_before_name_falls_back_to_text() {
        let result = collect_chunks(&[
            "before ",
            TOOL_CALL_START,
            r#"{"na":"broken","arguments":{"x":1"#,
            TOOL_CALL_END,
            " after",
        ]);

        assert_eq!(
            result.normal_text,
            r#"before <tool_call>{"na":"broken","arguments":{"x":1</tool_call> after"#
        );
        assert!(result.calls.is_empty());
    }

    #[test]
    fn hermes_default_parse_complete_uses_push_and_finish() {
        let mut parser = HermesToolParser::new(&[]);
        let result = parser
            .parse_complete(
                r#"Done <tool_call>{"name":"final_answer","arguments":{"trigger":true}}</tool_call>"#,
            )
            .unwrap();

        assert_eq!(result.normal_text, "Done ");
        assert_eq!(
            result.calls,
            vec![ToolCallDelta {
                tool_index: 0,
                name: Some("final_answer".to_string()),
                arguments: r#"{"trigger":true}"#.to_string(),
            }]
        );
    }
}
