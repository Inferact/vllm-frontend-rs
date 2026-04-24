use super::deepseek_v32::DeepSeekV32ToolParser;
use super::{Result, ToolParseResult, ToolParser};
use crate::request::{ChatRequest, ChatTool};

const TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";

/// Tool parser for DeepSeek V4 models.
///
/// V4 keeps the V3.2 DSML invoke/parameter grammar, but wraps tool calls in
/// `<｜DSML｜tool_calls>` instead of `<｜DSML｜function_calls>`.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/deepseekv4_tool_parser.py>
pub struct DeepSeekV4ToolParser(DeepSeekV32ToolParser);

impl ToolParser for DeepSeekV4ToolParser {
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self(DeepSeekV32ToolParser::with_start_token(
            tools,
            TOOL_CALLS_START,
        ))))
    }

    fn adjust_request(&self, request: &mut ChatRequest) -> Result<()> {
        self.0.adjust_request(request)
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.0.push(chunk)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        self.0.finish()
    }

    fn parse_complete(&mut self, output: &str) -> Result<ToolParseResult> {
        self.0.parse_complete(output)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{DeepSeekV4ToolParser, ToolParser};
    use crate::request::ChatTool;

    fn test_tools() -> Vec<ChatTool> {
        vec![ChatTool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" },
                    "date": { "type": "string" }
                }
            }),
            strict: None,
        }]
    }

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| {
                format!(
                    r#"<｜DSML｜parameter name="{name}" string="true">{value}</｜DSML｜parameter>"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"{function_name}\">\n{params}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )
    }

    fn make_parser() -> Box<dyn ToolParser> {
        DeepSeekV4ToolParser::create(&test_tools()).unwrap()
    }

    #[test]
    fn deepseek_v4_parse_complete_extracts_single_tool_call() {
        let mut parser = make_parser();
        let result = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "Beijing"), ("date", "2024-01-16")],
            ))
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "location": "Beijing",
                "date": "2024-01-16",
            })
        );
    }

    #[test]
    fn deepseek_v4_parse_complete_preserves_prefix_text() {
        let mut parser = make_parser();
        let output = format!(
            "Let me check. {}",
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let result = parser.parse_complete(&output).unwrap();

        assert_eq!(result.normal_text, "Let me check. ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v4_does_not_accept_v32_function_calls_block() {
        let mut parser = make_parser();
        let v32_block = build_tool_call("get_weather", &[("location", "NYC")])
            .replace("tool_calls", "function_calls");
        let result = parser.parse_complete(&v32_block).unwrap();

        assert_eq!(result.normal_text, v32_block);
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v4_streaming_extracts_complete_invokes() {
        let full_text = build_tool_call("get_weather", &[("location", "SF")]);

        // DSML markers contain multi-byte characters, so split on char boundaries.
        let mut chunks: Vec<&str> = Vec::new();
        let mut start = 0;
        let mut count = 0;
        for (idx, _) in full_text.char_indices() {
            if count == 5 {
                chunks.push(&full_text[start..idx]);
                start = idx;
                count = 0;
            }
            count += 1;
        }
        if start < full_text.len() {
            chunks.push(&full_text[start..]);
        }

        let mut parser = make_parser();
        let mut result = crate::parser::tool::ToolParseResult::default();
        for chunk in &chunks {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        let result = result.coalesce_calls();

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn deepseek_v4_adjust_request_keeps_special_tokens() {
        let parser = make_parser();
        let mut request = crate::request::ChatRequest::for_test();
        request.tools = test_tools();
        request.tool_choice = crate::request::ChatToolChoice::Auto;
        request.decode_options.skip_special_tokens = true;

        parser.adjust_request(&mut request).unwrap();
        assert!(!request.decode_options.skip_special_tokens);
    }
}
