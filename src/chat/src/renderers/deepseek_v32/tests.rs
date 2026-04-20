use std::fs;
use std::path::PathBuf;

use expect_test::{ExpectFile, expect_file};
use serde::Deserialize;
use serde_json::{Value, json};

use super::DeepSeekV32ChatRenderer;
use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::request::{ChatMessage, ChatRequest, ChatTool, ChatToolChoice};
use crate::{ChatRenderer, ChatRole};

#[derive(Debug, Deserialize)]
struct FixtureRequest {
    #[serde(default)]
    tools: Vec<FixtureTool>,
    messages: Vec<FixtureMessage>,
}

#[derive(Debug, Deserialize)]
struct FixtureTool {
    function: FixtureToolFunction,
}

#[derive(Debug, Deserialize)]
struct FixtureToolFunction {
    name: String,
    description: Option<String>,
    parameters: Value,
    #[serde(default)]
    strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
enum FixtureMessage {
    System {
        content: String,
    },
    Developer {
        content: String,
        #[serde(default)]
        tools: Vec<FixtureTool>,
    },
    User {
        content: String,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(default)]
        reasoning_content: String,
        #[serde(default)]
        tool_calls: Vec<FixtureToolCall>,
    },
    Tool {
        content: String,
        #[serde(default)]
        tool_call_id: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct FixtureToolCall {
    #[serde(default)]
    id: Option<String>,
    function: FixtureToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct FixtureToolCallFunction {
    name: String,
    arguments: String,
}

fn render_request(request: &ChatRequest) -> String {
    DeepSeekV32ChatRenderer::new()
        .render(request)
        .unwrap()
        .prompt
}

fn fixture_request(input_name: &str) -> ChatRequest {
    let fixture = fs::read_to_string(fixture_path(input_name)).unwrap();
    let fixture: FixtureRequest = serde_json::from_str(&fixture).unwrap();
    let mut request = ChatRequest {
        request_id: "deepseek-v32-fixture".to_string(),
        messages: fixture
            .messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| match message {
                FixtureMessage::System { content } => ChatMessage::system(content),
                FixtureMessage::Developer { content, tools } => ChatMessage::developer(
                    content,
                    (!tools.is_empty()).then(|| to_chat_tools(&tools)),
                ),
                FixtureMessage::User { content } => ChatMessage::user(content),
                FixtureMessage::Assistant {
                    content,
                    reasoning_content,
                    tool_calls,
                } => {
                    let mut blocks = Vec::new();
                    if !reasoning_content.is_empty() {
                        blocks.push(AssistantContentBlock::Reasoning {
                            text: reasoning_content,
                        });
                    }
                    if !content.is_empty() {
                        blocks.push(AssistantContentBlock::Text { text: content });
                    }
                    blocks.extend(tool_calls.into_iter().enumerate().map(
                        |(tool_index, tool_call)| {
                            AssistantContentBlock::ToolCall(AssistantToolCall {
                                id: tool_call.id.unwrap_or_else(|| {
                                    format!("fixture-tool-call-{index}-{tool_index}")
                                }),
                                name: tool_call.function.name,
                                arguments: tool_call.function.arguments,
                            })
                        },
                    ));
                    ChatMessage::assistant_blocks(blocks)
                }
                FixtureMessage::Tool {
                    content,
                    tool_call_id,
                } => ChatMessage::tool_response(
                    content,
                    tool_call_id.unwrap_or_else(|| format!("fixture-tool-response-{index}")),
                ),
            })
            .collect(),
        tools: to_chat_tools(&fixture.tools),
        tool_choice: if fixture.tools.is_empty() {
            ChatToolChoice::None
        } else {
            ChatToolChoice::Auto
        },
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));
    request
}

fn to_chat_tools(tools: &[FixtureTool]) -> Vec<ChatTool> {
    tools
        .iter()
        .map(|tool| ChatTool {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/renderers/deepseek_v32")
        .join("fixtures")
        .join(name)
}

fn assert_fixture(input_name: &str, expected: ExpectFile) {
    let request = fixture_request(input_name);
    let rendered = render_request(&request);
    expected.assert_eq(&rendered);
}

#[test]
fn renders_vllm_parity_prompt_for_request_level_tools_fixture() {
    assert_fixture(
        "test_input.json",
        expect_file!["fixtures/test_output_vllm_parity.txt"],
    );
}

#[test]
fn renders_official_search_fixture_without_date() {
    assert_fixture(
        "test_input_search_wo_date.json",
        expect_file!["fixtures/test_output_search_wo_date.txt"],
    );
}

#[test]
fn renders_official_search_fixture_with_date() {
    assert_fixture(
        "test_input_search_w_date.json",
        expect_file!["fixtures/test_output_search_w_date.txt"],
    );
}

#[test]
fn request_level_tools_are_lowered_as_synthetic_leading_system_message() {
    let mut request = ChatRequest {
        request_id: "deepseek-v32-tools".to_string(),
        messages: vec![
            ChatMessage::system("System prompt."),
            ChatMessage::text(ChatRole::User, "Hello"),
        ],
        tools: vec![ChatTool {
            name: "lookup".to_string(),
            description: Some("Look things up".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }),
            strict: None,
        }],
        tool_choice: ChatToolChoice::Auto,
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));

    let rendered = render_request(&request);

    assert!(rendered.starts_with("<｜begin▁of▁sentence｜>\n\n## Tools\n"));
    assert!(rendered.contains("</functions>\nSystem prompt."));
    assert!(rendered.ends_with("<｜User｜>Hello<｜Assistant｜><think>"));
}
