use std::fs;
use std::path::PathBuf;

use expect_test::{ExpectFile, expect_file};
use serde::Deserialize;
use serde_json::Value;

use super::DeepSeekV4ChatRenderer;
use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::request::{ChatMessage, ChatRequest, ChatTool, ChatToolChoice, GenerationPromptMode};
use crate::{ChatRenderer, ChatRole};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FixtureFile {
    WithTools(FixtureRequest),
    MessagesOnly(Vec<FixtureMessage>),
}

#[derive(Debug, Deserialize)]
struct FixtureRequest {
    #[serde(default)]
    tools: Vec<FixtureTool>,
    messages: Vec<FixtureMessage>,
}

impl FixtureFile {
    fn into_parts(self) -> (Vec<FixtureTool>, Vec<FixtureMessage>) {
        match self {
            Self::WithTools(req) => (req.tools, req.messages),
            Self::MessagesOnly(messages) => (Vec::new(), messages),
        }
    }
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
    DeepSeekV4ChatRenderer::new()
        .render(request)
        .unwrap()
        .prompt
        .into_text()
        .expect("deepseek v4 renderer should return text prompt")
}

fn fixture_request(input_name: &str) -> ChatRequest {
    let fixture = fs::read_to_string(fixture_path(input_name)).unwrap();
    let fixture: FixtureFile = serde_json::from_str(&fixture).unwrap();
    let (fixture_tools, fixture_messages) = fixture.into_parts();
    let mut request = ChatRequest {
        request_id: "deepseek-v4-fixture".to_string(),
        messages: fixture_messages
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
        tools: to_chat_tools(&fixture_tools),
        tool_choice: if fixture_tools.is_empty() {
            ChatToolChoice::None
        } else {
            ChatToolChoice::Auto
        },
        ..ChatRequest::for_test()
    };
    if matches!(
        request.messages.last().map(ChatMessage::role),
        Some(ChatRole::Assistant)
    ) {
        request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
    }
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
        .join("src/renderer/deepseek_v4")
        .join("fixtures")
        .join(name)
}

fn assert_fixture(input_name: &str, expected: ExpectFile) {
    let request = fixture_request(input_name);
    let rendered = render_request(&request);
    expected.assert_eq(&rendered);
}

#[test]
fn renders_v4_fixture_1_tool_call_round_trip() {
    assert_fixture(
        "test_input_1.json",
        expect_file!["fixtures/test_output_1.txt"],
    );
}

#[test]
fn renders_v4_fixture_2_multi_turn_drop_thinking() {
    assert_fixture(
        "test_input_2.json",
        expect_file!["fixtures/test_output_2.txt"],
    );
}
