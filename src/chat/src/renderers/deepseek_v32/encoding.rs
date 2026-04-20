use serde::Serialize;
use serde_json::Value;
use serde_json_fmt::JsonFormat;

use crate::error::{Error, Result};
use crate::request::{ChatMessage, ChatRequest, ChatRole, ChatTool};
use crate::{AssistantMessageExt, AssistantToolCall};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingMode {
    Chat,
    Thinking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EncodedMessage {
    System {
        content: String,
        tools: Vec<EncodedTool>,
    },
    Developer {
        content: String,
        tools: Vec<EncodedTool>,
    },
    User {
        content: String,
    },
    Assistant {
        reasoning: Option<String>,
        content: String,
        tool_calls: Vec<EncodedToolCall>,
        prefix: bool,
    },
    Tool {
        content: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EncodedTool {
    name: String,
    description: Option<String>,
    parameters: Value,
    strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EncodedToolCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct RenderedToolSchema<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

pub(super) fn render_request(request: &ChatRequest) -> Result<String> {
    let thinking_mode = thinking_mode_from_request(request);
    let drop_thinking = matches!(
        request.messages.last().map(ChatMessage::role),
        Some(ChatRole::User)
    );

    let mut messages = lower_messages(&request.messages)?;
    if request.tool_parsing_enabled() {
        messages.insert(
            0,
            EncodedMessage::System {
                content: String::new(),
                tools: lower_tools(&request.tools),
            },
        );
    }

    encode_messages(messages, thinking_mode, drop_thinking)
}

fn thinking_mode_from_request(request: &ChatRequest) -> ThinkingMode {
    let kwargs = &request.chat_options.template_kwargs;
    if kwargs.get("thinking").is_some_and(is_truthy)
        || kwargs.get("enable_thinking").is_some_and(is_truthy)
    {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Chat
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => match (value.as_i64(), value.as_u64(), value.as_f64()) {
            (Some(value), _, _) => value != 0,
            (_, Some(value), _) => value != 0,
            (_, _, Some(value)) => value != 0.0,
            _ => false,
        },
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn lower_messages(messages: &[ChatMessage]) -> Result<Vec<EncodedMessage>> {
    messages.iter().map(lower_message).collect()
}

fn lower_message(message: &ChatMessage) -> Result<EncodedMessage> {
    Ok(match message {
        ChatMessage::System { content } => EncodedMessage::System {
            content: content.try_flatten_to_text()?,
            tools: Vec::new(),
        },
        ChatMessage::Developer { content, tools } => EncodedMessage::Developer {
            content: content.try_flatten_to_text()?,
            tools: tools.as_deref().map(lower_tools).unwrap_or_default(),
        },
        ChatMessage::User { content } => EncodedMessage::User {
            content: content.try_flatten_to_text()?,
        },
        ChatMessage::Assistant { content } => EncodedMessage::Assistant {
            reasoning: content.reasoning(),
            content: content.text(),
            tool_calls: content.tool_calls().map(lower_tool_call).collect(),
            prefix: false,
        },
        ChatMessage::ToolResponse { content, .. } => EncodedMessage::Tool {
            content: content.try_flatten_to_text()?,
        },
    })
}

fn lower_tools(tools: &[ChatTool]) -> Vec<EncodedTool> {
    tools
        .iter()
        .map(|tool| EncodedTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict: tool.strict,
        })
        .collect()
}

fn lower_tool_call(tool_call: &AssistantToolCall) -> EncodedToolCall {
    EncodedToolCall {
        name: tool_call.name.clone(),
        arguments: tool_call.arguments.clone(),
    }
}

fn encode_messages(
    messages: Vec<EncodedMessage>,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
) -> Result<String> {
    let full_messages = if thinking_mode == ThinkingMode::Thinking && drop_thinking {
        drop_thinking_messages(&messages)
    } else {
        messages
    };

    let mut prompt = String::from(BOS_TOKEN);
    for index in 0..full_messages.len() {
        prompt.push_str(&render_message(index, &full_messages, thinking_mode)?);
    }
    Ok(prompt)
}

fn drop_thinking_messages(messages: &[EncodedMessage]) -> Vec<EncodedMessage> {
    let last_user_idx = find_last_user_index(messages);
    let mut stripped = Vec::with_capacity(messages.len());

    for (index, message) in messages.iter().enumerate() {
        if matches!(
            message.role(),
            MessageRole::User | MessageRole::System | MessageRole::Tool
        ) || index as isize >= last_user_idx
        {
            stripped.push(message.clone());
            continue;
        }

        if let EncodedMessage::Assistant {
            content,
            tool_calls,
            prefix,
            ..
        } = message
        {
            stripped.push(EncodedMessage::Assistant {
                reasoning: None,
                content: content.clone(),
                tool_calls: tool_calls.clone(),
                prefix: *prefix,
            });
        }
    }

    stripped
}

fn find_last_user_index(messages: &[EncodedMessage]) -> isize {
    messages
        .iter()
        .rposition(|message| matches!(message.role(), MessageRole::User | MessageRole::Developer))
        .map(|index| index as isize)
        .unwrap_or(-1)
}

fn render_message(
    index: usize,
    messages: &[EncodedMessage],
    thinking_mode: ThinkingMode,
) -> Result<String> {
    let message = messages.get(index).ok_or_else(|| {
        Error::ChatTemplate(format!(
            "DeepSeek V3.2 message index {index} out of range for {} messages",
            messages.len()
        ))
    })?;
    let last_user_idx = find_last_user_index(messages);

    let prompt = match message {
        EncodedMessage::System { content, tools } => {
            let mut prompt = content.clone();
            if !tools.is_empty() {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools)?);
            }
            prompt
        }
        EncodedMessage::Developer { content, tools } => {
            if content.is_empty() {
                return Err(Error::ChatTemplate(
                    "invalid DeepSeek V3.2 developer message: empty content".to_string(),
                ));
            }

            let mut content_developer = String::new();
            if !tools.is_empty() {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(tools)?);
            }
            content_developer.push_str("\n\n# The user's message is: ");
            content_developer.push_str(content);

            let mut prompt = format!("<｜User｜>{content_developer}<｜Assistant｜>");
            if index as isize == last_user_idx && thinking_mode == ThinkingMode::Thinking {
                prompt.push_str(THINKING_START_TOKEN);
            } else {
                prompt.push_str(THINKING_END_TOKEN);
            }
            prompt
        }
        EncodedMessage::User { content } => {
            let mut prompt = format!("<｜User｜>{content}<｜Assistant｜>");
            if index as isize == last_user_idx && thinking_mode == ThinkingMode::Thinking {
                prompt.push_str(THINKING_START_TOKEN);
            } else {
                prompt.push_str(THINKING_END_TOKEN);
            }
            prompt
        }
        EncodedMessage::Tool { content } => {
            render_tool_message(index, messages, last_user_idx, thinking_mode, content)?
        }
        EncodedMessage::Assistant {
            reasoning,
            content,
            tool_calls,
            prefix,
        } => render_assistant_message(
            index,
            last_user_idx,
            thinking_mode,
            reasoning.as_deref(),
            content,
            tool_calls,
            *prefix,
        )?,
    };

    Ok(prompt)
}

fn render_tool_message(
    index: usize,
    messages: &[EncodedMessage],
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
    content: &str,
) -> Result<String> {
    let Some(prev_assistant_idx) = previous_non_tool_index(index, messages) else {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: missing previous assistant message".to_string(),
        ));
    };

    let EncodedMessage::Assistant { tool_calls, .. } = &messages[prev_assistant_idx] else {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: previous non-tool message is not assistant"
                .to_string(),
        ));
    };

    let tool_call_order = index - prev_assistant_idx;
    if tool_calls.len() < tool_call_order {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: no matching assistant tool call".to_string(),
        ));
    }

    let mut prompt = String::new();
    if tool_call_order == 1 {
        prompt.push_str("\n\n<function_results>");
    }

    prompt.push_str("\n<result>");
    prompt.push_str(content);
    prompt.push_str("</result>");

    if tool_call_order == tool_calls.len() {
        prompt.push_str("\n</function_results>");
        if index as isize >= last_user_idx && thinking_mode == ThinkingMode::Thinking {
            prompt.push_str("\n\n");
            prompt.push_str(THINKING_START_TOKEN);
        } else {
            prompt.push_str("\n\n");
            prompt.push_str(THINKING_END_TOKEN);
        }
    }

    Ok(prompt)
}

fn previous_non_tool_index(index: usize, messages: &[EncodedMessage]) -> Option<usize> {
    let mut current = index.checked_sub(1)?;
    while matches!(messages[current], EncodedMessage::Tool { .. }) {
        current = current.checked_sub(1)?;
    }
    Some(current)
}

fn render_assistant_message(
    index: usize,
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
    reasoning: Option<&str>,
    content: &str,
    tool_calls: &[EncodedToolCall],
    prefix: bool,
) -> Result<String> {
    let mut tool_calls_content = String::new();
    if !tool_calls.is_empty() {
        let rendered_calls = tool_calls
            .iter()
            .map(render_tool_call)
            .collect::<Result<Vec<_>>>()?;
        tool_calls_content.push_str("\n\n<｜DSML｜function_calls>\n");
        tool_calls_content.push_str(&rendered_calls.join("\n"));
        tool_calls_content.push_str("\n</｜DSML｜function_calls>");
    }

    let mut thinking_part = String::new();
    if thinking_mode == ThinkingMode::Thinking && index as isize > last_user_idx {
        if reasoning.is_none() && tool_calls.is_empty() {
            return Err(Error::ChatTemplate(
                "invalid DeepSeek V3.2 assistant message after last user message: expected reasoning or tool calls"
                    .to_string(),
            ));
        }

        thinking_part.push_str(reasoning.unwrap_or_default());
        thinking_part.push_str(THINKING_END_TOKEN);
    }

    if tool_calls.is_empty() && prefix {
        return Ok(content.to_string());
    }

    Ok(format!(
        "{thinking_part}{content}{tool_calls_content}{EOS_TOKEN}"
    ))
}

fn render_tool_call(tool_call: &EncodedToolCall) -> Result<String> {
    Ok(format!(
        "<{DSML_TOKEN}invoke name=\"{}\">\n{}\n</{DSML_TOKEN}invoke>",
        tool_call.name,
        encode_arguments_to_dsml(tool_call)?,
    ))
}

fn encode_arguments_to_dsml(tool_call: &EncodedToolCall) -> Result<String> {
    let arguments: Value = serde_json::from_str(&tool_call.arguments).map_err(|error| {
        Error::ChatTemplate(format!(
            "assistant tool call has invalid JSON arguments for DeepSeek V3.2: {error}"
        ))
    })?;
    let Some(arguments) = arguments.as_object() else {
        return Err(Error::ChatTemplate(
            "assistant tool call arguments for DeepSeek V3.2 must be a JSON object".to_string(),
        ));
    };

    let mut rendered = Vec::with_capacity(arguments.len());
    for (key, value) in arguments {
        let value = match value {
            Value::String(value) => value.clone(),
            value => json_dumps(value)?,
        };
        rendered.push(format!(
            "<{DSML_TOKEN}parameter name=\"{key}\" string=\"{}\">{value}</{DSML_TOKEN}parameter>",
            if arguments[key].is_string() {
                "true"
            } else {
                "false"
            }
        ));
    }

    Ok(rendered.join("\n"))
}

fn render_tools(tools: &[EncodedTool]) -> Result<String> {
    let tool_schemas = tools
        .iter()
        .map(render_tool_schema)
        .collect::<Result<Vec<_>>>()?;

    Ok(format!(
        r#"## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<｜DSML｜function_calls>" block like the following as part of your reply to the user:
<｜DSML｜function_calls>
<｜DSML｜invoke name="$FUNCTION_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$FUNCTION_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<｜DSML｜function_calls>
...
</｜DSML｜function_calls>

<function_results>
...
</function_results>

<think>...thinking about results</think>

Here are the functions available in JSONSchema format:
<functions>
{}
</functions>
"#,
        tool_schemas.join("\n")
    ))
}

fn render_tool_schema(tool: &EncodedTool) -> Result<String> {
    json_dumps(&RenderedToolSchema {
        name: &tool.name,
        description: tool.description.as_deref(),
        parameters: &tool.parameters,
        strict: tool.strict,
    })
}

fn json_dumps<T: Serialize>(value: &T) -> Result<String> {
    JsonFormat::new()
        .comma(", ")
        .expect("literal comma separator is valid JSON")
        .colon(": ")
        .expect("literal colon separator is valid JSON")
        .ascii(false)
        .format_to_string(value)
        .map_err(|error| {
            Error::ChatTemplate(format!(
                "failed to serialize DeepSeek V3.2 JSON payload: {error}"
            ))
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

impl EncodedMessage {
    fn role(&self) -> MessageRole {
        match self {
            Self::System { .. } => MessageRole::System,
            Self::Developer { .. } => MessageRole::Developer,
            Self::User { .. } => MessageRole::User,
            Self::Assistant { .. } => MessageRole::Assistant,
            Self::Tool { .. } => MessageRole::Tool,
        }
    }
}
