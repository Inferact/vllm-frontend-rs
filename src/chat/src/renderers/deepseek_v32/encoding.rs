//! DeepSeek V3.2 prompt renderer.

use serde::Serialize;
use serde_json::Value;
use serde_json_fmt::JsonFormat;

use crate::error::{Error, Result};
use crate::request::{ChatMessage, ChatRequest, ChatRole, ChatTool};
use crate::{AssistantContentBlock, AssistantMessageExt, AssistantToolCall};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";

/// DeepSeek's renderer uses `"chat"` vs `"thinking"` mode names. Keep the split explicit here so
/// downstream render branches stay easy to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingMode {
    Chat,
    Thinking,
}

/// Tool schema shape rendered inside the `<functions>` block.
#[derive(Debug, Serialize)]
struct RenderedToolSchema<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// Render one chat request into the prompt string.
pub(super) fn render_request(request: &ChatRequest) -> Result<String> {
    let thinking_mode = match request.enable_thinking()?.unwrap_or(false) {
        true => ThinkingMode::Thinking,
        false => ThinkingMode::Chat,
    };
    let drop_thinking = matches!(
        request.messages.last().map(ChatMessage::role),
        Some(ChatRole::User)
    );
    let render_offset = isize::from(request.tool_parsing_enabled());
    let last_user_render_index =
        find_last_user_render_index(request.messages.as_slice(), render_offset);
    let mut prompt = String::from(BOS_TOKEN);

    if request.tool_parsing_enabled() {
        prompt.push_str(&render_system_message("", &request.tools)?);
    }

    for (message_index, message) in request.messages.iter().enumerate() {
        prompt.push_str(&render_message(
            request.messages.as_slice(),
            message_index,
            message,
            render_offset,
            last_user_render_index,
            thinking_mode,
            drop_thinking,
        )?);
    }

    Ok(prompt)
}

/// Find the last user-like turn in render order.
///
/// `render_offset` is `1` when a synthetic tool-only system turn is rendered
/// before the real request messages, and `0` otherwise.
fn find_last_user_render_index(messages: &[ChatMessage], render_offset: isize) -> isize {
    messages
        .iter()
        .rposition(|message| matches!(message.role(), ChatRole::User | ChatRole::Developer))
        .map(|index| index as isize + render_offset)
        .unwrap_or(-1)
}

/// Render one real request message, using `render_offset` to account for any
/// synthetic tool-only system turn that was already emitted before the loop.
fn render_message(
    messages: &[ChatMessage],
    message_index: usize,
    message: &ChatMessage,
    render_offset: isize,
    last_user_render_index: isize,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
) -> Result<String> {
    let render_index = message_index as isize + render_offset;
    let opens_thinking = render_index == last_user_render_index;
    let after_last_user_turn = render_index > last_user_render_index;
    let after_or_at_last_user_turn = render_index >= last_user_render_index;

    match message {
        ChatMessage::System { content } => {
            render_system_message(&content.try_flatten_to_text()?, &[])
        }
        ChatMessage::Developer { content, tools } => render_developer_message(
            &content.try_flatten_to_text()?,
            tools.as_deref().unwrap_or(&[]),
            thinking_mode == ThinkingMode::Thinking && opens_thinking,
        ),
        ChatMessage::User { content } => render_user_message(
            &content.try_flatten_to_text()?,
            thinking_mode == ThinkingMode::Thinking && opens_thinking,
        ),
        ChatMessage::Assistant { content } => {
            let reasoning = assistant_reasoning(
                content,
                message_index,
                messages,
                thinking_mode,
                drop_thinking,
            );
            let tool_calls = content.tool_calls().collect::<Vec<_>>();
            render_assistant_message(
                thinking_mode == ThinkingMode::Thinking && after_last_user_turn,
                reasoning.as_deref(),
                &content.text(),
                &tool_calls,
                false,
            )
        }
        ChatMessage::ToolResponse { content, .. } => render_tool_message(
            messages,
            message_index,
            thinking_mode == ThinkingMode::Thinking && after_or_at_last_user_turn,
            &content.try_flatten_to_text()?,
        ),
    }
}

/// Historical assistant reasoning is dropped in DeepSeek thinking mode when the
/// final request turn is a new user message. Keep that policy local to
/// assistant rendering instead of cloning a second stripped message list.
fn assistant_reasoning(
    content: &[AssistantContentBlock],
    actual_index: usize,
    messages: &[ChatMessage],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
) -> Option<String> {
    if thinking_mode == ThinkingMode::Thinking
        && drop_thinking
        && actual_index < find_last_user_actual_index(messages)
    {
        None
    } else {
        content.reasoning()
    }
}

/// Return the last user/developer turn in the real request message list.
fn find_last_user_actual_index(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .rposition(|message| matches!(message.role(), ChatRole::User | ChatRole::Developer))
        .unwrap_or(usize::MAX)
}

/// Render a system turn, optionally followed by the DeepSeek tool preamble.
fn render_system_message(content: &str, tools: &[ChatTool]) -> Result<String> {
    let mut prompt = content.to_string();
    if !tools.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&render_tools(tools)?);
    }
    Ok(prompt)
}

/// Developer messages follow DeepSeek's native "wrap as user turn" behavior and
/// may carry message-local tools that are distinct from request-level tools.
fn render_developer_message(
    content: &str,
    tools: &[ChatTool],
    opens_thinking: bool,
) -> Result<String> {
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

    Ok(render_user_like_message(&content_developer, opens_thinking))
}

/// Plain user turns share the same wrapper shape as developer turns minus the
/// extra developer preamble.
fn render_user_message(content: &str, opens_thinking: bool) -> Result<String> {
    Ok(render_user_like_message(content, opens_thinking))
}

/// Shared helper for the `<｜User｜>...<｜Assistant｜>` wrapper used by both
/// real user turns and native developer turns.
fn render_user_like_message(content: &str, opens_thinking: bool) -> String {
    let mut prompt = format!("<｜User｜>{content}<｜Assistant｜>");
    if opens_thinking {
        prompt.push_str(THINKING_START_TOKEN);
    } else {
        prompt.push_str(THINKING_END_TOKEN);
    }
    prompt
}

/// Render one tool result turn and decide whether it opens or closes the shared
/// `<function_results>` block for the preceding assistant tool-call message.
fn render_tool_message(
    messages: &[ChatMessage],
    message_index: usize,
    resumes_thinking: bool,
    content: &str,
) -> Result<String> {
    let Some(prev_assistant_idx) = previous_non_tool_actual_index(messages, message_index) else {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: missing previous assistant message".to_string(),
        ));
    };

    let ChatMessage::Assistant {
        content: assistant_content,
    } = &messages[prev_assistant_idx]
    else {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: previous non-tool message is not assistant"
                .to_string(),
        ));
    };

    let tool_call_count = assistant_content.tool_calls().count();
    let tool_call_order = message_index - prev_assistant_idx;
    if tool_call_count < tool_call_order {
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

    if tool_call_order == tool_call_count {
        prompt.push_str("\n</function_results>");
        prompt.push_str("\n\n");
        if resumes_thinking {
            prompt.push_str(THINKING_START_TOKEN);
        } else {
            prompt.push_str(THINKING_END_TOKEN);
        }
    }

    Ok(prompt)
}

/// Walk backwards from a tool result to the assistant turn that emitted the
/// corresponding tool calls, skipping any earlier sibling tool results in the
/// same batch.
fn previous_non_tool_actual_index(messages: &[ChatMessage], actual_index: usize) -> Option<usize> {
    let mut current = actual_index.checked_sub(1)?;
    while matches!(messages[current], ChatMessage::ToolResponse { .. }) {
        current = current.checked_sub(1)?;
    }
    Some(current)
}

/// Render one assistant turn, including optional reasoning, DSML tool calls,
/// and the trailing DeepSeek EOS marker.
fn render_assistant_message(
    after_last_user_turn: bool,
    reasoning: Option<&str>,
    content: &str,
    tool_calls: &[&AssistantToolCall],
    prefix: bool,
) -> Result<String> {
    let mut tool_calls_content = String::new();
    if !tool_calls.is_empty() {
        let rendered_calls = tool_calls
            .iter()
            .map(|tool_call| render_tool_call(tool_call))
            .collect::<Result<Vec<_>>>()?;
        tool_calls_content.push_str("\n\n<｜DSML｜function_calls>\n");
        tool_calls_content.push_str(&rendered_calls.join("\n"));
        tool_calls_content.push_str("\n</｜DSML｜function_calls>");
    }

    let mut thinking_part = String::new();
    if after_last_user_turn {
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

/// Render one assistant tool call in DeepSeek's DSML XML-like format.
fn render_tool_call(tool_call: &AssistantToolCall) -> Result<String> {
    Ok(format!(
        "<{DSML_TOKEN}invoke name=\"{}\">\n{}\n</{DSML_TOKEN}invoke>",
        tool_call.name,
        encode_arguments_to_dsml(tool_call)?,
    ))
}

/// Convert one assistant tool-call arguments object into DeepSeek's DSML
/// parameter form.
///
/// String values are emitted raw with `string="true"`, while all other JSON
/// values are rendered with JSON syntax and `string="false"`.
fn encode_arguments_to_dsml(tool_call: &AssistantToolCall) -> Result<String> {
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

/// Render the full DeepSeek tool preamble shown to the model.
fn render_tools(tools: &[ChatTool]) -> Result<String> {
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

/// Serialize one typed tool schema into the JSON shape embedded inside
/// `<functions>`.
fn render_tool_schema(tool: &ChatTool) -> Result<String> {
    json_dumps(&RenderedToolSchema {
        name: &tool.name,
        description: tool.description.as_deref(),
        parameters: &tool.parameters,
        strict: tool.strict,
    })
}

/// Python-compatible compact JSON serialization used by the upstream encoder.
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
