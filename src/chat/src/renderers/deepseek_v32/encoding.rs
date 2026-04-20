//! DeepSeek V3.2 prompt renderer.

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
///
/// The only extra bookkeeping beyond the typed request itself is:
/// - whether request-level tools should behave like a synthetic leading system turn
/// - where the last user/developer turn lands in that synthetic index space
/// - whether historical assistant reasoning should be dropped because the final request turn is a
///   new user question
pub(super) fn render_request(request: &ChatRequest) -> Result<String> {
    let thinking_mode = thinking_mode_from_request(request);
    let drop_thinking = matches!(
        request.messages.last().map(ChatMessage::role),
        Some(ChatRole::User)
    );
    let synthetic_tool_system = request.tool_parsing_enabled();
    let last_user_idx =
        find_last_user_virtual_index(request.messages.as_slice(), synthetic_tool_system);
    let mut prompt = String::from(BOS_TOKEN);

    for virtual_index in
        0..virtual_message_count(request.messages.as_slice(), synthetic_tool_system)
    {
        prompt.push_str(&render_message(
            request,
            virtual_index,
            synthetic_tool_system,
            last_user_idx,
            thinking_mode,
            drop_thinking,
        )?);
    }

    Ok(prompt)
}

/// DeepSeek V3.2 accepts loose truthiness for both `thinking` and
/// `enable_thinking`, so mirror that contract instead of requiring a strict
/// boolean.
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

/// Return the number of turns visible to the DeepSeek renderer.
///
/// When request-level tools are enabled, the renderer treats them as an empty
/// leading system message carrying those tools. We model that as one extra
/// "virtual" message slot instead of allocating a new `ChatMessage`.
fn virtual_message_count(messages: &[ChatMessage], synthetic_tool_system: bool) -> usize {
    messages.len() + usize::from(synthetic_tool_system)
}

/// Find the last user-like turn in the same virtual index space used during
/// rendering.
///
/// With request-level tools enabled:
/// - virtual index `0` is the synthetic tool-only system turn
/// - virtual index `1` maps to the real `messages[0]`
/// - and so on
///
/// This offset lets downstream logic talk about one consistent index space when
/// deciding whether to open `<think>` after the final user/developer turn.
fn find_last_user_virtual_index(messages: &[ChatMessage], synthetic_tool_system: bool) -> isize {
    let offset = isize::from(synthetic_tool_system);
    messages
        .iter()
        .rposition(|message| matches!(message.role(), ChatRole::User | ChatRole::Developer))
        .map(|index| index as isize + offset)
        .unwrap_or(-1)
}

/// Render one visible turn in virtual-index order.
///
/// The first virtual slot may be the synthetic request-level-tools system turn;
/// all later slots map back to a real message in `request.messages`.
fn render_message(
    request: &ChatRequest,
    virtual_index: usize,
    synthetic_tool_system: bool,
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
) -> Result<String> {
    if synthetic_tool_system && virtual_index == 0 {
        return render_system_message("", &request.tools);
    }

    let actual_index = actual_index(virtual_index, synthetic_tool_system);
    let message = request.messages.get(actual_index).ok_or_else(|| {
        Error::ChatTemplate(format!(
            "DeepSeek V3.2 message index {virtual_index} out of range for {} messages",
            virtual_message_count(request.messages.as_slice(), synthetic_tool_system)
        ))
    })?;

    match message {
        ChatMessage::System { content } => {
            render_system_message(&content.try_flatten_to_text()?, &[])
        }
        ChatMessage::Developer { content, tools } => render_developer_message(
            &content.try_flatten_to_text()?,
            tools.as_deref().unwrap_or(&[]),
            virtual_index as isize,
            last_user_idx,
            thinking_mode,
        ),
        ChatMessage::User { content } => render_user_message(
            &content.try_flatten_to_text()?,
            virtual_index as isize,
            last_user_idx,
            thinking_mode,
        ),
        ChatMessage::Assistant { content } => {
            let reasoning = assistant_reasoning(
                content,
                actual_index,
                request.messages.as_slice(),
                thinking_mode,
                drop_thinking,
            );
            let tool_calls = content.tool_calls().collect::<Vec<_>>();
            render_assistant_message(
                virtual_index as isize,
                last_user_idx,
                thinking_mode,
                reasoning.as_deref(),
                &content.text(),
                &tool_calls,
                false,
            )
        }
        ChatMessage::ToolResponse { content, .. } => render_tool_message(
            request.messages.as_slice(),
            actual_index,
            virtual_index as isize,
            last_user_idx,
            thinking_mode,
            &content.try_flatten_to_text()?,
        ),
    }
}

/// Convert a render-time virtual index back to the underlying `request.messages`
/// index. This is only valid for non-synthetic turns.
fn actual_index(virtual_index: usize, synthetic_tool_system: bool) -> usize {
    virtual_index - usize::from(synthetic_tool_system)
}

/// Historical assistant reasoning is dropped in DeepSeek thinking mode when the
/// final request turn is a new user message. Keep that policy local to
/// assistant rendering instead of cloning a second stripped message list.
fn assistant_reasoning(
    content: &[crate::AssistantContentBlock],
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
    virtual_index: isize,
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
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

    Ok(render_user_like_message(
        &content_developer,
        virtual_index,
        last_user_idx,
        thinking_mode,
    ))
}

/// Plain user turns share the same wrapper shape as developer turns minus the
/// extra developer preamble.
fn render_user_message(
    content: &str,
    virtual_index: isize,
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
) -> Result<String> {
    Ok(render_user_like_message(
        content,
        virtual_index,
        last_user_idx,
        thinking_mode,
    ))
}

/// Shared helper for the `<｜User｜>...<｜Assistant｜>` wrapper used by both
/// real user turns and native developer turns.
fn render_user_like_message(
    content: &str,
    virtual_index: isize,
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
) -> String {
    let mut prompt = format!("<｜User｜>{content}<｜Assistant｜>");
    if virtual_index == last_user_idx && thinking_mode == ThinkingMode::Thinking {
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
    actual_index: usize,
    virtual_index: isize,
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
    content: &str,
) -> Result<String> {
    let Some(prev_assistant_idx) = previous_non_tool_actual_index(messages, actual_index) else {
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
    let tool_call_order = actual_index - prev_assistant_idx;
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
        if virtual_index >= last_user_idx && thinking_mode == ThinkingMode::Thinking {
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
    virtual_index: isize,
    last_user_idx: isize,
    thinking_mode: ThinkingMode,
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
    if thinking_mode == ThinkingMode::Thinking && virtual_index > last_user_idx {
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
