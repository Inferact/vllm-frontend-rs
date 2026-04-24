//! DeepSeek V4 prompt renderer.
//!
//! Original Python implementation:
//! <https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/deepseek_v4_encoding.py>

use std::fmt::Write as _;

use serde::Serialize;
use serde_json::Value;
use serde_json_fmt::JsonFormat;

use crate::error::{Error, Result};
use crate::request::{ChatContent, ChatMessage, ChatRequest, ChatTool};
use crate::{AssistantContentBlock, AssistantMessageExt, AssistantToolCall};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingMode {
    Chat,
    Thinking,
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

/// One pre-processed entry, mirroring Python's post-merge message list.
///
/// Tool-response messages are folded into a preceding pseudo-`User` entry that
/// carries `<tool_result>...</tool_result>` blocks alongside any plain user
/// text, matching `merge_tool_messages()` in the Python encoder.
enum RenderEntry<'a> {
    System {
        content: Option<&'a ChatContent>,
        tools: &'a [ChatTool],
    },
    Developer {
        content: &'a ChatContent,
        tools: &'a [ChatTool],
    },
    User {
        blocks: Vec<UserBlock<'a>>,
    },
    Assistant {
        content: &'a [AssistantContentBlock],
    },
}

enum UserBlock<'a> {
    Text(&'a ChatContent),
    ToolResult(&'a ChatContent),
}

impl<'a> RenderEntry<'a> {
    fn is_user_like(&self) -> bool {
        matches!(
            self,
            RenderEntry::User { .. } | RenderEntry::Developer { .. }
        )
    }
}

/// Render one chat request into the final prompt string.
pub(super) fn render_request(request: &ChatRequest) -> Result<String> {
    let thinking_mode = match request.enable_thinking()?.unwrap_or(false) {
        true => ThinkingMode::Thinking,
        false => ThinkingMode::Chat,
    };

    let entries = preprocess_messages(request)?;

    // Mirror Python: if any rendered message carries tools, keep historical
    // assistant reasoning instead of dropping it.
    let any_tools = entries.iter().any(|entry| match entry {
        RenderEntry::System { tools, .. } | RenderEntry::Developer { tools, .. } => {
            !tools.is_empty()
        }
        _ => false,
    });
    let drop_thinking = !any_tools;

    let last_user_idx = entries
        .iter()
        .rposition(RenderEntry::is_user_like)
        .map(|idx| idx as isize)
        .unwrap_or(-1);

    let mut out = String::from(BOS_TOKEN);

    for (idx, entry) in entries.iter().enumerate() {
        let idx_isize = idx as isize;
        let opens_thinking = idx_isize >= last_user_idx;

        match entry {
            RenderEntry::System { content, tools } => {
                render_system_message(&mut out, *content, tools)?;
            }
            RenderEntry::Developer { content, tools } => {
                render_developer_message(&mut out, content, tools)?;
            }
            RenderEntry::User { blocks } => {
                render_user_message(&mut out, blocks)?;
            }
            RenderEntry::Assistant { content } => {
                // Mirror Python: thinking block (reasoning + </think>) is
                // emitted whenever thinking is active and reasoning isn't
                // dropped — i.e. drop_thinking is off OR this turn lies
                // strictly after the last user turn.
                let emit_thinking_block = thinking_mode == ThinkingMode::Thinking
                    && (!drop_thinking || idx_isize > last_user_idx);
                render_assistant_message(&mut out, emit_thinking_block, content)?;
            }
        }

        // Append the assistant transition after a user-like turn whose
        // following entry is an assistant turn (or end-of-list).
        let next_is_assistant = entries
            .get(idx + 1)
            .map(|e| matches!(e, RenderEntry::Assistant { .. }))
            .unwrap_or(true);

        if entry.is_user_like() && next_is_assistant {
            out.push_str(ASSISTANT_SP_TOKEN);
            let want_thinking_start =
                thinking_mode == ThinkingMode::Thinking && (!drop_thinking || opens_thinking);
            if want_thinking_start {
                out.push_str(THINKING_START_TOKEN);
            } else {
                out.push_str(THINKING_END_TOKEN);
            }
        }
    }

    Ok(out)
}

/// Build the post-merge entry list.
///
/// `request.tools` are attached to the first system message, mirroring the
/// fixture-generation convention where top-level tools live on the first
/// message. If no system message exists, a synthetic empty system entry is
/// prepended to carry the tools.
fn preprocess_messages(request: &ChatRequest) -> Result<Vec<RenderEntry<'_>>> {
    let mut entries: Vec<RenderEntry<'_>> = Vec::with_capacity(request.messages.len() + 1);
    let mut tools_attached = false;
    let request_tools = if request.tool_parsing_enabled() {
        request.tools.as_slice()
    } else {
        &[]
    };

    // Index of the last entry that owns an active tool-result run, or `None`
    // when the next ToolResponse should start a fresh user entry.
    let mut tool_run_idx: Option<usize> = None;

    for message in &request.messages {
        match message {
            ChatMessage::System { content } => {
                let tools = if !tools_attached {
                    tools_attached = true;
                    request_tools
                } else {
                    &[][..]
                };
                entries.push(RenderEntry::System {
                    content: Some(content),
                    tools,
                });
                tool_run_idx = None;
            }
            ChatMessage::Developer { content, tools } => {
                let dev_tools = tools.as_deref().unwrap_or(&[]);
                entries.push(RenderEntry::Developer {
                    content,
                    tools: dev_tools,
                });
                tool_run_idx = None;
            }
            ChatMessage::User { content } => {
                entries.push(RenderEntry::User {
                    blocks: vec![UserBlock::Text(content)],
                });
                tool_run_idx = None;
            }
            ChatMessage::Assistant { content } => {
                entries.push(RenderEntry::Assistant { content });
                tool_run_idx = None;
            }
            ChatMessage::ToolResponse { content, .. } => {
                let block = UserBlock::ToolResult(content);
                if let Some(idx) = tool_run_idx
                    && let Some(RenderEntry::User { blocks }) = entries.get_mut(idx)
                {
                    blocks.push(block);
                } else {
                    entries.push(RenderEntry::User {
                        blocks: vec![block],
                    });
                    tool_run_idx = Some(entries.len() - 1);
                }
            }
        }
    }

    // If request.tools were never attached (no system message exists), prepend
    // a synthetic empty system entry to carry them.
    if !tools_attached && !request_tools.is_empty() {
        entries.insert(
            0,
            RenderEntry::System {
                content: None,
                tools: request_tools,
            },
        );
    }

    Ok(entries)
}

/// Render the tool preamble shown to the model, V4 flavor.
fn render_tools(out: &mut String, tools: &[ChatTool]) -> Result<()> {
    out.push_str(
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"#,
    );

    for (index, tool) in tools.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        render_tool_schema(out, tool)?;
    }

    out.push_str(
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n",
    );
    Ok(())
}

fn render_tool_schema(out: &mut String, tool: &ChatTool) -> Result<()> {
    out.push_str(&json_dumps(&RenderedToolSchema {
        name: &tool.name,
        description: tool.description.as_deref(),
        parameters: &tool.parameters,
        strict: tool.strict,
    })?);
    Ok(())
}

fn render_system_message(
    out: &mut String,
    content: Option<&ChatContent>,
    tools: &[ChatTool],
) -> Result<()> {
    if let Some(content) = content {
        write_chat_content(out, content)?;
    }
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools)?;
    }
    Ok(())
}

fn render_developer_message(
    out: &mut String,
    content: &ChatContent,
    tools: &[ChatTool],
) -> Result<()> {
    if content.is_empty() {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V4 developer message: empty content".to_string(),
        ));
    }

    out.push_str(USER_SP_TOKEN);
    write_chat_content(out, content)?;
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools)?;
    }
    Ok(())
}

fn render_user_message(out: &mut String, blocks: &[UserBlock<'_>]) -> Result<()> {
    out.push_str(USER_SP_TOKEN);

    let only_text = blocks
        .iter()
        .all(|block| matches!(block, UserBlock::Text(_)));

    if only_text {
        for block in blocks {
            if let UserBlock::Text(content) = block {
                write_chat_content(out, content)?;
            }
        }
    } else {
        for (index, block) in blocks.iter().enumerate() {
            if index > 0 {
                out.push_str("\n\n");
            }
            match block {
                UserBlock::Text(content) => write_chat_content(out, content)?,
                UserBlock::ToolResult(content) => {
                    out.push_str("<tool_result>");
                    write_chat_content(out, content)?;
                    out.push_str("</tool_result>");
                }
            }
        }
    }

    Ok(())
}

fn render_assistant_message(
    out: &mut String,
    emit_thinking_block: bool,
    content: &[AssistantContentBlock],
) -> Result<()> {
    let has_tool_calls = content.has_tool_calls();

    if emit_thinking_block {
        if content.has_reasoning() {
            write_assistant_reasoning(out, content);
        }
        out.push_str(THINKING_END_TOKEN);
    }

    write_assistant_text(out, content);

    if has_tool_calls {
        out.push_str("\n\n<｜DSML｜tool_calls>\n");
        for (index, tool_call) in content.tool_calls().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            render_tool_call(out, tool_call)?;
        }
        out.push_str("\n</｜DSML｜tool_calls>");
    }

    out.push_str(EOS_TOKEN);
    Ok(())
}

fn render_tool_call(out: &mut String, tool_call: &AssistantToolCall) -> Result<()> {
    writeln!(out, "<{DSML_TOKEN}invoke name=\"{}\">", tool_call.name)
        .expect("writing to String cannot fail");
    encode_arguments_to_dsml(out, tool_call)?;
    write!(out, "\n</{DSML_TOKEN}invoke>").expect("writing to String cannot fail");
    Ok(())
}

fn encode_arguments_to_dsml(out: &mut String, tool_call: &AssistantToolCall) -> Result<()> {
    let arguments: Value = serde_json::from_str(&tool_call.arguments).map_err(|error| {
        Error::ChatTemplate(format!(
            "assistant tool call has invalid JSON arguments for DeepSeek V4: {error}"
        ))
    })?;
    let Some(arguments) = arguments.as_object() else {
        return Err(Error::ChatTemplate(
            "assistant tool call arguments for DeepSeek V4 must be a JSON object".to_string(),
        ));
    };

    let mut wrote_parameter = false;
    for (key, value) in arguments {
        if wrote_parameter {
            out.push('\n');
        }

        let is_string = matches!(value, Value::String(_));
        write!(
            out,
            "<{DSML_TOKEN}parameter name=\"{key}\" string=\"{}\">",
            if is_string { "true" } else { "false" }
        )
        .expect("writing to String cannot fail");

        match value {
            Value::String(value) => out.push_str(value),
            value => out.push_str(&json_dumps(value)?),
        }

        write!(out, "</{DSML_TOKEN}parameter>").expect("writing to String cannot fail");
        wrote_parameter = true;
    }

    Ok(())
}

fn write_chat_content(out: &mut String, content: &ChatContent) -> Result<()> {
    match content {
        ChatContent::Text(text) => out.push_str(text),
        ChatContent::Parts(parts) => {
            for part in parts {
                out.push_str(part.as_text());
            }
        }
    }
    Ok(())
}

fn write_assistant_reasoning(out: &mut String, content: &[AssistantContentBlock]) {
    for block in content {
        if let AssistantContentBlock::Reasoning { text } = block {
            out.push_str(text);
        }
    }
}

fn write_assistant_text(out: &mut String, content: &[AssistantContentBlock]) {
    for block in content {
        if let AssistantContentBlock::Text { text } = block {
            out.push_str(text);
        }
    }
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
                "failed to serialize DeepSeek V4 JSON payload: {error}"
            ))
        })
}
