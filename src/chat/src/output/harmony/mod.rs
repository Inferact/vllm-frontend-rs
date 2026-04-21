//! Native Harmony output processing for `gpt_oss`.
//!
//! Unlike the default text-first pipeline, this processor consumes `DecodedTextEvent`
//! token IDs directly and lets the official `openai-harmony` parser recover the
//! structured assistant message shape at token granularity.

use std::sync::LazyLock;

use anyhow::Context;
use futures::StreamExt as _;
use futures_async_stream::try_stream;
use openai_harmony::chat::Role;
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, StreamableParser, load_harmony_encoding,
};
use thiserror_ext::AsReport;

use crate::Result as ChatResult;
use crate::error::{Error, Result};
use crate::event::AssistantBlockKind;
use crate::output::{
    AssistantEvent, ChatOutputProcessor, DynChatEventStream, DynDecodedTextEventStream,
    generate_tool_call_id,
};
use crate::parser::ParserSelection;
use crate::request::ChatRequest;

/// Request-scoped Harmony output processor used for `model_type == "gpt_oss"`.
///
/// This processor keeps the existing northbound `ChatEvent` shape, but swaps the
/// parsed-assistant backend from generic text/reasoning/tool parsers to the
/// official Harmony token parser.
#[derive(Debug)]
pub struct HarmonyChatOutputProcessor {
    encoding: &'static HarmonyEncoding,
    tool_calls_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HarmonyGroupKey {
    serial: usize,
    channel: Option<String>,
    recipient: Option<String>,
}

#[derive(Debug)]
struct HarmonyGroup {
    key: HarmonyGroupKey,
    text: String,
}

#[derive(Debug)]
struct OpenHarmonyToolCall {
    recipient: String,
}

struct HarmonyState {
    parser: StreamableParser,
    tool_calls_enabled: bool,
    completed_visible_messages: usize,
    completed_reasoning_messages: usize,
    current_text_group: Option<HarmonyGroupKey>,
    open_tool_call: Option<OpenHarmonyToolCall>,
}

impl HarmonyChatOutputProcessor {
    /// Build the native Harmony output processor for one request after backend-level parser
    /// policy has already been validated.
    pub fn new(request: &ChatRequest) -> ChatResult<Self> {
        Ok(Self {
            encoding: harmony_encoding()?,
            tool_calls_enabled: request.tool_parsing_enabled(),
        })
    }
}

/// Validate that the generic parser selections are compatible with native Harmony output parsing.
///
/// `gpt_oss` uses a model-specific token-level parser, so any generic reasoning/tool parser
/// override is rejected instead of being silently ignored.
pub(crate) fn validate_harmony_parser_overrides(
    tool_call_parser: &ParserSelection,
    reasoning_parser: &ParserSelection,
) -> ChatResult<()> {
    validate_harmony_override("tool", tool_call_parser)?;
    validate_harmony_override("reasoning", reasoning_parser)?;
    Ok(())
}

fn validate_harmony_override(kind: &'static str, selection: &ParserSelection) -> ChatResult<()> {
    if matches!(selection, ParserSelection::Auto) {
        return Ok(());
    }

    Err(Error::HarmonyParserOverrideUnsupported {
        kind,
        selection: selection.to_string(),
    })
}

impl ChatOutputProcessor for HarmonyChatOutputProcessor {
    fn process(self: Box<Self>, decoded: DynDecodedTextEventStream) -> Result<DynChatEventStream> {
        let assistant =
            harmony_assistant_event_stream(decoded, self.encoding, self.tool_calls_enabled);
        Ok(crate::output::structured::structured_chat_event_stream(assistant).boxed())
    }
}

impl HarmonyState {
    fn new(encoding: HarmonyEncoding, tool_calls_enabled: bool) -> Result<Self> {
        Ok(Self {
            parser: StreamableParser::new(encoding, Some(Role::Assistant))
                .map_err(harmony_output_parsing_error)?,
            tool_calls_enabled,
            completed_visible_messages: 0,
            completed_reasoning_messages: 0,
            current_text_group: None,
            open_tool_call: None,
        })
    }

    fn process_token_ids(&mut self, token_ids: &[u32]) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();
        let mut pending_group: Option<HarmonyGroup> = None;

        for &token_id in token_ids {
            let completed_before = self.parser.messages().len();
            self.parser
                .process(token_id)
                .map_err(harmony_output_parsing_error)?;
            let completed_after = self.parser.messages().len();

            if let Some(delta) = self
                .parser
                .last_content_delta()
                .map_err(harmony_output_parsing_error)?
                .filter(|delta| !delta.is_empty())
            {
                let key = HarmonyGroupKey {
                    serial: completed_after,
                    channel: self.parser.current_channel(),
                    recipient: self.parser.current_recipient(),
                };

                match pending_group.as_mut() {
                    Some(group) if group.key == key => group.text.push_str(&delta),
                    _ => {
                        if let Some(group) = pending_group.take() {
                            self.emit_group(group, &mut events);
                        }
                        pending_group = Some(HarmonyGroup { key, text: delta });
                    }
                }
            }

            if completed_after > completed_before {
                if let Some(group) = pending_group.take() {
                    self.emit_group(group, &mut events);
                }

                for serial in completed_before..completed_after {
                    let key = {
                        let message = &self.parser.messages()[serial];
                        HarmonyGroupKey {
                            serial,
                            channel: message.channel.clone(),
                            recipient: message.recipient.clone(),
                        }
                    };
                    self.handle_completed_message(key);
                }
            }
        }

        if let Some(group) = pending_group {
            self.emit_group(group, &mut events);
        }

        Ok(events)
    }

    fn emit_group(&mut self, group: HarmonyGroup, events: &mut Vec<AssistantEvent>) {
        let channel = group.key.channel.as_deref();
        let recipient = group.key.recipient.as_deref();

        if let Some(kind) = text_block_kind(channel, recipient) {
            self.open_tool_call = None;

            if self.current_text_group.as_ref() != Some(&group.key) {
                let needs_newline = match kind {
                    AssistantBlockKind::Text => self.completed_visible_messages > 0,
                    AssistantBlockKind::Reasoning => self.completed_reasoning_messages > 0,
                    AssistantBlockKind::ToolCall => false,
                };

                if needs_newline {
                    events.push(AssistantEvent::TextDelta {
                        kind,
                        delta: "\n".to_string(),
                    });
                }

                self.current_text_group = Some(group.key.clone());
            }

            events.push(AssistantEvent::TextDelta {
                kind,
                delta: group.text,
            });
            return;
        }

        self.current_text_group = None;

        let Some(tool_name) = tool_name(channel, recipient) else {
            return;
        };
        if !self.tool_calls_enabled {
            return;
        }

        let recipient = recipient
            .expect("tool groups always have recipient")
            .to_string();
        let opens_same_call = match self.open_tool_call.as_ref() {
            Some(open_call) => open_call.recipient == recipient,
            None => false,
        };
        if !opens_same_call {
            let id = generate_tool_call_id();
            self.open_tool_call = Some(OpenHarmonyToolCall { recipient });
            events.push(AssistantEvent::ToolCallStart {
                id,
                name: tool_name.to_string(),
            });
        }

        if !group.text.is_empty() {
            events.push(AssistantEvent::ToolCallArgumentsDelta { delta: group.text });
        }
    }

    fn handle_completed_message(&mut self, key: HarmonyGroupKey) {
        if self.current_text_group.as_ref() == Some(&key) {
            self.current_text_group = None;
        }

        let channel = key.channel.as_deref();
        let recipient = key.recipient.as_deref();

        if text_block_kind(channel, recipient) == Some(AssistantBlockKind::Text) {
            self.completed_visible_messages += 1;
        } else if text_block_kind(channel, recipient) == Some(AssistantBlockKind::Reasoning) {
            self.completed_reasoning_messages += 1;
        } else if tool_name(channel, recipient).is_some() {
            self.open_tool_call = None;
        }
    }
}

#[try_stream(ok = AssistantEvent, error = Error)]
async fn harmony_assistant_event_stream(
    decoded: DynDecodedTextEventStream,
    encoding: &'static HarmonyEncoding,
    tool_calls_enabled: bool,
) {
    let mut state = HarmonyState::new(encoding.clone(), tool_calls_enabled)?;
    let decoded = decoded;
    futures::pin_mut!(decoded);

    while let Some(event) = decoded.next().await.transpose()? {
        match event {
            vllm_text::output::DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                yield AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                };
            }
            vllm_text::output::DecodedTextEvent::TextDelta {
                token_ids,
                logprobs,
                finished,
                ..
            } => {
                for semantic_event in state.process_token_ids(&token_ids)? {
                    yield semantic_event;
                }

                if logprobs.is_some() || !token_ids.is_empty() {
                    yield AssistantEvent::LogprobsDelta {
                        logprobs,
                        token_ids,
                    };
                }

                if let Some(finished) = finished {
                    yield AssistantEvent::Done {
                        prompt_token_count: finished.prompt_token_count,
                        output_token_count: finished.output_token_count,
                        finish_reason: finished.finish_reason,
                        kv_transfer_params: finished.kv_transfer_params,
                    };
                }
            }
        }
    }
}

fn harmony_encoding() -> Result<&'static HarmonyEncoding> {
    static ENCODING: LazyLock<anyhow::Result<HarmonyEncoding>> = LazyLock::new(|| {
        load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .context("failed to load harmony encoding for gpt-oss")
    });

    ENCODING
        .as_ref()
        .map_err(|error| Error::HarmonyOutputParsing {
            error: error.to_report_string().into(),
        })
}

fn harmony_output_parsing_error(
    error: impl Into<Box<dyn std::error::Error + Send + Sync>>,
) -> Error {
    Error::HarmonyOutputParsing {
        error: error.into(),
    }
}

fn text_block_kind(channel: Option<&str>, recipient: Option<&str>) -> Option<AssistantBlockKind> {
    match (channel, recipient) {
        (Some("final"), _) => Some(AssistantBlockKind::Text),
        (Some("analysis"), None) => Some(AssistantBlockKind::Reasoning),
        (Some("commentary"), None) => Some(AssistantBlockKind::Text),
        _ => None,
    }
}

fn tool_name<'a>(channel: Option<&str>, recipient: Option<&'a str>) -> Option<&'a str> {
    match (channel, recipient) {
        (Some("commentary" | "analysis"), Some(recipient)) => recipient.strip_prefix("functions."),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use futures::{TryStreamExt as _, stream};
    use openai_harmony::chat::{Message, Role};
    use vllm_text::output::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTextEvent, Finished};

    use super::*;
    use crate::output::ChatOutputProcessor;
    use crate::request::{ChatRequest, ChatTool, ChatToolChoice};
    use crate::{AssistantMessageExt, ChatEvent, FinishReason};

    fn assistant_prefix() -> Vec<u32> {
        harmony_encoding()
            .unwrap()
            .render_conversation_for_completion(
                std::iter::empty::<&Message>(),
                Role::Assistant,
                None,
            )
            .unwrap()
    }

    fn completion_tokens(messages: &[Message]) -> Vec<u32> {
        let encoding = harmony_encoding().unwrap();
        let prefix = assistant_prefix();
        let rendered = encoding.render_conversation(messages.iter(), None).unwrap();
        assert!(rendered.starts_with(&prefix));
        rendered[prefix.len()..].to_vec()
    }

    fn text_message(channel: &str, text: &str) -> Message {
        Message::from_role_and_content(Role::Assistant, text).with_channel(channel)
    }

    fn tool_message(name: &str, arguments: &str, channel: &str) -> Message {
        Message::from_role_and_content(Role::Assistant, arguments)
            .with_channel(channel)
            .with_recipient(format!("functions.{name}"))
            .with_content_type("json")
    }

    fn decoded_start() -> DecodedTextEvent {
        DecodedTextEvent::Start {
            prompt_token_ids: Arc::<[u32]>::from([]),
            prompt_logprobs: None,
        }
    }

    fn finished() -> Finished {
        Finished {
            prompt_token_count: 0,
            output_token_count: 0,
            finish_reason: FinishReason::stop_eos(),
            kv_transfer_params: None,
        }
    }

    async fn collect_events(
        processor: HarmonyChatOutputProcessor,
        events: Vec<DecodedTextEvent>,
    ) -> Vec<ChatEvent> {
        Box::new(processor)
            .process(Box::pin(stream::iter(events.into_iter().map(Ok))))
            .unwrap()
            .try_collect()
            .await
            .unwrap()
    }

    fn request_with_tools() -> ChatRequest {
        ChatRequest {
            tool_choice: ChatToolChoice::Auto,
            tools: vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }),
                strict: None,
            }],
            ..ChatRequest::for_test()
        }
    }

    #[test]
    fn interrupted_final_message_is_preserved() {
        let tokens = completion_tokens(&[text_message("final", "hello")]);
        let events = block_on(collect_events(
            HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
            vec![
                decoded_start(),
                DecodedTextEvent::TextDelta {
                    delta: String::new(),
                    token_ids: tokens[..tokens.len() - 1].to_vec(),
                    logprobs: None,
                    finished: Some(finished()),
                },
            ],
        ));

        assert_eq!(
            events.last(),
            Some(&ChatEvent::Done {
                message: crate::AssistantMessage {
                    content: vec![crate::AssistantContentBlock::Text {
                        text: "hello".to_string(),
                    }],
                },
                prompt_token_count: 0,
                output_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            })
        );
    }

    #[test]
    fn interrupted_analysis_message_is_preserved() {
        let tokens = completion_tokens(&[text_message("analysis", "think")]);
        let events = block_on(collect_events(
            HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
            vec![
                decoded_start(),
                DecodedTextEvent::TextDelta {
                    delta: String::new(),
                    token_ids: tokens[..tokens.len() - 1].to_vec(),
                    logprobs: None,
                    finished: Some(finished()),
                },
            ],
        ));

        assert_eq!(
            events.last(),
            Some(&ChatEvent::Done {
                message: crate::AssistantMessage {
                    content: vec![crate::AssistantContentBlock::Reasoning {
                        text: "think".to_string(),
                    }],
                },
                prompt_token_count: 0,
                output_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            })
        );
    }

    #[test]
    fn commentary_preamble_is_visible_but_commentary_tool_payload_is_not() {
        let tokens = completion_tokens(&[
            text_message("commentary", "Let me check."),
            tool_message("get_weather", r#"{"city":"Paris"}"#, "commentary"),
        ]);
        let events = block_on(collect_events(
            HarmonyChatOutputProcessor::new(&request_with_tools()).unwrap(),
            vec![
                decoded_start(),
                DecodedTextEvent::TextDelta {
                    delta: String::new(),
                    token_ids: tokens,
                    logprobs: None,
                    finished: Some(finished()),
                },
            ],
        ));

        let done = events.last().unwrap();
        let ChatEvent::Done { message, .. } = done else {
            panic!("expected done");
        };
        assert_eq!(message.text(), "Let me check.");
        assert_eq!(message.tool_calls().count(), 1);
    }

    #[test]
    fn multiple_messages_get_newline_separators() {
        let tokens = completion_tokens(&[
            text_message("analysis", "first think"),
            text_message("analysis", "second think"),
            text_message("final", "first answer"),
            text_message("final", "second answer"),
        ]);
        let events = block_on(collect_events(
            HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
            vec![
                decoded_start(),
                DecodedTextEvent::TextDelta {
                    delta: String::new(),
                    token_ids: tokens,
                    logprobs: None,
                    finished: Some(finished()),
                },
            ],
        ));

        let ChatEvent::Done { message, .. } = events.last().unwrap() else {
            panic!("expected done");
        };
        assert_eq!(
            message.reasoning().as_deref(),
            Some("first think\nsecond think")
        );
        assert_eq!(message.text(), "first answer\nsecond answer");
    }

    #[test]
    fn tool_calls_stream_arguments_and_finish_with_local_id_shape() {
        let tokens = completion_tokens(&[tool_message(
            "get_weather",
            r#"{"city":"Paris"}"#,
            "commentary",
        )]);
        let midpoint = tokens.len() / 2;
        let events = block_on(collect_events(
            HarmonyChatOutputProcessor::new(&request_with_tools()).unwrap(),
            vec![
                decoded_start(),
                DecodedTextEvent::TextDelta {
                    delta: String::new(),
                    token_ids: tokens[..midpoint].to_vec(),
                    logprobs: None,
                    finished: None,
                },
                DecodedTextEvent::TextDelta {
                    delta: String::new(),
                    token_ids: tokens[midpoint..].to_vec(),
                    logprobs: None,
                    finished: Some(finished()),
                },
            ],
        ));

        let mut saw_start = None;
        let mut saw_args = String::new();
        let mut saw_end = None;
        for event in &events {
            match event {
                ChatEvent::ToolCallStart { id, name, .. } => {
                    assert!(id.starts_with("call_"));
                    assert_eq!(name, "get_weather");
                    saw_start = Some(id.clone());
                }
                ChatEvent::ToolCallArgumentsDelta { delta, .. } => saw_args.push_str(delta),
                ChatEvent::ToolCallEnd { call, .. } => {
                    saw_end = Some(call.clone());
                }
                _ => {}
            }
        }

        let start_id = saw_start.expect("tool start");
        assert_eq!(saw_args, r#"{"city":"Paris"}"#);
        let end = saw_end.expect("tool end");
        assert_eq!(end.id, start_id);
        assert_eq!(end.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn semantic_events_precede_same_update_logprobs() {
        let tokens = completion_tokens(&[text_message("final", "hello")]);
        let events = block_on(collect_events(
            HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
            vec![
                decoded_start(),
                DecodedTextEvent::TextDelta {
                    delta: String::new(),
                    token_ids: tokens,
                    logprobs: Some(DecodedLogprobs {
                        positions: vec![DecodedPositionLogprobs { entries: vec![] }],
                    }),
                    finished: Some(finished()),
                },
            ],
        ));

        let block_delta_index = events
            .iter()
            .position(|event| matches!(event, ChatEvent::BlockDelta { .. }))
            .unwrap();
        let logprobs_index = events
            .iter()
            .position(|event| matches!(event, ChatEvent::LogprobsDelta { .. }))
            .unwrap();
        assert!(block_delta_index < logprobs_index);
    }

    #[test]
    fn rejects_generic_parser_overrides() {
        let reasoning_error =
            validate_harmony_parser_overrides(&ParserSelection::Auto, &ParserSelection::None)
                .unwrap_err();
        assert_eq!(
            reasoning_error.to_string(),
            "gpt_oss uses native Harmony output parsing; generic reasoning parser override `none` is not supported"
        );

        let tool_error = validate_harmony_parser_overrides(
            &ParserSelection::Explicit("json".to_string()),
            &ParserSelection::Auto,
        )
        .unwrap_err();
        assert_eq!(
            tool_error.to_string(),
            "gpt_oss uses native Harmony output parsing; generic tool parser override `json` is not supported"
        );
    }

    #[test]
    fn allows_auto_auto_only() {
        validate_harmony_parser_overrides(&ParserSelection::Auto, &ParserSelection::Auto).unwrap();
        let _ = HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap();
    }
}
