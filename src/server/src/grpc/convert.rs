//! Conversion between gRPC protobuf types and internal `vllm-text` request/response types.

use tonic::Status;
use uuid::Uuid;
use vllm_engine_core_client::protocol::{StopReason, StructuredOutputsParams};
use vllm_text::{
    DecodedLogprobs, DecodedPromptLogprobs, FinishReason, Finished, Prompt, SamplingParams,
    TextDecodeOptions, TextRequest,
};

use super::pb;

// ========================================================================================
// Request conversion
// ========================================================================================

/// Convert a gRPC `GenerateRequest` into the internal `TextRequest`.
///
/// If `req.model` is non-empty, it must match `configured_model`; otherwise the request is
/// rejected with `NotFound`. An empty string is treated as "unset" (proto3 default) and accepted.
pub fn to_text_request(
    req: pb::GenerateRequest,
    stream: bool,
    configured_model: &str,
) -> Result<TextRequest, Status> {
    if !req.model.is_empty() && req.model != configured_model {
        return Err(Status::not_found(format!(
            "model `{}` not found",
            req.model
        )));
    }

    let prompt = match req.prompt {
        Some(pb::generate_request::Prompt::Text(text)) => Prompt::Text(text),
        Some(pb::generate_request::Prompt::TokenIds(ids)) => Prompt::TokenIds(ids.ids),
        None => return Err(Status::invalid_argument("prompt is required")),
    };

    let request_id = if req.request_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        req.request_id
    };

    let sampling = req.sampling.as_ref();
    let decoding = req.decoding.as_ref();
    let stopping = req.stopping.as_ref();
    let response = req.response.as_ref();
    let kv = req.kv.as_ref();

    let mut sampling_params = build_sampling_params(sampling, decoding, stopping, response)?;

    // Thread kv_transfer_params through vllm_xargs, matching the HTTP route convention.
    if let Some(kv_struct) = kv.and_then(|k| k.kv_transfer_params.as_ref()) {
        let kv_json = proto_struct_to_json(kv_struct);
        let map = sampling_params
            .vllm_xargs
            .get_or_insert_with(Default::default);
        map.insert("kv_transfer_params".to_string(), kv_json);
    }

    let decode_options = TextDecodeOptions {
        skip_special_tokens: true,
        include_stop_str_in_output: stopping.is_some_and(|s| s.include_stop_strings),
        stop_strings: stopping
            .map(|s| &s.stop_strings)
            .filter(|ss| !ss.is_empty())
            .cloned(),
        min_tokens: stopping.map_or(0, |s| s.min_new_tokens),
    };

    Ok(TextRequest {
        request_id,
        prompt,
        sampling_params,
        decode_options,
        intermediate: stream,
        priority: req.priority,
        cache_salt: kv.map(|k| &k.cache_salt).filter(|s| !s.is_empty()).cloned(),
        add_special_tokens: true,
        data_parallel_rank: None,
    })
}

fn build_sampling_params(
    sampling: Option<&pb::RandomSampling>,
    decoding: Option<&pb::DecodingParameters>,
    stopping: Option<&pb::StoppingCriteria>,
    response: Option<&pb::ResponseOptions>,
) -> Result<SamplingParams, Status> {
    let mut params = SamplingParams::default();

    // RandomSampling
    if let Some(s) = sampling {
        if s.temperature != 0.0 {
            params.temperature = Some(s.temperature);
        }
        if s.top_k != 0 {
            params.top_k = Some(s.top_k);
        }
        if s.top_p != 0.0 {
            params.top_p = Some(s.top_p);
        }
        if s.min_p != 0.0 {
            params.min_p = Some(s.min_p);
        }
        params.seed = s.seed.map(|v| v as i64);
        // TODO: num_sequences (n > 1) is not supported yet by the TextLlm layer.
    }

    // DecodingParameters
    if let Some(d) = decoding {
        if d.presence_penalty != 0.0 {
            params.presence_penalty = Some(d.presence_penalty);
        }
        if d.frequency_penalty != 0.0 {
            params.frequency_penalty = Some(d.frequency_penalty);
        }
        if d.repetition_penalty != 0.0 {
            params.repetition_penalty = Some(d.repetition_penalty);
        }
        if !d.logit_bias.is_empty() {
            params.logit_bias = Some(d.logit_bias.clone());
        }
        if !d.allowed_token_ids.is_empty() {
            params.allowed_token_ids = Some(d.allowed_token_ids.clone());
        }
        params.structured_outputs = convert_structured_output(d)?;
    }

    // StoppingCriteria
    if let Some(s) = stopping {
        if s.max_new_tokens != 0 {
            params.max_tokens = Some(s.max_new_tokens);
        }
        if s.min_new_tokens != 0 {
            params.min_tokens = Some(s.min_new_tokens);
        }
        if !s.stop_token_ids.is_empty() {
            params.stop_token_ids = Some(s.stop_token_ids.clone());
        }
        params.ignore_eos = s.ignore_eos;
    }

    // ResponseOptions → logprobs
    if let Some(r) = response {
        if r.output_logprobs {
            let (count, token_ids) = candidate_logprob_spec(r.output_candidates.as_ref());
            params.logprobs = Some(count);
            params.logprob_token_ids = token_ids;
        }
        if r.prompt_logprobs {
            let (count, _) = candidate_logprob_spec(r.prompt_candidates.as_ref());
            params.prompt_logprobs = Some(count);
        }
    }

    Ok(params)
}

/// Map the proto `CandidateTokens` selector to a `(logprobs_count, logprob_token_ids)` pair.
///
/// - `top_n(k)` → `(k, None)` — return top-k candidates by probability
/// - `all` → `(-1, None)` — return the full vocabulary
/// - `token_ids(n)` → `(1, Some(vec of n token ids))` — return logprobs for specific tokens (the
///   count `n` is stored in the proto as the number of token IDs that follow, but the actual IDs
///   are carried via `logprob_token_ids` on `SamplingParams`)
/// - absent → `(1, None)` — just the sampled/scored token
fn candidate_logprob_spec(candidates: Option<&pb::CandidateTokens>) -> (i32, Option<Vec<u32>>) {
    match candidates.and_then(|c| c.select.as_ref()) {
        Some(pb::candidate_tokens::Select::TopN(n)) => (*n as i32, None),
        Some(pb::candidate_tokens::Select::All(true)) => (-1, None),
        Some(pb::candidate_tokens::Select::TokenIds(ids)) => (1, Some(ids.ids.clone())),
        _ => (1, None),
    }
}

fn convert_structured_output(
    d: &pb::DecodingParameters,
) -> Result<Option<StructuredOutputsParams>, Status> {
    let so = match d.structured_output.as_ref() {
        None => return Ok(None),
        Some(so) => so,
    };
    use pb::decoding_parameters::StructuredOutput;
    let params = match so {
        StructuredOutput::Json(schema) => {
            let json: serde_json::Value = serde_json::from_str(schema)
                .map_err(|e| Status::invalid_argument(format!("invalid json schema: {e}")))?;
            StructuredOutputsParams {
                json: Some(json),
                ..Default::default()
            }
        }
        StructuredOutput::Regex(regex) => StructuredOutputsParams {
            regex: Some(regex.clone()),
            ..Default::default()
        },
        StructuredOutput::Choice(choices) => StructuredOutputsParams {
            choice: Some(choices.choices.clone()),
            ..Default::default()
        },
        StructuredOutput::Grammar(grammar) => StructuredOutputsParams {
            grammar: Some(grammar.clone()),
            ..Default::default()
        },
        StructuredOutput::JsonObject(true) => StructuredOutputsParams {
            json_object: Some(true),
            ..Default::default()
        },
        StructuredOutput::JsonObject(false) => return Ok(None),
        StructuredOutput::StructuralTag(tag) => StructuredOutputsParams {
            structural_tag: Some(tag.clone()),
            ..Default::default()
        },
    };
    Ok(Some(params))
}

// ========================================================================================
// Response conversion
// ========================================================================================

/// Convert a `DecodedTextEvent::Start` into the prompt info portion of a gRPC response.
pub fn to_prompt_info(
    prompt_token_ids: &[u32],
    prompt_logprobs: Option<&DecodedPromptLogprobs>,
    opts: &ResponseOpts,
) -> pb::PromptInfo {
    let token_ids = if opts.prompt_token_ids {
        prompt_token_ids.to_vec()
    } else {
        vec![]
    };

    let (logprobs, ranks, candidate_tokens) = match prompt_logprobs {
        Some(plp) if opts.prompt_logprobs => prompt_logprobs_to_proto(plp),
        _ => (vec![], vec![], vec![]),
    };

    pb::PromptInfo {
        num_prompt_tokens: prompt_token_ids.len() as u32,
        token_ids,
        logprobs,
        ranks,
        candidate_tokens,
    }
}

/// Convert a `DecodedTextEvent::TextDelta` into a gRPC `SequenceOutput`.
pub fn to_sequence_output(
    delta: &str,
    token_ids: &[u32],
    logprobs: Option<&DecodedLogprobs>,
    finished: Option<&Finished>,
    opts: &ResponseOpts,
) -> pb::SequenceOutput {
    let (lp_values, rank_values, candidates) = match logprobs {
        Some(lp) if opts.output_logprobs => output_logprobs_to_proto(lp),
        _ => (vec![], vec![], vec![]),
    };

    pb::SequenceOutput {
        index: 0, // TODO: multi-sequence (n > 1) not supported
        text: if opts.output_text {
            delta.to_string()
        } else {
            String::new()
        },
        num_tokens: token_ids.len() as u32,
        token_ids: if opts.output_token_ids {
            token_ids.to_vec()
        } else {
            vec![]
        },
        logprobs: lp_values,
        ranks: rank_values,
        candidate_tokens: candidates,
        finish_info: finished.map(to_finish_info),
    }
}

fn to_finish_info(finished: &Finished) -> pb::FinishInfo {
    use pb::finish_info::FinishReason as PbFinishReason;

    let (finish_reason, stop_reason) = match &finished.finish_reason {
        FinishReason::Stop(reason) => {
            let sr = match reason {
                Some(StopReason::TokenId(id)) => {
                    Some(pb::finish_info::StopReason::StopTokenId(*id))
                }
                Some(StopReason::Text(s)) => {
                    Some(pb::finish_info::StopReason::StopString(s.clone()))
                }
                None => Some(pb::finish_info::StopReason::EosTokenId(0)),
            };
            (PbFinishReason::Stop as i32, sr)
        }
        FinishReason::Length => (PbFinishReason::Length as i32, None),
        FinishReason::Abort | FinishReason::Error | FinishReason::Repetition => {
            (PbFinishReason::Aborted as i32, None)
        }
    };

    pb::FinishInfo {
        num_output_tokens: finished.output_token_count as u32,
        finish_reason,
        stop_reason,
        kv_transfer_params: finished
            .kv_transfer_params
            .as_ref()
            .and_then(json_to_proto_struct),
    }
}

// ========================================================================================
// Logprobs helpers
// ========================================================================================

/// Convert output logprobs to the flat proto representation.
///
/// Returns (logprob_values, ranks, candidate_tokens) — all parallel arrays indexed by position.
fn output_logprobs_to_proto(
    lp: &DecodedLogprobs,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    positions_to_proto(&lp.positions)
}

/// Convert prompt logprobs to the flat proto representation.
fn prompt_logprobs_to_proto(
    plp: &DecodedPromptLogprobs,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    // The proto PromptInfo has flat parallel arrays covering all prompt positions.
    // DecodedPromptLogprobs has first_token separately + scored_positions for the rest.
    // The first prompt position has no scores, so we emit zeros for it.
    let (mut logprobs, mut ranks, mut candidates) = positions_to_proto(&plp.scored_positions);
    logprobs.insert(0, 0.0);
    ranks.insert(0, 0);
    candidates.insert(0, pb::CandidateTokenInfo { tokens: vec![] });
    (logprobs, ranks, candidates)
}

/// Shared helper: convert a slice of decoded position logprobs to flat proto arrays.
fn positions_to_proto(
    positions: &[vllm_text::DecodedPositionLogprobs],
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    let mut logprobs = Vec::with_capacity(positions.len());
    let mut ranks = Vec::with_capacity(positions.len());
    let mut candidates = Vec::with_capacity(positions.len());

    for pos in positions {
        // First entry is the sampled/scored token.
        if let Some(first) = pos.entries.first() {
            logprobs.push(first.logprob);
            ranks.push(first.rank);
        }

        // Extra candidates beyond the first.
        let entries = pos.entries.iter().skip(1);
        candidates.push(pb::CandidateTokenInfo {
            tokens: entries
                .map(|e| pb::candidate_token_info::TokenInfo {
                    id: e.token_id,
                    logprob: e.logprob,
                    rank: e.rank,
                })
                .collect(),
        });
    }

    (logprobs, ranks, candidates)
}

// ========================================================================================
// KV transfer params conversion (serde_json::Value ↔ prost_types::Struct)
// ========================================================================================

fn proto_struct_to_json(s: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        s.fields
            .iter()
            .map(|(k, v)| (k.clone(), proto_value_to_json(v)))
            .collect(),
    )
}

fn proto_value_to_json(v: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match v.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::NumberValue(n)) => serde_json::json!(*n),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(proto_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => proto_struct_to_json(s),
    }
}

fn json_to_proto_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_proto_value(v)))
                .collect(),
        }),
        _ => None,
    }
}

fn json_to_proto_value(v: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match v {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_proto_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_proto_value(v)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

// ========================================================================================
// Options extracted from the request for response building
// ========================================================================================

/// Response-shaping options extracted from the proto `ResponseOptions`.
#[derive(Default)]
pub struct ResponseOpts {
    pub prompt_token_ids: bool,
    pub prompt_logprobs: bool,
    pub output_text: bool,
    pub output_token_ids: bool,
    pub output_logprobs: bool,
}

impl ResponseOpts {
    pub fn from_proto(r: Option<&pb::ResponseOptions>) -> Self {
        match r {
            Some(r) => Self {
                prompt_token_ids: r.prompt_token_ids,
                prompt_logprobs: r.prompt_logprobs,
                output_text: r.output_text.unwrap_or(true),
                output_token_ids: r.output_token_ids,
                output_logprobs: r.output_logprobs,
            },
            None => Self {
                output_text: true,
                ..Default::default()
            },
        }
    }
}
