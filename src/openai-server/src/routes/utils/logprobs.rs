use std::collections::HashMap;

use openai_protocol::common::LogProbs;
use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs};

use crate::error::{ApiError, server_error};

/// Convert decoded token-position logprobs into the OpenAI completions `logprobs` shape.
pub fn decoded_logprobs_to_openai(
    logprobs: &DecodedLogprobs,
    initial_text_offset: u32,
) -> Result<LogProbs, ApiError> {
    let mut text_offset = Vec::with_capacity(logprobs.positions.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.positions.len());
    let mut tokens = Vec::with_capacity(logprobs.positions.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.positions.len());
    let mut current_offset = initial_text_offset;

    for position in &logprobs.positions {
        let chosen = position.entries.first().ok_or_else(|| {
            server_error!("decoded logprobs position unexpectedly had no token candidates")
        })?;

        text_offset.push(current_offset);
        token_logprobs.push(Some(clamp_logprob(chosen.logprob)));
        tokens.push(chosen.token.clone());
        top_logprobs.push(Some(position_top_logprobs_map(position)));
        current_offset = current_offset.saturating_add(text_len(&chosen.token));
    }

    Ok(LogProbs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    })
}

/// Convert decoded prompt logprobs into the vLLM-style prompt-logprobs response shape.
pub fn decoded_prompt_logprobs_to_maps(
    prompt_logprobs: &DecodedPromptLogprobs,
) -> Vec<Option<HashMap<String, f32>>> {
    prompt_logprobs
        .positions
        .iter()
        .map(|position| position.as_ref().map(position_top_logprobs_map))
        .collect()
}

/// Count visible text positions using OpenAI completions' character-offset convention.
pub fn text_len(text: &str) -> u32 {
    u32::try_from(text.chars().count()).unwrap_or(u32::MAX)
}

fn position_top_logprobs_map(position: &DecodedPositionLogprobs) -> HashMap<String, f32> {
    position
        .entries
        .iter()
        .map(|entry| (entry.token.clone(), clamp_logprob(entry.logprob)))
        .collect()
}

fn clamp_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}
