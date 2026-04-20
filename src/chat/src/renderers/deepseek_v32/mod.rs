mod encoding;

use super::{ChatRenderer, RenderedPrompt};
use crate::Result;
use crate::request::ChatRequest;

/// Dedicated DeepSeek V3.2 renderer that ports the local Python vLLM encoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekV32ChatRenderer;

impl DeepSeekV32ChatRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl ChatRenderer for DeepSeekV32ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        Ok(RenderedPrompt {
            prompt: encoding::render_request(request)?,
        })
    }
}

#[cfg(test)]
mod tests;
