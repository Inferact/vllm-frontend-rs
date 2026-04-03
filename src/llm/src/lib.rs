#![feature(coroutines)]
#![feature(trait_alias)]

use vllm_engine_core_client::EngineCoreClient;

mod batch;
mod error;
mod output;
mod request;
mod request_metrics;

pub use error::{Error, Result};
pub use output::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStream,
    GenerateOutputStreamExt, GeneratePromptInfo,
};
pub use request::GenerateRequest;
pub use vllm_engine_core_client::protocol::{Logprobs, PositionLogprobs, TokenLogprob};

use crate::request_metrics::RequestMetricsTracker;

/// Thin generate-only facade over [`EngineCoreClient`].
///
/// This mirrors the narrow public shape of Python `AsyncLLM.generate()` and `abort()`, but
/// keeps the boundary close to raw engine-core requests and outputs.
pub struct Llm {
    client: EngineCoreClient,
    stream_interval: usize,
}

impl Llm {
    /// Create a new minimal LLM facade from an already connected engine-core client.
    pub fn new(client: EngineCoreClient) -> Self {
        Self {
            client,
            stream_interval: 1,
        }
    }

    /// Override the per-request output stream interval in generated token count.
    pub fn with_stream_interval(mut self, stream_interval: usize) -> Self {
        assert!(stream_interval >= 1, "stream_interval must be >= 1");
        self.stream_interval = stream_interval;
        self
    }

    /// Expose the underlying engine-core client for low-level utility/admin calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        &self.client
    }

    /// Submit one tokenized generate request and return a per-request output stream.
    pub async fn generate(&self, req: GenerateRequest) -> Result<impl GenerateOutputStream> {
        let prepared = req.prepare()?;
        let prompt_token_ids = prepared.prompt_token_ids().into();

        let request_metrics = RequestMetricsTracker::new(
            self.client.model_name().to_string(),
            prepared.engine_request.arrival_time,
            prepared.prompt_token_ids().len() as u32,
            (prepared.engine_request.sampling_params.as_ref()).map(|p| p.max_tokens),
            1,
        );
        let stream = self.client.call(prepared.engine_request).await?;

        let gen_stream =
            output::GenerateOutputStreamImpl::new(prompt_token_ids, stream, request_metrics);
        let batched_stream =
            batch::batched_generate_output_stream(gen_stream, self.stream_interval);

        Ok(batched_stream)
    }

    /// Abort one in-flight request by request ID.
    pub async fn abort(&self, request_id: &str) -> Result<()> {
        self.client.abort(&[request_id.to_string()]).await?;
        Ok(())
    }

    /// Shut down the underlying engine-core client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.client.shutdown().await?;
        Ok(())
    }
}
