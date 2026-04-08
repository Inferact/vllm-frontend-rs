use std::time::{SystemTime, UNIX_EPOCH};

use vllm_engine_core_client::protocol::{EngineCoreEvent, EngineCoreEventType, EngineCoreOutput};
use vllm_metrics::{
    EngineLabels, FinishedReasonLabels, Histogram, METRICS, PromptTokenSourceLabels,
    RequestMetrics, U64Counter,
};

use crate::FinishReason;

fn metrics() -> &'static RequestMetrics {
    &METRICS.request
}

const PROMPT_TOKEN_SOURCE_LOCAL_COMPUTE: &str = "local_compute";
const PROMPT_TOKEN_SOURCE_LOCAL_CACHE_HIT: &str = "local_cache_hit";
const PROMPT_TOKEN_SOURCE_EXTERNAL_KV_TRANSFER: &str = "external_kv_transfer";

/// Pre-resolved metric handles for one `(model_name, engine)` pair.
///
/// Cloning a handle from a prometheus `Family` is cheap (just `Arc` reference bumps), and
/// subsequent observations go straight to the underlying atomic / `RwLock` without re-acquiring the
/// `Family`-level `RwLock` or allocating label `String`s on every call.
// Debug impl is intentionally opaque since metric handle internals are not useful in debug output.
struct CachedMetricHandles {
    // --- Per-output hot path ---
    generation_tokens: U64Counter,
    num_preemptions: U64Counter,
    inter_token_latency_seconds: Histogram,

    // --- Prefill (first output) ---
    prompt_tokens: U64Counter,
    prompt_tokens_computed: U64Counter,
    prompt_tokens_local_cache_hit: U64Counter,
    prompt_tokens_external_kv_transfer: U64Counter,
    prompt_tokens_cached: U64Counter,
    prompt_tokens_recomputed: U64Counter,
    time_to_first_token_seconds: Histogram,

    // --- Terminal (record_finished) ---
    request_prompt_tokens: Histogram,
    request_generation_tokens: Histogram,
    request_max_num_generation_tokens: Histogram,
    request_params_max_tokens: Histogram,
    request_params_n: Histogram,
    request_prefill_kv_computed_tokens: Histogram,
    e2e_request_latency_seconds: Histogram,
    request_queue_time_seconds: Histogram,
    request_prefill_time_seconds: Histogram,
    request_decode_time_seconds: Histogram,
    request_inference_time_seconds: Histogram,
    request_time_per_output_token_seconds: Histogram,

    // --- Finish-reason counters ---
    request_success_stop: U64Counter,
    request_success_length: U64Counter,
    request_success_abort: U64Counter,
    request_success_error: U64Counter,
    request_success_repetition: U64Counter,
}

impl std::fmt::Debug for CachedMetricHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedMetricHandles").finish_non_exhaustive()
    }
}

impl CachedMetricHandles {
    /// Resolve and clone all metric handles for the given engine from the global `Family` entries.
    fn resolve(model_name: &str, engine: u32) -> Self {
        let m = metrics();
        let el = EngineLabels {
            model_name: model_name.to_string(),
            engine,
        };

        let prompt_token_source =
            |source| PromptTokenSourceLabels {
                model_name: model_name.to_string(),
                engine,
                source,
            };
        let finished_reason =
            |finished_reason| FinishedReasonLabels {
                model_name: model_name.to_string(),
                engine,
                finished_reason,
            };

        Self {
            generation_tokens: m.generation_tokens.get_or_create_owned(&el),
            num_preemptions: m.num_preemptions.get_or_create_owned(&el),
            inter_token_latency_seconds: m.inter_token_latency_seconds.get_or_create_owned(&el),

            prompt_tokens: m.prompt_tokens.get_or_create_owned(&el),
            prompt_tokens_computed: m
                .prompt_tokens_by_source
                .get_or_create_owned(&prompt_token_source(PROMPT_TOKEN_SOURCE_LOCAL_COMPUTE)),
            prompt_tokens_local_cache_hit: m
                .prompt_tokens_by_source
                .get_or_create_owned(&prompt_token_source(PROMPT_TOKEN_SOURCE_LOCAL_CACHE_HIT)),
            prompt_tokens_external_kv_transfer: m
                .prompt_tokens_by_source
                .get_or_create_owned(&prompt_token_source(PROMPT_TOKEN_SOURCE_EXTERNAL_KV_TRANSFER)),
            prompt_tokens_cached: m.prompt_tokens_cached.get_or_create_owned(&el),
            prompt_tokens_recomputed: m.prompt_tokens_recomputed.get_or_create_owned(&el),
            time_to_first_token_seconds: m.time_to_first_token_seconds.get_or_create_owned(&el),

            request_prompt_tokens: m.request_prompt_tokens.get_or_create_owned(&el),
            request_generation_tokens: m.request_generation_tokens.get_or_create_owned(&el),
            request_max_num_generation_tokens: m
                .request_max_num_generation_tokens
                .get_or_create_owned(&el),
            request_params_max_tokens: m.request_params_max_tokens.get_or_create_owned(&el),
            request_params_n: m.request_params_n.get_or_create_owned(&el),
            request_prefill_kv_computed_tokens: m
                .request_prefill_kv_computed_tokens
                .get_or_create_owned(&el),
            e2e_request_latency_seconds: m.e2e_request_latency_seconds.get_or_create_owned(&el),
            request_queue_time_seconds: m.request_queue_time_seconds.get_or_create_owned(&el),
            request_prefill_time_seconds: m.request_prefill_time_seconds.get_or_create_owned(&el),
            request_decode_time_seconds: m.request_decode_time_seconds.get_or_create_owned(&el),
            request_inference_time_seconds: m
                .request_inference_time_seconds
                .get_or_create_owned(&el),
            request_time_per_output_token_seconds: m
                .request_time_per_output_token_seconds
                .get_or_create_owned(&el),

            request_success_stop: m
                .request_success
                .get_or_create_owned(&finished_reason("stop")),
            request_success_length: m
                .request_success
                .get_or_create_owned(&finished_reason("length")),
            request_success_abort: m
                .request_success
                .get_or_create_owned(&finished_reason("abort")),
            request_success_error: m
                .request_success
                .get_or_create_owned(&finished_reason("error")),
            request_success_repetition: m
                .request_success
                .get_or_create_owned(&finished_reason("repetition")),
        }
    }

    fn record_request_success(&self, finish_reason: &FinishReason) {
        match finish_reason {
            FinishReason::Stop(_) => self.request_success_stop.inc(),
            FinishReason::Length => self.request_success_length.inc(),
            FinishReason::Abort => self.request_success_abort.inc(),
            FinishReason::Error => self.request_success_error.inc(),
            FinishReason::Repetition => self.request_success_repetition.inc(),
        };
    }

    fn record_prompt_tokens(
        &self,
        prompt_len: u32,
        num_cached_tokens: u32,
        num_external_computed_tokens: u32,
    ) {
        let recomputed = u64::from(num_cached_tokens + 1 == prompt_len);
        let computed = prompt_len.saturating_sub(num_cached_tokens) as u64;
        let external_kv_transfer = num_external_computed_tokens as u64;
        let local_cache_hit = (num_cached_tokens as u64)
            .saturating_add(recomputed)
            .saturating_sub(external_kv_transfer);

        self.prompt_tokens.inc_by(prompt_len as u64);
        self.prompt_tokens_computed.inc_by(computed);
        self.prompt_tokens_local_cache_hit.inc_by(local_cache_hit);
        self.prompt_tokens_external_kv_transfer
            .inc_by(external_kv_transfer);
        self.prompt_tokens_cached.inc_by(num_cached_tokens as u64);
        self.prompt_tokens_recomputed.inc_by(recomputed);
    }
}

/// Request-scoped metrics state tracked across streamed engine-core updates.
///
/// This is the Rust-side counterpart of the Python frontend's request-lifecycle bookkeeping,
/// centered on `RequestStateStats` and the per-output/per-finished update flow.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L200-L237>
///
/// Original Python update flow:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/engine/output_processor.py#L600-L677>
#[derive(Debug)]
pub(crate) struct RequestMetricsTracker {
    model_name: String,
    arrival_time: f64,
    prompt_len: u32,
    max_tokens_param: Option<u32>,
    n_param: u32,
    is_prefilling: bool,
    queued_ts: f64,
    scheduled_ts: f64,
    first_token_ts: f64,
    last_token_ts: f64,
    first_token_latency: f64,
    num_generation_tokens: u32,
    latest_num_cached_tokens: u32,
    /// Pre-resolved metric handles, lazily initialized on the first output when the engine index
    /// becomes known.
    cached: Option<CachedMetricHandles>,
}

impl RequestMetricsTracker {
    /// Create the per-request tracker from the normalized `llm`-layer request context.
    pub(crate) fn new(
        model_name: String,
        arrival_time: f64,
        prompt_len: u32,
        max_tokens_param: Option<u32>,
        n_param: u32,
    ) -> Self {
        Self {
            model_name,
            arrival_time,
            prompt_len,
            max_tokens_param,
            n_param,
            is_prefilling: true,
            queued_ts: 0.0,
            scheduled_ts: 0.0,
            first_token_ts: 0.0,
            last_token_ts: 0.0,
            first_token_latency: 0.0,
            num_generation_tokens: 0,
            latest_num_cached_tokens: 0,
            cached: None,
        }
    }

    /// Lazily resolve and cache metric handles on first use.
    fn cached(&mut self, engine_index: u32) -> &CachedMetricHandles {
        self.cached
            .get_or_insert_with(|| CachedMetricHandles::resolve(&self.model_name, engine_index))
    }

    /// Update request-lifecycle state from one engine-core output item.
    ///
    /// Original Python stats logic:
    /// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L331-L384>
    pub(crate) fn observe_output(
        &mut self,
        engine_index: u32,
        batch_timestamp: f64,
        received_at: f64,
        output: &EngineCoreOutput,
    ) {
        self.latest_num_cached_tokens = output.num_cached_tokens;
        self.num_generation_tokens += output.new_token_ids.len() as u32;

        let cached = self.cached(engine_index);
        cached
            .generation_tokens
            .inc_by(output.new_token_ids.len() as u64);

        if let Some(events) = &output.events {
            self.observe_events(events);
        }

        if self.is_prefilling {
            let cached = self.cached.as_ref().unwrap();
            cached.record_prompt_tokens(
                self.prompt_len,
                output.num_cached_tokens,
                output.num_external_computed_tokens,
            );
            self.first_token_latency = received_at - self.arrival_time;
            cached
                .time_to_first_token_seconds
                .observe(self.first_token_latency);
            self.first_token_ts = batch_timestamp;
            self.is_prefilling = false;
        } else if self.last_token_ts > 0.0 {
            let cached = self.cached.as_ref().unwrap();
            cached
                .inter_token_latency_seconds
                .observe(batch_timestamp - self.last_token_ts);
        }

        self.last_token_ts = batch_timestamp;
    }

    /// Emit the terminal request metrics once a finished output has been observed.
    ///
    /// Original Python finished-request stats:
    /// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L222-L237>
    pub(crate) fn record_finished(&self, received_at: f64, finish_reason: FinishReason) {
        let Some(cached) = self.cached.as_ref() else {
            return;
        };

        let prefill_kv_computed_tokens = self
            .prompt_len
            .saturating_sub(self.latest_num_cached_tokens);
        let e2e_latency_seconds = received_at - self.arrival_time;
        let queue_time_seconds = diff_or_zero(self.scheduled_ts, self.queued_ts);
        let prefill_time_seconds = diff_or_zero(self.first_token_ts, self.scheduled_ts);
        let decode_time_seconds = diff_or_zero(self.last_token_ts, self.first_token_ts);
        let inference_time_seconds = diff_or_zero(self.last_token_ts, self.scheduled_ts);
        let time_per_output_token_seconds = if self.num_generation_tokens > 1 {
            diff_or_zero(self.last_token_ts, self.first_token_ts)
                / (self.num_generation_tokens - 1) as f64
        } else {
            0.0
        };

        cached.record_request_success(&finish_reason);
        cached
            .request_prompt_tokens
            .observe(self.prompt_len as f64);
        cached
            .request_generation_tokens
            .observe(self.num_generation_tokens as f64);
        cached
            .request_max_num_generation_tokens
            .observe(self.num_generation_tokens as f64);
        if let Some(max_tokens_param) = self.max_tokens_param {
            cached
                .request_params_max_tokens
                .observe(max_tokens_param as f64);
        }
        cached.request_params_n.observe(self.n_param as f64);
        cached
            .request_prefill_kv_computed_tokens
            .observe(prefill_kv_computed_tokens as f64);
        cached
            .e2e_request_latency_seconds
            .observe(e2e_latency_seconds);
        cached
            .request_queue_time_seconds
            .observe(queue_time_seconds);
        cached
            .request_prefill_time_seconds
            .observe(prefill_time_seconds);
        cached
            .request_decode_time_seconds
            .observe(decode_time_seconds);
        cached
            .request_inference_time_seconds
            .observe(inference_time_seconds);
        cached
            .request_time_per_output_token_seconds
            .observe(time_per_output_token_seconds);
    }

    fn observe_events(&mut self, events: &[EngineCoreEvent]) {
        for event in events {
            match event.r#type {
                EngineCoreEventType::Queued => {
                    self.queued_ts = event.timestamp;
                }
                EngineCoreEventType::Scheduled => {
                    if self.scheduled_ts == 0.0 {
                        self.scheduled_ts = event.timestamp;
                    }
                }
                EngineCoreEventType::Preempted => {
                    if let Some(cached) = self.cached.as_ref() {
                        cached.num_preemptions.inc();
                    }
                }
            }
        }
    }
}

fn diff_or_zero(end: f64, start: f64) -> f64 {
    if end > 0.0 && start > 0.0 && end >= start {
        end - start
    } else {
        0.0
    }
}

/// Return the current wall-clock time in seconds since the Unix epoch.
///
/// This is used for frontend-side latency measurements such as TTFT and E2E, matching the Python
/// frontend's use of wall-clock request arrival/iteration timestamps rather than engine-core's
/// monotonic scheduler timestamps.
///
/// Original Python request timestamp source:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L206-L216>
pub(crate) fn current_unix_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::protocol::{EngineCoreEvent, EngineCoreEventType};

    use super::{RequestMetricsTracker, diff_or_zero};

    #[test]
    fn tracker_updates_timing_state_across_prefill_decode_and_finish() {
        let mut tracker = RequestMetricsTracker::new("model".to_string(), 100.0, 64, Some(128), 1);

        tracker.observe_output(
            2,
            10.0,
            100.2,
            &vllm_engine_core_client::protocol::EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![1],
                finish_reason: None,
                events: Some(vec![
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Queued,
                        timestamp: 8.0,
                    },
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Scheduled,
                        timestamp: 9.0,
                    },
                ]),
                num_cached_tokens: 4,
                ..Default::default()
            },
        );
        tracker.observe_output(
            2,
            11.5,
            100.4,
            &vllm_engine_core_client::protocol::EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![2, 3],
                finish_reason: None,
                events: Some(vec![EngineCoreEvent {
                    r#type: EngineCoreEventType::Preempted,
                    timestamp: 10.5,
                }]),
                num_cached_tokens: 4,
                ..Default::default()
            },
        );

        assert!(!tracker.is_prefilling);
        assert_eq!(tracker.num_generation_tokens, 3);
        assert_eq!(tracker.queued_ts, 8.0);
        assert_eq!(tracker.scheduled_ts, 9.0);
        assert_eq!(tracker.first_token_ts, 10.0);
        assert_eq!(tracker.last_token_ts, 11.5);
        assert!((tracker.first_token_latency - 0.2).abs() < 1e-9);
        assert_eq!(
            diff_or_zero(tracker.last_token_ts, tracker.first_token_ts),
            1.5
        );
    }
}
