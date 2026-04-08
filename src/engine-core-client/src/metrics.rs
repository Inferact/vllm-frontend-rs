use std::collections::HashMap;

use vllm_metrics::{
    EngineLabels, EnginePositionLabels, F64Gauge, Histogram, SchedulerMetrics, U64Counter, U64Gauge,
};

use crate::protocol::stats::SchedulerStats;

/// Pre-resolved metric handles for one `(model_name, engine)` pair, used by the output dispatcher
/// to avoid repeated `Family` RwLock lookups and label `String` allocations on every stats update.
struct CachedSchedulerHandles {
    // Scheduler state gauges.
    scheduler_running: U64Gauge,
    scheduler_waiting: U64Gauge,
    kv_cache_usage: F64Gauge,

    // Prefix-cache counters.
    prefix_cache_queries: U64Counter,
    prefix_cache_hits: U64Counter,
    external_prefix_cache_queries: U64Counter,
    external_prefix_cache_hits: U64Counter,

    // Speculative decoding counters.
    spec_decode_num_drafts: U64Counter,
    spec_decode_num_draft_tokens: U64Counter,
    spec_decode_num_accepted_tokens: U64Counter,

    // Per-engine performance / MFU counters.
    estimated_flops_per_gpu: U64Counter,
    estimated_read_bytes_per_gpu: U64Counter,
    estimated_write_bytes_per_gpu: U64Counter,

    // Sampled KV-cache residency histograms.
    kv_block_lifetime_seconds: Histogram,
    kv_block_idle_before_evict_seconds: Histogram,
    kv_block_reuse_gap_seconds: Histogram,
}

impl CachedSchedulerHandles {
    fn resolve(metrics: &SchedulerMetrics, model_name: &str, engine: u32) -> Self {
        let labels = EngineLabels {
            model_name: model_name.to_string(),
            engine,
        };
        Self {
            scheduler_running: metrics.scheduler_running.get_or_create_owned(&labels),
            scheduler_waiting: metrics.scheduler_waiting.get_or_create_owned(&labels),
            kv_cache_usage: metrics.kv_cache_usage.get_or_create_owned(&labels),

            prefix_cache_queries: metrics.prefix_cache_queries.get_or_create_owned(&labels),
            prefix_cache_hits: metrics.prefix_cache_hits.get_or_create_owned(&labels),
            external_prefix_cache_queries: metrics
                .external_prefix_cache_queries
                .get_or_create_owned(&labels),
            external_prefix_cache_hits: metrics
                .external_prefix_cache_hits
                .get_or_create_owned(&labels),

            spec_decode_num_drafts: metrics.spec_decode_num_drafts.get_or_create_owned(&labels),
            spec_decode_num_draft_tokens: metrics
                .spec_decode_num_draft_tokens
                .get_or_create_owned(&labels),
            spec_decode_num_accepted_tokens: metrics
                .spec_decode_num_accepted_tokens
                .get_or_create_owned(&labels),

            estimated_flops_per_gpu: metrics.estimated_flops_per_gpu.get_or_create_owned(&labels),
            estimated_read_bytes_per_gpu: metrics
                .estimated_read_bytes_per_gpu
                .get_or_create_owned(&labels),
            estimated_write_bytes_per_gpu: metrics
                .estimated_write_bytes_per_gpu
                .get_or_create_owned(&labels),

            kv_block_lifetime_seconds: metrics
                .kv_block_lifetime_seconds
                .get_or_create_owned(&labels),
            kv_block_idle_before_evict_seconds: metrics
                .kv_block_idle_before_evict_seconds
                .get_or_create_owned(&labels),
            kv_block_reuse_gap_seconds: metrics
                .kv_block_reuse_gap_seconds
                .get_or_create_owned(&labels),
        }
    }

    fn record(&self, metrics: &SchedulerMetrics, model_name: &str, engine: u32, stats: &SchedulerStats) {
        // Scheduler state gauges.
        self.scheduler_running.set(stats.num_running_reqs);
        self.scheduler_waiting.set(stats.num_waiting_reqs);
        self.kv_cache_usage.set(stats.kv_cache_usage);

        // Prefix-cache counters.
        self.prefix_cache_queries
            .inc_by(stats.prefix_cache_stats.base.queries);
        self.prefix_cache_hits
            .inc_by(stats.prefix_cache_stats.base.hits);

        if let Some(connector_prefix_cache_stats) = &stats.connector_prefix_cache_stats {
            self.external_prefix_cache_queries
                .inc_by(connector_prefix_cache_stats.base.queries);
            self.external_prefix_cache_hits
                .inc_by(connector_prefix_cache_stats.base.hits);
        }

        // Speculative decoding counters.
        if let Some(spec_decoding_stats) = &stats.spec_decoding_stats {
            self.spec_decode_num_drafts
                .inc_by(spec_decoding_stats.num_drafts);
            self.spec_decode_num_draft_tokens
                .inc_by(spec_decoding_stats.num_draft_tokens);
            self.spec_decode_num_accepted_tokens
                .inc_by(spec_decoding_stats.num_accepted_tokens);

            // Per-position counters use dynamic labels and are not worth caching since the number
            // of positions varies and the spec-decoding path is already conditional.
            for (position, accepted_tokens) in spec_decoding_stats
                .num_accepted_tokens_per_pos
                .iter()
                .copied()
                .enumerate()
            {
                metrics
                    .spec_decode_num_accepted_tokens_per_pos
                    .get_or_create(&EnginePositionLabels {
                        model_name: model_name.to_string(),
                        engine,
                        position: position as u32,
                    })
                    .inc_by(accepted_tokens);
            }
        }

        // Per-engine performance / MFU counters.
        if let Some(perf_stats) = &stats.perf_stats
            && (perf_stats.num_flops_per_gpu != 0
                || perf_stats.num_read_bytes_per_gpu != 0
                || perf_stats.num_write_bytes_per_gpu != 0)
        {
            self.estimated_flops_per_gpu
                .inc_by(perf_stats.num_flops_per_gpu);
            self.estimated_read_bytes_per_gpu
                .inc_by(perf_stats.num_read_bytes_per_gpu);
            self.estimated_write_bytes_per_gpu
                .inc_by(perf_stats.num_write_bytes_per_gpu);
        }

        // Sampled KV-cache residency histograms.
        if !stats.kv_cache_eviction_events.is_empty() {
            for event in &stats.kv_cache_eviction_events {
                self.kv_block_lifetime_seconds
                    .observe(event.lifetime_seconds);
                self.kv_block_idle_before_evict_seconds
                    .observe(event.idle_seconds);
                for reuse_gap_seconds in &event.reuse_gaps_seconds {
                    self.kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
                }
            }
        }
    }
}

/// Per-model cache of pre-resolved scheduler metric handles, keyed by engine index.
///
/// Maintains one [`CachedSchedulerHandles`] per engine, lazily resolved on the first stats update
/// for that engine. Intended to live inside the single output dispatcher task.
pub(crate) struct SchedulerStatsRecorder {
    model_name: String,
    per_engine: HashMap<u32, CachedSchedulerHandles>,
}

impl SchedulerStatsRecorder {
    pub(crate) fn new(model_name: String) -> Self {
        Self {
            model_name,
            per_engine: HashMap::new(),
        }
    }

    /// Record the scheduler-stats-backed metrics for one engine at one point in time.
    pub(crate) fn record(
        &mut self,
        metrics: &SchedulerMetrics,
        engine_index: u32,
        stats: &SchedulerStats,
    ) {
        let cached = self
            .per_engine
            .entry(engine_index)
            .or_insert_with(|| CachedSchedulerHandles::resolve(metrics, &self.model_name, engine_index));
        cached.record(metrics, &self.model_name, engine_index, stats);
    }
}
