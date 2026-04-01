use vllm_engine_core_client::{CoordinatorMode, TransportMode};

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    Bind { host: String, port: u16 },
    /// Adopt an already-open listening socket inherited from a supervisor process.
    InheritedFd { fd: i32 },
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Frontend-to-engine transport setup.
    pub transport_mode: TransportMode,
    /// Requested frontend-side coordinator behavior.
    pub coordinator_mode: CoordinatorMode,
    /// Backend model identifier and exposed OpenAI model ID.
    pub model: String,
    /// HTTP listener setup.
    pub listener_mode: HttpListenerMode,
    /// Explicit tool call parser name, or `None` for model-based auto-detection.
    pub tool_call_parser: Option<String>,
    /// Explicit reasoning parser name, or `None` for model-based auto-detection.
    pub reasoning_parser: Option<String>,
    /// Override for the maximum model context length. Takes priority over the model's
    /// `max_position_embeddings` from `config.json`.
    pub max_model_len: Option<u32>,
}

impl Config {
    /// Return the number of engines implied by the configured transport mode.
    pub fn engine_count(&self) -> usize {
        match &self.transport_mode {
            TransportMode::HandshakeOwner { engine_count, .. }
            | TransportMode::Bootstrapped { engine_count, .. } => *engine_count,
        }
    }

    /// Resolve the effective coordinator mode after applying model-specific safeguards.
    ///
    /// Phase 1 only allows in-process coordination for MoE-managed `serve` deployments. For
    /// non-MoE models we silently degrade `InProc` to `None` so the caller can build configs
    /// without duplicating that check.
    pub fn resolve_coordinator_mode(&self, model_is_moe: bool) -> CoordinatorMode {
        match &self.coordinator_mode {
            CoordinatorMode::InProc if !model_is_moe => CoordinatorMode::None,
            mode => mode.clone(),
        }
    }
}
