use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::Value;
use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::{CoordinatorMode as EngineCoreCoordinatorMode, TransportMode};

/// TLS/SSL configuration for the HTTP server.
///
/// Equivalent to the `ssl_keyfile`, `ssl_certfile`, `ssl_ca_certs`, `ssl_cert_reqs`, and
/// `enable_ssl_refresh` arguments in Python vLLM's `FrontendArgs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsConfig {
    /// Path to the SSL private key file (PEM format).
    pub keyfile: String,
    /// Path to the SSL certificate file (PEM format).
    pub certfile: String,
    /// Path to the CA certificates file for client certificate verification.
    pub ca_certs: Option<String>,
    /// Client certificate requirement level: 0 = none, 1 = optional, 2 = required.
    pub cert_reqs: i32,
    /// Whether to watch certificate files and reload on change.
    pub enable_refresh: bool,
}

impl TlsConfig {
    /// Build a `TlsConfig` from the individual SSL arguments.
    ///
    /// Returns `Ok(None)` when neither keyfile nor certfile is set, and errors if only one of
    /// the two is provided.
    pub fn from_args(
        ssl_keyfile: Option<String>,
        ssl_certfile: Option<String>,
        ssl_ca_certs: Option<String>,
        ssl_cert_reqs: i32,
        enable_ssl_refresh: bool,
    ) -> Result<Option<Self>> {
        match (ssl_keyfile, ssl_certfile) {
            (Some(keyfile), Some(certfile)) => Ok(Some(Self {
                keyfile,
                certfile,
                ca_certs: ssl_ca_certs,
                cert_reqs: ssl_cert_reqs,
                enable_refresh: enable_ssl_refresh,
            })),
            (None, None) => Ok(None),
            _ => bail!("both --ssl-keyfile and --ssl-certfile must be provided for TLS"),
        }
    }
}

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    BindTcp { host: String, port: u16 },
    /// Bind a fresh Unix domain listener on the given filesystem path.
    BindUnix { path: String },
    /// Adopt an already-open listening socket inherited from a supervisor process.
    InheritedFd { fd: i32 },
}

/// Which coordinator implementation should be active when one is present for a frontend client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorMode {
    /// Do not run a coordinator at all.
    None,
    /// Run the Rust in-process coordinator for managed `serve` deployments, if there are mutliple
    /// engines and the model is MoE.
    MaybeInProc,
    /// Connect to an external coordinator owned by another process.
    External { address: String },
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
    /// Optional TLS configuration. When `Some`, the server serves HTTPS.
    pub tls: Option<TlsConfig>,
    /// Tool-call parser selection.
    pub tool_call_parser: ParserSelection,
    /// Reasoning parser selection.
    pub reasoning_parser: ParserSelection,
    /// Chat renderer selection.
    pub renderer: RendererSelection,
    /// Server-default chat template override, as a file path or inline template.
    pub chat_template: Option<String>,
    /// Server-default keyword arguments merged into every chat-template render.
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,
    /// How to serialize `message.content` for chat-template rendering.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// When `true`, suppress periodic stats logging (throughput, queue depth, cache usage).
    pub disable_log_stats: bool,
    /// TCP port for the gRPC Generate service. When `None`, no gRPC server is started.
    pub grpc_port: Option<u16>,
}

impl Config {
    /// Validate frontend configuration that can be checked before engine startup.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(&self.tool_call_parser, &self.reasoning_parser)?;

        Ok(())
    }

    /// Return the number of engines implied by the configured transport mode.
    pub fn engine_count(&self) -> usize {
        match &self.transport_mode {
            TransportMode::HandshakeOwner { engine_count, .. }
            | TransportMode::Bootstrapped { engine_count, .. } => *engine_count,
        }
    }

    /// Resolve the effective coordinator mode.
    pub fn effective_coordinator_mode(
        &self,
        model_is_moe: bool,
    ) -> Option<EngineCoreCoordinatorMode> {
        match &self.coordinator_mode {
            CoordinatorMode::None => None,
            CoordinatorMode::MaybeInProc => {
                if model_is_moe && self.engine_count() > 1 {
                    Some(EngineCoreCoordinatorMode::InProc)
                } else {
                    None
                }
            }
            CoordinatorMode::External { address } => Some(EngineCoreCoordinatorMode::External {
                address: address.clone(),
            }),
        }
    }
}
