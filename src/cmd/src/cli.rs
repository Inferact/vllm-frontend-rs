//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

use std::ffi::OsString;
use std::time::Duration;

use bpaf::{Args, Bpaf, ParseFailure};
use vllm_openai_server::Config;

use crate::managed_engine::ManagedEngineConfig;

/// Top-level parser for the `vllm-rs` binary.
#[derive(Debug, Clone, PartialEq, Eq, Bpaf)]
#[bpaf(
    options("vllm-rs"),
    descr("Rust frontend and managed-engine CLI for vLLM.")
)]
pub struct Cli {
    #[bpaf(external(command))]
    pub command: Command,
}

impl Cli {
    pub fn parse() -> Self {
        cli().run()
    }

    #[allow(dead_code)]
    fn try_parse_from<I, T>(itr: I) -> Result<Self, ParseFailure>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        let (name, args) = split_name_from_args(itr);
        cli().run_inner(Args::from(args.as_slice()).set_name(&name))
    }
}

/// Supported top-level CLI commands.
#[derive(Debug, Clone, PartialEq, Eq, Bpaf)]
pub enum Command {
    /// Run the Rust OpenAI frontend against an already running headless Python engine.
    #[bpaf(command("frontend"))]
    Frontend(#[bpaf(external(frontend_args))] FrontendArgs),
    /// Launch a managed Python headless engine, then run the Rust OpenAI frontend.
    #[bpaf(command("serve"))]
    Serve(#[bpaf(external(serve_args))] ServeArgs),
}

/// Runtime arguments shared by the external-engine and managed-engine paths.
#[derive(Debug, Clone, PartialEq, Eq, Bpaf)]
pub struct FrontendRuntimeArgs {
    /// HTTP bind host for the OpenAI-compatible server.
    #[bpaf(long("host"), argument("HOST"), fallback(String::from("127.0.0.1")))]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[bpaf(long("port"), argument("PORT"), fallback(8000))]
    pub port: u16,
    /// Maximum time to wait for the engine handshake to complete.
    #[bpaf(
        long("ready-timeout-secs"),
        argument("SECONDS"),
        env("VLLM_ENGINE_READY_TIMEOUT_S"),
        fallback(300)
    )]
    pub ready_timeout_secs: u64,
    /// Select the tool call parser depending on the model that you're using.
    /// When not specified, the parser is auto-detected from the model.
    #[bpaf(long("tool-call-parser"), argument("NAME"))]
    pub tool_call_parser: Option<String>,
    /// Select the reasoning parser depending on the model that you're using.
    /// When not specified, the parser is auto-detected from the model.
    #[bpaf(long("reasoning-parser"), argument("NAME"))]
    pub reasoning_parser: Option<String>,
    /// Override the maximum model context length. When set, the frontend uses this value
    /// instead of the model's `max_position_embeddings` from `config.json`.
    #[bpaf(long("max-model-len"), argument("TOKENS"))]
    pub max_model_len: Option<u32>,
    /// Hugging Face model identifier used both for backend loading and public model ID.
    // Note: this positional arg must be put after all the known flags and right before the
    // passthrough Python args.
    #[bpaf(positional("MODEL"))]
    pub model: String,
}

impl FrontendRuntimeArgs {
    /// Build one OpenAI-server runtime config for the resolved handshake address.
    fn into_config(
        self,
        handshake_address: String,
        engine_count: usize,
        advertised_host: String,
    ) -> Config {
        Config {
            handshake_address,
            engine_count,
            model: self.model,
            host: self.host,
            port: self.port,
            advertised_host,
            ready_timeout: Duration::from_secs(self.ready_timeout_secs),
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            max_model_len: self.max_model_len,
        }
    }
}

/// Arguments for connecting the Rust frontend to an already running headless engine.
#[derive(Debug, Clone, PartialEq, Eq, Bpaf)]
pub struct FrontendArgs {
    /// Host/IP advertised by the frontend to headless engines for shared input/output ZMQ sockets.
    #[bpaf(
        long("advertised-host"),
        argument("HOST"),
        env("VLLM_HOST_IP"),
        fallback(String::from("127.0.0.1"))
    )]
    pub advertised_host: String,
    /// Headless vLLM engine handshake endpoint, for example `tcp://127.0.0.1:62100`.
    #[bpaf(long("handshake-address"), argument("ADDR"))]
    pub handshake_address: String,
    /// Number of engines expected to connect on the shared handshake socket.
    #[bpaf(long("engine-count"), argument("COUNT"), fallback(1))]
    pub engine_count: usize,
    #[bpaf(external(frontend_runtime_args))]
    pub runtime: FrontendRuntimeArgs,
}

impl FrontendArgs {
    /// Convert the CLI arguments into the OpenAI server's runtime config.
    pub fn into_config(self) -> Config {
        self.runtime.into_config(
            self.handshake_address,
            self.engine_count,
            self.advertised_host,
        )
    }
}

/// Arguments for the managed-engine mode that spawns Python on behalf of the user.
#[derive(Debug, Clone, PartialEq, Eq, Bpaf)]
pub struct ServeArgs {
    /// Only launch the managed Python headless engine and do not start the Rust frontend.
    #[bpaf(long("headless"))]
    pub headless: bool,
    /// Python executable used to launch the managed headless vLLM engine.
    #[bpaf(
        long("python"),
        argument("PYTHON"),
        env("VLLM_RS_PYTHON"),
        fallback(String::from("python3"))
    )]
    pub python: String,
    /// Host/IP used both for the managed-engine handshake endpoint and the frontend-advertised
    /// input/output ZMQ socket addresses.
    #[bpaf(
        long("data-parallel-address"),
        long("handshake-host"),
        argument("HOST"),
        fallback(String::from("127.0.0.1"))
    )]
    pub handshake_host: String,
    /// Optional TCP port for the managed-engine handshake / data-parallel RPC endpoint.
    ///
    /// When omitted, the CLI allocates an ephemeral port automatically.
    #[bpaf(
        long("data-parallel-rpc-port"),
        long("handshake-port"),
        argument("PORT")
    )]
    pub handshake_port: Option<u16>,
    /// Total number of data-parallel engines expected to join the shared handshake socket.
    #[bpaf(
        long("data-parallel-size"),
        long("engine-count"),
        argument("COUNT"),
        fallback(1)
    )]
    pub engine_count: usize,
    #[bpaf(external(frontend_runtime_args))]
    pub runtime: FrontendRuntimeArgs,
    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main serve ...` to launch
    /// the managed engine.
    /// NOTE: Arguments will first be attempted to be parsed as Rust-side `serve` options, and if
    /// not recognized, treated as Python engine flags. To explicitly separate Rust and Python
    /// flags, specify a separator `--` before Python flags.
    #[bpaf(any("PYTHON_ARG", not_help), many)]
    pub python_args: Vec<String>,
}

impl ServeArgs {
    /// Build the OpenAI-server runtime config that should connect to the managed engine.
    pub fn to_frontend_config(&self, handshake_address: String) -> Config {
        self.runtime.clone().into_config(
            handshake_address,
            self.engine_count,
            self.handshake_host.clone(),
        )
    }

    /// Build the managed Python-engine spawn configuration for one resolved handshake port.
    pub fn into_managed_engine_config(self, handshake_port: u16) -> ManagedEngineConfig {
        let mut python_args = self.python_args;
        // Forward `--max-model-len` to the Python engine so both sides agree on the limit.
        if let Some(max_model_len) = self.runtime.max_model_len {
            python_args.push("--max-model-len".to_string());
            python_args.push(max_model_len.to_string());
        }
        ManagedEngineConfig {
            python: self.python,
            model: self.runtime.model,
            handshake_host: self.handshake_host,
            handshake_port,
            engine_count: self.engine_count,
            python_args,
        }
    }
}

fn split_name_from_args<I, T>(itr: I) -> (String, Vec<OsString>)
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let args: Vec<OsString> = itr.into_iter().map(Into::into).collect();
    let name = args
        .first()
        .map(|arg| arg.to_string_lossy().into_owned())
        .unwrap_or_else(|| "vllm-rs".to_string());
    (name, args.into_iter().skip(1).collect())
}

fn not_help(arg: String) -> Option<String> {
    (!matches!(arg.as_str(), "--help" | "-h")).then_some(arg)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{Cli, Command};

    #[test]
    fn serve_args_mix_python_and_rust_flags_without_separator() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--python",
            "../vllm/.venv/bin/python",
            "--dtype",
            "float16",
            "--port",
            "9123",
        ])
        .unwrap();

        expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "../vllm/.venv/bin/python",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        engine_count: 1,
                        runtime: FrontendRuntimeArgs {
                            host: "127.0.0.1",
                            port: 9123,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                            model: "Qwen/Qwen3-0.6B",
                        },
                        python_args: [
                            "--dtype",
                            "float16",
                        ],
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }

    #[test]
    fn serve_args_support_equals_syntax_for_mixed_flags() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--dtype=float16",
            "--port=9123",
        ])
        .unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve args");
        };
        assert_eq!(args.runtime.port, 9123);
        assert_eq!(args.python_args, ["--dtype=float16"]);
    }

    #[test]
    fn serve_args_explicit_user_separator() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--python",
            "python3",
            "--dtype",
            "float16",
            "--",
            "--port",
            "9123",
        ])
        .unwrap();

        expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        engine_count: 1,
                        runtime: FrontendRuntimeArgs {
                            host: "127.0.0.1",
                            port: 8000,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                            model: "Qwen/Qwen3-0.6B",
                        },
                        python_args: [
                            "--dtype",
                            "float16",
                            "--port",
                            "9123",
                        ],
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }

    #[test]
    fn serve_args_leave_python_multi_char_single_dash_aliases_in_passthrough() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--python",
            "python3",
            "-dp",
            "4",
            "--port",
            "9123",
        ])
        .unwrap();

        expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        engine_count: 1,
                        runtime: FrontendRuntimeArgs {
                            host: "127.0.0.1",
                            port: 9123,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                            model: "Qwen/Qwen3-0.6B",
                        },
                        python_args: [
                            "-dp",
                            "4",
                        ],
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }

    #[test]
    fn serve_args_accept_handshake_aliases() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--python",
            "python3",
            "--handshake-host",
            "10.99.48.128",
            "--handshake-port",
            "13345",
            "--engine-count",
            "4",
        ])
        .unwrap();

        expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3",
                        handshake_host: "10.99.48.128",
                        handshake_port: Some(
                            13345,
                        ),
                        engine_count: 4,
                        runtime: FrontendRuntimeArgs {
                            host: "127.0.0.1",
                            port: 8000,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                            model: "Qwen/Qwen3-0.6B",
                        },
                        python_args: [],
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }

    #[test]
    fn serve_args_accept_known_flags_before_model() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "--python",
            "python3.12",
            "--data-parallel-size",
            "2",
            "Qwen/Qwen3-0.6B",
        ])
        .unwrap();

        expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        headless: false,
                        python: "python3.12",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        engine_count: 2,
                        runtime: FrontendRuntimeArgs {
                            host: "127.0.0.1",
                            port: 8000,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                            model: "Qwen/Qwen3-0.6B",
                        },
                        python_args: [],
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }

    #[test]
    fn serve_args_accept_headless_mode() {
        let cli =
            Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--headless"]).unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve args");
        };
        assert!(args.headless);
    }

    #[test]
    fn serve_frontend_config_uses_dp_address_for_both_handshake_and_transport_host() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--handshake-host",
            "10.99.48.128",
            "--engine-count",
            "4",
        ])
        .unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve args");
        };
        let config = args.to_frontend_config("tcp://10.99.48.128:29550".to_string());

        expect![[r#"
            Config {
                handshake_address: "tcp://10.99.48.128:29550",
                engine_count: 4,
                model: "Qwen/Qwen3-0.6B",
                host: "127.0.0.1",
                port: 8000,
                advertised_host: "10.99.48.128",
                ready_timeout: 300s,
                tool_call_parser: None,
                reasoning_parser: None,
                max_model_len: None,
            }
        "#]]
        .assert_debug_eq(&config);
    }
}
