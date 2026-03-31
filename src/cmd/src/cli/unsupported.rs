use std::str::FromStr;

use clap::Args;

/// Marker type for frontend-owned `serve` arguments that `vllm-rs` recognizes but does not
/// support yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {}

impl FromStr for Unsupported {
    type Err = String;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Err("argument is not implemented in Rust frontend yet".to_string())
    }
}

/// Frontend-owned Python `serve` arguments that `vllm-rs` recognizes but does not support yet.
#[derive(Debug, Clone, PartialEq, Eq, Default, Args)]
#[command(next_help_heading = "Options not implemented in Rust frontend yet")]
pub struct UnsupportedArgs {
    /// Name or path of the Hugging Face tokenizer to use. If unspecified, model
    /// name or path will be used.
    #[arg(long)]
    pub tokenizer: Option<Unsupported>,

    /// Tokenizer mode:
    ///
    /// - "auto" will use the tokenizer from `mistral_common` for Mistral models if available,
    ///   otherwise it will use the "hf" tokenizer.
    ///
    /// - "hf" will use the fast tokenizer if available.
    ///
    /// - "slow" will always use the slow tokenizer.
    ///
    /// - "mistral" will always use the tokenizer from `mistral_common`.
    ///
    /// - "deepseek_v32" will always use the tokenizer from `deepseek_v32`.
    ///
    /// - "qwen_vl" will always use the tokenizer from `qwen_vl`.
    ///
    /// - Other custom values can be supported via plugins.
    #[arg(long)]
    pub tokenizer_mode: Option<Unsupported>,

    /// Trust remote code (e.g., from HuggingFace) when downloading the model
    /// and tokenizer.
    #[arg(
        long,
        visible_alias = "no-trust-remote-code",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub trust_remote_code: Option<Unsupported>,

    /// Random seed for reproducibility.
    ///
    /// We must set the global seed because otherwise,
    /// different tensor parallel workers would sample different tokens,
    /// leading to inconsistent results.
    #[arg(long)]
    pub seed: Option<Unsupported>,

    /// Name or path of the Hugging Face config to use. If unspecified, model
    /// name or path will be used.
    #[arg(long)]
    pub hf_config_path: Option<Unsupported>,

    /// Allowing API requests to read local images or videos from directories
    /// specified by the server file system. This is a security risk. Should only
    /// be enabled in trusted environments.
    #[arg(long)]
    pub allowed_local_media_path: Option<Unsupported>,

    /// If set, only media URLs that belong to this domain can be used for
    /// multi-modal inputs.
    #[arg(long)]
    pub allowed_media_domains: Option<Unsupported>,

    /// The specific revision to use for the tokenizer on the Hugging Face Hub.
    /// It can be a branch name, a tag name, or a commit id. If unspecified, will
    /// use the default version.
    #[arg(long)]
    pub tokenizer_revision: Option<Unsupported>,

    /// Maximum number of log probabilities to return when `logprobs` is
    /// specified in `SamplingParams`. The default value comes the default for the
    /// OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length *
    /// vocab_size) logprobs are allowed to be returned and it may cause OOM.
    #[arg(long)]
    pub max_logprobs: Option<Unsupported>,

    /// Indicates the content returned in the logprobs and prompt_logprobs.
    /// Supported mode:
    /// 1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    /// Raw means the values before applying any logit processors, like bad words.
    /// Processed means the values after applying all processors, including
    /// temperature and top_k/top_p.
    #[arg(long)]
    pub logprobs_mode: Option<Unsupported>,

    /// Skip initialization of tokenizer and detokenizer. Expects valid
    /// `prompt_token_ids` and `None` for prompt from the input. The generated
    /// output will contain token ids.
    #[arg(
        long,
        visible_alias = "no-skip-tokenizer-init",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub skip_tokenizer_init: Option<Unsupported>,

    /// If `True`, enables passing text embeddings as inputs via the
    /// `prompt_embeds` key.
    ///
    /// WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed.
    /// Only enable this flag for trusted users!
    #[arg(
        long,
        visible_alias = "no-enable-prompt-embeds",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_prompt_embeds: Option<Unsupported>,

    /// The model name(s) used in the API. If multiple names are provided, the
    /// server will respond to any of the provided names. The model name in the
    /// model field of a response will be the first name in this list. If not
    /// specified, the model name will be the same as the `--model` argument. Noted
    /// that this name(s) will also be used in `model_name` tag content of
    /// prometheus metrics, if multiple names provided, metrics tag will take the
    /// first one.
    #[arg(long)]
    pub served_model_name: Option<Unsupported>,

    /// The token to use as HTTP bearer authorization for remote files. If
    /// `True`, will use the token generated when running `hf auth login`
    /// (stored in `~/.cache/huggingface/token`).
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub hf_token: Option<Unsupported>,

    /// If a dictionary, contains arguments to be forwarded to the Hugging Face
    /// config. If a callable, it is called to update the HuggingFace config.
    #[arg(long)]
    pub hf_overrides: Option<Unsupported>,

    /// The folder path to the generation config. Defaults to `"auto"`, the
    /// generation config will be loaded from model path. If set to `"vllm"`, no
    /// generation config is loaded, vLLM defaults will be used. If set to a folder
    /// path, the generation config will be loaded from the specified folder path.
    /// If `max_new_tokens` is specified in generation config, then it sets a
    /// server-wide limit on the number of output tokens for all requests.
    #[arg(long)]
    pub generation_config: Option<Unsupported>,

    /// IOProcessor plugin name to load at model startup
    #[arg(long)]
    pub io_processor_plugin: Option<Unsupported>,

    /// Path to a dynamically reasoning parser plugin that can be dynamically
    /// loaded and registered.
    #[arg(long)]
    pub reasoning_parser_plugin: Option<Unsupported>,

    /// Rank of the data parallel group.
    #[arg(long, env = "VLLM_DP_RANK")]
    pub data_parallel_rank: Option<Unsupported>,

    /// Whether to use "hybrid" DP LB mode. Applies only to online serving
    /// and when data_parallel_size > 0. Enables running an AsyncLLM
    /// and API server on a "per-node" basis where vLLM load balances
    /// between local data parallel ranks, but an external LB balances
    /// between vLLM nodes/replicas. Set explicitly in conjunction with
    /// --data-parallel-start-rank.
    #[arg(
        long,
        visible_alias = "no-data-parallel-hybrid-lb",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub data_parallel_hybrid_lb: Option<Unsupported>,

    /// Whether to use "external" DP LB mode. Applies only to online serving
    /// and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    /// wide-EP setup in Kubernetes. Set implicitly when --data-parallel-rank
    /// is provided explicitly to vllm serve.
    #[arg(
        long,
        visible_alias = "no-data-parallel-external-lb",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub data_parallel_external_lb: Option<Unsupported>,

    /// This feature is work in progress and no prefill optimization takes place
    /// with this flag enabled currently.
    #[arg(
        long,
        visible_alias = "no-kv-sharing-fast-prefill",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub kv_sharing_fast_prefill: Option<Unsupported>,

    /// The maximum number of input items and options allowed per
    /// prompt for each modality.
    #[arg(long)]
    pub limit_mm_per_prompt: Option<Unsupported>,

    /// Additional args passed to process media inputs, keyed by modalities.
    #[arg(long)]
    pub media_io_kwargs: Option<Unsupported>,

    /// Arguments to be forwarded to the model's processor for multi-modal data,
    /// e.g., image processor.
    #[arg(long)]
    pub mm_processor_kwargs: Option<Unsupported>,

    /// The size (in GiB) of the multi-modal processor cache.
    #[arg(long)]
    pub mm_processor_cache_gb: Option<Unsupported>,

    /// Type of cache to use for the multi-modal preprocessor/mapper.
    #[arg(long)]
    pub mm_processor_cache_type: Option<Unsupported>,

    /// If True, enable handling of LoRA adapters.
    #[arg(
        long,
        visible_alias = "no-enable-lora",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_lora: Option<Unsupported>,

    /// Dictionary mapping specific modalities to LoRA model paths.
    #[arg(long)]
    pub default_mm_loras: Option<Unsupported>,

    /// Target URL to which OpenTelemetry traces will be sent.
    #[arg(long)]
    pub otlp_traces_endpoint: Option<Unsupported>,

    /// It makes sense to set this only if `--otlp-traces-endpoint` is set.
    #[arg(long)]
    pub collect_detailed_traces: Option<Unsupported>,

    /// Maximum number of sequences to be processed in a single iteration.
    #[arg(long)]
    pub max_num_seqs: Option<Unsupported>,

    /// The interval (or buffer size) for streaming in terms of token length.
    #[arg(long)]
    pub stream_interval: Option<Unsupported>,

    /// Structured outputs configuration.
    #[arg(long)]
    pub structured_outputs_config: Option<Unsupported>,

    /// Profiling configuration.
    #[arg(long)]
    pub profiler_config: Option<Unsupported>,

    /// Disable logging statistics.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub disable_log_stats: Option<Unsupported>,

    /// Log aggregate rather than per-engine statistics when using data parallelism.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub aggregate_engine_logging: Option<Unsupported>,

    /// Log requests.
    #[arg(
        long,
        visible_alias = "no-enable-log-requests",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_log_requests: Option<Unsupported>,
}
