use bpaf::Bpaf;

/// Marker type for frontend-owned `serve` arguments that `vllm-rs` recognizes but does not
/// support yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {}

const TOKENIZER_UNSUPPORTED: &str =
    "frontend-owned Python argument '--tokenizer' is not implemented in vllm-rs yet";
const TOKENIZER_MODE_UNSUPPORTED: &str =
    "frontend-owned Python argument '--tokenizer-mode' is not implemented in vllm-rs yet";
const TRUST_REMOTE_CODE_UNSUPPORTED: &str =
    "frontend-owned Python argument '--trust-remote-code' is not implemented in vllm-rs yet";
const SEED_UNSUPPORTED: &str =
    "frontend-owned Python argument '--seed' is not implemented in vllm-rs yet";
const HF_CONFIG_PATH_UNSUPPORTED: &str =
    "frontend-owned Python argument '--hf-config-path' is not implemented in vllm-rs yet";
const ALLOWED_LOCAL_MEDIA_PATH_UNSUPPORTED: &str =
    "frontend-owned Python argument '--allowed-local-media-path' is not implemented in vllm-rs yet";
const ALLOWED_MEDIA_DOMAINS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--allowed-media-domains' is not implemented in vllm-rs yet";
const TOKENIZER_REVISION_UNSUPPORTED: &str =
    "frontend-owned Python argument '--tokenizer-revision' is not implemented in vllm-rs yet";
const MAX_LOGPROBS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--max-logprobs' is not implemented in vllm-rs yet";
const LOGPROBS_MODE_UNSUPPORTED: &str =
    "frontend-owned Python argument '--logprobs-mode' is not implemented in vllm-rs yet";
const SKIP_TOKENIZER_INIT_UNSUPPORTED: &str =
    "frontend-owned Python argument '--skip-tokenizer-init' is not implemented in vllm-rs yet";
const ENABLE_PROMPT_EMBEDS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--enable-prompt-embeds' is not implemented in vllm-rs yet";
const SERVED_MODEL_NAME_UNSUPPORTED: &str =
    "frontend-owned Python argument '--served-model-name' is not implemented in vllm-rs yet";
const HF_TOKEN_UNSUPPORTED: &str =
    "frontend-owned Python argument '--hf-token' is not implemented in vllm-rs yet";
const HF_OVERRIDES_UNSUPPORTED: &str =
    "frontend-owned Python argument '--hf-overrides' is not implemented in vllm-rs yet";
const GENERATION_CONFIG_UNSUPPORTED: &str =
    "frontend-owned Python argument '--generation-config' is not implemented in vllm-rs yet";
const IO_PROCESSOR_PLUGIN_UNSUPPORTED: &str =
    "frontend-owned Python argument '--io-processor-plugin' is not implemented in vllm-rs yet";
const REASONING_PARSER_PLUGIN_UNSUPPORTED: &str =
    "frontend-owned Python argument '--reasoning-parser-plugin' is not implemented in vllm-rs yet";
const DATA_PARALLEL_RANK_UNSUPPORTED: &str =
    "frontend-owned Python argument '--data-parallel-rank' is not implemented in vllm-rs yet";
const DATA_PARALLEL_HYBRID_LB_UNSUPPORTED: &str =
    "frontend-owned Python argument '--data-parallel-hybrid-lb' is not implemented in vllm-rs yet";
const DATA_PARALLEL_EXTERNAL_LB_UNSUPPORTED: &str = "frontend-owned Python argument '--data-parallel-external-lb' is not implemented in vllm-rs yet";
const KV_SHARING_FAST_PREFILL_UNSUPPORTED: &str =
    "frontend-owned Python argument '--kv-sharing-fast-prefill' is not implemented in vllm-rs yet";
const LIMIT_MM_PER_PROMPT_UNSUPPORTED: &str =
    "frontend-owned Python argument '--limit-mm-per-prompt' is not implemented in vllm-rs yet";
const MEDIA_IO_KWARGS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--media-io-kwargs' is not implemented in vllm-rs yet";
const MM_PROCESSOR_KWARGS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--mm-processor-kwargs' is not implemented in vllm-rs yet";
const MM_PROCESSOR_CACHE_GB_UNSUPPORTED: &str =
    "frontend-owned Python argument '--mm-processor-cache-gb' is not implemented in vllm-rs yet";
const MM_PROCESSOR_CACHE_TYPE_UNSUPPORTED: &str =
    "frontend-owned Python argument '--mm-processor-cache-type' is not implemented in vllm-rs yet";
const ENABLE_LORA_UNSUPPORTED: &str =
    "frontend-owned Python argument '--enable-lora' is not implemented in vllm-rs yet";
const DEFAULT_MM_LORAS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--default-mm-loras' is not implemented in vllm-rs yet";
const OTLP_TRACES_ENDPOINT_UNSUPPORTED: &str =
    "frontend-owned Python argument '--otlp-traces-endpoint' is not implemented in vllm-rs yet";
const COLLECT_DETAILED_TRACES_UNSUPPORTED: &str =
    "frontend-owned Python argument '--collect-detailed-traces' is not implemented in vllm-rs yet";
const MAX_NUM_SEQS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--max-num-seqs' is not implemented in vllm-rs yet";
const STREAM_INTERVAL_UNSUPPORTED: &str =
    "frontend-owned Python argument '--stream-interval' is not implemented in vllm-rs yet";
const STRUCTURED_OUTPUTS_CONFIG_UNSUPPORTED: &str = "frontend-owned Python argument '--structured-outputs-config' is not implemented in vllm-rs yet";
const PROFILER_CONFIG_UNSUPPORTED: &str =
    "frontend-owned Python argument '--profiler-config' is not implemented in vllm-rs yet";
const DISABLE_LOG_STATS_UNSUPPORTED: &str =
    "frontend-owned Python argument '--disable-log-stats' is not implemented in vllm-rs yet";
const AGGREGATE_ENGINE_LOGGING_UNSUPPORTED: &str =
    "frontend-owned Python argument '--aggregate-engine-logging' is not implemented in vllm-rs yet";
const ENABLE_LOG_REQUESTS_UNSUPPORTED: &str =
    "frontend-owned Python argument is not implemented in vllm-rs yet";

macro_rules! unsupported_value_parser {
    ($fn_name:ident, $msg:expr) => {
        fn $fn_name(_: String) -> Result<Unsupported, String> {
            Err($msg.to_string())
        }
    };
}

unsupported_value_parser!(reject_tokenizer, TOKENIZER_UNSUPPORTED);
unsupported_value_parser!(reject_tokenizer_mode, TOKENIZER_MODE_UNSUPPORTED);
unsupported_value_parser!(reject_seed, SEED_UNSUPPORTED);
unsupported_value_parser!(reject_hf_config_path, HF_CONFIG_PATH_UNSUPPORTED);
unsupported_value_parser!(
    reject_allowed_local_media_path,
    ALLOWED_LOCAL_MEDIA_PATH_UNSUPPORTED
);
unsupported_value_parser!(
    reject_allowed_media_domains,
    ALLOWED_MEDIA_DOMAINS_UNSUPPORTED
);
unsupported_value_parser!(reject_tokenizer_revision, TOKENIZER_REVISION_UNSUPPORTED);
unsupported_value_parser!(reject_max_logprobs, MAX_LOGPROBS_UNSUPPORTED);
unsupported_value_parser!(reject_logprobs_mode, LOGPROBS_MODE_UNSUPPORTED);
unsupported_value_parser!(reject_served_model_name, SERVED_MODEL_NAME_UNSUPPORTED);
unsupported_value_parser!(reject_hf_token, HF_TOKEN_UNSUPPORTED);
unsupported_value_parser!(reject_hf_overrides, HF_OVERRIDES_UNSUPPORTED);
unsupported_value_parser!(reject_generation_config, GENERATION_CONFIG_UNSUPPORTED);
unsupported_value_parser!(reject_io_processor_plugin, IO_PROCESSOR_PLUGIN_UNSUPPORTED);
unsupported_value_parser!(
    reject_reasoning_parser_plugin,
    REASONING_PARSER_PLUGIN_UNSUPPORTED
);
unsupported_value_parser!(reject_data_parallel_rank, DATA_PARALLEL_RANK_UNSUPPORTED);
unsupported_value_parser!(reject_limit_mm_per_prompt, LIMIT_MM_PER_PROMPT_UNSUPPORTED);
unsupported_value_parser!(reject_media_io_kwargs, MEDIA_IO_KWARGS_UNSUPPORTED);
unsupported_value_parser!(reject_mm_processor_kwargs, MM_PROCESSOR_KWARGS_UNSUPPORTED);
unsupported_value_parser!(
    reject_mm_processor_cache_gb,
    MM_PROCESSOR_CACHE_GB_UNSUPPORTED
);
unsupported_value_parser!(
    reject_mm_processor_cache_type,
    MM_PROCESSOR_CACHE_TYPE_UNSUPPORTED
);
unsupported_value_parser!(reject_default_mm_loras, DEFAULT_MM_LORAS_UNSUPPORTED);
unsupported_value_parser!(
    reject_otlp_traces_endpoint,
    OTLP_TRACES_ENDPOINT_UNSUPPORTED
);
unsupported_value_parser!(
    reject_collect_detailed_traces,
    COLLECT_DETAILED_TRACES_UNSUPPORTED
);
unsupported_value_parser!(reject_max_num_seqs, MAX_NUM_SEQS_UNSUPPORTED);
unsupported_value_parser!(reject_stream_interval, STREAM_INTERVAL_UNSUPPORTED);
unsupported_value_parser!(
    reject_structured_outputs_config,
    STRUCTURED_OUTPUTS_CONFIG_UNSUPPORTED
);
unsupported_value_parser!(reject_profiler_config, PROFILER_CONFIG_UNSUPPORTED);

fn reject_flag_presence(present: &bool) -> bool {
    !*present
}

fn drop_flag_value(_: bool) -> Option<Unsupported> {
    None
}

/// Frontend-owned Python `serve` arguments that `vllm-rs` recognizes but does not support yet.
#[derive(Debug, Clone, PartialEq, Eq, Default, Bpaf)]
pub struct UnsupportedServeArgs {
    /// Name or path of the Hugging Face tokenizer to use. If unspecified, model
    /// name or path will be used.
    #[bpaf(long("tokenizer"), argument::<String>("TOKENIZER"), parse(reject_tokenizer), optional)]
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
    #[bpaf(long("tokenizer-mode"), argument::<String>("MODE"), parse(reject_tokenizer_mode), optional)]
    pub tokenizer_mode: Option<Unsupported>,

    /// Trust remote code (e.g., from HuggingFace) when downloading the model
    /// and tokenizer.
    #[bpaf(
        long("trust-remote-code"),
        long("no-trust-remote-code"),
        switch,
        guard(reject_flag_presence, TRUST_REMOTE_CODE_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub trust_remote_code: Option<Unsupported>,

    /// Random seed for reproducibility.
    ///
    /// We must set the global seed because otherwise,
    /// different tensor parallel workers would sample different tokens,
    /// leading to inconsistent results.
    #[bpaf(long("seed"), argument::<String>("SEED"), parse(reject_seed), optional)]
    pub seed: Option<Unsupported>,

    /// Name or path of the Hugging Face config to use. If unspecified, model
    /// name or path will be used.
    #[bpaf(long("hf-config-path"), argument::<String>("PATH"), parse(reject_hf_config_path), optional)]
    pub hf_config_path: Option<Unsupported>,

    /// Allowing API requests to read local images or videos from directories
    /// specified by the server file system. This is a security risk. Should only
    /// be enabled in trusted environments.
    #[bpaf(long("allowed-local-media-path"), argument::<String>("PATH"), parse(reject_allowed_local_media_path), optional)]
    pub allowed_local_media_path: Option<Unsupported>,

    /// If set, only media URLs that belong to this domain can be used for
    /// multi-modal inputs.
    #[bpaf(long("allowed-media-domains"), argument::<String>("DOMAIN"), parse(reject_allowed_media_domains), optional)]
    pub allowed_media_domains: Option<Unsupported>,

    /// The specific revision to use for the tokenizer on the Hugging Face Hub.
    /// It can be a branch name, a tag name, or a commit id. If unspecified, will
    /// use the default version.
    #[bpaf(long("tokenizer-revision"), argument::<String>("REVISION"), parse(reject_tokenizer_revision), optional)]
    pub tokenizer_revision: Option<Unsupported>,

    /// Maximum number of log probabilities to return when `logprobs` is
    /// specified in `SamplingParams`. The default value comes the default for the
    /// OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length *
    /// vocab_size) logprobs are allowed to be returned and it may cause OOM.
    #[bpaf(long("max-logprobs"), argument::<String>("COUNT"), parse(reject_max_logprobs), optional)]
    pub max_logprobs: Option<Unsupported>,

    /// Indicates the content returned in the logprobs and prompt_logprobs.
    /// Supported mode:
    /// 1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    /// Raw means the values before applying any logit processors, like bad words.
    /// Processed means the values after applying all processors, including
    /// temperature and top_k/top_p.
    #[bpaf(long("logprobs-mode"), argument::<String>("MODE"), parse(reject_logprobs_mode), optional)]
    pub logprobs_mode: Option<Unsupported>,

    /// Skip initialization of tokenizer and detokenizer. Expects valid
    /// `prompt_token_ids` and `None` for prompt from the input. The generated
    /// output will contain token ids.
    #[bpaf(
        long("skip-tokenizer-init"),
        long("no-skip-tokenizer-init"),
        switch,
        guard(reject_flag_presence, SKIP_TOKENIZER_INIT_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub skip_tokenizer_init: Option<Unsupported>,

    /// If `True`, enables passing text embeddings as inputs via the
    /// `prompt_embeds` key.
    ///
    /// WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed.
    /// Only enable this flag for trusted users!
    #[bpaf(
        long("enable-prompt-embeds"),
        long("no-enable-prompt-embeds"),
        switch,
        guard(reject_flag_presence, ENABLE_PROMPT_EMBEDS_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub enable_prompt_embeds: Option<Unsupported>,

    /// The model name(s) used in the API. If multiple names are provided, the
    /// server will respond to any of the provided names. The model name in the
    /// model field of a response will be the first name in this list. If not
    /// specified, the model name will be the same as the `--model` argument. Noted
    /// that this name(s) will also be used in `model_name` tag content of
    /// prometheus metrics, if multiple names provided, metrics tag will take the
    /// first one.
    #[bpaf(long("served-model-name"), argument::<String>("MODEL_NAME"), parse(reject_served_model_name), optional)]
    pub served_model_name: Option<Unsupported>,

    /// The token to use as HTTP bearer authorization for remote files. If
    /// `True`, will use the token generated when running `hf auth login`
    /// (stored in `~/.cache/huggingface/token`).
    #[bpaf(long("hf-token"), argument::<String>("TOKEN"), parse(reject_hf_token), optional)]
    pub hf_token: Option<Unsupported>,

    /// If a dictionary, contains arguments to be forwarded to the Hugging Face
    /// config. If a callable, it is called to update the HuggingFace config.
    #[bpaf(long("hf-overrides"), argument::<String>("JSON"), parse(reject_hf_overrides), optional)]
    pub hf_overrides: Option<Unsupported>,

    /// The folder path to the generation config. Defaults to `"auto"`, the
    /// generation config will be loaded from model path. If set to `"vllm"`, no
    /// generation config is loaded, vLLM defaults will be used. If set to a folder
    /// path, the generation config will be loaded from the specified folder path.
    /// If `max_new_tokens` is specified in generation config, then it sets a
    /// server-wide limit on the number of output tokens for all requests.
    #[bpaf(long("generation-config"), argument::<String>("PATH"), parse(reject_generation_config), optional)]
    pub generation_config: Option<Unsupported>,

    /// IOProcessor plugin name to load at model startup
    #[bpaf(long("io-processor-plugin"), argument::<String>("NAME"), parse(reject_io_processor_plugin), optional)]
    pub io_processor_plugin: Option<Unsupported>,

    /// Path to a dynamically reasoning parser plugin that can be dynamically
    /// loaded and registered.
    #[bpaf(long("reasoning-parser-plugin"), argument::<String>("PATH"), parse(reject_reasoning_parser_plugin), optional)]
    pub reasoning_parser_plugin: Option<Unsupported>,

    /// Rank of the data parallel group.
    #[bpaf(
        long("data-parallel-rank"),
        argument::<String>("RANK"),
        env("VLLM_DP_RANK"),
        parse(reject_data_parallel_rank),
        optional
    )]
    pub data_parallel_rank: Option<Unsupported>,

    /// Whether to use "hybrid" DP LB mode. Applies only to online serving
    /// and when data_parallel_size > 0. Enables running an AsyncLLM
    /// and API server on a "per-node" basis where vLLM load balances
    /// between local data parallel ranks, but an external LB balances
    /// between vLLM nodes/replicas. Set explicitly in conjunction with
    /// --data-parallel-start-rank.
    #[bpaf(
        long("data-parallel-hybrid-lb"),
        long("no-data-parallel-hybrid-lb"),
        switch,
        guard(reject_flag_presence, DATA_PARALLEL_HYBRID_LB_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub data_parallel_hybrid_lb: Option<Unsupported>,

    /// Whether to use "external" DP LB mode. Applies only to online serving
    /// and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    /// wide-EP setup in Kubernetes. Set implicitly when --data-parallel-rank
    /// is provided explicitly to vllm serve.
    #[bpaf(
        long("data-parallel-external-lb"),
        long("no-data-parallel-external-lb"),
        switch,
        guard(reject_flag_presence, DATA_PARALLEL_EXTERNAL_LB_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub data_parallel_external_lb: Option<Unsupported>,

    /// This feature is work in progress and no prefill optimization takes place
    /// with this flag enabled currently.
    #[bpaf(
        long("kv-sharing-fast-prefill"),
        long("no-kv-sharing-fast-prefill"),
        switch,
        guard(reject_flag_presence, KV_SHARING_FAST_PREFILL_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub kv_sharing_fast_prefill: Option<Unsupported>,

    /// The maximum number of input items and options allowed per
    /// prompt for each modality.
    #[bpaf(long("limit-mm-per-prompt"), argument::<String>("JSON"), parse(reject_limit_mm_per_prompt), optional)]
    pub limit_mm_per_prompt: Option<Unsupported>,

    /// Additional args passed to process media inputs, keyed by modalities.
    #[bpaf(long("media-io-kwargs"), argument::<String>("JSON"), parse(reject_media_io_kwargs), optional)]
    pub media_io_kwargs: Option<Unsupported>,

    /// Arguments to be forwarded to the model's processor for multi-modal data,
    /// e.g., image processor.
    #[bpaf(long("mm-processor-kwargs"), argument::<String>("JSON"), parse(reject_mm_processor_kwargs), optional)]
    pub mm_processor_kwargs: Option<Unsupported>,

    /// The size (in GiB) of the multi-modal processor cache.
    #[bpaf(long("mm-processor-cache-gb"), argument::<String>("GIB"), parse(reject_mm_processor_cache_gb), optional)]
    pub mm_processor_cache_gb: Option<Unsupported>,

    /// Type of cache to use for the multi-modal preprocessor/mapper.
    #[bpaf(long("mm-processor-cache-type"), argument::<String>("TYPE"), parse(reject_mm_processor_cache_type), optional)]
    pub mm_processor_cache_type: Option<Unsupported>,

    /// If True, enable handling of LoRA adapters.
    #[bpaf(
        long("enable-lora"),
        long("no-enable-lora"),
        switch,
        guard(reject_flag_presence, ENABLE_LORA_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub enable_lora: Option<Unsupported>,

    /// Dictionary mapping specific modalities to LoRA model paths.
    #[bpaf(long("default-mm-loras"), argument::<String>("JSON"), parse(reject_default_mm_loras), optional)]
    pub default_mm_loras: Option<Unsupported>,

    /// Target URL to which OpenTelemetry traces will be sent.
    #[bpaf(long("otlp-traces-endpoint"), argument::<String>("URL"), parse(reject_otlp_traces_endpoint), optional)]
    pub otlp_traces_endpoint: Option<Unsupported>,

    /// It makes sense to set this only if `--otlp-traces-endpoint` is set.
    #[bpaf(long("collect-detailed-traces"), argument::<String>("MODULES"), parse(reject_collect_detailed_traces), optional)]
    pub collect_detailed_traces: Option<Unsupported>,

    /// Maximum number of sequences to be processed in a single iteration.
    #[bpaf(long("max-num-seqs"), argument::<String>("COUNT"), parse(reject_max_num_seqs), optional)]
    pub max_num_seqs: Option<Unsupported>,

    /// The interval (or buffer size) for streaming in terms of token length.
    #[bpaf(long("stream-interval"), argument::<String>("TOKENS"), parse(reject_stream_interval), optional)]
    pub stream_interval: Option<Unsupported>,

    /// Structured outputs configuration.
    #[bpaf(long("structured-outputs-config"), argument::<String>("JSON"), parse(reject_structured_outputs_config), optional)]
    pub structured_outputs_config: Option<Unsupported>,

    /// Profiling configuration.
    #[bpaf(long("profiler-config"), argument::<String>("JSON"), parse(reject_profiler_config), optional)]
    pub profiler_config: Option<Unsupported>,

    /// Disable logging statistics.
    #[bpaf(
        long("disable-log-stats"),
        switch,
        guard(reject_flag_presence, DISABLE_LOG_STATS_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub disable_log_stats: Option<Unsupported>,

    /// Log aggregate rather than per-engine statistics when using data parallelism.
    #[bpaf(
        long("aggregate-engine-logging"),
        switch,
        guard(reject_flag_presence, AGGREGATE_ENGINE_LOGGING_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub aggregate_engine_logging: Option<Unsupported>,

    /// Log requests.
    #[bpaf(
        long("enable-log-requests"),
        switch,
        guard(reject_flag_presence, ENABLE_LOG_REQUESTS_UNSUPPORTED),
        map(drop_flag_value)
    )]
    pub enable_log_requests: Option<Unsupported>,
}
