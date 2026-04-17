use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::json;
use tokio::runtime::Runtime;
use vllm_chat::{
    AssistantMessageExt as _, ChatLlm, ChatMessage, ChatRequest, ChatTool, ChatToolChoice,
    LoadModelBackendsOptions, ParserSelection, load_model_backends,
};
use vllm_engine_core_client::test_utils::{
    IpcNamespace, ScriptedFakeEngine, ScriptedFakeEngineHandle, ScriptedFakeEngineScenario,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::TextLlm;
use vllm_text::tokenizers::Tokenizer;

const MODEL_ID: &str = "Qwen/Qwen3-0.6B";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScenarioKind {
    PreprocessHeavy,
    StreamHeavy,
    StructuredHeavy,
}

impl ScenarioKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::PreprocessHeavy => "preprocess_heavy",
            Self::StreamHeavy => "stream_heavy",
            Self::StructuredHeavy => "structured_heavy",
        }
    }
}

#[derive(Debug, Clone)]
struct FrontendScenarioSpec {
    kind: ScenarioKind,
    fake_engine: FakeEngineScriptSpec,
    request: ChatRequest,
    tool_parser: ParserSelection,
    reasoning_parser: ParserSelection,
}

#[derive(Debug, Clone)]
struct FakeEngineScriptSpec {
    output_text: String,
    chunk_size: usize,
    include_logprobs: bool,
}

impl FakeEngineScriptSpec {
    fn new(output_text: impl Into<String>) -> Self {
        Self {
            output_text: output_text.into(),
            chunk_size: usize::MAX,
            include_logprobs: false,
        }
    }

    fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    fn with_logprobs(mut self, include_logprobs: bool) -> Self {
        self.include_logprobs = include_logprobs;
        self
    }
}

impl FrontendScenarioSpec {
    fn preprocess_heavy() -> Self {
        let prompt = long_preprocess_prompt();
        let mut request = ChatRequest::for_test();
        request.request_id = "bench-preprocess".to_string();
        request.messages = vec![
            ChatMessage::system("You are a performance benchmark assistant."),
            ChatMessage::user(prompt),
        ];
        request.sampling_params.max_tokens = Some(4);
        request.intermediate = true;

        Self {
            kind: ScenarioKind::PreprocessHeavy,
            fake_engine: FakeEngineScriptSpec::new("ok").with_chunk_size(1),
            request,
            tool_parser: ParserSelection::None,
            reasoning_parser: ParserSelection::None,
        }
    }

    fn stream_heavy() -> Self {
        let mut request = ChatRequest::for_test();
        request.request_id = "bench-stream".to_string();
        request.messages = vec![
            ChatMessage::system("You stream short decode chunks as fast as possible."),
            ChatMessage::user("Count upward in a terse stream."),
        ];
        request.sampling_params.max_tokens = Some(256);
        request.intermediate = true;

        Self {
            kind: ScenarioKind::StreamHeavy,
            fake_engine: FakeEngineScriptSpec::new("token ".repeat(192)).with_chunk_size(1),
            request,
            tool_parser: ParserSelection::None,
            reasoning_parser: ParserSelection::None,
        }
    }

    fn structured_heavy() -> Self {
        let mut request = ChatRequest::for_test();
        request.request_id = "bench-structured".to_string();
        request.messages = vec![
            ChatMessage::system("Reason briefly, then call a tool in the Qwen tool-call format."),
            ChatMessage::user("Check the weather in Paris and use the available tool."),
        ];
        request.sampling_params.max_tokens = Some(128);
        request.intermediate = true;
        request.tools = vec![weather_tool()];
        request.tool_choice = ChatToolChoice::Auto;

        Self {
            kind: ScenarioKind::StructuredHeavy,
            fake_engine: FakeEngineScriptSpec::new(
                "<think>Need the weather tool.</think><tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"unit\":\"celsius\"}}\n</tool_call>",
            )
            .with_chunk_size(7)
            .with_logprobs(true),
            request,
            tool_parser: ParserSelection::Auto,
            reasoning_parser: ParserSelection::Auto,
        }
    }
}

struct FrontendBenchFixture {
    runtime: Runtime,
    _ipc: IpcNamespace,
    _fake_engine: ScriptedFakeEngineHandle,
    chat: ChatLlm,
    request: ChatRequest,
    kind: ScenarioKind,
}

impl FrontendBenchFixture {
    fn new(spec: FrontendScenarioSpec) -> Self {
        let runtime = Runtime::new().expect("create tokio runtime for benchmark fixture");
        let (ipc, fake_engine, chat) = runtime.block_on(async {
            let ipc = IpcNamespace::new().expect("create ipc namespace");
            let handshake_address = ipc.handshake_endpoint();
            let loaded = load_model_backends(MODEL_ID, LoadModelBackendsOptions::default())
                .await
                .expect("load benchmark model backends from Hugging Face");
            let scripted_scenario = tokenize_fake_engine_script(
                loaded.text_backend.tokenizer().as_ref(),
                &spec.fake_engine,
            );
            let fake_engine = ScriptedFakeEngine::new(scripted_scenario)
                .spawn(handshake_address.clone(), b"bench-engine".to_vec());

            let client = EngineCoreClient::connect_with_input_output_addresses(
                EngineCoreClientConfig::new_single(handshake_address.clone())
                    .with_model_name(MODEL_ID),
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            )
            .await
            .expect("connect benchmark engine-core client");
            let text = TextLlm::new(Llm::new(client), loaded.text_backend);
            let chat = ChatLlm::new(text, loaded.chat_backend)
                .with_tool_call_parser(spec.tool_parser.clone())
                .with_reasoning_parser(spec.reasoning_parser.clone());

            (ipc, fake_engine, chat)
        });

        Self {
            runtime,
            _ipc: ipc,
            _fake_engine: fake_engine,
            chat,
            request: spec.request,
            kind: spec.kind,
        }
    }

    fn run_once(&self) {
        let collected = self.runtime.block_on(async {
            self.chat
                .chat(self.request.clone())
                .await
                .expect("start benchmark chat request")
                .collect_message()
                .await
                .expect("collect benchmark chat output")
        });

        let tool_call_count = collected.message.tool_calls().count();
        let reasoning = collected.message.reasoning();
        criterion::black_box(self.kind);
        criterion::black_box(collected.prompt_token_count);
        criterion::black_box(collected.output_token_count);
        criterion::black_box(collected.message.text());
        criterion::black_box(reasoning);
        criterion::black_box(tool_call_count);
        criterion::black_box(
            collected
                .logprobs
                .as_ref()
                .map(|logprobs| logprobs.positions.len()),
        );
    }
}

fn tokenize_fake_engine_script(
    tokenizer: &dyn Tokenizer,
    script: &FakeEngineScriptSpec,
) -> ScriptedFakeEngineScenario {
    let output_token_ids = tokenizer
        .encode(&script.output_text, false)
        .expect("tokenize fake-engine scripted output with model tokenizer");
    let mut scenario =
        ScriptedFakeEngineScenario::new(output_token_ids).with_chunk_size(script.chunk_size);
    if script.include_logprobs {
        scenario = scenario.with_logprobs(true);
    }
    scenario
}

fn weather_tool() -> ChatTool {
    ChatTool {
        name: "get_weather".to_string(),
        description: Some("Fetch the current weather for a city.".to_string()),
        parameters: json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "unit"]
        }),
        strict: None,
    }
}

fn long_preprocess_prompt() -> String {
    let mut prompt = String::new();
    for round in 0..8 {
        prompt.push_str("Round ");
        prompt.push_str(&(round + 1).to_string());
        prompt.push_str(
            ": summarize this request, preserve structured details, and keep the answer terse. ",
        );
        prompt.push_str(
            "Need low scheduling overhead, stable streaming, and accurate tool-call surfaces. ",
        );
        prompt.push_str("Inputs include long chat history, large tool schema blocks, and multilingual instructions. ");
        prompt.push_str("Also preserve literals like {\"mode\":\"benchmark\"} and XML-ish tags such as <tool_call>.\n");
    }
    prompt
}

fn bench_frontend_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("chat_frontend_scenarios");
    for spec in [
        FrontendScenarioSpec::preprocess_heavy(),
        FrontendScenarioSpec::stream_heavy(),
        FrontendScenarioSpec::structured_heavy(),
    ] {
        let bench_name = spec.kind.as_str().to_string();
        let fixture = FrontendBenchFixture::new(spec);
        group.bench_function(bench_name, |b| b.iter(|| fixture.run_once()));
    }
    group.finish();
}

criterion_group!(benches, bench_frontend_scenarios);
criterion_main!(benches);
