use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use futures::FutureExt as _;
use openai_protocol::common::{Function as OpenAiFunction, Tool as OpenAiTool};
use serde_json::json;
use tool_parser::parsers::QwenCoderParser as ExternalQwenCoderParser;
use tool_parser::traits::ToolParser as ExternalToolParser;
use vllm_chat::{ChatTool, ToolParser, ToolParserFactory};

const PARSER_NAME: &str = "qwen3_coder";
const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_CHUNK_CHARS: usize = 37;
const LONG_NORMAL_TEXT_REPEATS: usize = 4096;

fn tools() -> Vec<ChatTool> {
    vec![ChatTool {
        name: "get_weather".to_string(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" },
                "date": { "type": "string" },
                "unit": { "type": "string" },
                "days": { "type": "integer" }
            }
        }),
        strict: None,
    }]
}

fn openai_tools(tools: &[ChatTool]) -> Vec<OpenAiTool> {
    tools
        .iter()
        .map(|tool| OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            },
        })
        .collect()
}

fn mixed_fixture() -> String {
    concat!(
        "I will check two cities before answering.\n",
        "<tool_call>\n",
        "<function=get_weather>\n",
        "<parameter=location>Hangzhou</parameter>\n",
        "<parameter=date>2026-04-29</parameter>\n",
        "<parameter=unit>celsius</parameter>\n",
        "<parameter=days>3</parameter>\n",
        "</function>\n",
        "</tool_call>\n",
        "<tool_call>\n",
        "<function=get_weather>\n",
        "<parameter=location>San Francisco</parameter>\n",
        "<parameter=date>2026-04-29</parameter>\n",
        "<parameter=unit>fahrenheit</parameter>\n",
        "<parameter=days>2</parameter>\n",
        "</function>\n",
        "</tool_call>",
    )
    .to_string()
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no Qwen Coder tool markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn split_by_chars(text: &str, chunk_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut count = 0;

    for (index, _) in text.char_indices() {
        if count == chunk_chars {
            chunks.push(text[start..index].to_string());
            start = index;
            count = 0;
        }
        count += 1;
    }

    if start < text.len() {
        chunks.push(text[start..].to_string());
    }

    chunks
}

fn native_parser(tools: &[ChatTool]) -> Box<dyn ToolParser> {
    ToolParserFactory::global()
        .create(PARSER_NAME, tools)
        .expect("Qwen Coder parser should be registered")
}

fn feed_native_parser(parser: &mut dyn ToolParser, chunks: &[String]) -> (String, usize) {
    let mut normal_text = String::new();
    let mut calls_len = 0;
    for chunk in chunks {
        let delta = parser.push(chunk).expect("chunk should parse");
        normal_text.push_str(&delta.normal_text);
        calls_len += delta.calls.len();
    }
    let delta = parser.finish().expect("stream should finish");
    normal_text.push_str(&delta.normal_text);
    calls_len += delta.calls.len();
    (normal_text, calls_len)
}

fn feed_external_parser(
    parser: &mut ExternalQwenCoderParser,
    tools: &[OpenAiTool],
    chunks: &[String],
) -> (String, usize) {
    ExternalToolParser::reset(parser);

    let mut normal_text = String::new();
    let mut calls_len = 0;
    for chunk in chunks {
        let delta = parser
            .parse_incremental(chunk, tools)
            .now_or_never()
            .expect("external parser should not suspend")
            .expect("chunk should parse");
        normal_text.push_str(&delta.normal_text);
        calls_len += delta.calls.len();
    }
    calls_len += parser.get_unstreamed_tool_args().unwrap_or_default().len();
    (normal_text, calls_len)
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[ChatTool],
    text: &str,
    chunk_chars: usize,
    expected_normal_text: &str,
    expected_native_calls_len: usize,
) {
    let chunks = split_by_chars(text, chunk_chars);
    let openai_tools = openai_tools(tools);

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("native_reuse_parser", |b| {
        let mut parser = native_parser(tools);
        b.iter(|| {
            let result = feed_native_parser(&mut *parser, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_native_calls_len);
            black_box(result);
        })
    });

    group.bench_function("native_create_parser", |b| {
        b.iter_batched(
            || native_parser(tools),
            |mut parser| {
                let result = feed_native_parser(&mut *parser, black_box(&chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                debug_assert_eq!(result.1, expected_native_calls_len);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("external_reuse_parser", |b| {
        let mut parser = ExternalQwenCoderParser::new();
        b.iter(|| {
            let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            black_box(result);
        })
    });

    group.bench_function("external_create_parser", |b| {
        b.iter_batched(
            ExternalQwenCoderParser::new,
            |mut parser| {
                let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_qwen3_coder_tool_parser(c: &mut Criterion) {
    let tools = tools();
    let mixed_text = mixed_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "qwen3_coder_tool_parser/mixed_text_tool_call",
        &tools,
        &mixed_text,
        CHUNK_CHARS,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "qwen3_coder_tool_parser/long_normal_text",
        &tools,
        &long_normal_text,
        LONG_NORMAL_TEXT_CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_qwen3_coder_tool_parser);
criterion_main!(benches);
