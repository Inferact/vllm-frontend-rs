use futures_async_stream::try_stream;

use crate::error::Error;
use crate::output::{GenerateOutput, GenerateOutputStreamImpl};

#[try_stream(ok = GenerateOutput, error = Error)]
pub(crate) async fn batched_generate_output_stream(
    raw_stream: GenerateOutputStreamImpl,
    stream_interval: usize,
) {
    let mut emitted_first_output = false;
    let mut buffered: Option<GenerateOutput> = None;

    #[for_await]
    for next in raw_stream {
        let output = next?;

        if !emitted_first_output {
            emitted_first_output = true;
            yield output;
            continue;
        }

        if let Some(existing) = buffered.as_mut() {
            merge_generate_output(existing, output);
        } else {
            buffered = Some(output);
        }

        // Match Python vLLM `OutputProcessor.make_request_output()` semantics:
        // 1. the first output is streamed immediately,
        // 2. later outputs are held until they accumulate at least `stream_interval` tokens, and
        // 3. terminal metadata is always flushed immediately even if the buffered token count stays
        //    below the interval.
        if let Some(output) = buffered
            .take_if(|output| output.finished() || output.token_ids.len() >= stream_interval)
        {
            yield output;
        }
    }

    debug_assert!(
        buffered.is_none(),
        "batched generate stream should not retain buffered output after termination"
    );
}

fn merge_generate_output(buffered: &mut GenerateOutput, next: GenerateOutput) {
    let GenerateOutput {
        request_id,
        prompt_info,
        token_ids,
        logprobs,
        finish_reason,
        kv_transfer_params,
    } = buffered;

    debug_assert_eq!(*request_id, next.request_id);
    debug_assert!(
        prompt_info.is_none(),
        "only the buffered first generate output should carry prompt info"
    );
    debug_assert!(
        next.prompt_info.is_none(),
        "only the first incoming generate output should carry prompt info"
    );

    token_ids.extend(next.token_ids);

    match (logprobs.as_mut(), next.logprobs) {
        (Some(logprobs), Some(mut next_logprobs)) => {
            logprobs.positions.append(&mut next_logprobs.positions);
        }
        (None, Some(next_logprobs)) => {
            *logprobs = Some(next_logprobs);
        }
        _ => {}
    }

    if let Some(next_finish_reason) = next.finish_reason {
        *finish_reason = Some(next_finish_reason);
        *kv_transfer_params = next.kv_transfer_params;
    }
}
