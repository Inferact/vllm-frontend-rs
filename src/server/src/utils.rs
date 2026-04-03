use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use serde_json::Value;
use thiserror_ext::AsReport;

use crate::error::ApiError;

/// Return the current Unix timestamp in seconds for OpenAI response objects.
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Construct an API error for a failed utility call to the engine core.
pub fn utility_call_error(method: &str, error: impl AsReport) -> ApiError {
    ApiError::server_error(format!("failed to call {method}: {}", error.as_report()))
}

/// Extract the optional `X-data-parallel-rank` header from an HTTP request.
///
/// Returns `None` when the header is absent or cannot be parsed as a `u32`, matching the Python
/// vLLM behavior in `_get_data_parallel_rank()`.
pub fn get_data_parallel_rank(headers: &HeaderMap) -> Option<u32> {
    headers
        .get("x-data-parallel-rank")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
}

/// Merge `kv_transfer_params` into the `vllm_xargs` map, mirroring the Python vLLM behavior
/// where `kv_transfer_params` is injected into `extra_args` for engine-core consumption.
pub fn merge_kv_transfer_params(
    mut xargs: Option<HashMap<String, Value>>,
    kv_transfer_params: Option<&HashMap<String, Value>>,
) -> Option<HashMap<String, Value>> {
    if let Some(kv_params) = kv_transfer_params {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.insert(
            "kv_transfer_params".to_string(),
            // This is safe because we know that `kv_params` is already valid JSON.
            serde_json::to_value(kv_params).unwrap(),
        );
    }
    xargs
}

/// Convert OpenAI-style `logit_bias` with string token-ID keys into the internal
/// `HashMap<u32, f32>` representation, validating that every key parses as a `u32`.
pub fn convert_logit_bias(
    logit_bias: Option<HashMap<String, f32>>,
) -> Result<Option<HashMap<u32, f32>>, ApiError> {
    logit_bias
        .map(|bias| {
            bias.into_iter()
                .map(|(key, value)| {
                    key.parse().map(|k| (k, value)).map_err(|_| {
                        ApiError::invalid_request(
                            format!(
                                "Invalid key in 'logit_bias': '{key}' is not a valid token ID. \
                                 Token IDs must be non-negative integers."
                            ),
                            Some("logit_bias"),
                        )
                    })
                })
                .collect()
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;

    use super::get_data_parallel_rank;

    #[test]
    fn parses_valid_rank() {
        let mut headers = HeaderMap::new();
        headers.insert("x-data-parallel-rank", "3".parse().unwrap());
        assert_eq!(get_data_parallel_rank(&headers), Some(3));
    }

    #[test]
    fn parses_rank_zero() {
        let mut headers = HeaderMap::new();
        headers.insert("x-data-parallel-rank", "0".parse().unwrap());
        assert_eq!(get_data_parallel_rank(&headers), Some(0));
    }

    #[test]
    fn returns_none_when_header_absent() {
        assert_eq!(get_data_parallel_rank(&HeaderMap::new()), None);
    }

    #[test]
    fn returns_none_for_non_integer_value() {
        let mut headers = HeaderMap::new();
        headers.insert("x-data-parallel-rank", "abc".parse().unwrap());
        assert_eq!(get_data_parallel_rank(&headers), None);
    }

    #[test]
    fn returns_none_for_negative_value() {
        let mut headers = HeaderMap::new();
        headers.insert("x-data-parallel-rank", "-1".parse().unwrap());
        assert_eq!(get_data_parallel_rank(&headers), None);
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Data-Parallel-Rank", "5".parse().unwrap());
        assert_eq!(get_data_parallel_rank(&headers), Some(5));
    }
}
