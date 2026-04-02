//! Validated JSON extractor for automatic request validation.

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use serde_json::json;
use validator::Validate;

use super::types::Normalizable;

/// A JSON extractor that automatically validates and normalizes the request body.
///
/// This extractor deserializes the request body and automatically calls `.validate()`
/// on types that implement the `Validate` trait. If validation fails, it returns
/// a 400 Bad Request with detailed error information.
pub struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(mut data) =
            Json::<T>::from_request(req, state)
                .await
                .map_err(|err: JsonRejection| {
                    let error_message = match err {
                        JsonRejection::JsonDataError(e) => {
                            format!("Invalid JSON data: {e}")
                        }
                        JsonRejection::JsonSyntaxError(e) => {
                            format!("JSON syntax error: {e}")
                        }
                        JsonRejection::MissingJsonContentType(_) => {
                            "Missing Content-Type: application/json header".to_string()
                        }
                        _ => format!("Failed to parse JSON: {err}"),
                    };

                    (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": error_message,
                                "type": "invalid_request_error",
                                "code": "json_parse_error"
                            }
                        })),
                    )
                        .into_response()
                })?;

        data.normalize();

        data.validate().map_err(|validation_errors| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": validation_errors.to_string(),
                        "type": "invalid_request_error",
                        "code": 400
                    }
                })),
            )
                .into_response()
        })?;

        Ok(ValidatedJson(data))
    }
}

impl<T> std::ops::Deref for ValidatedJson<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for ValidatedJson<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
