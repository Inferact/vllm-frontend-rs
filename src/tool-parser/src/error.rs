use thiserror::Error;

/// Result alias for tool parser operations.
pub type Result<T> = std::result::Result<T, ToolParserError>;

/// Errors produced while creating or running tool parsers.
#[derive(Debug, Error)]
pub enum ToolParserError {
    #[error("tool parser parsing failed: {message}")]
    ParsingFailed { message: String },
}
