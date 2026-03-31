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
}
