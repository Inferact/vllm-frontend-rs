use vllm_text::DynTextBackend;

use crate::error::Result;
use crate::{ChatTemplateContentFormatOption, DynChatBackend};

pub mod hf;

/// Frontend-side chat backend loading options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadModelBackendsOptions {
    pub chat_template_content_format: ChatTemplateContentFormatOption,
}

/// Shared backends loaded from a model id.
pub struct LoadedModelBackends {
    pub text_backend: DynTextBackend,
    pub chat_backend: DynChatBackend,
}

/// Load text and chat backends for the given model id.
pub async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    // Currently, we only have HuggingFace backends.
    hf::load_model_backends(model_id, options).await
}
