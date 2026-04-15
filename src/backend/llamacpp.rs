// llama.cpp backend for Windows/Linux (CUDA, ROCm, CPU)
// Uses llama-cpp-rs crate
// Reference: https://github.com/utilityai/llama-cpp-rs
//
// TODO: Implement full InferenceBackend with llama-cpp-rs

use super::traits::*;
use async_trait::async_trait;
use std::path::Path;

pub struct LlamaCppBackend;

impl LlamaCppBackend {
    pub fn new() -> Self { Self }
    pub fn available() -> bool { true } // Always available as CPU fallback
}

#[async_trait]
impl InferenceBackend for LlamaCppBackend {
    async fn load_model(&self, path: &Path) -> anyhow::Result<ModelHandle> {
        tracing::info!("Loading GGUF model from {:?}", path);
        // TODO: Use llama-cpp-rs to load model
        Ok(ModelHandle {
            id: uuid::Uuid::new_v4().to_string(),
            path: path.to_string_lossy().to_string(),
            model_type: ModelType::Llm,
            params_total: 0,
            params_active: 0,
        })
    }

    async fn unload_model(&self, _handle: &ModelHandle) -> anyhow::Result<()> { Ok(()) }

    async fn generate(&self, _handle: &ModelHandle, _request: &GenerateRequest) -> anyhow::Result<GenerateResult> {
        Ok(GenerateResult {
            tokens: vec![],
            text: "llama.cpp backend not yet connected".into(),
            finish_reason: "stop".into(),
            prompt_tokens: 0,
            completion_tokens: 0,
        })
    }

    async fn prefill(&self, _handle: &ModelHandle, tokens: &[u32]) -> anyhow::Result<PrefillResult> {
        Ok(PrefillResult { tokens_processed: tokens.len() as u32, cache_blocks: vec![], time_ms: 0.0 })
    }

    async fn embed(&self, _handle: &ModelHandle, _text: &str) -> anyhow::Result<Vec<f32>> { Ok(vec![]) }

    fn backend_name(&self) -> &str { "llamacpp" }
    fn available() -> bool { true }
}
