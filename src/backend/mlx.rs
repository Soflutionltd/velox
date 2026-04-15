// MLX backend for Apple Silicon
// Uses mlx-rs crate to call Apple's MLX framework from Rust
// Reference: /tmp/omlx-install/omlx/engine/ (batched engine, VLM engine)
//
// TODO: When mlx-rs is mature enough, implement full InferenceBackend.
// For now, this is a stub that Cursor will fill in.

use super::traits::*;
use async_trait::async_trait;
use std::path::Path;

pub struct MlxBackend;

impl MlxBackend {
    pub fn new() -> Self { Self }

    pub fn available() -> bool {
        // Check if running on Apple Silicon
        cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
    }
}

#[async_trait]
impl InferenceBackend for MlxBackend {
    async fn load_model(&self, path: &Path) -> anyhow::Result<ModelHandle> {
        tracing::info!("Loading MLX model from {:?}", path);
        // TODO: Use mlx-rs to load the model
        // let model = mlx_nn::Module::load(path)?;
        Ok(ModelHandle {
            id: uuid::Uuid::new_v4().to_string(),
            path: path.to_string_lossy().to_string(),
            model_type: ModelType::Llm,
            params_total: 0,
            params_active: 0,
        })
    }

    async fn unload_model(&self, _handle: &ModelHandle) -> anyhow::Result<()> {
        // TODO: Free MLX model memory
        Ok(())
    }

    async fn generate(&self, _handle: &ModelHandle, _request: &GenerateRequest) -> anyhow::Result<GenerateResult> {
        // TODO: Call mlx-rs generate
        Ok(GenerateResult {
            tokens: vec![],
            text: "MLX backend not yet connected".into(),
            finish_reason: "stop".into(),
            prompt_tokens: 0,
            completion_tokens: 0,
        })
    }

    async fn prefill(&self, _handle: &ModelHandle, tokens: &[u32]) -> anyhow::Result<PrefillResult> {
        Ok(PrefillResult { tokens_processed: tokens.len() as u32, cache_blocks: vec![], time_ms: 0.0 })
    }

    async fn embed(&self, _handle: &ModelHandle, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![])
    }

    fn backend_name(&self) -> &str { "mlx" }
    fn available() -> bool { cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") }
}
