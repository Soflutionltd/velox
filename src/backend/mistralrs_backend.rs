//! mistral.rs alternative backend.
//!
//! This is a thin wrapper that exposes [`mistralrs::Model`] through
//! the Velox `InferenceBackend` trait so users can opt into it with
//! `--backend mistralrs` (or `VELOX_BACKEND=mistralrs`).
//!
//! Trade-offs vs the default Candle backend:
//!
//! * **Pros**: zero-config — auto-detects architecture, quantization,
//!   chat template. Supports any HF text model out of the box.
//!   Mature single-stream perf with ISQ (in-situ quant) on Metal.
//!
//! * **Cons**: bypasses Velox's paged scheduler entirely. mistral.rs
//!   has its own engine, so all the paged-attention / continuous
//!   batching / prefix-cache work doesn't apply. Memory accounting
//!   isn't visible to our `EnginePool`. Use this when you want
//!   correctness on a model Velox doesn't natively support, not
//!   when you want our throughput characteristics.
//!
//! Compiled only when the `mistralrs` Cargo feature is enabled.

use super::traits::*;
use async_trait::async_trait;
use dashmap::DashMap;
use std::path::Path;
use std::sync::Arc;

use mistralrs::{
    IsqType, Model, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

/// One backend instance manages a pool of loaded models keyed by
/// their handle UUID. mistral.rs's `Model` is `Send + Sync` and owns
/// its own engine, so multiple concurrent requests against the same
/// model are routed through the in-process scheduler it embeds.
pub struct MistralRsBackend {
    models: DashMap<String, Arc<Model>>,
}

impl MistralRsBackend {
    pub fn new() -> Self {
        Self { models: DashMap::new() }
    }

    /// Always available; mistral.rs falls back to CPU if no GPU is
    /// present, so the constructor never fails.
    pub fn available() -> bool {
        true
    }
}

impl Default for MistralRsBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for MistralRsBackend {
    async fn load_model(&self, path: &Path) -> anyhow::Result<ModelHandle> {
        // mistral.rs accepts both HF repo IDs and local directories.
        // We pass through whatever was given.
        let path_str = path.to_string_lossy().to_string();
        tracing::info!("mistralrs: loading {}", path_str);

        let model = TextModelBuilder::new(&path_str)
            .with_isq(IsqType::Q4_0)
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("mistralrs build: {e}"))?;

        let id = uuid::Uuid::new_v4().to_string();
        self.models.insert(id.clone(), Arc::new(model));

        Ok(ModelHandle {
            id,
            path: path_str,
            model_type: ModelType::Llm,
            params_total: 0,
            params_active: 0,
        })
    }

    async fn unload_model(&self, handle: &ModelHandle) -> anyhow::Result<()> {
        self.models.remove(&handle.id);
        Ok(())
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        request: &GenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        let model = self
            .models
            .get(&handle.id)
            .ok_or_else(|| anyhow::anyhow!("mistralrs: model {} not loaded", handle.id))?
            .clone();

        // Translate our chat history into mistral.rs's TextMessages.
        let mut messages = TextMessages::new();
        for m in &request.messages {
            let role = match m.role.as_str() {
                "system" => TextMessageRole::System,
                "user" => TextMessageRole::User,
                "assistant" => TextMessageRole::Assistant,
                _ => TextMessageRole::User,
            };
            messages = messages.add_message(role, &m.content);
        }

        // RequestBuilder lets us set sampling params + max tokens.
        let req = RequestBuilder::from(messages)
            .set_sampler_max_len(request.max_tokens as usize)
            .set_sampler_temperature(request.temperature as f64)
            .set_sampler_topp(request.top_p as f64);

        let response = model
            .send_chat_request(req)
            .await
            .map_err(|e| anyhow::anyhow!("mistralrs send_chat_request: {e}"))?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("mistralrs: empty choices"))?;
        let text = choice
            .message
            .content
            .clone()
            .unwrap_or_default();
        let finish_reason = choice
            .finish_reason
            .clone();

        Ok(GenerateResult {
            tokens: vec![], // mistral.rs doesn't surface raw token ids in its
            // high-level chat response — that's fine, the HTTP API
            // doesn't need them.
            text,
            finish_reason,
            prompt_tokens: response.usage.prompt_tokens as u32,
            completion_tokens: response.usage.completion_tokens as u32,
        })
    }

    async fn prefill(
        &self,
        _handle: &ModelHandle,
        tokens: &[u32],
    ) -> anyhow::Result<PrefillResult> {
        // mistral.rs handles prefill internally on every request.
        // We expose a no-op so the trait stays satisfied.
        Ok(PrefillResult {
            tokens_processed: tokens.len() as u32,
            cache_blocks: vec![],
            time_ms: 0.0,
        })
    }

    async fn embed(&self, _handle: &ModelHandle, _text: &str) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("mistralrs backend: embed() not yet wired (use TextEmbeddingBuilder upstream)")
    }

    fn backend_name(&self) -> &str {
        "mistralrs"
    }

    fn available() -> bool {
        true
    }
}
