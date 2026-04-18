// Inference backend trait: common interface for MLX and llama.cpp

use async_trait::async_trait;
use futures::stream::BoxStream;
use std::path::Path;

/// Handle to a loaded model
pub struct ModelHandle {
    pub id: String,
    pub path: String,
    pub model_type: ModelType,
    pub params_total: u64,
    pub params_active: u64,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    Llm,
    Vlm,
    Embedding,
    Reranker,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct GenerateRequest {
    /// Pre-tokenized prompt (used when caller already tokenized)
    pub prompt_tokens: Vec<u32>,
    /// Chat messages (backend will apply chat template + tokenize)
    /// If non-empty, takes precedence over prompt_tokens
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_sequences: Vec<String>,
}

pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct PrefillResult {
    pub tokens_processed: u32,
    pub cache_blocks: Vec<u64>,
    pub time_ms: f64,
}

/// One event emitted by an in-flight streaming generation.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// A newly sampled token, plus its incremental decoded text (the diff).
    /// `text_delta` may be empty for tokens that don't produce visible bytes
    /// yet (typical for multi-byte UTF-8 sequences mid-encoding).
    Token { token_id: u32, text_delta: String },
    /// Generation finished cleanly. Includes final usage accounting.
    Done {
        finish_reason: String,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// Fatal error mid-stream.
    Error(String),
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Load a model from disk
    async fn load_model(&self, path: &Path) -> anyhow::Result<ModelHandle>;

    /// Unload a model from memory
    async fn unload_model(&self, handle: &ModelHandle) -> anyhow::Result<()>;

    /// Generate tokens (non-streaming)
    async fn generate(&self, handle: &ModelHandle, request: &GenerateRequest) -> anyhow::Result<GenerateResult>;

    /// Generate tokens as a stream of chunks. Default implementation runs the
    /// non-streaming generator and emits the whole result as one chunk; backends
    /// that support real per-token streaming should override.
    async fn generate_stream(
        &self,
        handle: &ModelHandle,
        request: &GenerateRequest,
    ) -> anyhow::Result<BoxStream<'static, StreamChunk>> {
        let result = self.generate(handle, request).await?;
        let chunks = vec![
            StreamChunk::Token {
                token_id: 0,
                text_delta: result.text,
            },
            StreamChunk::Done {
                finish_reason: result.finish_reason,
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.completion_tokens,
            },
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    /// Prefill: process prompt tokens and populate KV cache
    async fn prefill(&self, handle: &ModelHandle, tokens: &[u32]) -> anyhow::Result<PrefillResult>;

    /// Generate embeddings
    async fn embed(&self, handle: &ModelHandle, text: &str) -> anyhow::Result<Vec<f32>>;

    /// Backend name
    fn backend_name(&self) -> &str;

    /// Check if this backend is available on the current system
    fn available() -> bool where Self: Sized;
}
