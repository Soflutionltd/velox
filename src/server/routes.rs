use axum::{routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};

// ── OpenAI-compatible routes ──

pub fn openai_routes() -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))
}

pub fn anthropic_routes() -> Router {
    Router::new()
        .route("/v1/messages", post(messages))
}

pub fn admin_routes() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/admin/status", get(status))
}

// ── Request/Response types ──

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub tools: Option<Vec<serde_json::Value>>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> u32 { 4096 }

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Handlers (stubs for Cursor to fill) ──

async fn chat_completions(Json(req): Json<ChatCompletionRequest>) -> Json<ChatCompletionResponse> {
    // TODO: Route to engine pool, get model, generate
    tracing::info!("POST /v1/chat/completions model={}", req.model);
    Json(ChatCompletionResponse {
        id: uuid::Uuid::new_v4().to_string(),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: Message { role: "assistant".into(), content: serde_json::json!("AURA inference server ready. Connect backend to generate.") },
            finish_reason: Some("stop".into()),
        }],
        usage: Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    })
}

async fn completions(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    // TODO: Text completion endpoint
    Json(serde_json::json!({"error": "not yet implemented"}))
}

async fn embeddings(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    // TODO: Embedding endpoint
    Json(serde_json::json!({"error": "not yet implemented"}))
}

async fn list_models() -> Json<serde_json::Value> {
    // TODO: List loaded models from engine pool
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "aura",
            "object": "model",
            "owned_by": "soflution"
        }]
    }))
}

async fn messages(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    // TODO: Anthropic Messages API
    Json(serde_json::json!({"error": "not yet implemented"}))
}

async fn health() -> &'static str { "ok" }

async fn status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}
