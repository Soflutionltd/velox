use axum::{extract::State, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use crate::server::AppState;
use crate::backend::traits::*;

pub fn openai_routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))
}

pub fn anthropic_routes() -> Router<AppState> {
    Router::new()
        .route("/v1/messages", post(messages))
}

pub fn admin_routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/admin/status", get(status))
}

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

#[derive(Debug, Serialize, Deserialize, Clone)]
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

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<serde_json::Value> {
    tracing::info!("POST /v1/chat/completions model={}", req.model);

    // Resolve and load the model through the engine pool
    let handle = match state.pool.get_model(&req.model).await {
        Ok(h) => h,
        Err(e) => {
            return Json(serde_json::json!({
                "error": {
                    "message": format!("Model load failed: {e}"),
                    "type": "model_not_found",
                }
            }));
        }
    };

    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: extract_text_content(&m.content),
        })
        .collect();

    let gen_req = GenerateRequest {
        prompt_tokens: vec![],
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: 0.95,
        stop_sequences: vec![],
    };

    let backend = state.pool.backend_arc();
    let result = match backend.generate(&handle, &gen_req).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("generate failed: {e:#}");
            return Json(serde_json::json!({
                "error": {
                    "message": format!("Generation failed: {e}"),
                    "type": "inference_error",
                }
            }));
        }
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: serde_json::Value::String(result.text),
            },
            finish_reason: Some(result.finish_reason),
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        },
    };

    Json(serde_json::to_value(response).unwrap())
}

/// Extract plain text from an OpenAI message content field.
/// Supports both string content and the array-of-parts format.
fn extract_text_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                let t = p.get("type")?.as_str()?;
                if t == "text" {
                    p.get("text").and_then(|v| v.as_str()).map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

async fn completions(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"error": "not yet implemented"}))
}

async fn embeddings(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"error": "not yet implemented"}))
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models = state.pool.list_models();
    let data: Vec<serde_json::Value> = models.iter().map(|m| {
        serde_json::json!({"id": m, "object": "model", "owned_by": "aura"})
    }).collect();
    Json(serde_json::json!({"object": "list", "data": data}))
}

async fn messages(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"error": "not yet implemented"}))
}

async fn health() -> &'static str { "ok" }

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "models_discovered": state.pool.list_models().len(),
        "models_loaded": state.pool.loaded_models().len(),
        "cache_efficiency": state.metrics.cache_efficiency(),
    }))
}
