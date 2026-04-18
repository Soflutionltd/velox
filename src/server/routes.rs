use crate::api::tool_calling;
use crate::backend::traits::*;
use crate::server::sse::{anthropic_messages_sse, openai_chat_sse};
use crate::server::AppState;
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

pub fn openai_routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))
}

pub fn anthropic_routes() -> Router<AppState> {
    Router::new().route("/v1/messages", post(messages))
}

pub fn admin_routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/admin/status", get(status))
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stop: Option<StopValue>,
    pub tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopValue {
    One(String),
    Many(Vec<String>),
}
impl StopValue {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopValue::One(s) => vec![s],
            StopValue::Many(v) => v,
        }
    }
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.95 }
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
) -> Response {
    tracing::info!(
        "POST /v1/chat/completions model={} stream={}",
        req.model,
        req.stream
    );

    let handle = match state.pool.get_model(&req.model).await {
        Ok(h) => h,
        Err(e) => return error_json("model_not_found", format!("Model load failed: {e}")),
    };

    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: extract_text_content(&m.content),
        })
        .collect();

    let stop_sequences = req
        .stop
        .map(|s| s.into_vec())
        .unwrap_or_default();

    let gen_req = GenerateRequest {
        prompt_tokens: vec![],
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop_sequences,
    };

    let backend = state.pool.backend_arc();

    if req.stream {
        let stream = match backend.generate_stream(&handle, &gen_req).await {
            Ok(s) => s,
            Err(e) => return error_json("inference_error", format!("generate_stream: {e}")),
        };
        return openai_chat_sse(req.model.clone(), stream).into_response();
    }

    let result = match backend.generate(&handle, &gen_req).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("generate failed: {e:#}");
            return error_json("inference_error", format!("Generation failed: {e}"));
        }
    };

    // Tool-calling extraction (Hermes / OpenAI format) when caller passed tools.
    let (clean_text, tool_calls) = if req.tools.is_some() {
        let parsed = tool_calling::parse_tool_calls(&result.text);
        if !parsed.calls.is_empty() {
            (
                parsed.cleaned_text,
                Some(
                    parsed
                        .calls
                        .into_iter()
                        .map(|c| {
                            serde_json::json!({
                                "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": c.arguments_json,
                                },
                            })
                        })
                        .collect::<Vec<_>>(),
                ),
            )
        } else {
            (result.text.clone(), None)
        }
    } else {
        (result.text.clone(), None)
    };

    let mut message_value = serde_json::json!({
        "role": "assistant",
        "content": if tool_calls.is_some() && clean_text.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(clean_text)
        },
    });
    if let Some(tc) = tool_calls.as_ref() {
        message_value["tool_calls"] = serde_json::Value::Array(tc.clone());
    }

    let finish_reason = if tool_calls.is_some() {
        "tool_calls".to_string()
    } else {
        result.finish_reason
    };

    let response = serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": message_value,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    });

    Json(response).into_response()
}

// ---------------------------------------------------------------------------
// OpenAI legacy /v1/completions (text completion)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LegacyCompletionRequest {
    model: String,
    prompt: PromptValue,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    stop: Option<StopValue>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PromptValue {
    Text(String),
    Array(Vec<String>),
}
impl PromptValue {
    fn into_string(self) -> String {
        match self {
            PromptValue::Text(s) => s,
            PromptValue::Array(v) => v.join("\n"),
        }
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<LegacyCompletionRequest>,
) -> Response {
    let handle = match state.pool.get_model(&req.model).await {
        Ok(h) => h,
        Err(e) => return error_json("model_not_found", format!("Model load failed: {e}")),
    };

    // Treat the legacy `prompt` field as a single user message so the chat
    // template still applies; this is what most OSS chat models expect.
    let user_text = req.prompt.into_string();
    let stop_sequences = req.stop.map(|s| s.into_vec()).unwrap_or_default();

    let gen_req = GenerateRequest {
        prompt_tokens: vec![],
        messages: vec![ChatMessage {
            role: "user".into(),
            content: user_text,
        }],
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop_sequences,
    };

    let backend = state.pool.backend_arc();
    if req.stream {
        let stream = match backend.generate_stream(&handle, &gen_req).await {
            Ok(s) => s,
            Err(e) => return error_json("inference_error", format!("generate_stream: {e}")),
        };
        return openai_chat_sse(req.model.clone(), stream).into_response();
    }

    let result = match backend.generate(&handle, &gen_req).await {
        Ok(r) => r,
        Err(e) => return error_json("inference_error", format!("Generation failed: {e}")),
    };

    Json(serde_json::json!({
        "id": format!("cmpl-{}", uuid::Uuid::new_v4()),
        "object": "text_completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model": req.model,
        "choices": [{
            "text": result.text,
            "index": 0,
            "logprobs": null,
            "finish_reason": result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    })).into_response()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn error_json(kind: &str, msg: String) -> Response {
    Json(serde_json::json!({
        "error": { "message": msg, "type": kind }
    }))
    .into_response()
}

/// Extract plain text from an OpenAI-shaped `content` field.
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

async fn embeddings(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"error": "not yet implemented"}))
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models = state.pool.list_models();
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| serde_json::json!({"id": m, "object": "model", "owned_by": "velox"}))
        .collect();
    Json(serde_json::json!({"object": "list", "data": data}))
}

// ---------------------------------------------------------------------------
// Anthropic Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AnthropicReq {
    model: String,
    messages: Vec<AnthroMsg>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    system: Option<serde_json::Value>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct AnthroMsg {
    role: String,
    content: serde_json::Value,
}

async fn messages(State(state): State<AppState>, Json(req): Json<AnthropicReq>) -> Response {
    tracing::info!(
        "POST /v1/messages model={} stream={}",
        req.model,
        req.stream
    );

    let handle = match state.pool.get_model(&req.model).await {
        Ok(h) => h,
        Err(e) => return error_json("model_not_found", format!("Model load failed: {e}")),
    };

    // Anthropic puts the system prompt outside of the messages array; merge
    // it back as the first chat message so the underlying chat template
    // sees it.
    let mut chat_messages: Vec<ChatMessage> = Vec::new();
    if let Some(system) = req.system.as_ref() {
        let sys_text = extract_anthropic_text(system);
        if !sys_text.is_empty() {
            chat_messages.push(ChatMessage {
                role: "system".into(),
                content: sys_text,
            });
        }
    }
    for m in &req.messages {
        chat_messages.push(ChatMessage {
            role: m.role.clone(),
            content: extract_anthropic_text(&m.content),
        });
    }

    let gen_req = GenerateRequest {
        prompt_tokens: vec![],
        messages: chat_messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop_sequences: req.stop_sequences.unwrap_or_default(),
    };

    let backend = state.pool.backend_arc();

    if req.stream {
        let stream = match backend.generate_stream(&handle, &gen_req).await {
            Ok(s) => s,
            Err(e) => return error_json("inference_error", format!("generate_stream: {e}")),
        };
        return anthropic_messages_sse(req.model.clone(), stream).into_response();
    }

    let result = match backend.generate(&handle, &gen_req).await {
        Ok(r) => r,
        Err(e) => return error_json("inference_error", format!("Generation failed: {e}")),
    };

    let mut content_blocks: Vec<serde_json::Value> = Vec::new();

    if req.tools.is_some() {
        let parsed = tool_calling::parse_tool_calls(&result.text);
        if !parsed.cleaned_text.trim().is_empty() {
            content_blocks.push(serde_json::json!({
                "type": "text",
                "text": parsed.cleaned_text,
            }));
        }
        for c in parsed.calls {
            let args: serde_json::Value =
                serde_json::from_str(&c.arguments_json).unwrap_or(serde_json::json!({}));
            content_blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                "name": c.name,
                "input": args,
            }));
        }
    }

    if content_blocks.is_empty() {
        content_blocks.push(serde_json::json!({
            "type": "text",
            "text": result.text,
        }));
    }

    let stop_reason = if content_blocks
        .iter()
        .any(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_use"))
    {
        "tool_use"
    } else {
        match result.finish_reason.as_str() {
            "stop" => "end_turn",
            "length" => "max_tokens",
            _ => "end_turn",
        }
    };

    Json(serde_json::json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": result.prompt_tokens,
            "output_tokens": result.completion_tokens,
        },
    })).into_response()
}

/// Anthropic content can be a plain string or an array of content blocks.
fn extract_anthropic_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(|v| v.as_str()) {
                Some("text") => p.get("text").and_then(|v| v.as_str()).map(String::from),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Admin
// ---------------------------------------------------------------------------

async fn health() -> &'static str { "ok" }

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "backend": state.pool.backend_ref().backend_name(),
        "models_discovered": state.pool.list_models().len(),
        "models_loaded": state.pool.loaded_models().len(),
        "cache_efficiency": state.metrics.cache_efficiency(),
    }))
}
