// Anthropic Messages API compatibility layer
// Reference: /tmp/omlx-install/omlx/api/anthropic_utils.py (947 lines)

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    pub r#type: String,
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct ContentBlock {
    pub r#type: String,
    pub text: Option<String>,
    pub thinking: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
