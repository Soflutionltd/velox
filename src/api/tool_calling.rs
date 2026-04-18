//! Tool calling extraction from raw model output.
//!
//! Supports the two formats most OSS chat models actually use:
//!
//!   * **Hermes / Qwen format**: `<tool_call>{"name": "...", "arguments": ...}</tool_call>`
//!   * **OpenAI-style**: a top-level JSON object `{"name": "...", "arguments": "..."}`
//!     or `{"function": {"name": "...", "arguments": "..."}}`
//!
//! The parser is deliberately tolerant: it scans the text for the markers,
//! returns whatever it can extract, and reports the remaining "natural-language"
//! text separately so it can still be returned to the caller.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Result of scanning a generation for tool calls.
#[derive(Debug, Default)]
pub struct ParsedToolCalls {
    pub calls: Vec<ParsedCall>,
    /// The original text with all `<tool_call>` blocks stripped.
    pub cleaned_text: String,
}

#[derive(Debug, Clone)]
pub struct ParsedCall {
    pub name: String,
    /// Arguments serialised as JSON text (matches OpenAI's wire format).
    pub arguments_json: String,
}

/// Try to extract tool calls from a model's raw output. Always returns; an
/// empty `calls` list means "no tool call detected".
pub fn parse_tool_calls(text: &str) -> ParsedToolCalls {
    let mut out = ParsedToolCalls::default();
    let mut cleaned = String::with_capacity(text.len());

    // 1) Hermes / Qwen / Mistral pattern: <tool_call>...</tool_call>
    let mut cursor = 0usize;
    while let Some(start) = text[cursor..].find("<tool_call>") {
        let abs_start = cursor + start;
        cleaned.push_str(&text[cursor..abs_start]);
        let inner_start = abs_start + "<tool_call>".len();
        let abs_end = match text[inner_start..].find("</tool_call>") {
            Some(rel) => inner_start + rel,
            None => {
                // Unterminated: keep the rest of the text and stop.
                cleaned.push_str(&text[abs_start..]);
                cursor = text.len();
                break;
            }
        };
        let payload = text[inner_start..abs_end].trim();
        if let Some(call) = parse_single_call(payload) {
            out.calls.push(call);
        }
        cursor = abs_end + "</tool_call>".len();
    }
    cleaned.push_str(&text[cursor..]);

    // 2) Bare-JSON fallback: only attempt if no Hermes blocks were found.
    if out.calls.is_empty() {
        if let Some(call) = parse_bare_json_call(cleaned.trim()) {
            out.calls.push(call);
            cleaned.clear();
        }
    }

    out.cleaned_text = cleaned.trim().to_string();
    out
}

/// Parse a single JSON payload like
///   `{"name": "X", "arguments": {...}}`
/// or `{"function": {"name": "X", "arguments": {...}}}`
fn parse_single_call(payload: &str) -> Option<ParsedCall> {
    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    let inner = v.get("function").unwrap_or(&v);
    let name = inner.get("name").and_then(|x| x.as_str())?.to_string();
    let args = inner.get("arguments").cloned().unwrap_or(serde_json::json!({}));
    let arguments_json = match args {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    };
    Some(ParsedCall {
        name,
        arguments_json,
    })
}

/// Bare-JSON fallback: trim whitespace then try to parse the entire output as
/// a single JSON object that looks like a tool call.
fn parse_bare_json_call(text: &str) -> Option<ParsedCall> {
    if !(text.starts_with('{') && text.ends_with('}')) {
        return None;
    }
    parse_single_call(text)
}

/// Validate a tool call's name against the declared tools (no schema validation
/// yet — Phase 2).
pub fn validate_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> bool {
    tools.iter().any(|t| t.function.name == call.function.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hermes_format() {
        let text = "Sure, let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n";
        let out = parse_tool_calls(text);
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "get_weather");
        assert!(out.calls[0].arguments_json.contains("Paris"));
        assert_eq!(out.cleaned_text, "Sure, let me check.");
    }

    #[test]
    fn parses_bare_json_fallback() {
        let text = "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}";
        let out = parse_tool_calls(text);
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "add");
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn no_call_returns_original_text() {
        let text = "Just a plain answer.";
        let out = parse_tool_calls(text);
        assert!(out.calls.is_empty());
        assert_eq!(out.cleaned_text, text);
    }

    #[test]
    fn handles_multiple_calls() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\nthen<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let out = parse_tool_calls(text);
        assert_eq!(out.calls.len(), 2);
        assert_eq!(out.calls[0].name, "a");
        assert_eq!(out.calls[1].name, "b");
        assert_eq!(out.cleaned_text, "then");
    }
}
