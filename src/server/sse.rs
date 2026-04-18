//! Server-Sent Events streaming for `chat/completions` and `messages`.
//!
//! Two response shapes are produced from the same underlying `StreamChunk`
//! source: the OpenAI Chat Completions delta format, and the Anthropic
//! Messages event format.

use crate::backend::traits::StreamChunk;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::{Stream, StreamExt};
use serde_json::json;
use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Wrap a `StreamChunk` stream into an OpenAI-compatible `chat.completion.chunk`
/// SSE response.
pub fn openai_chat_sse(
    model: String,
    upstream: impl Stream<Item = StreamChunk> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_seconds();
    let mut sent_role = false;

    let stream = upstream
        .map(move |chunk| {
            let payload = match chunk {
                StreamChunk::Token { text_delta, .. } => {
                    let mut delta = json!({ "content": text_delta });
                    if !sent_role {
                        delta["role"] = json!("assistant");
                        sent_role = true;
                    }
                    json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": null,
                        }],
                    })
                }
                StreamChunk::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                } => json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }),
                StreamChunk::Error(msg) => json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "error": { "message": msg, "type": "inference_error" },
                }),
            };
            Event::default().data(payload.to_string())
        })
        // Final OpenAI sentinel.
        .chain(futures::stream::once(async {
            Event::default().data("[DONE]")
        }))
        .map(Ok::<_, Infallible>);

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// Wrap a `StreamChunk` stream into Anthropic Messages SSE events.
///
/// Emits the canonical sequence:
///   message_start → content_block_start → content_block_delta* →
///   content_block_stop → message_delta → message_stop
pub fn anthropic_messages_sse(
    model: String,
    upstream: impl Stream<Item = StreamChunk> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    let header = futures::stream::iter([
        anthropic_event(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 },
                },
            }),
        ),
        anthropic_event(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" },
            }),
        ),
    ]);

    let body = upstream.flat_map(move |chunk| match chunk {
        StreamChunk::Token { text_delta, .. } => {
            futures::stream::iter(vec![anthropic_event(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": text_delta },
                }),
            )])
        }
        StreamChunk::Done {
            finish_reason,
            prompt_tokens,
            completion_tokens,
        } => futures::stream::iter(vec![
            anthropic_event(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }),
            ),
            anthropic_event(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": map_stop_reason(&finish_reason),
                        "stop_sequence": null,
                    },
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                    },
                }),
            ),
            anthropic_event("message_stop", json!({ "type": "message_stop" })),
        ]),
        StreamChunk::Error(msg) => futures::stream::iter(vec![anthropic_event(
            "error",
            json!({
                "type": "error",
                "error": { "type": "inference_error", "message": msg },
            }),
        )]),
    });

    let stream = header.chain(body).map(Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn anthropic_event(name: &'static str, payload: serde_json::Value) -> Event {
    Event::default().event(name).data(payload.to_string())
}

/// Convert OpenAI-style finish_reason → Anthropic stop_reason.
fn map_stop_reason(finish: &str) -> &'static str {
    match finish {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
