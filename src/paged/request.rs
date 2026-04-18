//! Per-request state and lifecycle for the batch scheduler.

use crate::backend::traits::StreamChunk;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::mpsc;

/// Sentinel codepoint Unicode emits when bytes can't form a valid char yet.
const UNICODE_REPLACEMENT: char = '\u{FFFD}';

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Opaque request identifier (monotonic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

impl RequestId {
    pub fn new() -> Self {
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "req#{}", self.0)
    }
}

/// Lifecycle state of a single request inside the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    /// In the wait queue, not yet admitted to the running batch.
    Waiting,
    /// Admitted; KV pages allocated; participates in every batch step.
    Running,
    /// Generation finished (EOS, max_tokens, stop seq, or client cancelled).
    Finished,
    /// Aborted (OOM during prefill, or backend error).
    Aborted,
}

/// One in-flight generation request.
pub struct Request {
    pub id: RequestId,
    /// Tokenised prompt (after chat template). Treated as immutable.
    pub prompt_tokens: Vec<u32>,
    /// Tokens generated so far (excludes prompt).
    pub generated_tokens: Vec<u32>,
    /// `prompt_tokens.len() + generated_tokens.len()` once the request has
    /// at least its full prompt cached. While prefill is happening this
    /// counts the number of prompt tokens already pushed into KV.
    pub seq_len: usize,
    /// Sampling parameters.
    pub temperature: f64,
    pub top_p: f64,
    pub max_new_tokens: usize,
    pub stop_sequences: Vec<String>,
    pub eos_token_ids: Vec<u32>,
    /// Block table: logical position `i` lives in physical page
    /// `block_table[i / page_size]`, slot `i % page_size`.
    pub block_table: Vec<u32>,
    pub status: RequestStatus,
    pub created_at: Instant,
    pub admitted_at: Option<Instant>,
    /// Channel to push streaming chunks back to the HTTP handler. The
    /// handler drops its receiver when the client disconnects, which we
    /// detect via `send_chunk()` returning `Err`.
    pub tx: mpsc::Sender<StreamChunk>,
    /// Running concatenation of decoded text, used for stop-sequence
    /// detection AND for incremental streaming. We re-decode the entire
    /// `prompt_tokens + generated_tokens` slice each step and emit the
    /// suffix that wasn't there before; this is robust to BPE merges and
    /// multi-byte UTF-8 sequences (we wait until the diff no longer
    /// ends with U+FFFD before emitting).
    pub decoded_text: String,
}

impl Request {
    /// Push a chunk to the client. Returns `false` if the receiver is gone
    /// (client disconnected) — the scheduler treats this as a cancel.
    pub fn send_chunk(&self, chunk: StreamChunk) -> bool {
        // We use try_send so the inference loop never blocks on slow
        // clients; backpressure manifests as the channel becoming full,
        // which we treat as "client too slow, drop and cancel".
        match self.tx.try_send(chunk) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                // Channel full → consumer can't keep up. Bail rather than
                // back-pressuring the entire batch.
                tracing::warn!("{}: stream channel full, cancelling", self.id);
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                // Receiver dropped (HTTP client disconnected).
                false
            }
        }
    }

    /// True when the request still needs a prefill pass (any prompt
    /// tokens not yet pushed into KV).
    pub fn needs_prefill(&self) -> bool {
        self.seq_len < self.prompt_tokens.len()
    }

    /// Bytes / pages required to admit this request right now.
    pub fn pages_required(&self, page_size: usize, expected_extra_tokens: usize) -> usize {
        let total = self.prompt_tokens.len() + expected_extra_tokens;
        total.div_ceil(page_size)
    }
}

/// Compute the deterministic delta between `previous` and `current`. Returns
/// `Some(delta)` if the new text is a clean extension of the previous one
/// AND doesn't end on an unfinished UTF-8 sequence; `None` otherwise (caller
/// should wait for more tokens).
pub fn safe_text_delta(previous: &str, current: &str) -> Option<String> {
    let suffix = current.strip_prefix(previous)?;
    if suffix.ends_with(UNICODE_REPLACEMENT) {
        return None;
    }
    Some(suffix.to_string())
}

/// Lightweight snapshot of request state for logging / metrics.
#[derive(Debug, Clone)]
pub struct RequestSnapshot {
    pub id: RequestId,
    pub status: RequestStatus,
    pub prompt_len: usize,
    pub generated: usize,
    pub seq_len: usize,
    pub waited_ms: u128,
}

impl Request {
    pub fn snapshot(&self) -> RequestSnapshot {
        RequestSnapshot {
            id: self.id,
            status: self.status,
            prompt_len: self.prompt_tokens.len(),
            generated: self.generated_tokens.len(),
            seq_len: self.seq_len,
            waited_ms: self
                .admitted_at
                .unwrap_or(self.created_at)
                .duration_since(self.created_at)
                .as_millis(),
        }
    }
}
