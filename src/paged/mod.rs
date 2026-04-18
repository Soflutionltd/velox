//! Continuous batching + paged KV cache.
//!
//! This module is what turns Velox from a one-request-at-a-time toy into a
//! real LLM inference server. It combines two ideas pioneered by vLLM
//! (Kwon et al., SOSP 2023):
//!
//!   * **Continuous batching**: the active batch composition can change at
//!     every decode step. As soon as a request finishes, we slot a new one
//!     in its place — keeping GPU utilisation near 100%.
//!
//!   * **Paged KV cache**: instead of pre-allocating a contiguous
//!     `max_seq_len` block per request (massive waste — typically 60-80% of
//!     KV memory is unused), we allocate KV memory as fixed-size pages
//!     (default 16 tokens). Each request holds a block table mapping its
//!     logical positions to physical pages. This lets us pack 2-4× more
//!     concurrent requests into the same VRAM.
//!
//! Submodules:
//!   * [`pages`] — page allocator + per-layer KV tensor pools
//!   * [`request`] — request lifecycle and block tables
//!   * [`qwen3`] — Qwen3 forked with paged attention
//!   * [`scheduler`] — batch admission, decode loop, streaming dispatch

pub mod pages;
pub mod qwen3;
pub mod request;
pub mod scheduler;

pub use pages::{PagedKvCache, PagedKvConfig};
pub use request::{Request, RequestId, RequestSnapshot, RequestStatus};
pub use scheduler::{BatchScheduler, SchedulerConfig};
