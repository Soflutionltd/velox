use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Default, Clone)]
pub struct ServerMetrics {
    inner: Arc<MetricsInner>,
}

#[derive(Debug, Default)]
struct MetricsInner {
    pub total_tokens_processed: AtomicU64,
    pub cached_tokens: AtomicU64,
    pub total_requests: AtomicU64,
    pub active_requests: AtomicU64,
    pub prefill_tokens_per_sec: AtomicU64,
    pub generation_tokens_per_sec: AtomicU64,
}

impl ServerMetrics {
    pub fn new() -> Self { Self::default() }

    pub fn add_tokens(&self, count: u64) {
        self.inner.total_tokens_processed.fetch_add(count, Ordering::Relaxed);
    }

    pub fn add_cached(&self, count: u64) {
        self.inner.cached_tokens.fetch_add(count, Ordering::Relaxed);
    }

    pub fn cache_efficiency(&self) -> f64 {
        let total = self.inner.total_tokens_processed.load(Ordering::Relaxed);
        let cached = self.inner.cached_tokens.load(Ordering::Relaxed);
        if total == 0 { 0.0 } else { cached as f64 / total as f64 * 100.0 }
    }
}
