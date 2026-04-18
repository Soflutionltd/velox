pub mod traits;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "mlx")]
pub mod mlx;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "mistralrs")]
pub mod mistralrs_backend;

use std::sync::Arc;

/// Auto-detect and return the best available backend.
///
/// Selection order (when no explicit override is set):
///   1. Candle (default — pure Rust, Metal-accelerated on macOS aarch64)
///   2. MLX    (opt-in via `--features mlx`, currently broken upstream)
///   3. mistral.rs (opt-in via `--features mistralrs`, bypasses paged engine)
///   4. llama.cpp (opt-in fallback for non-Apple platforms)
///
/// Override with the `VELOX_BACKEND` environment variable: one of
/// `candle`, `mlx`, `mistralrs`, `llamacpp`.
pub fn detect_backend() -> Arc<dyn traits::InferenceBackend> {
    let override_name = std::env::var("VELOX_BACKEND").ok();
    if let Some(name) = override_name.as_deref() {
        match name {
            #[cfg(feature = "candle")]
            "candle" => return Arc::new(candle::CandleBackend::new()),
            #[cfg(feature = "mlx")]
            "mlx" => return Arc::new(mlx::MlxBackend::new()),
            #[cfg(feature = "mistralrs")]
            "mistralrs" => return Arc::new(mistralrs_backend::MistralRsBackend::new()),
            #[cfg(feature = "llamacpp")]
            "llamacpp" => return Arc::new(llamacpp::LlamaCppBackend::new()),
            other => tracing::warn!(
                "VELOX_BACKEND={} not enabled at build time; falling back to default",
                other
            ),
        }
    }
    #[cfg(feature = "candle")]
    {
        if candle::CandleBackend::available() {
            return Arc::new(candle::CandleBackend::new());
        }
    }
    #[cfg(feature = "mlx")]
    {
        if mlx::MlxBackend::available() {
            return Arc::new(mlx::MlxBackend::new());
        }
    }
    #[cfg(feature = "mistralrs")]
    {
        return Arc::new(mistralrs_backend::MistralRsBackend::new());
    }
    #[cfg(feature = "llamacpp")]
    {
        return Arc::new(llamacpp::LlamaCppBackend::new());
    }
    #[allow(unreachable_code)]
    panic!(
        "No inference backend available. Compile with one of: \
         --features candle, --features mlx, --features mistralrs, --features llamacpp"
    );
}
