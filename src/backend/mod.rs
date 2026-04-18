pub mod traits;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "mlx")]
pub mod mlx;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

use std::sync::Arc;

/// Auto-detect and return the best available backend.
///
/// Selection order:
///   1. Candle (default — pure Rust, Metal-accelerated on macOS aarch64)
///   2. MLX    (opt-in via `--features mlx`, currently broken upstream)
///   3. llama.cpp (opt-in fallback for non-Apple platforms)
pub fn detect_backend() -> Arc<dyn traits::InferenceBackend> {
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
    #[cfg(feature = "llamacpp")]
    {
        return Arc::new(llamacpp::LlamaCppBackend::new());
    }
    #[allow(unreachable_code)]
    panic!(
        "No inference backend available. Compile with one of: \
         --features candle, --features mlx, --features llamacpp"
    );
}
