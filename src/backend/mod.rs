pub mod traits;

#[cfg(feature = "mlx")]
pub mod mlx;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

use std::sync::Arc;

/// Auto-detect and return the best available backend
pub fn detect_backend() -> Arc<dyn traits::InferenceBackend> {
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
    panic!("No inference backend available. Compile with --features mlx or --features llamacpp");
}
