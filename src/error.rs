use thiserror::Error;

#[derive(Error, Debug)]
pub enum AuraError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    #[error("Request timeout")]
    Timeout,
    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}
