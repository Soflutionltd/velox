//! Velox - The world's first Rust-native LLM inference server for Apple Silicon
//!
//! Usage from Tauri:
//! ```ignore
//! let config = velox::ServerConfig::default();
//! tokio::spawn(velox::run_server(config));
//! ```

pub mod config;
pub mod error;
pub mod metrics;

pub mod api;
pub mod backend;
pub mod cache;
pub mod engine;
pub mod memory;
pub mod model;
pub mod server;

pub use config::ServerConfig;

/// Start the inference server (for embedding in Tauri)
pub async fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    server::run(config).await
}
