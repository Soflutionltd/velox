//! AURA Inference Server - library interface for Tauri integration
//!
//! Usage from Tauri:
//! ```rust
//! let config = aura_inference::ServerConfig { ... };
//! tokio::spawn(aura_inference::run_server(config));
//! ```

pub mod config;
pub mod error;
pub mod metrics;

pub mod api;
pub mod backend;
pub mod cache;
pub mod engine;
pub mod model;
pub mod server;

pub use config::ServerConfig;

/// Start the inference server (for embedding in Tauri)
pub async fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    server::run(config).await
}
