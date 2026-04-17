pub mod routes;
pub mod sse;
pub mod middleware;

use crate::config::ServerConfig;
use crate::backend;
use crate::engine::pool::EnginePool;
use crate::engine::scheduler::Scheduler;
use crate::metrics::ServerMetrics;
use axum::Router;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<EnginePool>,
    pub scheduler: Arc<Scheduler>,
    pub metrics: ServerMetrics,
    pub config: ServerConfig,
}

pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    // Detect and create the inference backend
    let inference_backend = backend::detect_backend();
    let backend_name = inference_backend.backend_name().to_string();
    tracing::info!("Using inference backend: {backend_name}");

    // Expand ~ in model_dir
    let model_dir = shellexpand::tilde(&config.model_dir).to_string();
    let model_path = PathBuf::from(&model_dir);
    std::fs::create_dir_all(&model_path)?;

    // Create engine pool
    let pool = Arc::new(EnginePool::new(
        inference_backend,
        model_path,
        parse_memory_limit(&config.max_model_memory),
    ));

    // Create scheduler
    let scheduler = Arc::new(Scheduler::new(config.max_concurrent));

    let state = AppState {
        pool,
        scheduler,
        metrics: ServerMetrics::new(),
        config: config.clone(),
    };

    let app = Router::new()
        .merge(routes::openai_routes())
        .merge(routes::anthropic_routes())
        .merge(routes::admin_routes())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("AURA Inference Server listening on http://{addr}");
    tracing::info!("OpenAI API: http://{addr}/v1");
    tracing::info!("Anthropic API: http://{addr}/v1/messages");
    tracing::info!("Admin: http://{addr}/admin/status");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn parse_memory_limit(s: &str) -> u64 {
    if s == "auto" {
        // Use 80% of available RAM
        let total = sysinfo_total_memory();
        (total as f64 * 0.8) as u64
    } else if s.ends_with("GB") {
        s.trim_end_matches("GB").parse::<u64>().unwrap_or(32) * 1024 * 1024 * 1024
    } else {
        32 * 1024 * 1024 * 1024 // Default 32 GB
    }
}

fn sysinfo_total_memory() -> u64 {
    // Simple fallback: assume 96 GB if we can't detect
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                return s.trim().parse().unwrap_or(96 * 1024 * 1024 * 1024);
            }
        }
    }
    96 * 1024 * 1024 * 1024
}
