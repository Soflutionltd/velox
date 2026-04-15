pub mod routes;
pub mod sse;
pub mod middleware;

use crate::config::ServerConfig;
use axum::Router;
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;

pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    let app = Router::new()
        .merge(routes::openai_routes())
        .merge(routes::anthropic_routes())
        .merge(routes::admin_routes())
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("Listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
