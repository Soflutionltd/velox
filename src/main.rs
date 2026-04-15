use clap::Parser;
use tracing_subscriber::EnvFilter;

mod config;
mod error;
mod metrics;

mod api;
mod backend;
mod cache;
mod engine;
mod model;
mod server;

#[derive(Parser)]
#[command(name = "aura-inference", about = "High-performance LLM inference server")]
enum Cli {
    /// Start the inference server
    Serve {
        /// Directory containing MLX/GGUF models
        #[arg(long, default_value = "~/.aura/models")]
        model_dir: String,

        /// Port to listen on
        #[arg(long, default_value = "8000")]
        port: u16,

        /// Max memory for loaded models (e.g., "32GB", "80%")
        #[arg(long, default_value = "auto")]
        max_model_memory: String,

        /// SSD cache directory for KV blocks
        #[arg(long, default_value = "~/.aura/cache")]
        ssd_cache_dir: String,

        /// Hot cache size as percentage of available RAM
        #[arg(long, default_value = "20")]
        hot_cache_pct: u8,

        /// Max concurrent requests
        #[arg(long, default_value = "8")]
        max_concurrent: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("aura_inference=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli {
        Cli::Serve {
            model_dir,
            port,
            max_model_memory,
            ssd_cache_dir,
            hot_cache_pct,
            max_concurrent,
        } => {
            tracing::info!("Starting AURA Inference Server on port {port}");
            tracing::info!("Model directory: {model_dir}");
            tracing::info!("SSD cache: {ssd_cache_dir}");

            let config = config::ServerConfig {
                model_dir,
                port,
                max_model_memory,
                ssd_cache_dir,
                hot_cache_pct,
                max_concurrent,
            };

            server::run(config).await?;
        }
    }

    Ok(())
}
