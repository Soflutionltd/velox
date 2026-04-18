use clap::Parser;
use tracing_subscriber::EnvFilter;

mod config;
mod error;
mod metrics;

mod api;
mod backend;
mod cache;
mod engine;
mod memory;
mod model;
#[cfg(feature = "candle")]
mod paged;
mod server;

#[derive(Parser)]
#[command(name = "velox", about = "Velox - The world's first Rust-native LLM inference server for Apple Silicon")]
enum Cli {
    /// Download a model into the local model directory.
    ///
    /// Accepts either a curated alias (e.g. `qwen3-0.6b`, `llama3-8b`) or a
    /// full HuggingFace repo id (e.g. `mlx-community/Qwen3-4B-4bit`). Run
    /// `velox pull --list` to see the curated catalog.
    Pull {
        /// Alias (e.g. `qwen3-0.6b`) or HF repo id. Omit when using --list.
        model: Option<String>,

        /// Where to install models. Each model lands in <dir>/<repo_id>.
        #[arg(long, default_value = "~/.velox/models")]
        model_dir: String,

        /// Print the curated catalog and exit.
        #[arg(long)]
        list: bool,
    },

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

        /// Optional Unix domain socket path. When set, the server
        /// listens on this path *in addition to* TCP. Use when local
        /// apps want to skip the localhost TCP stack (~30µs/req
        /// saved). Example: --socket /tmp/velox.sock
        #[arg(long)]
        socket: Option<String>,

        /// Optional gRPC port. When set, the server exposes a typed
        /// tonic-based gRPC API in parallel to HTTP. Schema lives in
        /// proto/velox.proto. Example: --grpc-port 50051
        #[arg(long)]
        grpc_port: Option<u16>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("velox=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli {
        Cli::Pull { model, model_dir, list } => {
            if list || model.is_none() {
                model::catalog::print_catalog();
                if model.is_none() && !list {
                    eprintln!("\nNo model specified. Pass an alias or repo_id, or use --list.");
                    std::process::exit(2);
                }
                return Ok(());
            }
            let model = model.unwrap();
            let (repo_id, entry) = model::catalog::resolve(&model);
            let model_dir = expand_tilde(&model_dir);
            // Each model lives in its own subdir named after the repo's
            // last path segment so multiple checkpoints don't collide.
            let leaf = repo_id.rsplit('/').next().unwrap_or(&repo_id);
            let target_dir = std::path::PathBuf::from(&model_dir).join(leaf);
            std::fs::create_dir_all(&target_dir)?;

            tracing::info!("Pulling {} → {:?}", repo_id, target_dir);
            if let Some(e) = entry {
                tracing::info!("  family={} approx_size={:.1}GB", e.family, e.size_gb);
            }

            let req = model::download::DownloadRequest {
                repo_id: repo_id.clone(),
                target_dir: target_dir.clone(),
                revision: None,
            };
            let snapshot = model::download::download_model(&req, |p| {
                eprintln!(
                    "  [{:>3}/{:>3}] {}",
                    p.files_done, p.files_total, p.current_file
                );
            })
            .await?;

            // hf-hub stores files inside `target_dir/models--<user>--<repo>/snapshots/<rev>/`
            // as symlinks to the blob store. The Velox model loader expects
            // a flat layout (`config.json`, `model.safetensors`, ...) directly
            // inside the model dir, so we mirror the snapshot via symlinks.
            // Symlinks (not copies) are intentional: zero extra disk and the
            // blob store is canonical.
            let mut linked = 0usize;
            for entry in std::fs::read_dir(&snapshot)? {
                let entry = entry?;
                let src = entry.path();
                let name = entry.file_name();
                let dst = target_dir.join(&name);
                if dst.exists() {
                    let _ = std::fs::remove_file(&dst);
                }
                if let Err(e) = std::os::unix::fs::symlink(&src, &dst) {
                    eprintln!("  warning: symlink {name:?} → {e}");
                } else {
                    linked += 1;
                }
            }

            println!();
            println!("✓ Model installed:");
            println!("  repo:     {}", repo_id);
            println!("  path:     {}", target_dir.display());
            println!("  files:    {} linked from snapshot", linked);
            println!();
            println!("Start the server with:");
            println!("  velox serve --model-dir {}", model_dir);
        }
        Cli::Serve {
            model_dir,
            port,
            max_model_memory,
            ssd_cache_dir,
            hot_cache_pct,
            max_concurrent,
            socket,
            grpc_port,
        } => {
            tracing::info!("Starting Velox Inference Server on port {port}");
            tracing::info!("Model directory: {model_dir}");
            tracing::info!("SSD cache: {ssd_cache_dir}");
            if let Some(s) = &socket {
                tracing::info!("Unix socket: {s}");
            }
            if let Some(g) = grpc_port {
                tracing::info!("gRPC port: {g}");
            }

            let config = config::ServerConfig {
                model_dir,
                port,
                max_model_memory,
                ssd_cache_dir,
                hot_cache_pct,
                max_concurrent,
                socket_path: socket,
                grpc_port,
            };

            server::run(config).await?;
        }
    }

    Ok(())
}

/// Expand a leading `~` in a path to `$HOME`. We keep this self-contained
/// (rather than pulling in `dirs`) because it's the only place we need it.
fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home)
                .join(rest)
                .to_string_lossy()
                .into_owned();
        }
    }
    p.to_string()
}
