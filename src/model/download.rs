// HuggingFace model downloader
// Reference: /tmp/omlx-install/omlx/admin/hf_downloader.py (849 lines)

use std::path::PathBuf;

pub struct DownloadRequest {
    pub repo_id: String,       // e.g. "mlx-community/gemma-4-26B-A4B-it-4bit"
    pub target_dir: PathBuf,
    pub revision: Option<String>,
}

pub struct DownloadProgress {
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub current_file: String,
    pub files_done: u32,
    pub files_total: u32,
}

/// Download a model from HuggingFace
pub async fn download_model(
    request: &DownloadRequest,
    _on_progress: impl Fn(DownloadProgress),
) -> anyhow::Result<PathBuf> {
    // TODO: Implement HuggingFace API download
    // Use hf-hub crate or manual HTTPS requests
    tracing::info!("Downloading {} to {:?}", request.repo_id, request.target_dir);
    Ok(request.target_dir.clone())
}
