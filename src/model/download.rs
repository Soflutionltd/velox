// HuggingFace model downloader using hf-hub crate
// Reference: /tmp/omlx-install/omlx/admin/hf_downloader.py (849 lines)

use hf_hub::api::tokio::ApiBuilder;
use std::path::PathBuf;

pub struct DownloadRequest {
    pub repo_id: String,
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

/// Files we always need for an MLX text-generation model
const MLX_REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// Optional files (downloaded if present in the repo).
///
/// Note: `model.safetensors.index.json` is intentionally NOT in this list — we
/// probe for it explicitly in the shard-discovery step below. Listing it here
/// would cause a second download attempt that wrongly classifies it as
/// required (because the filename contains "safetensors").
const MLX_OPTIONAL_FILES: &[&str] = &[
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// Download an MLX model from HuggingFace into the target directory.
///
/// Discovers safetensors shards from `model.safetensors.index.json` (sharded models)
/// or falls back to the single-file `model.safetensors`.
pub async fn download_model(
    request: &DownloadRequest,
    on_progress: impl Fn(DownloadProgress),
) -> anyhow::Result<PathBuf> {
    tracing::info!("Downloading {} → {:?}", request.repo_id, request.target_dir);
    std::fs::create_dir_all(&request.target_dir)?;

    let api = ApiBuilder::new()
        .with_cache_dir(request.target_dir.clone())
        .build()?;

    let repo = if let Some(rev) = &request.revision {
        api.repo(hf_hub::Repo::with_revision(
            request.repo_id.clone(),
            hf_hub::RepoType::Model,
            rev.clone(),
        ))
    } else {
        api.model(request.repo_id.clone())
    };

    // Step 1: download required files
    let mut downloaded_paths: Vec<PathBuf> = Vec::new();
    let mut required: Vec<String> = MLX_REQUIRED_FILES.iter().map(|s| s.to_string()).collect();

    // Step 2: discover safetensors shards
    let mut shard_files: Vec<String> = Vec::new();
    let index_filename = "model.safetensors.index.json";
    match repo.get(index_filename).await {
        Ok(index_path) => {
            // Sharded model: parse index for the shard list
            let index_text = std::fs::read_to_string(&index_path)?;
            let index_json: serde_json::Value = serde_json::from_str(&index_text)?;
            if let Some(map) = index_json.get("weight_map").and_then(|v| v.as_object()) {
                let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                for (_, v) in map {
                    if let Some(s) = v.as_str() {
                        shards.insert(s.to_string());
                    }
                }
                shard_files.extend(shards.into_iter());
            }
            downloaded_paths.push(index_path);
        }
        Err(_) => {
            // Single-file model
            shard_files.push("model.safetensors".to_string());
        }
    }

    let mut all_files: Vec<String> = Vec::new();
    all_files.append(&mut required);
    for f in MLX_OPTIONAL_FILES {
        all_files.push((*f).to_string());
    }
    all_files.extend(shard_files);

    let total = all_files.len() as u32;
    let mut done: u32 = 0;
    for filename in all_files {
        if MLX_REQUIRED_FILES.contains(&filename.as_str()) || filename.contains("safetensors") {
            // Required: must succeed
            let path = repo.get(&filename).await
                .map_err(|e| anyhow::anyhow!("Failed to download {}: {e}", filename))?;
            downloaded_paths.push(path);
        } else {
            // Optional: ignore failures
            match repo.get(&filename).await {
                Ok(p) => downloaded_paths.push(p),
                Err(_) => {}
            }
        }
        done += 1;
        on_progress(DownloadProgress {
            total_bytes: 0,
            downloaded_bytes: 0,
            current_file: filename,
            files_done: done,
            files_total: total,
        });
    }

    // hf-hub stores files inside its own snapshot layout. Find the snapshot dir.
    let snapshot_dir = downloaded_paths
        .first()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .ok_or_else(|| anyhow::anyhow!("No files downloaded for {}", request.repo_id))?;

    tracing::info!(
        "Downloaded {} files for {} → {:?}",
        downloaded_paths.len(),
        request.repo_id,
        snapshot_dir
    );

    Ok(snapshot_dir)
}
