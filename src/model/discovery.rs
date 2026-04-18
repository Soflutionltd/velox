// Model auto-discovery from directory
// Reference: /tmp/omlx-install/omlx/model_discovery.py (813 lines)
// Scans model directory for MLX and GGUF models

use std::path::{Path, PathBuf};
use crate::backend::traits::ModelType;

#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    pub name: String,
    pub path: PathBuf,
    pub model_type: ModelType,
    pub format: ModelFormat,
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
pub enum ModelFormat {
    Mlx,      // MLX safetensors format
    Gguf,     // llama.cpp GGUF format
    Unknown,
}

/// Scan a directory for available models.
/// Recurses up to 4 levels deep to support both flat layouts and the
/// HuggingFace cache layout (`models--<org>--<repo>/snapshots/<commit>/`).
pub fn discover_models(model_dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    if !model_dir.exists() {
        return models;
    }
    walk(model_dir, model_dir, 0, 4, &mut models);
    tracing::info!("Discovered {} models in {:?}", models.len(), model_dir);
    models
}

fn walk(
    root: &Path,
    path: &Path,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<DiscoveredModel>,
) {
    if let Some(model) = detect_model(root, path) {
        out.push(model);
        return;
    }
    if depth >= max_depth {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(root, &p, depth + 1, max_depth, out);
            }
        }
    }
}

fn detect_model(root: &Path, path: &Path) -> Option<DiscoveredModel> {
    // Resolve symlinks (HF cache uses snapshots/<commit>/ -> blobs/<sha>)
    let real_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());

    // Check for MLX model: config.json + at least one *.safetensors
    let has_config = real_path.join("config.json").exists();
    let has_safetensors = std::fs::read_dir(&real_path).ok()?.flatten().any(|e| {
        let p = std::fs::canonicalize(e.path()).unwrap_or(e.path());
        p.extension().map_or(false, |ext| ext == "safetensors")
            || e.file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
    });

    if has_config && has_safetensors {
        let name = friendly_name(root, path);
        return Some(DiscoveredModel {
            name,
            path: path.to_path_buf(),
            model_type: ModelType::Llm,
            format: ModelFormat::Mlx,
            size_bytes: dir_size(&real_path),
        });
    }

    // GGUF model (single .gguf file inside a directory)
    let gguf = std::fs::read_dir(&real_path).ok()?.flatten().find(|e| {
        e.path()
            .extension()
            .map_or(false, |ext| ext == "gguf")
    });
    if let Some(gguf_file) = gguf {
        let name = friendly_name(root, path);
        return Some(DiscoveredModel {
            name,
            path: gguf_file.path(),
            model_type: ModelType::Llm,
            format: ModelFormat::Gguf,
            size_bytes: gguf_file.metadata().ok()?.len(),
        });
    }
    None
}

/// Build a human-friendly model name from the path.
/// For HF cache layout `models--mlx-community--Qwen3-0.6B-4bit/snapshots/<sha>/`
/// returns `Qwen3-0.6B-4bit`.
fn friendly_name(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy();
        if let Some(rest) = s.strip_prefix("models--") {
            // models--<org>--<repo>
            if let Some((_, repo)) = rest.rsplit_once("--") {
                return repo.to_string();
            }
            return rest.to_string();
        }
    }
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn dir_size(path: &Path) -> u64 {
    std::fs::read_dir(path).ok().map_or(0, |entries| {
        entries.flatten().map(|e| e.metadata().ok().map_or(0, |m| m.len())).sum()
    })
}
