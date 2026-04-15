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

/// Scan a directory for available models
pub fn discover_models(model_dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    if !model_dir.exists() { return models; }

    // Scan top-level and two-level directories (org/model-name)
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(model) = detect_model(&path) {
                    models.push(model);
                }
                // Check subdirectories (org/model-name pattern)
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub in sub_entries.flatten() {
                        if sub.path().is_dir() {
                            if let Some(model) = detect_model(&sub.path()) {
                                models.push(model);
                            }
                        }
                    }
                }
            }
        }
    }
    tracing::info!("Discovered {} models in {:?}", models.len(), model_dir);
    models
}

fn detect_model(path: &Path) -> Option<DiscoveredModel> {
    let name = path.file_name()?.to_string_lossy().to_string();
    // Check for MLX model (has config.json + *.safetensors)
    let has_config = path.join("config.json").exists();
    let has_safetensors = std::fs::read_dir(path).ok()?
        .flatten()
        .any(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"));

    if has_config && has_safetensors {
        return Some(DiscoveredModel {
            name,
            path: path.to_path_buf(),
            model_type: ModelType::Llm, // TODO: detect VLM, embedding, reranker
            format: ModelFormat::Mlx,
            size_bytes: dir_size(path),
        });
    }

    // Check for GGUF model (single .gguf file)
    let gguf = std::fs::read_dir(path).ok()?
        .flatten()
        .find(|e| e.path().extension().map_or(false, |ext| ext == "gguf"));

    if let Some(gguf_file) = gguf {
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

fn dir_size(path: &Path) -> u64 {
    std::fs::read_dir(path).ok().map_or(0, |entries| {
        entries.flatten().map(|e| e.metadata().ok().map_or(0, |m| m.len())).sum()
    })
}
