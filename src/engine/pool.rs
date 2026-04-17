// Engine pool: multi-model serving with LRU eviction, TTL, memory limits
// Reference: /tmp/omlx-install/omlx/engine_pool.py (921 lines)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use crate::backend::traits::*;
use crate::model::discovery::{DiscoveredModel, ModelFormat};

struct LoadedModel {
    handle: ModelHandle,
    last_used: std::time::Instant,
    pinned: bool,
    size_bytes: u64,
}

pub struct EnginePool {
    models: RwLock<HashMap<String, LoadedModel>>,
    discovered: RwLock<Vec<DiscoveredModel>>,
    backend: Arc<dyn InferenceBackend>,
    max_memory_bytes: u64,
    model_dir: PathBuf,
}

impl EnginePool {
    pub fn backend_ref(&self) -> &dyn InferenceBackend {
        &*self.backend
    }

    pub fn backend_arc(&self) -> std::sync::Arc<dyn InferenceBackend> {
        self.backend.clone()
    }

    pub fn new(backend: Arc<dyn InferenceBackend>, model_dir: PathBuf, max_memory_bytes: u64) -> Self {
        let discovered = crate::model::discovery::discover_models(&model_dir);
        tracing::info!("Engine pool: discovered {} models", discovered.len());
        Self {
            models: RwLock::new(HashMap::new()),
            discovered: RwLock::new(discovered),
            backend,
            max_memory_bytes,
            model_dir,
        }
    }

    /// Get or load a model by name
    pub async fn get_model(&self, name: &str) -> anyhow::Result<ModelHandle> {
        // Check if already loaded
        {
            let mut models = self.models.write();
            if let Some(loaded) = models.get_mut(name) {
                loaded.last_used = std::time::Instant::now();
                return Ok(ModelHandle {
                    id: loaded.handle.id.clone(),
                    path: loaded.handle.path.clone(),
                    model_type: loaded.handle.model_type.clone(),
                    params_total: loaded.handle.params_total,
                    params_active: loaded.handle.params_active,
                });
            }
        }
        // Find in discovered models
        let model_info = {
            let disc = self.discovered.read();
            disc.iter().find(|m| m.name == name || m.name.contains(name)).cloned()
        };
        let info = model_info.ok_or_else(|| anyhow::anyhow!("Model '{}' not found", name))?;

        // Evict LRU if needed
        self.evict_if_needed(info.size_bytes).await;

        // Load the model
        tracing::info!("Loading model: {} ({:?})", info.name, info.format);
        let handle = self.backend.load_model(&info.path).await?;
        let loaded = LoadedModel {
            handle: ModelHandle {
                id: handle.id.clone(),
                path: handle.path.clone(),
                model_type: handle.model_type.clone(),
                params_total: handle.params_total,
                params_active: handle.params_active,
            },
            last_used: std::time::Instant::now(),
            pinned: false,
            size_bytes: info.size_bytes,
        };
        self.models.write().insert(name.to_string(), loaded);
        Ok(handle)
    }

    async fn evict_if_needed(&self, needed_bytes: u64) {
        let mut models = self.models.write();
        let current: u64 = models.values().map(|m| m.size_bytes).sum();
        if current + needed_bytes <= self.max_memory_bytes { return; }

        // Sort by last_used (oldest first), skip pinned
        let mut evict_candidates: Vec<(String, std::time::Instant)> = models.iter()
            .filter(|(_, m)| !m.pinned)
            .map(|(k, m)| (k.clone(), m.last_used))
            .collect();
        evict_candidates.sort_by_key(|(_, t)| *t);

        for (name, _) in evict_candidates {
            if current + needed_bytes <= self.max_memory_bytes { break; }
            if let Some(removed) = models.remove(&name) {
                tracing::info!("Evicting model: {} (LRU)", name);
                let _ = self.backend.unload_model(&removed.handle).await;
            }
        }
    }

    pub fn list_models(&self) -> Vec<String> {
        let disc = self.discovered.read();
        disc.iter().map(|m| m.name.clone()).collect()
    }

    pub fn loaded_models(&self) -> Vec<String> {
        self.models.read().keys().cloned().collect()
    }
}
