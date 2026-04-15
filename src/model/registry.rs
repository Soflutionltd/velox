// Model registry: track loaded models and their settings
// Reference: /tmp/omlx-install/omlx/model_registry.py

use dashmap::DashMap;
use crate::model::discovery::DiscoveredModel;

pub struct ModelRegistry {
    models: DashMap<String, RegisteredModel>,
}

pub struct RegisteredModel {
    pub info: DiscoveredModel,
    pub loaded: bool,
    pub ttl_seconds: Option<u64>,
    pub pinned: bool,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self { models: DashMap::new() }
    }

    pub fn register(&self, model: DiscoveredModel) {
        let name = model.name.clone();
        self.models.insert(name, RegisteredModel {
            info: model, loaded: false, ttl_seconds: None, pinned: false,
        });
    }

    pub fn list(&self) -> Vec<String> {
        self.models.iter().map(|e| e.key().clone()).collect()
    }

    pub fn get(&self, name: &str) -> Option<DiscoveredModel> {
        self.models.get(name).map(|e| e.info.clone())
    }
}
