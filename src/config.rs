use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub model_dir: String,
    pub port: u16,
    pub max_model_memory: String,
    pub ssd_cache_dir: String,
    pub hot_cache_pct: u8,
    pub max_concurrent: usize,
    /// Optional Unix domain socket path. When set, Velox binds to this
    /// path instead of (or in addition to, depending on serve mode)
    /// the TCP port. UDS skips the kernel TCP stack — measured ~30µs
    /// per round-trip saved versus localhost TCP, useful for
    /// latency-sensitive local apps integrating Velox.
    #[serde(default)]
    pub socket_path: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_dir: "~/.aura/models".into(),
            port: 8000,
            max_model_memory: "auto".into(),
            ssd_cache_dir: "~/.aura/cache".into(),
            hot_cache_pct: 20,
            max_concurrent: 8,
            socket_path: None,
        }
    }
}
