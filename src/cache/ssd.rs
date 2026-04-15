// SSD cold tier: persisted KV blocks in safetensors format
// Reference: /tmp/omlx-install/omlx/cache/paged_ssd_cache.py (1984 lines)
//
// Key feature: blocks evicted from hot cache are written to SSD.
// On next request with matching prefix, restored from SSD (1-3s)
// instead of recomputed from scratch (30-90s).
// Persists across server restarts.

use std::path::{Path, PathBuf};

pub struct SsdCache {
    cache_dir: PathBuf,
    max_size_bytes: u64,
}

impl SsdCache {
    pub fn new(cache_dir: impl AsRef<Path>, max_size_bytes: u64) -> std::io::Result<Self> {
        let dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { cache_dir: dir, max_size_bytes })
    }

    /// Store a KV block to SSD in safetensors format
    pub fn store(&self, block_id: u64, data: &[u8]) -> std::io::Result<()> {
        let path = self.block_path(block_id);
        // TODO: Use safetensors crate for proper serialization
        std::fs::write(path, data)
    }

    /// Restore a KV block from SSD
    pub fn restore(&self, block_id: u64) -> std::io::Result<Vec<u8>> {
        let path = self.block_path(block_id);
        std::fs::read(path)
    }

    /// Check if a block exists on SSD
    pub fn contains(&self, block_id: u64) -> bool {
        self.block_path(block_id).exists()
    }

    /// Delete a block from SSD
    pub fn delete(&self, block_id: u64) -> std::io::Result<()> {
        let path = self.block_path(block_id);
        if path.exists() { std::fs::remove_file(path)?; }
        Ok(())
    }

    fn block_path(&self, block_id: u64) -> PathBuf {
        self.cache_dir.join(format!("kv_block_{block_id}.safetensors"))
    }

    pub fn total_size(&self) -> u64 {
        // TODO: Sum file sizes in cache_dir
        0
    }
}
