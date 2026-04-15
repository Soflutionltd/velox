// Hot cache: in-memory tier with write-back to SSD
// Reference: /tmp/omlx-install/omlx/cache/paged_cache.py (hot cache section)

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// In-memory hot cache with LRU eviction
pub struct HotCache {
    entries: DashMap<u64, HotEntry>,
    max_size_bytes: u64,
    current_size: AtomicU64,
}

struct HotEntry {
    data: Vec<u8>,
    last_access: std::time::Instant,
    size_bytes: u64,
}

impl HotCache {
    pub fn new(max_size_bytes: u64) -> Self {
        Self {
            entries: DashMap::new(),
            max_size_bytes,
            current_size: AtomicU64::new(0),
        }
    }

    pub fn get(&self, block_id: u64) -> Option<Vec<u8>> {
        if let Some(mut entry) = self.entries.get_mut(&block_id) {
            entry.last_access = std::time::Instant::now();
            Some(entry.data.clone())
        } else {
            None
        }
    }

    pub fn put(&self, block_id: u64, data: Vec<u8>) {
        let size = data.len() as u64;
        // Evict LRU if needed
        while self.current_size.load(Ordering::Relaxed) + size > self.max_size_bytes {
            if !self.evict_lru() { break; }
        }
        self.current_size.fetch_add(size, Ordering::Relaxed);
        self.entries.insert(block_id, HotEntry {
            data,
            last_access: std::time::Instant::now(),
            size_bytes: size,
        });
    }

    fn evict_lru(&self) -> bool {
        // Find oldest entry
        let oldest = self.entries.iter()
            .min_by_key(|e| e.last_access)
            .map(|e| (*e.key(), e.size_bytes));
        if let Some((key, size)) = oldest {
            self.entries.remove(&key);
            self.current_size.fetch_sub(size, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        // TODO: Track hits/misses
        0.0
    }
}
