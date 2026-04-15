// Paged KV cache manager (GPU blocks, Copy-on-Write, prefix sharing)
// Reference: /tmp/omlx-install/omlx/cache/paged_cache.py (1732 lines)

use std::collections::HashMap;
use parking_lot::RwLock;

/// A block of KV cache data
#[derive(Debug, Clone)]
pub struct CacheBlock {
    pub block_id: u64,
    pub token_ids: Vec<u32>,
    pub ref_count: u32,
    pub dirty: bool,
}

/// Paged KV cache with Copy-on-Write semantics
pub struct PagedCacheManager {
    blocks: RwLock<HashMap<u64, CacheBlock>>,
    block_size: usize,
    max_blocks: usize,
    next_block_id: std::sync::atomic::AtomicU64,
}

impl PagedCacheManager {
    pub fn new(block_size: usize, max_blocks: usize) -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
            block_size,
            max_blocks,
            next_block_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Allocate a new block
    pub fn allocate(&self) -> Option<u64> {
        let id = self.next_block_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut blocks = self.blocks.write();
        if blocks.len() >= self.max_blocks {
            return None; // OOM, need eviction
        }
        blocks.insert(id, CacheBlock {
            block_id: id,
            token_ids: Vec::with_capacity(self.block_size),
            ref_count: 1,
            dirty: false,
        });
        Some(id)
    }

    /// Copy-on-Write: clone block if shared
    pub fn cow_clone(&self, block_id: u64) -> Option<u64> {
        let blocks = self.blocks.read();
        let block = blocks.get(&block_id)?;
        if block.ref_count <= 1 {
            return Some(block_id); // No need to clone
        }
        drop(blocks);
        let new_id = self.allocate()?;
        // TODO: Copy KV data from old block to new block
        Some(new_id)
    }

    /// Free a block (decrement ref count)
    pub fn free(&self, block_id: u64) {
        let mut blocks = self.blocks.write();
        if let Some(block) = blocks.get_mut(&block_id) {
            block.ref_count = block.ref_count.saturating_sub(1);
            if block.ref_count == 0 {
                blocks.remove(&block_id);
            }
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.read().len()
    }

    pub fn num_free(&self) -> usize {
        self.max_blocks.saturating_sub(self.num_blocks())
    }
}
