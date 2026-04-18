//! Paged KV cache: a fixed-size pool of pages shared across all in-flight
//! requests for a given model.
//!
//! Memory layout (per layer, separately for K and V):
//!
//!   `[num_pages, num_kv_heads, page_size, head_dim]`
//!
//! At inference time, each request holds a `block_table: Vec<u32>` mapping
//! its logical position `i` to physical page `block_table[i / page_size]`,
//! offset `i % page_size`. This is exactly the analogue of OS-level virtual
//! memory translation, applied to KV cache.
//!
//! We allocate ALL the K/V storage up-front in one big tensor pool per
//! layer. Allocation/free of pages is just integer index manipulation
//! (a free-list stack) — zero tensor allocation in the hot path.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use parking_lot::Mutex;
use std::sync::Arc;

/// Static configuration of the paged KV pool. Once a [`PagedKvCache`] is
/// built it cannot grow.
#[derive(Debug, Clone)]
pub struct PagedKvConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub num_pages: usize,
    pub dtype: DType,
}

impl PagedKvConfig {
    /// Total bytes of KV memory this config will consume across all layers,
    /// for both K and V.
    pub fn total_bytes(&self) -> usize {
        let elem_bytes = match self.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::I64 | DType::U32 => 8, // conservative
            _ => 4,
        };
        self.num_layers
            * 2 // K and V
            * self.num_pages
            * self.num_kv_heads
            * self.page_size
            * self.head_dim
            * elem_bytes
    }
}

/// One layer's K and V storage. We keep ONE Tensor per physical page
/// (shape `[num_kv_heads, page_size, head_dim]`) instead of a single big
/// `[num_pages, ...]` tensor: this turns each scatter from O(num_pages) to
/// O(page_size). All pages are pre-allocated as `Tensor::zeros` once, and
/// scatter replaces just the slot inside one page tensor.
struct LayerPool {
    k_pages: Mutex<Vec<Tensor>>,
    v_pages: Mutex<Vec<Tensor>>,
}

/// The full paged KV cache: one [`LayerPool`] per transformer layer, plus a
/// shared free-list of physical pages.
pub struct PagedKvCache {
    cfg: PagedKvConfig,
    layers: Vec<LayerPool>,
    /// Free-list of physical page IDs, used as a stack (Vec::pop = O(1)).
    free: Mutex<Vec<u32>>,
    device: Device,
}

impl PagedKvCache {
    pub fn new(cfg: PagedKvConfig, device: &Device) -> Result<Self> {
        if cfg.num_layers == 0 || cfg.num_pages == 0 {
            return Err(anyhow!("PagedKvCache must have ≥1 layer and ≥1 page"));
        }
        if cfg.num_pages > u32::MAX as usize {
            return Err(anyhow!("num_pages exceeds u32 range"));
        }

        let page_shape = (cfg.num_kv_heads, cfg.page_size, cfg.head_dim);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let mut k_pages = Vec::with_capacity(cfg.num_pages);
            let mut v_pages = Vec::with_capacity(cfg.num_pages);
            for _ in 0..cfg.num_pages {
                k_pages.push(
                    Tensor::zeros(page_shape, cfg.dtype, device)
                        .map_err(|e| anyhow!("alloc K page layer {layer_idx}: {e}"))?,
                );
                v_pages.push(
                    Tensor::zeros(page_shape, cfg.dtype, device)
                        .map_err(|e| anyhow!("alloc V page layer {layer_idx}: {e}"))?,
                );
            }
            layers.push(LayerPool {
                k_pages: Mutex::new(k_pages),
                v_pages: Mutex::new(v_pages),
            });
        }

        // Free list: all pages start free, pushed in reverse so page 0 is
        // popped first (nicer for debugging / determinism).
        let mut free: Vec<u32> = (0..cfg.num_pages as u32).rev().collect();
        free.shrink_to_fit();

        tracing::info!(
            "PagedKvCache: {} layers × {} pages × {} kv_heads × {} tokens × {} dim ({} dtype, {:.1} MB)",
            cfg.num_layers,
            cfg.num_pages,
            cfg.num_kv_heads,
            cfg.page_size,
            cfg.head_dim,
            format!("{:?}", cfg.dtype),
            cfg.total_bytes() as f64 / 1024.0 / 1024.0,
        );

        Ok(Self {
            cfg,
            layers,
            free: Mutex::new(free),
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &PagedKvConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn page_size(&self) -> usize {
        self.cfg.page_size
    }

    pub fn num_layers(&self) -> usize {
        self.cfg.num_layers
    }

    pub fn num_kv_heads(&self) -> usize {
        self.cfg.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.cfg.head_dim
    }

    pub fn num_free_pages(&self) -> usize {
        self.free.lock().len()
    }

    pub fn num_total_pages(&self) -> usize {
        self.cfg.num_pages
    }

    /// Allocate `n` physical pages. Returns `None` if not enough are free.
    /// All-or-nothing: if it can't satisfy the full request, no pages are
    /// taken (so the caller can decide to evict / wait).
    pub fn alloc(&self, n: usize) -> Option<Vec<u32>> {
        let mut free = self.free.lock();
        if free.len() < n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(free.pop().expect("checked length above"));
        }
        Some(out)
    }

    /// Return a single physical page to the free list. Idempotent on
    /// out-of-range IDs (logged as a warning).
    pub fn free_page(&self, page_id: u32) {
        if (page_id as usize) >= self.cfg.num_pages {
            tracing::warn!("free_page: page_id {} out of range", page_id);
            return;
        }
        self.free.lock().push(page_id);
    }

    pub fn free_pages<I: IntoIterator<Item = u32>>(&self, pages: I) {
        let mut free = self.free.lock();
        for p in pages {
            if (p as usize) >= self.cfg.num_pages {
                tracing::warn!("free_pages: page_id {} out of range", p);
                continue;
            }
            free.push(p);
        }
    }

    /// Lock per-page K storage for a layer. The returned guard is a
    /// `Vec<Tensor>` of length `num_pages`. To scatter, replace one entry.
    /// To gather, clone a few entries and `cat` them.
    pub fn lock_layer_k_pages(&self, layer: usize) -> parking_lot::MutexGuard<'_, Vec<Tensor>> {
        self.layers[layer].k_pages.lock()
    }

    pub fn lock_layer_v_pages(&self, layer: usize) -> parking_lot::MutexGuard<'_, Vec<Tensor>> {
        self.layers[layer].v_pages.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(num_pages: usize) -> PagedKvConfig {
        PagedKvConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 8,
            page_size: 4,
            num_pages,
            dtype: DType::F32,
        }
    }

    #[test]
    fn alloc_and_free_round_trip() {
        let dev = Device::Cpu;
        let pool = PagedKvCache::new(cfg(8), &dev).unwrap();
        assert_eq!(pool.num_free_pages(), 8);

        let pages = pool.alloc(3).unwrap();
        assert_eq!(pages.len(), 3);
        assert_eq!(pool.num_free_pages(), 5);

        pool.free_pages(pages);
        assert_eq!(pool.num_free_pages(), 8);
    }

    #[test]
    fn alloc_fails_when_not_enough_free() {
        let dev = Device::Cpu;
        let pool = PagedKvCache::new(cfg(2), &dev).unwrap();
        let _grabbed = pool.alloc(2).unwrap();
        assert!(pool.alloc(1).is_none());
        assert_eq!(pool.num_free_pages(), 0);
    }

    #[test]
    fn config_total_bytes_is_sane() {
        let c = cfg(16);
        // 2 layers * 2 (K+V) * 16 pages * 4 heads * 4 page_size * 8 head_dim * 4 bytes (f32)
        // = 2 * 2 * 16 * 4 * 4 * 8 * 4 = 32_768
        assert_eq!(c.total_bytes(), 32_768);
    }
}

/// Convenience for tests / construction sites that want an Arc.
impl PagedKvCache {
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}
