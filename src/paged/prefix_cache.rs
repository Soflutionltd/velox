//! Prefix KV cache.
//!
//! Caches **page-aligned** KV pages keyed by a chained hash of the tokens
//! that fill them. When two requests share the same prompt prefix (system
//! prompt + few-shot exemplars + tool descriptions are common cases), the
//! second request reuses the first's KV pages instead of re-running the
//! prefill — a 5-50× speedup at the prefill stage on long shared prefixes.
//!
//! Hashing scheme:
//!   `h_n = hash(h_{n-1} || tokens_in_page_n)`
//!
//! That ensures matching is correct: same prefix tokens → same chained
//! hashes → same cached pages.
//!
//! Lifetime:
//!   * On `lookup` hit, the matched pages have their refcount incremented
//!     once each (the caller becomes a co-owner). The cache itself also
//!     holds 1 ref per cached page.
//!   * `insert` registers new pages and incref's them once.
//!   * LRU eviction removes the LRU entry and decref's its page (which
//!     may free it if no in-flight request is using it).
//!
//! Single-threaded for simplicity (the scheduler is single-threaded
//! anyway). Wrap in `Mutex` if you need shared access.

use super::pages::PagedKvCache;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Initial seed for the chained hash. Just an arbitrary non-zero u64.
const HASH_SEED: u64 = 0x9E3779B97F4A7C15;

/// Chain one more page's worth of tokens onto an existing prefix hash.
fn chain(prev_hash: u64, tokens: &[u32]) -> u64 {
    // FNV-1a flavour. Cheap, no deps, avalanche is good enough since we
    // only need collision resistance across realistic prompt prefixes.
    let mut h = prev_hash;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

/// One cached entry: the chained-hash key and the physical page that
/// holds the KV for the tokens that produced this hash.
#[derive(Debug, Clone, Copy)]
struct CachedPage {
    page_id: u32,
}

pub struct PrefixCache {
    pages: Arc<PagedKvCache>,
    page_size: usize,
    capacity: usize,
    /// chained-hash → cached page
    entries: HashMap<u64, CachedPage>,
    /// LRU order: front = LRU, back = MRU
    lru: VecDeque<u64>,
    hits: u64,
    misses: u64,
}

/// Result of a cache lookup against a prompt.
#[derive(Debug, Clone)]
pub struct PrefixHit {
    /// Physical pages, in order, covering the matched prefix. Each one's
    /// refcount has been incremented; the caller MUST eventually
    /// `pages.free_pages(...)` them.
    pub matched_pages: Vec<u32>,
    /// Number of prompt tokens covered by `matched_pages`. Always equals
    /// `matched_pages.len() * page_size`.
    pub matched_tokens: usize,
    /// The chained hashes for each matched page, useful for the caller
    /// to keep extending the chain when registering NEW pages on insert.
    pub matched_hashes: Vec<u64>,
    /// The chained hash to pass to `insert` as `start_hash` when
    /// registering pages that come AFTER the matched prefix. Equals the
    /// last entry of `matched_hashes` on a hit, or [`HASH_SEED`] on a
    /// total miss. Always non-zero so callers can use `0` to mean
    /// "no lookup performed".
    pub next_chain_hash: u64,
}

impl PrefixCache {
    pub fn new(pages: Arc<PagedKvCache>, capacity: usize) -> Self {
        let page_size = pages.page_size();
        Self {
            pages,
            page_size,
            capacity,
            entries: HashMap::with_capacity(capacity),
            lru: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Walk the prompt one page at a time, return the longest matching
    /// prefix. Increments refcount on every matched page.
    pub fn lookup(&mut self, prompt_tokens: &[u32]) -> PrefixHit {
        let mut h = HASH_SEED;
        let mut matched_pages: Vec<u32> = Vec::new();
        let mut matched_hashes: Vec<u64> = Vec::new();
        let n_full_pages = prompt_tokens.len() / self.page_size;

        for page_idx in 0..n_full_pages {
            let lo = page_idx * self.page_size;
            let hi = lo + self.page_size;
            h = chain(h, &prompt_tokens[lo..hi]);
            match self.entries.get(&h).copied() {
                Some(entry) => {
                    matched_pages.push(entry.page_id);
                    matched_hashes.push(h);
                }
                None => break,
            }
        }

        if matched_pages.is_empty() {
            self.misses += 1;
        } else {
            self.hits += 1;
            // Refresh LRU and incref all matched pages.
            for h in &matched_hashes {
                if let Some(pos) = self.lru.iter().position(|x| x == h) {
                    self.lru.remove(pos);
                }
                self.lru.push_back(*h);
            }
            self.pages.incref_pages(matched_pages.iter().copied());
        }

        let matched_tokens = matched_pages.len() * self.page_size;
        let next_chain_hash = matched_hashes.last().copied().unwrap_or(HASH_SEED);
        PrefixHit {
            matched_pages,
            matched_tokens,
            matched_hashes,
            next_chain_hash,
        }
    }

    /// Register newly-filled pages into the cache. The caller has just
    /// finished prefilling pages `new_pages` covering tokens
    /// `prompt_tokens[start_token..start_token + new_pages.len()*page_size]`
    /// (all aligned). `start_hash` is the chained hash *up to but not
    /// including* the first new page (i.e. `HASH_SEED` if no cached
    /// prefix preceded this insert, or the last hash returned by
    /// `lookup`).
    ///
    /// New pages are incref'd once each (cache becomes a co-owner).
    pub fn insert(
        &mut self,
        prompt_tokens: &[u32],
        start_token: usize,
        start_hash: u64,
        new_pages: &[u32],
    ) {
        debug_assert!(start_token % self.page_size == 0);
        let mut h = start_hash;
        for (i, &page_id) in new_pages.iter().enumerate() {
            let lo = start_token + i * self.page_size;
            let hi = lo + self.page_size;
            if hi > prompt_tokens.len() {
                // Don't cache an underfilled trailing page.
                break;
            }
            h = chain(h, &prompt_tokens[lo..hi]);
            if self.entries.contains_key(&h) {
                // Same chunk already cached → skip (don't double-incref).
                continue;
            }
            self.pages.incref_pages([page_id]);
            self.entries.insert(h, CachedPage { page_id });
            self.lru.push_back(h);
            self.evict_to_capacity();
        }
    }

    fn evict_to_capacity(&mut self) {
        while self.entries.len() > self.capacity {
            let Some(victim_hash) = self.lru.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim_hash) {
                self.pages.free_page(entry.page_id);
            }
        }
    }

    /// Drop everything (used on shutdown / model swap). Decrefs every
    /// cached page.
    pub fn clear(&mut self) {
        let pages: Vec<u32> = self.entries.values().map(|e| e.page_id).collect();
        self.entries.clear();
        self.lru.clear();
        self.pages.free_pages(pages);
    }
}

impl Drop for PrefixCache {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged::pages::{PagedKvCache, PagedKvConfig};
    use candle_core::{DType, Device};

    fn build_cache(num_pages: usize) -> Arc<PagedKvCache> {
        let cfg = PagedKvConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            page_size: 4,
            num_pages,
            dtype: DType::F32,
        };
        Arc::new(PagedKvCache::new(cfg, &Device::Cpu).unwrap())
    }

    #[test]
    fn miss_then_hit_reuses_pages() {
        let pages = build_cache(8);
        let mut cache = PrefixCache::new(pages.clone(), 16);
        let prompt: Vec<u32> = (0..16).collect(); // 4 page_size=4 chunks

        // First request: miss → alloc all 4 pages, register in cache
        let hit1 = cache.lookup(&prompt);
        assert_eq!(hit1.matched_pages.len(), 0);
        assert_eq!(hit1.next_chain_hash, HASH_SEED);
        let allocated = pages.alloc(4).unwrap();
        cache.insert(&prompt, 0, hit1.next_chain_hash, &allocated);

        // Each new page: cache holds 1 ref + alloc holds 1 ref = 2
        for &p in &allocated {
            assert_eq!(pages.page_refcount(p), 2);
        }

        // Second identical request: full hit
        let hit2 = cache.lookup(&prompt);
        assert_eq!(hit2.matched_pages.len(), 4);
        assert_eq!(hit2.matched_tokens, 16);
        assert_eq!(hit2.matched_pages, allocated);
        for &p in &allocated {
            // cache(1) + req1(1) + req2(1) = 3
            assert_eq!(pages.page_refcount(p), 3);
        }

        // First request finishes → drop its refs
        pages.free_pages(allocated.clone());
        for &p in &allocated {
            assert_eq!(pages.page_refcount(p), 2); // cache + req2
        }

        // Second request finishes → drop its refs (cache still holds them)
        pages.free_pages(hit2.matched_pages.clone());
        for &p in &allocated {
            assert_eq!(pages.page_refcount(p), 1); // only cache
        }

        // Cache cleared → all pages back to free
        cache.clear();
        assert_eq!(pages.num_free_pages(), 8);
    }

    #[test]
    fn partial_hit_returns_longest_prefix() {
        let pages = build_cache(8);
        let mut cache = PrefixCache::new(pages.clone(), 16);

        let prompt_a: Vec<u32> = (0..16).collect();
        let prompt_b: Vec<u32> = (0..8).chain([99, 99, 99, 99, 50, 51, 52, 53]).collect();

        // Cache prompt_a fully
        let alloc_a = pages.alloc(4).unwrap();
        let h = cache.lookup(&prompt_a).next_chain_hash;
        cache.insert(&prompt_a, 0, h, &alloc_a);

        // Lookup B: shares first 8 tokens (2 pages) with A
        let hit = cache.lookup(&prompt_b);
        assert_eq!(hit.matched_pages.len(), 2);
        assert_eq!(hit.matched_tokens, 8);
        assert_eq!(&hit.matched_pages[..], &alloc_a[..2]);
    }

    #[test]
    fn lru_eviction_frees_pages() {
        let pages = build_cache(4);
        let mut cache = PrefixCache::new(pages.clone(), 2);

        let prompt_a: Vec<u32> = (0..4).collect();
        let prompt_b: Vec<u32> = (10..14).collect();
        let prompt_c: Vec<u32> = (20..24).collect();

        let pa = pages.alloc(1).unwrap();
        let ha = cache.lookup(&prompt_a).next_chain_hash;
        cache.insert(&prompt_a, 0, ha, &pa);
        let pb = pages.alloc(1).unwrap();
        let hb = cache.lookup(&prompt_b).next_chain_hash;
        cache.insert(&prompt_b, 0, hb, &pb);
        let pc = pages.alloc(1).unwrap();
        let hc = cache.lookup(&prompt_c).next_chain_hash;
        cache.insert(&prompt_c, 0, hc, &pc);

        // Cache cap is 2 → A (oldest) was evicted.
        assert_eq!(cache.len(), 2);

        // After dropping the original alloc refs, A should be fully free,
        // B/C still cached.
        pages.free_pages(pa.clone());
        pages.free_pages(pb.clone());
        pages.free_pages(pc.clone());
        assert_eq!(pages.page_refcount(pa[0]), 0);
        assert_eq!(pages.page_refcount(pb[0]), 1);
        assert_eq!(pages.page_refcount(pc[0]), 1);
    }
}
