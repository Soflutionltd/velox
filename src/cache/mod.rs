pub mod paged;
pub mod hot;
pub mod ssd;
pub mod prefix;

// Two-tier KV cache: hot (RAM) + cold (SSD)
// Reference: /tmp/omlx-install/omlx/cache/ (6000+ lines)
// TODO: Implement PagedCacheManager, HotCache, SSDCache, PrefixTrie
