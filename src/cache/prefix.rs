// Prefix trie for sharing common prefixes between requests
// Reference: /tmp/omlx-install/omlx/cache/prefix_cache.py (2259 lines)
//
// When multiple requests share the same system prompt or context,
// the prefix cache avoids recomputing the KV cache for the shared part.
// This is what gives oMLX the 10x prefill speedup.

use std::collections::HashMap;

pub struct PrefixTrie {
    root: TrieNode,
}

struct TrieNode {
    children: HashMap<u32, TrieNode>,
    block_ids: Vec<u64>,   // KV cache blocks for this prefix
    is_terminal: bool,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self { root: TrieNode::new() }
    }

    /// Find the longest matching prefix for a token sequence
    pub fn find_prefix(&self, tokens: &[u32]) -> (usize, Vec<u64>) {
        let mut node = &self.root;
        let mut depth = 0;
        let mut blocks = Vec::new();
        for &token in tokens {
            match node.children.get(&token) {
                Some(child) => {
                    node = child;
                    depth += 1;
                    if !node.block_ids.is_empty() {
                        blocks = node.block_ids.clone();
                    }
                }
                None => break,
            }
        }
        (depth, blocks)
    }

    /// Insert a prefix with its associated cache blocks
    pub fn insert(&mut self, tokens: &[u32], block_ids: Vec<u64>) {
        let mut node = &mut self.root;
        for &token in tokens {
            node = node.children.entry(token).or_insert_with(TrieNode::new);
        }
        node.block_ids = block_ids;
        node.is_terminal = true;
    }
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            block_ids: Vec::new(),
            is_terminal: false,
        }
    }
}
