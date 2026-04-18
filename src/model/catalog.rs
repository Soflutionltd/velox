//! Curated catalog of known-good models for `velox pull`.
//!
//! These are MLX-Int4 (or BF16 where Int4 isn't published) checkpoints that
//! we've actually loaded and run end-to-end through the paged backend.
//! Adding a model here is a deliberate "this works today" claim.
//!
//! Aliases are short, memorable names users type on the CLI; the resolved
//! `repo_id` is the HuggingFace path we hand to `download_model`.

#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub alias: &'static str,
    pub repo_id: &'static str,
    pub family: &'static str,
    pub size_gb: f32,
    pub note: &'static str,
}

pub const CATALOG: &[CatalogEntry] = &[
    // Qwen3 (native Velox support, paged backend, fastest path).
    CatalogEntry {
        alias: "qwen3-0.6b",
        repo_id: "mlx-community/Qwen3-0.6B-4bit",
        family: "qwen3",
        size_gb: 0.4,
        note: "Smallest, fastest, best for testing.",
    },
    CatalogEntry {
        alias: "qwen3-1.7b",
        repo_id: "mlx-community/Qwen3-1.7B-4bit",
        family: "qwen3",
        size_gb: 1.0,
        note: "Smart enough for chat, runs on 8GB Macs.",
    },
    CatalogEntry {
        alias: "qwen3-4b",
        repo_id: "mlx-community/Qwen3-4B-4bit",
        family: "qwen3",
        size_gb: 2.4,
        note: "Sweet spot for 16GB Macs.",
    },
    CatalogEntry {
        alias: "qwen3-8b",
        repo_id: "mlx-community/Qwen3-8B-4bit",
        family: "qwen3",
        size_gb: 4.5,
        note: "Strong general-purpose, needs 16GB+.",
    },
    // Llama 3.x (paged backend via load_paged_llama).
    CatalogEntry {
        alias: "llama3-8b",
        repo_id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        family: "llama",
        size_gb: 4.6,
        note: "Meta Llama 3 8B Instruct, 4-bit.",
    },
    CatalogEntry {
        alias: "llama3.1-8b",
        repo_id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        family: "llama",
        size_gb: 4.6,
        note: "Llama 3.1 with longer context (128k).",
    },
    CatalogEntry {
        alias: "llama3.2-3b",
        repo_id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        family: "llama",
        size_gb: 1.8,
        note: "Smaller Llama 3.2, fits anywhere.",
    },
    // Mistral 7B (paged backend via load_paged_llama).
    CatalogEntry {
        alias: "mistral-7b",
        repo_id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        family: "mistral",
        size_gb: 4.1,
        note: "Mistral 7B v0.3 (no sliding window).",
    },
    CatalogEntry {
        alias: "mistral-7b-v0.2",
        repo_id: "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        family: "mistral",
        size_gb: 4.1,
        note: "Mistral 7B v0.2 (sliding window 4096).",
    },
];

pub fn resolve(alias_or_repo: &str) -> (String, Option<&'static CatalogEntry>) {
    if let Some(entry) = CATALOG.iter().find(|e| e.alias.eq_ignore_ascii_case(alias_or_repo)) {
        return (entry.repo_id.to_string(), Some(entry));
    }
    (alias_or_repo.to_string(), None)
}

pub fn print_catalog() {
    println!("Available models (use either the alias or the full repo_id):\n");
    println!("  {:<14} {:<8} {:>8}  {}", "ALIAS", "FAMILY", "SIZE", "DESCRIPTION");
    println!("  {:<14} {:<8} {:>8}  {}", "-----", "------", "----", "-----------");
    for e in CATALOG {
        println!(
            "  {:<14} {:<8} {:>6.1}GB  {}",
            e.alias, e.family, e.size_gb, e.note
        );
    }
    println!("\nExamples:");
    println!("  velox pull qwen3-0.6b");
    println!("  velox pull mlx-community/Qwen3-7B-4bit");
}
