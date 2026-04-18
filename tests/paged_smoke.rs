//! End-to-end smoke test: load Qwen3-0.6B, run a single batched forward
//! through the paged backend, verify we sample a sane next token.
//!
//! Skipped (passes silently) when the model isn't on disk.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use velox::paged::pages::{PagedKvCache, PagedKvConfig};
use velox::paged::qwen3::{BatchStep, PagedQwen3, Qwen3Config, SeqSlice};

fn find_model() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let direct: PathBuf = format!("{}/.velox/models/Qwen3-0.6B", home).into();
    if direct.join("model.safetensors").exists() {
        return Some(direct);
    }
    None
}

fn pick_dev() -> Device {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "candle-metal"))]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

#[test]
fn paged_qwen3_one_step() {
    let Some(model_dir) = find_model() else {
        eprintln!("model not on disk → skipping");
        return;
    };

    let device = pick_dev();
    let dtype = if matches!(device, Device::Metal(_)) {
        DType::BF16
    } else {
        DType::F32
    };

    let cfg_path = model_dir.join("config.json");
    let cfg_json = std::fs::read_to_string(&cfg_path).unwrap();
    let cfg: Qwen3Config = serde_json::from_str(&cfg_json).unwrap();

    let st = model_dir.join("model.safetensors");
    assert!(st.exists(), "missing safetensors at {:?}", st);

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[st], dtype, &device).expect("VarBuilder failed")
    };
    let model = Arc::new(PagedQwen3::load(&cfg, vb).expect("load PagedQwen3"));

    // Build a tiny KV pool: 32 pages × 16 tokens = 512 tokens of capacity.
    let pages_cfg = PagedKvConfig {
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        page_size: 16,
        num_pages: 32,
        dtype,
    };
    let pages = Arc::new(PagedKvCache::new(pages_cfg, &device).expect("build pages"));

    // Tokens for "Hello" — we just hard-code a sane id range; we're checking
    // that forward runs and produces a finite token, not that the text is
    // meaningful.
    let prompt: Vec<u32> = vec![9707, 11, 1879]; // "Hello, world" approx
    let n_pages_needed = (prompt.len() + 16).div_ceil(16);
    let block_table = pages.alloc(n_pages_needed).expect("alloc pages");

    let input = Tensor::from_vec(prompt.clone(), (prompt.len(),), &device).unwrap();
    let seqs = vec![SeqSlice {
        new_tokens: prompt.len(),
        kv_offset: 0,
        block_table: &block_table,
    }];
    let step = BatchStep {
        input_ids: &input,
        seqs: &seqs,
    };

    let logits = model.forward(&step, &pages).expect("forward");
    assert_eq!(logits.dims2().unwrap(), (1, cfg.vocab_size));
    let logits_v: Vec<f32> = logits
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        logits_v.iter().all(|x| x.is_finite()),
        "logits contain NaN/Inf"
    );
    let argmax = logits_v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("paged qwen3 sampled token: {argmax}");
    assert!(argmax < cfg.vocab_size);
}
