//! End-to-end smoke test for MLX-quant 4-bit Qwen3.
//!
//! Loads `~/.velox/models/Qwen3-0.6B-4bit` (downloaded from
//! mlx-community/Qwen3-0.6B-4bit), runs a single batched forward through
//! the paged backend, and verifies we sample a finite token.

#![cfg(all(feature = "candle-metal", target_os = "macos"))]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use velox::paged::pages::{PagedKvCache, PagedKvConfig};
use velox::paged::qwen3::{BatchStep, PagedQwen3, Qwen3Config, SeqSlice};

fn find_quant_model() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let direct: PathBuf = format!("{}/.velox/models/Qwen3-0.6B-4bit", home).into();
    if direct.join("model.safetensors").exists() {
        return Some(direct);
    }
    None
}

#[test]
fn paged_qwen3_4bit_one_step() {
    let Some(model_dir) = find_quant_model() else {
        eprintln!("Qwen3-0.6B-4bit not on disk → skipping");
        return;
    };

    let device = Device::new_metal(0).expect("metal device");
    let dtype = DType::BF16;

    let cfg_path = model_dir.join("config.json");
    let cfg_json = std::fs::read_to_string(&cfg_path).unwrap();
    let cfg: Qwen3Config = serde_json::from_str(&cfg_json).unwrap();
    assert!(cfg.quantization.is_some(), "expected quant config in config.json");

    let st = model_dir.join("model.safetensors");
    assert!(st.exists());

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[st], dtype, &device).expect("VarBuilder failed")
    };
    let model = Arc::new(PagedQwen3::load(&cfg, vb).expect("load PagedQwen3 4-bit"));

    let pages_cfg = PagedKvConfig {
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        page_size: 16,
        num_pages: 32,
        dtype,
    };
    let pages = Arc::new(PagedKvCache::new(pages_cfg, &device).expect("build pages"));

    let prompt: Vec<u32> = vec![9707, 11, 1879];
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
    assert!(logits_v.iter().all(|x| x.is_finite()), "logits NaN/Inf");
    let argmax = logits_v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("4-bit paged qwen3 sampled token: {argmax}");
    assert!(argmax < cfg.vocab_size);
}

/// Larger 4-bit model — demonstrates the **capability unlock**: a 4B
/// model that wouldn't fit in BF16 on small machines still loads and
/// runs at ~4 GB. Ignored by default (downloads ~2 GB).
#[test]
#[ignore = "downloads ~2 GB; run with --ignored"]
fn paged_qwen3_4b_4bit_one_step() {
    let home = std::env::var("HOME").expect("HOME");
    let model_dir: PathBuf = format!("{}/.velox/models/Qwen3-4B-4bit", home).into();
    let st = model_dir.join("model.safetensors");
    if !st.exists() {
        eprintln!("Qwen3-4B-4bit not on disk → skipping. Download with: hf download mlx-community/Qwen3-4B-4bit --local-dir {}", model_dir.display());
        return;
    }

    let device = Device::new_metal(0).expect("metal device");
    let dtype = DType::BF16;

    let cfg: Qwen3Config = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("config.json")).unwrap(),
    )
    .unwrap();
    assert!(cfg.quantization.is_some());

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[st], dtype, &device).expect("VarBuilder failed")
    };
    let model = Arc::new(PagedQwen3::load(&cfg, vb).expect("load PagedQwen3 4B 4-bit"));

    let pages_cfg = PagedKvConfig {
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        page_size: 16,
        num_pages: 32,
        dtype,
    };
    let pages = Arc::new(PagedKvCache::new(pages_cfg, &device).expect("build pages"));

    let prompt: Vec<u32> = vec![9707, 11, 1879];
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
    assert!(logits_v.iter().all(|x| x.is_finite()), "logits NaN/Inf");
    eprintln!(
        "Qwen3-4B-4bit OK ({} layers, {} hidden, vocab {})",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size
    );
}
