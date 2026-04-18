//! End-to-end smoke for Mistral 7B v0.3 routed through the paged backend.
//!
//! Mistral and Llama share the same architecture aside from the
//! checkpoint layout, so this test is essentially the Llama one with
//! a different model dir and chat template (Mistral uses `[INST]…[/INST]`).

#![cfg(all(feature = "candle-metal", target_os = "macos"))]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use velox::paged::llama::load_paged_llama;
use velox::paged::pages::{PagedKvCache, PagedKvConfig};
use velox::paged::qwen3::{BatchStep, SeqSlice};

fn find_mistral() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    for name in &[
        "Mistral-7B-Instruct-v0.3-4bit",
        "Mistral-7B-Instruct-v0.3",
    ] {
        let p: PathBuf = format!("{}/.velox/models/{}", home, name).into();
        if p.join("model.safetensors").exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn paged_mistral_one_step() {
    let Some(model_dir) = find_mistral() else {
        eprintln!("no Mistral model in ~/.velox/models → skipping");
        return;
    };
    eprintln!("loading {}", model_dir.display());

    let device = Device::new_metal(0).expect("metal device");
    let dtype = DType::BF16;
    let st = model_dir.join("model.safetensors");
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[st], dtype, &device).expect("VarBuilder") };
    let model = Arc::new(load_paged_llama(&model_dir, vb).expect("load Mistral"));
    let cfg = model.config().clone();

    // Mistral 7B has 32 layers, 32 heads, 8 KV heads, head_dim 128.
    // 7B at 4-bit ≈ 4 GB on disk → fits 16 GB unified mem comfortably.
    let pages_cfg = PagedKvConfig {
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        page_size: 16,
        num_pages: 16,
        dtype,
    };
    let pages = Arc::new(PagedKvCache::new(pages_cfg, &device).expect("pages"));

    // Mistral v0.3 BOS = 1, then "Hello, world." token-ish IDs.
    let prompt: Vec<u32> = vec![1, 22557, 28725, 1526, 28723];
    let n_pages = (prompt.len() + 16).div_ceil(16);
    let block_table = pages.alloc(n_pages).expect("alloc");

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
    let v: Vec<f32> = logits
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "Mistral logits NaN/Inf");
    let argmax = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("Mistral paged sampled token id: {argmax}");
    assert!(argmax < cfg.vocab_size);
}
