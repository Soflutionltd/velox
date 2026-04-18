//! Smoke test for the Phi-3 paged backend.
//!
//! Skipped when no Phi-3 checkpoint is present locally. To exercise:
//!
//!     velox pull phi3-mini      # not yet wired into catalog; pull manually:
//!     hf download mlx-community/Phi-3-mini-4k-instruct-4bit \
//!         --local-dir ~/.velox/models/Phi-3-mini-4k-instruct-4bit
//!     cargo test --release --features candle-metal --test paged_phi3_smoke

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use velox::paged::pages::{PagedKvCache, PagedKvConfig};
use velox::paged::phi3::load_paged_phi3;
use velox::paged::qwen3::{BatchStep, SeqSlice};

fn find_phi3() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    for name in &[
        "Phi-3-mini-4k-instruct-4bit",
        "Phi-3-mini-4k-instruct",
    ] {
        for root in &[".velox/models", ".aura/models"] {
            let p: PathBuf = format!("{}/{}/{}", home, root, name).into();
            if p.join("config.json").exists() && p.join("model.safetensors").exists() {
                return Some(p);
            }
        }
    }
    None
}

#[test]
fn paged_phi3_one_step() {
    let model_dir = match find_phi3() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Phi-3 checkpoint found in ~/.velox/models/");
            return;
        }
    };
    let device = Device::new_metal(0).expect("metal device");
    let dtype = DType::BF16;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_dir.join("model.safetensors")],
            dtype,
            &device,
        )
        .expect("VarBuilder::from_mmaped_safetensors")
    };
    let model = load_paged_phi3(&model_dir, vb).expect("load_paged_phi3");

    let cfg = model.config().clone();
    let pages_cfg = PagedKvConfig {
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        page_size: 16,
        num_pages: 32,
        dtype,
    };
    let pages = PagedKvCache::new(pages_cfg, &device).expect("kv cache");

    // Single 4-token prompt, kv_offset=0, fresh request id 0.
    let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], 4, &device).expect("ids");
    let block_table: Vec<u32> = vec![0]; // request id 0 → page 0
    let seqs = vec![SeqSlice {
        new_tokens: 4,
        kv_offset: 0,
        block_table: &block_table,
    }];
    let step = BatchStep {
        input_ids: &input_ids,
        seqs: &seqs,
    };

    let logits = model.forward(&step, &pages).expect("forward");
    let dims = logits.dims().to_vec();
    assert_eq!(dims[0], 1, "expected one logit row per request");
    assert_eq!(dims[1], cfg.vocab_size, "vocab dim mismatch");

    // Logits must be finite — the Phi-3 split path is the most error-prone
    // step (any wrong slice = NaN/Inf cascade through softmax).
    let v = logits
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let n_finite = v.iter().filter(|x| x.is_finite()).count();
    assert_eq!(n_finite, v.len(), "non-finite logits — phi3 split is wrong");
}
