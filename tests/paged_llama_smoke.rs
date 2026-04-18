//! End-to-end smoke test for Llama 3.x routed through the paged backend.
//!
//! Loads `~/.velox/models/Llama-3.2-1B-Instruct-4bit` (downloaded
//! from `mlx-community/Llama-3.2-1B-Instruct-4bit`), runs a single
//! batched forward through the paged backend, and verifies a finite
//! token is sampled. Exercises the same code path as Qwen3 — proves
//! the architecture-agnostic refactor (optional q_norm/k_norm) works.
//!
//! A second, more demanding test fires up the full BatchScheduler with
//! a Llama 3 chat template and verifies end-to-end generation.

#![cfg(all(feature = "candle-metal", target_os = "macos"))]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use velox::backend::traits::{ChatMessage, StreamChunk};
use velox::paged::llama::load_paged_llama;
use velox::paged::pages::{PagedKvCache, PagedKvConfig};
use velox::paged::qwen3::{BatchStep, SeqSlice};
use velox::paged::scheduler::{BatchScheduler, SchedulerConfig, SubmitRequest};

fn find_llama() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    for name in &[
        "Llama-3.2-1B-Instruct-4bit",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct-4bit",
    ] {
        let p: PathBuf = format!("{}/.velox/models/{}", home, name).into();
        if p.join("model.safetensors").exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn paged_llama_one_step() {
    let Some(model_dir) = find_llama() else {
        eprintln!("no Llama model in ~/.velox/models → skipping");
        return;
    };
    eprintln!("loading {}", model_dir.display());

    let device = Device::new_metal(0).expect("metal device");
    let dtype = DType::BF16;

    let st = model_dir.join("model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[st], dtype, &device).expect("VarBuilder")
    };
    let model =
        Arc::new(load_paged_llama(&model_dir, vb).expect("load Llama"));
    let cfg = model.config().clone();

    let pages_cfg = PagedKvConfig {
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        page_size: 16,
        num_pages: 32,
        dtype,
    };
    let pages = Arc::new(PagedKvCache::new(pages_cfg, &device).expect("pages"));

    // Llama 3 BOS = 128000, then "Hello, world." in tokens
    // (using Llama tokenizer-agnostic IDs as smoke; correctness is
    // checked via the finite-logit assertion below).
    let prompt: Vec<u32> = vec![128000, 9906, 11, 1917, 13];
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
    assert!(v.iter().all(|x| x.is_finite()), "Llama logits NaN/Inf");
    let argmax = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("Llama paged sampled token id: {argmax}");
    assert!(argmax < cfg.vocab_size);
}

/// Llama 3 chat template (the official one shipped with `mlx-community/Llama-3.2-*-Instruct`).
/// Plain MiniJinja string — the scheduler will render it.
fn llama3_chat_template() -> String {
    // Reduced version of the official template. Skips the system message
    // (none in this test), wraps a single user turn, then opens the
    // assistant header for generation.
    "{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n".to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paged_scheduler_generates_with_llama() {
    let Some(model_dir) = find_llama() else {
        eprintln!("no Llama model in ~/.velox/models → skipping");
        return;
    };
    let device = Device::new_metal(0).expect("metal device");
    let dtype = DType::BF16;

    let st = model_dir.join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[st], dtype, &device).unwrap() };
    let model = Arc::new(load_paged_llama(&model_dir, vb).expect("load Llama"));
    let cfg = model.config().clone();

    let pages = Arc::new(
        PagedKvCache::new(
            PagedKvConfig {
                num_layers: cfg.num_hidden_layers,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                page_size: 16,
                num_pages: 64,
                dtype,
            },
            &device,
        )
        .unwrap(),
    );

    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    // Llama 3 EOS family: <|end_of_text|>=128001, <|eom_id|>=128008,
    // <|eot_id|>=128009. The chat template emits <|eot_id|> after the
    // assistant turn, so 128009 is the practical stop token.
    let eos = vec![128001u32, 128008, 128009];

    let scheduler = Arc::new(BatchScheduler::new(
        model,
        pages,
        tokenizer,
        llama3_chat_template(),
        eos,
        SchedulerConfig {
            max_running_requests: 4,
            max_batch_tokens: 256,
            idle_sleep: Duration::from_millis(1),
            prefix_cache_capacity: 0,
        },
    ));

    let sched_clone = scheduler.clone();
    let join = std::thread::spawn(move || sched_clone.run().unwrap());

    let (tx, mut rx) = mpsc::channel::<StreamChunk>(64);
    scheduler
        .submit(SubmitRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Reply with the single word PONG and nothing else.".into(),
            }],
            prompt_tokens: vec![],
            max_tokens: 16,
            temperature: 0.0,
            top_p: 1.0,
            stop_sequences: vec![],
            tx,
        })
        .unwrap();

    let mut text = String::new();
    let mut completion_tokens: u32 = 0;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .ok()
        .flatten()
    {
        match chunk {
            StreamChunk::Token { text_delta, .. } => text.push_str(&text_delta),
            StreamChunk::Done {
                completion_tokens: ct,
                finish_reason,
                ..
            } => {
                completion_tokens = ct;
                eprintln!("LLAMA DONE: finish={finish_reason}, tokens={ct}, text={text:?}");
                break;
            }
            StreamChunk::Error(e) => panic!("err: {e}"),
        }
        if std::time::Instant::now() > deadline {
            panic!("timeout");
        }
    }

    scheduler.shutdown();
    let _ = join.join();

    assert!(completion_tokens > 0, "Llama generated zero tokens");
    assert!(!text.trim().is_empty(), "Llama returned empty text");
}
