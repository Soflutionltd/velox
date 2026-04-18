//! Integration test: spin up a real BatchScheduler with Qwen3-0.6B and
//! generate from one request through the public API.

#![cfg(feature = "candle")]

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use velox::backend::traits::{ChatMessage, StreamChunk};
use velox::paged::pages::{PagedKvCache, PagedKvConfig};
use velox::paged::qwen3::{PagedQwen3, Qwen3Config};
use velox::paged::scheduler::{BatchScheduler, SchedulerConfig, SubmitRequest};

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

fn load_chat_template(dir: &PathBuf) -> String {
    let p = dir.join("tokenizer_config.json");
    let raw = std::fs::read_to_string(&p).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null);
    v.get("chat_template")
        .and_then(|x| x.as_str())
        .unwrap_or("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n")
        .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paged_scheduler_generates_one_request() {
    let Some(model_dir) = find_model() else {
        eprintln!("model not on disk → skipping");
        return;
    };

    let device = pick_dev();
    let dtype = if matches!(device, Device::Metal(_)) { DType::BF16 } else { DType::F32 };

    let cfg: Qwen3Config = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("config.json")).unwrap(),
    )
    .unwrap();

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_dir.join("model.safetensors")],
            dtype,
            &device,
        )
        .unwrap()
    };
    let model = Arc::new(PagedQwen3::load(&cfg, vb).unwrap());

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
    let template = load_chat_template(&model_dir);

    let scheduler = Arc::new(BatchScheduler::new(
        model,
        pages,
        tokenizer,
        template,
        vec![151645, 151643], // Qwen3 EOS / im_end
        SchedulerConfig {
            max_running_requests: 4,
            max_batch_tokens: 256,
            idle_sleep: Duration::from_millis(1),
        },
    ));

    let sched_clone = scheduler.clone();
    let join = std::thread::spawn(move || {
        sched_clone.run().unwrap();
    });

    let (tx, mut rx) = mpsc::channel::<StreamChunk>(64);
    scheduler
        .submit(SubmitRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Reply with the single word 'PONG' and nothing else.".into(),
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
            StreamChunk::Done { completion_tokens: ct, finish_reason, .. } => {
                completion_tokens = ct;
                eprintln!("DONE: finish={finish_reason}, tokens={ct}, text={text:?}");
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

    assert!(completion_tokens > 0, "no tokens generated");
    assert!(!text.trim().is_empty(), "empty text");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paged_scheduler_handles_concurrent_requests() {
    let Some(model_dir) = find_model() else {
        return;
    };
    let device = pick_dev();
    let dtype = if matches!(device, Device::Metal(_)) { DType::BF16 } else { DType::F32 };

    let cfg: Qwen3Config = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_dir.join("model.safetensors")],
            dtype,
            &device,
        )
        .unwrap()
    };
    let model = Arc::new(PagedQwen3::load(&cfg, vb).unwrap());
    let pages = Arc::new(
        PagedKvCache::new(
            PagedKvConfig {
                num_layers: cfg.num_hidden_layers,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                page_size: 16,
                num_pages: 128,
                dtype,
            },
            &device,
        )
        .unwrap(),
    );
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    let template = load_chat_template(&model_dir);
    let scheduler = Arc::new(BatchScheduler::new(
        model,
        pages,
        tokenizer,
        template,
        vec![151645, 151643],
        SchedulerConfig {
            max_running_requests: 8,
            max_batch_tokens: 1024,
            idle_sleep: Duration::from_millis(1),
        },
    ));

    let sched_clone = scheduler.clone();
    let join = std::thread::spawn(move || sched_clone.run().unwrap());

    // Submit 4 different requests at the same time.
    let prompts = [
        "Say the digit 1.",
        "Say the digit 2.",
        "Say the digit 3.",
        "Say the digit 4.",
    ];
    let mut tasks = Vec::new();
    for p in prompts {
        let (tx, mut rx) = mpsc::channel::<StreamChunk>(64);
        scheduler
            .submit(SubmitRequest {
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: p.into(),
                }],
                prompt_tokens: vec![],
                max_tokens: 8,
                temperature: 0.0,
                top_p: 1.0,
                stop_sequences: vec![],
                tx,
            })
            .unwrap();
        tasks.push(tokio::spawn(async move {
            let mut text = String::new();
            let mut completion_tokens = 0u32;
            while let Some(chunk) = tokio::time::timeout(Duration::from_secs(15), rx.recv())
                .await
                .ok()
                .flatten()
            {
                match chunk {
                    StreamChunk::Token { text_delta, .. } => text.push_str(&text_delta),
                    StreamChunk::Done { completion_tokens: ct, .. } => {
                        completion_tokens = ct;
                        break;
                    }
                    StreamChunk::Error(e) => panic!("err: {e}"),
                }
            }
            (text, completion_tokens)
        }));
    }

    let start = std::time::Instant::now();
    let results: Vec<(String, u32)> = futures::future::join_all(tasks)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    let elapsed = start.elapsed();

    scheduler.shutdown();
    let _ = join.join();

    let total_tokens: u32 = results.iter().map(|(_, ct)| ct).sum();
    let throughput = total_tokens as f64 / elapsed.as_secs_f64();
    eprintln!(
        "concurrent: 4 requests, {} total tokens in {:.2}s = {:.1} tok/s",
        total_tokens,
        elapsed.as_secs_f64(),
        throughput
    );
    for (i, (text, ct)) in results.iter().enumerate() {
        eprintln!("  req {i}: {} tokens, {:?}", ct, text);
    }
    assert_eq!(results.len(), 4);
    assert!(results.iter().all(|(_, ct)| *ct > 0));
}
