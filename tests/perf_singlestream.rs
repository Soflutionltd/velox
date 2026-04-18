//! Single-stream throughput probe — measures pure decode tok/s without
//! any HTTP / JSON / tokenizer roundtrip. Compare against `mlx_lm.generate`
//! on the same model to isolate where the perf gap lives.

#![cfg(all(feature = "candle-metal", target_os = "macos"))]

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use velox::backend::traits::{ChatMessage, StreamChunk};
use velox::paged::pages::{PagedKvCache, PagedKvConfig};
use velox::paged::qwen3::{PagedQwen3, Qwen3Config};
use velox::paged::scheduler::{BatchScheduler, SchedulerConfig, SubmitRequest};

fn find_model() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let p: PathBuf = format!("{}/.velox/models/Qwen3-0.6B-4bit", home).into();
    p.join("model.safetensors").exists().then_some(p)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "perf probe; run with --ignored"]
async fn perf_qwen3_06b_4bit_singlestream() {
    let Some(model_dir) = find_model() else {
        eprintln!("no model");
        return;
    };
    let device = Device::new_metal(0).unwrap();
    let dtype = DType::BF16;

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
    let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n".to_string();

    let scheduler = Arc::new(BatchScheduler::new(
        model,
        pages,
        tokenizer,
        template,
        vec![151645, 151643],
        SchedulerConfig {
            max_running_requests: 1,
            max_batch_tokens: 256,
            idle_sleep: Duration::from_micros(100),
            prefix_cache_capacity: 0,
        },
    ));

    let sched_clone = scheduler.clone();
    let join = std::thread::spawn(move || sched_clone.run().unwrap());

    // Warmup.
    eprintln!("warmup...");
    run_one(&scheduler, 50).await;

    // 3 timed runs of 200 tokens each.
    let target = 200u32;
    let mut total_t = Duration::ZERO;
    let mut total_n: u32 = 0;
    let runs = 3;
    for i in 1..=runs {
        let t0 = Instant::now();
        let n = run_one(&scheduler, target).await;
        let dt = t0.elapsed();
        eprintln!("run {i}: {n} tok in {:.2}s = {:.1} tok/s", dt.as_secs_f64(), n as f64 / dt.as_secs_f64());
        total_t += dt;
        total_n += n;
    }
    eprintln!(
        "AVG: {:.1} tok/s ({} tok in {:.2}s over {} runs)",
        total_n as f64 / total_t.as_secs_f64(),
        total_n,
        total_t.as_secs_f64(),
        runs
    );

    scheduler.shutdown();
    let _ = join.join();
}

/// Same probe but isolates *just* the sampling cost — no tokenizer
/// decode at all. If this is significantly faster than the test above,
/// the O(N²) tokenizer.decode is part of the gap.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "perf probe; run with --ignored"]
async fn perf_qwen3_06b_4bit_singlestream_no_decode() {
    let Some(model_dir) = find_model() else {
        return;
    };
    let device = Device::new_metal(0).unwrap();
    let dtype = DType::BF16;
    let cfg: Qwen3Config =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
            .unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], dtype, &device)
            .unwrap()
    };
    let model = Arc::new(PagedQwen3::load(&cfg, vb).unwrap());

    use velox::paged::qwen3::{BatchStep, SeqSlice};

    let pages = Arc::new(
        PagedKvCache::new(
            PagedKvConfig {
                num_layers: cfg.num_hidden_layers,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                page_size: 16,
                num_pages: 32,
                dtype,
            },
            &device,
        )
        .unwrap(),
    );

    // Direct loop, no scheduler / no tokenizer / no JSON / no HTTP.
    // Just: prefill once, then N decode steps with GPU argmax.
    use candle_core::Tensor;
    let prompt: Vec<u32> = (0..30u32).collect(); // 30-tok prompt
    let n_pages = (prompt.len() + 200 + 16) / 16 + 1;
    let block_table = pages.alloc(n_pages).unwrap();

    // Prefill.
    let inp = Tensor::from_vec(prompt.clone(), (prompt.len(),), &device).unwrap();
    let seqs = vec![SeqSlice {
        new_tokens: prompt.len(),
        kv_offset: 0,
        block_table: &block_table,
    }];
    let logits = model
        .forward(&BatchStep { input_ids: &inp, seqs: &seqs }, &pages)
        .unwrap();
    // GPU argmax → 1 u32 sync, not 152K floats.
    let next = logits
        .argmax_keepdim(candle_core::D::Minus1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()[0];

    // Warmup decode (5 tokens).
    let mut cur_tok = next;
    let mut kv_off = prompt.len();
    for _ in 0..5 {
        let inp = Tensor::from_vec(vec![cur_tok], (1,), &device).unwrap();
        let seqs = vec![SeqSlice { new_tokens: 1, kv_offset: kv_off, block_table: &block_table }];
        let l = model
            .forward(&BatchStep { input_ids: &inp, seqs: &seqs }, &pages)
            .unwrap();
        let nxt = l
            .argmax_keepdim(candle_core::D::Minus1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];
        cur_tok = nxt;
        kv_off += 1;
    }

    // Timed: 200 decode steps.
    let n = 200usize;
    let t0 = Instant::now();
    for _ in 0..n {
        let inp = Tensor::from_vec(vec![cur_tok], (1,), &device).unwrap();
        let seqs = vec![SeqSlice { new_tokens: 1, kv_offset: kv_off, block_table: &block_table }];
        let l = model
            .forward(&BatchStep { input_ids: &inp, seqs: &seqs }, &pages)
            .unwrap();
        let nxt = l
            .argmax_keepdim(candle_core::D::Minus1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];
        cur_tok = nxt;
        kv_off += 1;
    }
    let dt = t0.elapsed();
    eprintln!(
        "RAW DECODE: {} tok in {:.2}s = {:.1} tok/s (single seq, GPU argmax, no tokenizer)",
        n,
        dt.as_secs_f64(),
        n as f64 / dt.as_secs_f64()
    );
}

/// **The headline number.** This is what Velox is actually built for:
/// many concurrent requests sharing one model. Continuous batching +
/// paged KV cache makes per-token cost flat as batch grows, so
/// aggregate throughput should be ~B× single-stream throughput up to
/// the GPU's saturation point.
#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
#[ignore = "perf probe; run with --ignored"]
async fn perf_qwen3_06b_4bit_concurrent() {
    let Some(model_dir) = find_model() else {
        return;
    };
    let device = Device::new_metal(0).unwrap();
    let dtype = DType::BF16;
    let cfg: Qwen3Config =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
            .unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], dtype, &device)
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
                num_pages: 256,
                dtype,
            },
            &device,
        )
        .unwrap(),
    );
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n".to_string();

    let scheduler = Arc::new(BatchScheduler::new(
        model,
        pages,
        tokenizer,
        template,
        vec![151645, 151643],
        SchedulerConfig {
            max_running_requests: 16,
            max_batch_tokens: 512,
            idle_sleep: Duration::from_micros(50),
            prefix_cache_capacity: 0,
        },
    ));
    let sched_clone = scheduler.clone();
    let join = std::thread::spawn(move || sched_clone.run().unwrap());

    // Warmup.
    run_one(&scheduler, 30).await;

    // 16 concurrent requests, each generating 100 tokens.
    let concurrency = 16usize;
    let target = 100u32;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let s = scheduler.clone();
            tokio::spawn(async move { run_one(&s, target).await })
        })
        .collect();
    let mut total_n = 0u32;
    for h in handles {
        total_n += h.await.unwrap();
    }
    let dt = t0.elapsed();
    eprintln!(
        "CONCURRENT @{}: {} tok in {:.2}s = {:.1} aggregate tok/s ({:.1} per request)",
        concurrency,
        total_n,
        dt.as_secs_f64(),
        total_n as f64 / dt.as_secs_f64(),
        (total_n as f64 / dt.as_secs_f64()) / concurrency as f64
    );

    scheduler.shutdown();
    let _ = join.join();
}

async fn run_one(scheduler: &Arc<BatchScheduler>, max_tokens: u32) -> u32 {
    let (tx, mut rx) = mpsc::channel::<StreamChunk>(64);
    scheduler
        .submit(SubmitRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Write a 200-word essay about Rust.".into(),
            }],
            prompt_tokens: vec![],
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            stop_sequences: vec![],
            tx,
        })
        .unwrap();
    let mut completion = 0u32;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { .. } => {}
            StreamChunk::Done { completion_tokens, .. } => {
                completion = completion_tokens;
                break;
            }
            StreamChunk::Error(e) => panic!("err: {e}"),
        }
    }
    completion
}
