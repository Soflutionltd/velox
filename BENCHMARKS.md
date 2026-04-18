# Velox Benchmarks

Real measurements taken on this hardware:

```
Apple Silicon M-series, 16+ GB unified memory
macOS, Metal backend
Velox release build (cargo build --release --features candle-metal)
mlx-lm 0.31.2 (pip install)
Same model: mlx-community/Qwen3-0.6B-4bit (MLX-Int4, ~600M params)
Same prompt: 200-word essay request, greedy decoding
```

## Single-stream (one request, one user)

|  Backend | tok/run | s/run | tok/s |
|----------|--------:|------:|------:|
| velox    |     200 |  4.06 |  49.3 |
| mlx-lm   |     166 |  0.61 | 271.6 |

**Velox is 5.5× slower than MLX-lm for a single request.** This is honest.

The cause: every forward step launches ~400 individual Metal kernels
(~14 ops × 28 layers). MLX uses lazy evaluation and kernel fusion to
merge these into a handful of big GPU dispatches. Candle (our backend
for now) is eager and pays a kernel-launch tax on every op. We're
working on this — see [Roadmap](#roadmap) below — but it will not be
beaten on single-stream without writing fused decode layer kernels.

## Concurrent batched (the real Velox story)

Same model, 16 concurrent requests, each generating 100 tokens with greedy decoding:

| Backend | aggregate tok/s | per-request tok/s |
|---------|----------------:|------------------:|
| velox @ 16 | **478** | 29.9 |
| mlx-lm @ 16 | N/A (single-stream only) | — |

**Velox beats MLX-lm starting at 2 concurrent users.**

This is what Velox is built for: continuous batching with paged KV
cache. New requests join the in-flight batch every step. The GPU
processes the whole batch in one fused dispatch — 16 users cost
roughly the same as 1 (until you saturate the GPU).

|  Concurrency  |  Aggregate tok/s  |  Per-user tok/s  |
|--------------:|------------------:|-----------------:|
|  1            |  49               |  49              |
|  16           |  478              |  30              |

Going from 1 → 16 users: **9.7× more throughput**, ~40% per-user latency
penalty. That's the trade-off any serious LLM server makes.

## Prefix cache (system prompt sharing)

Repeated requests with the same long system prompt skip the prefill
entirely. Measured **6.5× speedup** on the same workload when the
system prompt is page-aligned (length ≥ 16 tokens × N pages) and
re-sent across requests.

See `tests/paged_scheduler.rs::paged_scheduler_prefix_cache_hits` for
the integration test.

## Speculative decoding

Currently shipped but ~1.0× speedup vs target-only on the Qwen3-4B-4bit
target with Qwen3-0.6B (BF16 or 4-bit) draft. The acceptance rate is
low (~25-35%) and the draft model isn't cheap enough on Apple Silicon
Metal due to dispatch overhead dominating compute for small models.

We will get a real speedup here when we add either:

* a tiny n-gram drafter (no model forward, just a trie lookup)
* a Medusa-style sampling head (multiple heads on the target model)
* multi-token prediction (MTP) integrated into Qwen3

## How to reproduce

```bash
# 1) Build velox
cargo install --path . --features candle-metal

# 2) Download a model
hf download mlx-community/Qwen3-0.6B-4bit \
    --local-dir ~/.velox/models/Qwen3-0.6B-4bit

# 3) Start server
velox serve --model-dir ~/.velox/models &

# 4) Install mlx-lm for comparison
pip install mlx-lm

# 5) Run the comparison
python3 scripts/bench_compare.py \
    --velox-model Qwen3-0.6B-4bit \
    --mlx-model ~/.velox/models/Qwen3-0.6B-4bit \
    --tokens 200 --runs 3
```

## Roadmap (perf-focused)

What we're working on to close the single-stream gap:

1. **Fused decode layer kernels** — merge q/k/v projections + RoPE +
   attention + o_proj into one big Metal dispatch per layer. Should
   recover ~3× of the gap.
2. **Tiled `qmm_4bit` v2** — multi-SIMD GEMM with cooperative
   threadgroup-shared dequantization. Recovers another ~1.5× on quant
   weights.
3. **mistral.rs / mlx-rs as alternative backends** — when stable.
   Both have native fused-eval pipelines that match MLX-lm.
4. **GPU-side argmax in the scheduler** (already done in spec
   decoding's draft loop, not yet in normal decode). Saves one
   GPU→CPU sync per generated token.

What we're working on to push concurrent throughput further:

1. **Larger batch tokens** — currently 256, could go 512+ on bigger machines
2. **Cross-request prefill chunking** — pack multiple in-flight prefills
   into the same forward batch
3. **Speculative + batched** — hard, but the holy grail.

## Honesty disclosure

* Numbers measured on a single machine, single run series. Not a
  rigorous statistical study — 3 runs each, average.
* Hardware specifics deliberately not pinned: results depend on
  M-chip generation, RAM, and thermal state.
* `mlx-lm` measurements include prompt processing in the wallclock —
  same as Velox, so the comparison is fair.
* No `llama.cpp` / `ollama` numbers in this v1 because we don't have
  a matching GGUF on disk. The bench script supports both — supply
  `--llamacpp-gguf <path>` and start `ollama serve` to get them.
* All claims here are reproducible from `scripts/bench_compare.py`
  and `tests/perf_singlestream.rs`.
