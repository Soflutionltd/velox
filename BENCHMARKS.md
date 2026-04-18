# Velox Benchmarks

> **All numbers below are reproducible.** No marketing math, no synthetic best-cases. Scripts are in `scripts/`, charts are regenerated from real measurements.

## Hardware & methodology

```
Chip:   Apple M2 Max (12 cores, 8P + 4E)
RAM:    96 GB unified
OS:     macOS 26.4 (arm64)
Velox:  release build, commit 2a458a6
        cargo build --release --features candle-metal
        VELOX_QMM_BACKEND=mlx
Ollama: 0.6+ (llama.cpp backend, GGUF Q4_K_M)
MLX-LM: 0.31+ (Python, reference for single-stream)
Model:  Qwen3-0.6B 4-bit (same model family/size on every engine)
        Velox  → mlx-community/Qwen3-0.6B-4bit (MLX-Int4, group_size=64)
        Ollama → qwen3:0.6b (GGUF Q4_K_M, llama.cpp pipeline)
```

Each benchmark: warmup pass, then 3 timed runs per regime, p50/p95 aggregated across runs and across users in the concurrent regimes. Greedy decoding (`temperature=0`), 128-output-token completions over the same prompt.

---

## 1. End-to-end throughput vs Ollama

![Throughput](assets/bench_throughput.png)

|  Concurrency  |  Velox tok/s  |  Ollama tok/s  |  Velox advantage  |
|--------------:|--------------:|---------------:|:-----------------:|
|  1 user       |        92.7   |     **182.9**  | Ollama 1.97×      |
|  4 users      |    **223.0**  |       201.0    | **Velox +11%**    |
|  8 users      |    **394.8**  |       197.7    | **Velox +100%**   |
|  16 users     |    **546.9**  |       200.5    | **Velox +173%**   |

**Key finding:** Ollama plateaus at ~200 tok/s regardless of load — it's a sequential request pipeline. Velox's continuous batching + batched GPU-side argmax sampling scale **5.9×** from 1→16 users.

**Update vs previous bench (commit `2a458a6` → this commit):** the GPU-side argmax patch landed in `scheduler.rs::sample_batch` collapses N per-token GPU→CPU syncs (one per running request, ~600 KB each on Qwen3 vocab=151K) into **a single 4·N-byte sync**. Effect is proportional to concurrency — single-stream gained +3.6%, while 16-user batches gained +16% on top of the already-strong baseline.

---

## 2. Tail latency (p95, lower is better)

![Latency](assets/bench_latency.png)

| Concurrency | Velox p95 | Ollama p95 | Velox advantage |
|------------:|----------:|-----------:|:---------------:|
| 1 user      |    1.4 s  |     0.7 s  | Ollama 2.0×     |
| 4 users     |    2.3 s  |     2.6 s  | **Velox 1.13×** |
| 8 users     |  **2.6 s**|     5.4 s  | **Velox 2.06×** |
| 16 users    |  **3.7 s**|    10.4 s  | **Velox 2.77×** |

Above 4 users, Ollama's latency grows linearly with load. Velox's stays nearly flat.

---

## 3. Where MLX-LM wins (and why)

We previously published this single-stream comparison vs `mlx-lm` (Apple's reference Python serving library):

|  Backend |  tok/s (1 user, 200-token essay)  |
|----------|-----------------------------------:|
|  velox  |   ~89   |
|  mlx-lm |  **~270** |

**MLX-LM wins solo because Apple has years of hand-tuned MSL kernels** (`simdgroup_matrix_multiply`, fused decode layers, lazy graph evaluation). MLX-LM has **no continuous batching, no prefix cache, no paged KV** — the moment you add a second concurrent user, throughput stays flat at ~270 while Velox keeps scaling.

We are progressively closing this gap by **vendoring and porting MLX's Metal kernels** into pure Rust. See section 4.

---

## 4. MLX-ported kernel speedup (single-stream decode)

![Kernel speedup](assets/bench_kernel.png)

| Projection shape (M=1) | Velox-native | MLX-ported `qmv_fast_g64` | Speedup |
|---|---:|---:|:---:|
| Qwen3-0.6B q_proj (1024×1024)   | 1.00× | 1.65× | **1.65×** |
| Qwen3-4B mlp_up (2560×9728)     | 1.00× | 2.09× | **2.09×** |
| Llama-3.1-8B q_proj (4096×4096) | 1.00× | 1.81× | **1.81×** |
| Llama-3.1-8B mlp_up (4096×14336)| 1.00× | 1.92× | **1.92×** |
| Phi-3-mini fused_qkv (3072×9216)| 1.00× | 1.41× | **1.41×** |
| Mistral-7B mlp_down (14336×4096)| 1.00× | 1.33× | **1.33×** |
| **Geomean**                     |       |       | **1.65×** |

**Numerical parity:** all six shapes pass `tests/metal_qmv_mlx_parity.rs` for `f32` / `f16` / `bf16` within `atol + rtol·|x|`. Bench source: `tests/metal_qmv_mlx_bench.rs`.

This is **commit 2/6** of the MLX-port roadmap. After commits 3-6 (general `qmm_n`, tiled GEMM, dispatcher), Velox single-stream should be within 10-20% of MLX-LM while keeping its multi-user lead.

---

## 5. Prefix cache (system prompt sharing)

Repeated requests with the same long system prompt skip the prefill entirely. Measured **6.5× speedup** on the same workload when the system prompt is page-aligned (length ≥ 16 tokens × N pages) and re-sent across requests.

See `tests/paged_scheduler.rs::paged_scheduler_prefix_cache_hits` for the integration test.

---

## 6. Speculative decoding

Currently shipped but ~1.0× speedup vs target-only on the Qwen3-4B-4bit target with Qwen3-0.6B (BF16 or 4-bit) draft. The acceptance rate is low (~25-35%) and the draft model isn't cheap enough on Apple Silicon Metal due to dispatch overhead dominating compute for small models.

We will get a real speedup here when we add either:

* a tiny n-gram drafter (no model forward, just a trie lookup)
* a Medusa-style sampling head (multiple heads on the target model)
* multi-token prediction (MTP) integrated into Qwen3

---

## 7. How to reproduce

```bash
# 1) Build
cargo install --path . --features candle-metal --locked

# 2) Pull the same model on both engines
velox pull Qwen3-0.6B-4bit
ollama pull qwen3:0.6b

# 3) Start both servers
VELOX_QMM_BACKEND=mlx velox serve \
    --model-dir ~/.velox/models/Qwen3-0.6B-4bit \
    --port 8080 &
ollama serve > /tmp/ollama.log 2>&1 &

# 4) Run the head-to-head
pip install aiohttp matplotlib
python3 scripts/bench_vs_ollama.py \
    --max-tokens 128 --n-runs 3 \
    --concurrencies 1,4,8,16 \
    --out scripts/bench_results.json

# 5) Render the charts
python3 scripts/render_charts.py
# → assets/bench_throughput.png, bench_latency.png, bench_kernel.png

# 6) Internal kernel benchmark (MLX-ported vs Velox-native qmm_4bit)
cargo test --release --features candle-metal \
    --test metal_qmv_mlx_bench -- --ignored --nocapture
```

---

## 8. Honesty disclosure

* Numbers measured on a single machine, single 3-run series per regime. Not a rigorous statistical study.
* `MLX-LM` numbers in section 3 come from earlier in-house testing on the same hardware with the same model. Reproducible via `pip install mlx-lm && python -m mlx_lm.generate --model mlx-community/Qwen3-0.6B-4bit --prompt "<200-word essay request>" --max-tokens 200`.
* **No vLLM / TGI / SGLang numbers** because those engines are CUDA-only — they do not run on Apple Silicon at all. The relevant comparison there will be when we ship the Velox CUDA fork (post-commits-3-to-6).
* `llama.cpp` direct numbers not included separately because Ollama uses the same llama.cpp pipeline. Direct `llama-server` should match Ollama within 5-10%.
* The Ollama Q4_K_M format is ~10% larger per weight than MLX-Int4 g64, so Ollama gets a small inherent quality advantage and a small inherent speed disadvantage. Best honest apples-to-apples we can do without forcing one engine to use the other's quant format.

All claims here are reproducible from `scripts/bench_vs_ollama.py`, `scripts/render_charts.py`, and the kernel benches in `tests/metal_qmv_mlx_*.rs`.
