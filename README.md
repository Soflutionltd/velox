# Velox

**The world's first Rust-native LLM inference server for Apple Silicon.**

Single static binary. No Python, no Node, no GC. Continuous batching, paged KV cache,
fused Metal kernels, MLX-Int4 quantization, prefix cache, speculative decoding.
OpenAI- and Anthropic-compatible HTTP API.

```
$ velox serve --model Qwen3-4B-4bit
listening on http://0.0.0.0:8000

$ curl localhost:8000/v1/chat/completions -d '{
    "model": "Qwen3-4B-4bit",
    "messages": [{"role":"user","content":"Hello!"}]
  }'
```

---

## Why Velox

| | Velox | MLX (Python) | llama.cpp | Ollama |
|---|---|---|---|---|
| Single static binary | ✅ | ❌ (Python deps) | ✅ | ✅ |
| Memory-safe | ✅ Rust | ❌ Python | ⚠️ C++ | ⚠️ Go + C++ |
| Continuous batching | ✅ | ⚠️ partial | ⚠️ basic | ❌ |
| Paged KV cache | ✅ | ❌ | ❌ | ❌ |
| Apple Silicon native | ✅ Metal | ✅ MLX | ✅ Metal | ✅ Metal |
| MLX 4-bit weights | ✅ | ✅ | ❌ | ❌ |
| Prefix cache (system prompts) | ✅ | ❌ | ❌ | ⚠️ basic |
| Speculative decoding | ✅ | ❌ | ✅ | ❌ |
| OpenAI + Anthropic API | ✅ both | OpenAI only | OpenAI only | Custom + OpenAI |

Velox is built for **production single-user prosumer** and **multi-tenant
batched** workloads on Apple Silicon. If you want to serve a real chatbot, IDE
agent, or pipeline from a Mac mini or M-series workstation, this is for you.

## Quick start

### From source

```bash
git clone https://github.com/Soflutionltd/velox
cd velox
cargo install --path . --features candle-metal --locked
```

### Run a model

```bash
# Download an MLX checkpoint into ~/.velox/models/
hf download mlx-community/Qwen3-4B-4bit --local-dir ~/.velox/models/Qwen3-4B-4bit

# Start the server
velox serve --model Qwen3-4B-4bit
```

### Talk to it

OpenAI client (drop-in):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="Qwen3-4B-4bit",
    messages=[{"role": "user", "content": "Pourquoi Rust pour l'IA?"}],
).choices[0].message.content)
```

Anthropic client (drop-in):

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")
print(client.messages.create(
    model="Qwen3-4B-4bit",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hi!"}],
).content[0].text)
```

## Supported models

Native paged backend (continuous batching, paged KV, fused Metal kernels):

* **Qwen3** — 0.6B, 1.7B, 4B, 7B (BF16 and MLX-4bit)
* **Llama 3.x** — 1B, 3B, 8B (Llama 3.2 and 3.1, BF16 and MLX-4bit)
* **Mistral 7B** — Instruct v0.3+ (BF16 and MLX-4bit)

Sequential per-request backend (Candle's default, no paged optimisations):

* Phi-3, Gemma 2/3 (Gemma sliding-window not yet supported in paged mode)

## Architecture highlights

* **Paged KV cache** — vLLM-style block table, page size 16, configurable
  pool. Enables continuous batching and prefix sharing.
* **Continuous batching** — new requests join the in-flight batch every
  forward step. No head-of-line blocking, no per-request streams.
* **Fused Metal kernels** — `paged_decode_attention`, `paged_prefill_attention`,
  `batched_rope_decode`, `batched_scatter`, `qmm_4bit`. Hand-written MSL,
  benchmarked against MLX equivalents.
* **MLX-Int4 quantization** — native support for the `mlx-community` MLX
  4-bit format (packed uint32 weights, group_size=64). 4× memory reduction.
* **Prefix cache** — refcounted, page-level. Skips redundant prefill on
  shared system prompts. Measured ~6.5× speedup on chat-style repeated
  prompts.
* **Speculative decoding** — small draft model + greedy verify with a
  larger target. Algorithmically bit-identical to target-only greedy on
  the first verified tokens; numerical drift afterwards due to per-kernel
  reduction order.
* **OpenAI + Anthropic APIs** — full chat completions, streaming SSE,
  tool calling, vision, thinking blocks.

## Benchmarks

```bash
# Run the head-to-head suite (auto-detects mlx-lm, llama.cpp, ollama):
python3 scripts/bench_compare.py --tokens 200 --runs 3
```

Results table is appended to `scripts/bench_compare.md` for sharing.

## Roadmap

Next up (in order):

1. Tiled `qmm_4bit` v2 (multi-SIMD cooperative dequant) — batch-decode throughput
2. KV cache quantization (int8 then int4) — bigger contexts, less RAM
3. Long-context Llama 3.1 (NTK-scaling for 128K)
4. Sliding-window attention (unlocks Gemma 2/3)
5. Speculative decoding with a tiny dedicated draft (n-gram or Medusa)
6. CUDA / NVIDIA support (separate backend, shared scheduler)
7. Homebrew tap and signed macOS binary

## Project layout

```
src/
├── api/            # OpenAI + Anthropic types and routing
├── backend/        # Inference backends (Candle for Apple, llama.cpp fallback)
├── cache/          # Tiered KV cache (paged GPU + LRU + SSD + prefix trie)
├── paged/          # Native paged backend (Qwen3 / Llama / Mistral)
│   ├── pages.rs            # PagedKvCache + refcount
│   ├── prefix_cache.rs     # Chained-hash, LRU prefix cache
│   ├── scheduler.rs        # Continuous-batching scheduler
│   ├── spec.rs             # Speculative decoding engine
│   ├── metal_kernels.rs    # All custom MSL kernels
│   ├── qwen3.rs            # Qwen3 model + shared transformer engine
│   └── llama.rs            # Llama / Mistral config parser
├── server/         # HTTP server (axum), SSE streaming
└── model/          # Model discovery + HF downloader

tests/              # Integration + smoke + Metal kernel parity
scripts/            # Bench scripts + ops
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Status

Velox is **alpha**. APIs are stable but expect rough edges. Not yet
production-ready for adversarial multi-tenant. Single-user / trusted
multi-user workloads are well-tested.

Built by [@AntoinePinelli](https://github.com/antoinepinelli) at
[Soflution](https://soflution.com).
