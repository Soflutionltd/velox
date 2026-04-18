# Velox

**The world's first Rust-native LLM inference server for Apple Silicon.**

Single static binary. No Python, no Node, no GC. Built for **multi-tenant
serving** — continuous batching, paged KV cache, fused Metal kernels, prefix
cache for shared system prompts, MLX-Int4 quantization, speculative decoding.
OpenAI- and Anthropic-compatible HTTP API.

> **Where Velox shines:** serving N concurrent users from one model.
> Measured **478 tok/s aggregate at 16 concurrent requests** on Qwen3-0.6B-4bit,
> beating mlx-lm starting at 2 concurrent users.
> See [BENCHMARKS.md](./BENCHMARKS.md) for the honest single-stream comparison
> (mlx-lm wins solo; we're 5.5× slower until we ship fused decode kernels).

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

## Transports

Velox can serve over both TCP and Unix domain sockets simultaneously:

```bash
velox serve \
    --model-dir ~/.velox/models \
    --port 8000 \
    --socket /tmp/velox.sock          # optional, can run alongside TCP
```

```bash
# TCP
curl http://localhost:8000/v1/models

# UDS — local apps can skip the kernel TCP stack
curl --unix-socket /tmp/velox.sock http://localhost/v1/models
```

UDS shaves ~30µs per round-trip versus localhost TCP. Useful if your app
makes many small requests (token-by-token streaming, embeddings batches).

## Benchmarks

See [BENCHMARKS.md](./BENCHMARKS.md) for the full head-to-head numbers
(single-stream and concurrent), reproduction commands, and a transparent
discussion of where each backend wins.

Quick summary on Qwen3-0.6B-4bit (Apple Silicon, MLX-Int4):

| Mode | Velox | mlx-lm |
|---|---:|---:|
| Single-stream (1 user) | 49 tok/s | **271 tok/s** |
| Concurrent @16 (16 users) | **478 tok/s aggregate** | N/A (single-stream only) |

Run the comparison yourself:

```bash
python3 scripts/bench_compare.py \
    --velox-model Qwen3-0.6B-4bit \
    --mlx-model ~/.velox/models/Qwen3-0.6B-4bit \
    --tokens 200 --runs 3
```

## Roadmap

Single-stream perf (close the gap with mlx-lm):

1. **Fused decode layer kernels** — one big Metal dispatch per layer instead
   of ~14 small ones. Targets ~3× single-stream speedup.
2. **Tiled `qmm_4bit` v2** — multi-SIMD cooperative dequant. Targets ~1.5×
   on quant weights.
3. **mistral.rs / mlx-rs alternative backends** — when stable.

Concurrent perf (push the lead further):

4. Larger `max_batch_tokens`, cross-request prefill chunking
5. Speculative decoding with a tiny n-gram or Medusa draft

Coverage / features:

6. Phi-3 (split int4-aware fused weights)
7. Sliding-window attention (unlocks Gemma 2/3, Mistral v0.1/0.2, Phi-3 long-ctx)
8. Long-context Llama 3.1 (NTK-aware RoPE bands for 128K)
9. KV cache quantization (int8 then int4) — bigger contexts, less RAM
10. CUDA / NVIDIA backend (separate, shares the scheduler)

Distribution:

11. ~~Unix domain socket transport~~ ✅ shipped — `--socket /tmp/velox.sock`
12. gRPC server (`tonic`) alongside HTTP
13. Homebrew tap and signed macOS binary
14. Landing page + Show HN

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
