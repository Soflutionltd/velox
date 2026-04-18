#!/usr/bin/env python3
"""
Velox vs Ollama benchmark — same machine, same model, same prompt.

Both servers are hit through their respective HTTP APIs (chat completion
endpoints) to measure end-user-visible throughput.

Reproducibility:
  * Model: Qwen3-0.6B 4-bit MLX-quant on both sides
    - Velox loads:  ~/.velox/models/Qwen3-0.6B-4bit
    - Ollama loads: qwen3:0.6b   (same model, GGUF Q4_K_M)
  * Hardware: any Apple Silicon (script prints chip + RAM).
  * Three regimes:
      1. Single-stream decode    — 1 user, 256 tokens.
      2. Multi-stream throughput — N concurrent users, 256 tokens each.
      3. TTFT                    — time to first token, 1 user.

Note on fairness:
  * Ollama uses llama.cpp's GGUF Q4_K_M (5 bits/weight effective).
  * Velox uses MLX 4-bit (4.5 bits/weight effective with scales/biases).
  * Quality is comparable; weight-byte counts differ by ~10%. Quoted
    elsewhere: this is the cleanest direct comparison we can make
    without forcing one engine to use the other's format.

Outputs:
  * Console table.
  * JSON dump in BENCH_OUT (default: bench_results.json).
"""

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import aiohttp


# --------------------------------------------------------------------- #
# Hardware probe
# --------------------------------------------------------------------- #
def hw_info() -> dict:
    info = {"os": platform.platform()}
    if sys.platform == "darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            mem_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            info["chip"] = chip
            info["ram_gb"] = round(mem_bytes / (1024**3))
        except Exception:
            pass
    return info


# --------------------------------------------------------------------- #
# Result aggregation
# --------------------------------------------------------------------- #
@dataclass
class RunResult:
    backend: str
    regime: str
    concurrency: int
    n_requests: int
    tokens_per_request: int
    total_tokens: int
    wall_seconds: float
    tokens_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    ttft_ms: Optional[float] = None


# --------------------------------------------------------------------- #
# Single request workers
# --------------------------------------------------------------------- #
async def velox_request(
    session: aiohttp.ClientSession,
    base: str,
    prompt: str,
    max_tokens: int,
    measure_ttft: bool = False,
) -> tuple[int, float, Optional[float]]:
    """Returns (tokens_generated, total_seconds, ttft_seconds_or_None)."""
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": "Qwen3-0.6B-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": measure_ttft,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    if measure_ttft:
        ttft = None
        n_tok = 0
        async with session.post(url, json=payload) as r:
            async for line in r.content:
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                if line == b"data: [DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tok += 1
        return n_tok, time.perf_counter() - t0, ttft
    else:
        async with session.post(url, json=payload) as r:
            data = await r.json()
        tot = data.get("usage", {}).get("completion_tokens", max_tokens)
        return tot, time.perf_counter() - t0, None


async def ollama_request(
    session: aiohttp.ClientSession,
    base: str,
    prompt: str,
    max_tokens: int,
    measure_ttft: bool = False,
) -> tuple[int, float, Optional[float]]:
    """Returns (tokens_generated, total_seconds, ttft_seconds_or_None)."""
    url = f"{base}/api/chat"
    payload = {
        "model": "qwen3:0.6b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": measure_ttft,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }
    t0 = time.perf_counter()
    if measure_ttft:
        ttft = None
        n_tok = 0
        async with session.post(url, json=payload) as r:
            async for line in r.content:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if ttft is None and obj.get("message", {}).get("content"):
                    ttft = time.perf_counter() - t0
                if obj.get("message", {}).get("content"):
                    n_tok += 1
                if obj.get("done"):
                    n_tok = obj.get("eval_count", n_tok)
                    break
        return n_tok, time.perf_counter() - t0, ttft
    else:
        async with session.post(url, json=payload) as r:
            data = await r.json()
        tot = data.get("eval_count", max_tokens)
        return tot, time.perf_counter() - t0, None


REQUESTERS = {"velox": velox_request, "ollama": ollama_request}


# --------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------- #
async def regime_single(
    backend: str, base: str, prompt: str, max_tokens: int, n_runs: int
) -> RunResult:
    fn = REQUESTERS[backend]
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        # Warmup.
        await fn(s, base, prompt, 32)
        latencies = []
        toks = 0
        t0 = time.perf_counter()
        for _ in range(n_runs):
            n_tok, sec, _ = await fn(s, base, prompt, max_tokens)
            latencies.append(sec * 1000)
            toks += n_tok
        wall = time.perf_counter() - t0
    return RunResult(
        backend=backend,
        regime="single_stream",
        concurrency=1,
        n_requests=n_runs,
        tokens_per_request=max_tokens,
        total_tokens=toks,
        wall_seconds=wall,
        tokens_per_second=toks / wall,
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)],
    )


async def regime_concurrent(
    backend: str,
    base: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    n_runs_per_user: int,
) -> RunResult:
    fn = REQUESTERS[backend]
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        await fn(s, base, prompt, 32)  # warmup
        latencies: List[float] = []
        total_tokens = 0

        async def user_loop():
            nonlocal total_tokens
            for _ in range(n_runs_per_user):
                n_tok, sec, _ = await fn(s, base, prompt, max_tokens)
                latencies.append(sec * 1000)
                total_tokens += n_tok

        t0 = time.perf_counter()
        await asyncio.gather(*[user_loop() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    return RunResult(
        backend=backend,
        regime="multi_stream",
        concurrency=concurrency,
        n_requests=concurrency * n_runs_per_user,
        tokens_per_request=max_tokens,
        total_tokens=total_tokens,
        wall_seconds=wall,
        tokens_per_second=total_tokens / wall,
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)],
    )


async def regime_ttft(
    backend: str, base: str, prompt: str, n_runs: int
) -> RunResult:
    fn = REQUESTERS[backend]
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        await fn(s, base, prompt, 16, measure_ttft=True)  # warmup
        ttfts = []
        for _ in range(n_runs):
            _, _, ttft = await fn(s, base, prompt, 32, measure_ttft=True)
            if ttft is not None:
                ttfts.append(ttft * 1000)
    return RunResult(
        backend=backend,
        regime="ttft",
        concurrency=1,
        n_requests=n_runs,
        tokens_per_request=32,
        total_tokens=0,
        wall_seconds=0.0,
        tokens_per_second=0.0,
        p50_latency_ms=statistics.median(ttfts),
        p95_latency_ms=sorted(ttfts)[int(len(ttfts) * 0.95)],
        ttft_ms=statistics.median(ttfts),
    )


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velox-base", default="http://127.0.0.1:8080")
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    ap.add_argument("--prompt", default="Write a 200-word essay about why Apple Silicon is well-suited for LLM inference.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrencies", default="1,4,8,16")
    ap.add_argument("--n-runs", type=int, default=4)
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--backends", default="velox,ollama")
    args = ap.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    concurrencies = [int(c) for c in args.concurrencies.split(",")]

    bases = {"velox": args.velox_base, "ollama": args.ollama_base}
    results: List[RunResult] = []
    hw = hw_info()
    print(f"\nHardware: {hw}")
    print(f"Prompt:   {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Tokens:   {args.max_tokens} per request, {args.n_runs} runs each")
    print()

    for backend in backends:
        base = bases[backend]
        print(f"=== {backend.upper()} ({base}) ===")

        try:
            print(f"  ttft ...", end="", flush=True)
            r = await regime_ttft(backend, base, args.prompt, args.n_runs)
            results.append(r)
            print(f" p50={r.p50_latency_ms:.0f}ms  p95={r.p95_latency_ms:.0f}ms")
        except Exception as e:
            print(f" FAILED: {e}")

        try:
            print(f"  single ...", end="", flush=True)
            r = await regime_single(backend, base, args.prompt, args.max_tokens, args.n_runs)
            results.append(r)
            print(f" {r.tokens_per_second:.1f} tok/s  p50={r.p50_latency_ms:.0f}ms")
        except Exception as e:
            print(f" FAILED: {e}")

        for c in concurrencies:
            if c == 1:
                continue
            try:
                print(f"  conc={c:>3} ...", end="", flush=True)
                r = await regime_concurrent(
                    backend, base, args.prompt, args.max_tokens, c, args.n_runs
                )
                results.append(r)
                print(
                    f" {r.tokens_per_second:.1f} tok/s total"
                    f"  p50={r.p50_latency_ms:.0f}ms  p95={r.p95_latency_ms:.0f}ms"
                )
            except Exception as e:
                print(f" FAILED: {e}")

        print()

    out = {
        "hardware": hw,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "results": [asdict(r) for r in results],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(results)} runs → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
