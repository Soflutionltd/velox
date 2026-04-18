#!/usr/bin/env python3
"""
Velox vs MLX vs llama.cpp vs Ollama — single-stream throughput shootout.

Same prompt, same model family, same generation length, greedy decoding.
Each backend gets warmed up first, then we average over N runs.

Backends are auto-detected. Missing ones are skipped with a clear message
rather than failing the whole comparison.

Usage:
    python3 scripts/bench_compare.py
    python3 scripts/bench_compare.py --tokens 500 --runs 5
    python3 scripts/bench_compare.py --model Llama-3.2-1B-Instruct-4bit

Backends and how they're invoked:
    velox      → HTTP, expects `velox` running on localhost:8000
    mlx        → Python `mlx_lm.generate` (pip install mlx-lm)
    llamacpp   → CLI `llama-cli` (brew install llama.cpp)
    ollama     → HTTP, expects `ollama serve` on localhost:11434

Output: a markdown-friendly table you can paste into a Show HN post.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

PROMPT = (
    "Write a 200-word essay about the importance of pure-Rust LLM "
    "inference servers, focusing on memory safety, single-binary "
    "deployment, and Apple Silicon performance."
)


@dataclass
class Result:
    backend: str
    tokens: int
    seconds: float

    @property
    def tps(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


def time_it(fn: Callable[[], int]) -> Result:
    t0 = time.time()
    n = fn()
    dt = time.time() - t0
    return Result(backend="<unset>", tokens=n, seconds=dt)


# ---- Velox ---------------------------------------------------------------


def have_velox(host: str) -> bool:
    try:
        urllib.request.urlopen(f"{host}/health", timeout=2)
        return True
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        return False


def run_velox(model: str, max_tokens: int, host: str) -> int:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return int(data["usage"]["completion_tokens"])


# ---- MLX -----------------------------------------------------------------


def have_mlx() -> bool:
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


_MLX_CACHE: dict[str, tuple] = {}


def run_mlx(model_dir: str, max_tokens: int) -> int:
    """Run mlx-lm generate inline (faster than CLI subprocess — avoids
    re-loading the model on every run)."""
    from mlx_lm import generate, load

    if model_dir not in _MLX_CACHE:
        _MLX_CACHE[model_dir] = load(model_dir)
    model, tokenizer = _MLX_CACHE[model_dir]

    out = generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=max_tokens,
        verbose=False,
    )
    # mlx-lm returns the decoded string; we count via re-tokenising.
    # This includes the prompt — so we tokenise both and subtract.
    full = tokenizer.encode(out)
    prompt_ids = tokenizer.encode(PROMPT)
    return max(0, len(full) - len(prompt_ids))


# ---- llama.cpp -----------------------------------------------------------


def have_llamacpp() -> bool:
    return shutil.which("llama-cli") is not None


def run_llamacpp(gguf_path: str, max_tokens: int) -> int:
    out = subprocess.run(
        [
            "llama-cli",
            "-m", gguf_path,
            "-p", PROMPT,
            "-n", str(max_tokens),
            "--temp", "0",
            "-ngl", "99",  # all layers on GPU (Metal)
            "--no-warmup",
            "-no-cnv",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # llama.cpp prints timing info on stderr; we grab the "predicted"
    # count from "n_predict =" or "n_eval = N runs".
    for line in out.stderr.splitlines():
        if "eval time" in line and "tokens" in line:
            # Format: "llama_perf: ... eval time = ... ms / N tokens (...)"
            try:
                parts = line.split("/")
                tok_part = parts[1].strip().split()[0]
                return int(tok_part)
            except (IndexError, ValueError):
                continue
    return max_tokens  # fallback


# ---- Ollama --------------------------------------------------------------


def have_ollama() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        return False


def run_ollama(model: str, max_tokens: int) -> int:
    body = json.dumps(
        {
            "model": model,
            "prompt": PROMPT,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return int(data.get("eval_count", max_tokens))


# ---- main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--velox-model",
        default="Qwen3-0.6B",
        help="model name as registered in velox",
    )
    ap.add_argument(
        "--mlx-model",
        default=os.path.expanduser("~/.velox/models/Qwen3-0.6B"),
        help="mlx-lm model dir",
    )
    ap.add_argument(
        "--llamacpp-gguf",
        default="",
        help="path to .gguf file (skips llama.cpp if empty)",
    )
    ap.add_argument(
        "--ollama-model",
        default="qwen2.5:0.5b",
        help="ollama model tag (e.g. llama3.2:1b)",
    )
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--velox-host", default="http://localhost:8000")
    args = ap.parse_args()

    print(f"prompt length (chars): {len(PROMPT)}")
    print(f"max_tokens / run     : {args.tokens}")
    print(f"runs                 : {args.runs}")
    print()

    results: list[Result] = []

    def bench(name: str, fn: Callable[[], int]) -> None:
        # 1 warmup, then N timed runs.
        try:
            fn()
        except Exception as e:
            print(f"{name}: warmup failed ({e}); skipping")
            return
        runs = []
        for _ in range(args.runs):
            try:
                r = time_it(fn)
                runs.append(r)
            except Exception as e:
                print(f"{name}: run failed ({e})")
                return
        if not runs:
            return
        # Aggregate.
        total_t = sum(r.seconds for r in runs)
        total_n = sum(r.tokens for r in runs)
        avg = Result(backend=name, tokens=total_n, seconds=total_t)
        results.append(avg)
        print(f"{name:<12} {avg.tokens / args.runs:>6.0f} tok  {avg.seconds / args.runs:>5.2f} s/run  {avg.tps:>6.1f} tok/s")

    if have_velox(args.velox_host):
        bench("velox", lambda: run_velox(args.velox_model, args.tokens, args.velox_host))
    else:
        print(f"velox        offline at {args.velox_host} — start `velox serve` first")

    if have_mlx() and os.path.isdir(args.mlx_model):
        bench("mlx-lm", lambda: run_mlx(args.mlx_model, args.tokens))
    else:
        print(f"mlx-lm       not installed or model dir missing ({args.mlx_model})")

    if args.llamacpp_gguf and have_llamacpp() and os.path.exists(args.llamacpp_gguf):
        bench("llama.cpp", lambda: run_llamacpp(args.llamacpp_gguf, args.tokens))
    else:
        print("llama.cpp    skipped (--llamacpp-gguf empty or binary missing)")

    if have_ollama():
        bench("ollama", lambda: run_ollama(args.ollama_model, args.tokens))
    else:
        print("ollama       not running on :11434")

    if not results:
        sys.exit("no backends available — start at least one and rerun")

    print()
    print("| backend     | avg tok/run | avg s/run | tok/s |")
    print("|-------------|------------:|----------:|------:|")
    for r in results:
        per_run_t = r.tokens / args.runs
        per_run_s = r.seconds / args.runs
        print(f"| {r.backend:<11} | {per_run_t:>11.0f} | {per_run_s:>9.2f} | {r.tps:>5.1f} |")


if __name__ == "__main__":
    main()
