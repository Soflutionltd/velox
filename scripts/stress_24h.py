#!/usr/bin/env python3
"""
24h stress test harness for Velox.

Hammers the running server with a configurable mix of concurrent
chat-completion requests for a long duration (default 24h), tracking:

  * RSS memory growth (via psutil) — flags leaks
  * P50/P95/P99 latency per minute
  * Error rate per minute
  * Total tokens generated

Output:
  * Live console summary every 60s
  * stress_24h_<timestamp>.csv with per-minute aggregated metrics
  * stress_24h_<timestamp>.log with raw per-request samples

Failure conditions (exit code 1):
  * Error rate > 1% in any 10-minute window
  * RSS growth > 50% over the run (simple leak heuristic)
  * Any 5xx response

Run:
  python3 scripts/stress_24h.py                      # 24h default
  python3 scripts/stress_24h.py --duration 3600      # 1h smoke
  python3 scripts/stress_24h.py --concurrency 16 --tokens 200

Designed to live alongside the server (same machine). Pin it to a
separate core if you really want clean numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_PROMPTS = [
    "Explain what an LLM inference server does in two sentences.",
    "Write a haiku about Apple Silicon.",
    "List five reasons to write systems software in Rust.",
    "Summarise the difference between continuous batching and naive batching.",
    "What is paged attention and why does it matter for throughput?",
    "Translate 'Le serveur d'inférence Velox est rapide' into English.",
    "Give me a one-paragraph history of Metal Shader Language.",
    "Describe the role of a KV cache in transformer inference.",
]


@dataclass
class Sample:
    t_start: float
    latency_ms: float
    tokens: int
    status: int          # HTTP status (0 if connection error)
    error: str = ""


@dataclass
class WindowStats:
    minute: int
    n: int
    errors: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    tokens: int
    rss_mb: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen3-0.6B-4bit")
    p.add_argument("--duration", type=int, default=24 * 3600,
                   help="Total run duration in seconds (default 86400 = 24h)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Number of concurrent worker threads")
    p.add_argument("--tokens", type=int, default=128,
                   help="max_tokens per request")
    p.add_argument("--server-pid", type=int, default=None,
                   help="PID of velox server (for RSS tracking)")
    p.add_argument("--out-dir", type=str, default=".",
                   help="Where to write stress_24h_*.csv / *.log")
    return p.parse_args()


def get_rss_mb(pid: int | None) -> float:
    """Return RSS of the given pid in MiB. 0.0 if pid is None or process gone."""
    if pid is None:
        return 0.0
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip()) / 1024.0
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def find_server_pid() -> int | None:
    """Best-effort autodetect a running `velox serve` process."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "target/release/velox serve"],
            stderr=subprocess.DEVNULL,
        )
        pids = [int(x) for x in out.split() if x.strip().isdigit()]
        return pids[0] if pids else None
    except subprocess.CalledProcessError:
        return None


def fire_request(host: str, model: str, prompt: str, max_tokens: int) -> Sample:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
            elapsed_ms = (time.time() - t0) * 1000.0
            usage = payload.get("usage") or {}
            toks = int(usage.get("completion_tokens") or 0)
            return Sample(t_start=t0, latency_ms=elapsed_ms, tokens=toks, status=resp.status)
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.time() - t0) * 1000.0
        return Sample(t_start=t0, latency_ms=elapsed_ms, tokens=0, status=e.code, error=str(e))
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000.0
        return Sample(t_start=t0, latency_ms=elapsed_ms, tokens=0, status=0, error=str(e))


class Harness:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.samples: list[Sample] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.start_time = time.time()
        self.server_pid = args.server_pid or find_server_pid()
        self.initial_rss = get_rss_mb(self.server_pid)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / f"stress_24h_{ts}.csv"
        self.log_path = out_dir / f"stress_24h_{ts}.log"

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "minute", "wallclock_iso", "n", "errors", "p50_ms", "p95_ms",
            "p99_ms", "tokens", "rss_mb"
        ])
        self.log_file = open(self.log_path, "w")

        # Failure tracking: rolling 10-minute windows.
        self.recent_windows: list[WindowStats] = []
        self.fatal_5xx = False

    def worker(self, wid: int):
        rng = random.Random(wid * 1234567 + int(self.start_time))
        while not self.stop.is_set():
            prompt = rng.choice(DEFAULT_PROMPTS)
            s = fire_request(self.args.host, self.args.model, prompt, self.args.tokens)
            with self.lock:
                self.samples.append(s)
                self.log_file.write(json.dumps({
                    "t": s.t_start,
                    "latency_ms": s.latency_ms,
                    "tokens": s.tokens,
                    "status": s.status,
                    "err": s.error,
                }) + "\n")
                if 500 <= s.status < 600:
                    self.fatal_5xx = True

    def aggregate_window(self, minute: int) -> WindowStats:
        cutoff = self.start_time + minute * 60
        prev_cutoff = self.start_time + (minute - 1) * 60
        with self.lock:
            window = [s for s in self.samples if prev_cutoff <= s.t_start < cutoff]
        if not window:
            rss = get_rss_mb(self.server_pid)
            return WindowStats(minute, 0, 0, 0.0, 0.0, 0.0, 0, rss)
        latencies = sorted(s.latency_ms for s in window)
        errors = sum(1 for s in window if s.status == 0 or s.status >= 400)
        tokens = sum(s.tokens for s in window)
        rss = get_rss_mb(self.server_pid)
        def pct(p: float) -> float:
            idx = max(0, min(len(latencies) - 1, int(round(p * (len(latencies) - 1)))))
            return latencies[idx]
        return WindowStats(
            minute=minute, n=len(window), errors=errors,
            p50_ms=pct(0.50), p95_ms=pct(0.95), p99_ms=pct(0.99),
            tokens=tokens, rss_mb=rss,
        )

    def report_window(self, w: WindowStats):
        wallclock = datetime.now().isoformat(timespec="seconds")
        self.csv_writer.writerow([
            w.minute, wallclock, w.n, w.errors,
            f"{w.p50_ms:.1f}", f"{w.p95_ms:.1f}", f"{w.p99_ms:.1f}",
            w.tokens, f"{w.rss_mb:.1f}",
        ])
        self.csv_file.flush()
        err_pct = (100.0 * w.errors / w.n) if w.n > 0 else 0.0
        rss_delta = w.rss_mb - self.initial_rss
        print(
            f"[t+{w.minute:>4}min] n={w.n:>4} err={w.errors:>3} ({err_pct:>5.1f}%)"
            f"  p50={w.p50_ms:>7.1f}ms p95={w.p95_ms:>7.1f}ms p99={w.p99_ms:>7.1f}ms"
            f"  tok={w.tokens:>5}  rss={w.rss_mb:>7.1f}MB (Δ{rss_delta:+.1f})",
            flush=True,
        )

    def check_failure(self) -> str | None:
        if self.fatal_5xx:
            return "5xx response detected"
        # Rolling 10-minute error rate
        recent = self.recent_windows[-10:]
        total_n = sum(w.n for w in recent)
        total_err = sum(w.errors for w in recent)
        if total_n >= 100 and total_err / total_n > 0.01:
            return f"error rate {100*total_err/total_n:.2f}% over last 10 min"
        # RSS growth check (only if we have a server pid)
        if self.server_pid and self.initial_rss > 0:
            current = self.recent_windows[-1].rss_mb if self.recent_windows else self.initial_rss
            if current > self.initial_rss * 1.5:
                return f"RSS grew {current/self.initial_rss:.2f}× ({self.initial_rss:.0f}→{current:.0f}MB)"
        return None

    def run(self) -> int:
        print(f"Stress harness")
        print(f"  host        : {self.args.host}")
        print(f"  model       : {self.args.model}")
        print(f"  duration    : {self.args.duration}s ({self.args.duration/3600:.1f}h)")
        print(f"  concurrency : {self.args.concurrency}")
        print(f"  tokens/req  : {self.args.tokens}")
        print(f"  server pid  : {self.server_pid or 'not detected'}")
        print(f"  initial RSS : {self.initial_rss:.1f}MB")
        print(f"  csv         : {self.csv_path}")
        print(f"  log         : {self.log_path}")
        print()

        threads = []
        for wid in range(self.args.concurrency):
            t = threading.Thread(target=self.worker, args=(wid,), daemon=True)
            t.start()
            threads.append(t)

        signal.signal(signal.SIGINT, lambda *a: self.stop.set())

        minute = 0
        deadline = self.start_time + self.args.duration
        try:
            while time.time() < deadline and not self.stop.is_set():
                next_tick = self.start_time + (minute + 1) * 60
                while time.time() < next_tick and not self.stop.is_set():
                    time.sleep(min(1.0, next_tick - time.time()))
                minute += 1
                w = self.aggregate_window(minute)
                self.recent_windows.append(w)
                self.report_window(w)
                fail = self.check_failure()
                if fail:
                    print(f"\nFATAL: {fail}", file=sys.stderr)
                    self.stop.set()
                    self.csv_file.close()
                    self.log_file.close()
                    return 1
        finally:
            self.stop.set()
            for t in threads:
                t.join(timeout=5.0)
            self.csv_file.close()
            self.log_file.close()

        print(f"\nDone. Wrote {self.csv_path}")
        return 0


def main() -> int:
    args = parse_args()
    h = Harness(args)
    return h.run()


if __name__ == "__main__":
    sys.exit(main())
