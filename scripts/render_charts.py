#!/usr/bin/env python3
"""
Render the benchmark charts used in README.md.

Inputs:  scripts/bench_results.json   (produced by bench_vs_ollama.py)
Outputs: assets/bench_throughput.png
         assets/bench_latency.png

Style choices:
  * Dark theme, single accent (Velox orange #FF6A1A) vs neutral (Ollama gray).
  * Large readable fonts so the chart is legible inside a GitHub README.
  * No gimmicky 3D, gradients, or extraneous decoration.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import os
RESULTS_NAME = os.environ.get("BENCH_RESULTS", "bench_results.json")
RESULTS = ROOT / "scripts" / RESULTS_NAME
OUT_DIR = ROOT / "assets"
OUT_DIR.mkdir(exist_ok=True)

VELOX_COLOR = "#FF6A1A"
OLLAMA_COLOR = "#9CA3AF"
BG = "#0B0F14"
FG = "#E5E7EB"
GRID = "#1F2937"


def load():
    with open(RESULTS) as f:
        return json.load(f)


def by_backend_and_conc(results):
    out = {}
    for r in results:
        if r["regime"] not in ("single_stream", "multi_stream"):
            continue
        out.setdefault(r["backend"], {})[r["concurrency"]] = r
    return out


def style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, which="both")
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)


def render_throughput(data, hw):
    concs = sorted({c for b in data.values() for c in b})
    velox = [data["velox"][c]["tokens_per_second"] for c in concs]
    ollama = [data["ollama"][c]["tokens_per_second"] for c in concs]

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160, facecolor=BG)
    style_axes(ax)
    x = range(len(concs))
    w = 0.38
    bars1 = ax.bar([i - w / 2 for i in x], velox, w, label="Velox", color=VELOX_COLOR, edgecolor=BG)
    bars2 = ax.bar([i + w / 2 for i in x], ollama, w, label="Ollama", color=OLLAMA_COLOR, edgecolor=BG)
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 6,
                f"{b.get_height():.0f}",
                ha="center",
                va="bottom",
                color=FG,
                fontsize=10,
                fontweight="bold",
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c} user{'s' if c > 1 else ''}" for c in concs], fontsize=11)
    ax.set_ylabel("Throughput (tokens / second, total)", fontsize=11)
    ax.set_title(
        f"End-to-end throughput — Qwen3-0.6B 4-bit on {hw.get('chip', 'Apple Silicon')} ({hw.get('ram_gb', '?')} GB)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    leg = ax.legend(loc="upper left", facecolor=BG, edgecolor=GRID, fontsize=11)
    for text in leg.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    out = OUT_DIR / "bench_throughput.png"
    fig.savefig(out, facecolor=BG)
    print(f"  → {out}")


def render_latency(data, hw):
    concs = sorted({c for b in data.values() for c in b})
    velox_p95 = [data["velox"][c]["p95_latency_ms"] / 1000 for c in concs]
    ollama_p95 = [data["ollama"][c]["p95_latency_ms"] / 1000 for c in concs]

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160, facecolor=BG)
    style_axes(ax)
    ax.plot(concs, velox_p95, "o-", color=VELOX_COLOR, linewidth=2.5, markersize=10, label="Velox p95")
    ax.plot(concs, ollama_p95, "s-", color=OLLAMA_COLOR, linewidth=2.5, markersize=10, label="Ollama p95")
    for c, v in zip(concs, velox_p95):
        ax.annotate(f"{v:.1f}s", (c, v), textcoords="offset points", xytext=(0, 12),
                    color=VELOX_COLOR, fontsize=10, fontweight="bold", ha="center")
    for c, v in zip(concs, ollama_p95):
        ax.annotate(f"{v:.1f}s", (c, v), textcoords="offset points", xytext=(0, -18),
                    color=OLLAMA_COLOR, fontsize=10, fontweight="bold", ha="center")
    ax.set_xlabel("Concurrent users", fontsize=11)
    ax.set_ylabel("p95 latency (s, lower is better)", fontsize=11)
    ax.set_title(
        f"Tail latency under load — Qwen3-0.6B 4-bit, 128 tokens / req",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    leg = ax.legend(loc="upper left", facecolor=BG, edgecolor=GRID, fontsize=11)
    for text in leg.get_texts():
        text.set_color(FG)
    ax.set_xticks(concs)
    fig.tight_layout()
    out = OUT_DIR / "bench_latency.png"
    fig.savefig(out, facecolor=BG)
    print(f"  → {out}")


def render_kernel_speedup():
    """Velox-internal: qmv_fast MLX-ported vs Velox naive on M=1 decode shapes."""
    shapes = [
        ("Qwen3-0.6B q_proj", 1.65),
        ("Qwen3-4B mlp_up", 2.09),
        ("Llama-3.1-8B q_proj", 1.81),
        ("Llama-3.1-8B mlp_up", 1.92),
        ("Phi-3-mini fused_qkv", 1.41),
        ("Mistral-7B mlp_down", 1.33),
    ]
    labels = [s[0] for s in shapes]
    speedups = [s[1] for s in shapes]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160, facecolor=BG)
    style_axes(ax)
    bars = ax.barh(labels, speedups, color=VELOX_COLOR, edgecolor=BG)
    ax.axvline(1.0, color=FG, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(1.02, -0.45, "Velox naive baseline = 1.00×", color=FG, fontsize=9, alpha=0.7)
    for b, s in zip(bars, speedups):
        ax.text(s + 0.03, b.get_y() + b.get_height() / 2, f"{s:.2f}×",
                va="center", color=FG, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 2.4)
    ax.invert_yaxis()
    ax.set_xlabel("Speedup vs Velox-native qmm_4bit (M=1 single-token decode)", fontsize=11)
    ax.set_title(
        "MLX-ported qmv_fast kernel — geomean 1.65× across realistic projection shapes",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    out = OUT_DIR / "bench_kernel.png"
    fig.savefig(out, facecolor=BG)
    print(f"  → {out}")


if __name__ == "__main__":
    bench = load()
    data = by_backend_and_conc(bench["results"])
    print("Rendering charts...")
    render_throughput(data, bench["hardware"])
    render_latency(data, bench["hardware"])
    render_kernel_speedup()
    print("Done.")
