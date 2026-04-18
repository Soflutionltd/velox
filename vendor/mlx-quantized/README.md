# MLX quantized kernels (vendored)

Source: [ml-explore/mlx](https://github.com/ml-explore/mlx)
License: MIT (Copyright © 2023 Apple Inc.) — see `LICENSE`
Pinned commit: `fa4320d5fa4df4896640dabced1630e9b715a95b` (2026-04-18)

## What is in here

These are the original Metal Shading Language (MSL) sources for MLX's
4-bit / 6-bit / 8-bit quantized matmul/matvec kernels. They are the
reference implementation we are progressively porting into Velox's
embedded MSL backend (`src/paged/mlx_kernels.rs`).

| File | Purpose |
|------|---------|
| `kernels/quantized.metal` | Top-level instantiations (group_size × bits × dtype × variant). |
| `kernels/quantized.h`     | Kernel bodies: `qmm_n`, `qmv_fast`, `qmv`, `qvm`, `qmm_t_aligned_n_b`, `gather_qmm`, `qmm_split_k`, etc. |
| `kernels/quantized_utils.h` | Pack/unpack helpers for 4-bit interleaved storage. |

## Why vendored vs git submodule

* **Pinning by content**, not by network access — clean offline builds.
* **Audit trail**: any local edit (e.g. bridging types to Velox's runtime)
  shows up as a git diff in this directory only.
* **Single binary**: at build time `build.rs` (or the embedded MSL string
  in Rust) only consumes the kernel bodies we have ported and validated;
  unused kernels are never compiled into the Velox runtime.

## Velox port status

The full MLX kernel collection is large (~2.6 kLOC of MSL plus the
`steel/gemm` framework). We port kernels one at a time, with strict
parity tests against the existing `qmm_4bit_cpu` reference. Every
ported kernel becomes selectable via the `VELOX_QMM_BACKEND` env var
(`velox` = current homemade, `mlx` = ported Apple kernels).

| Kernel | Velox status | Used in | Speedup vs Velox-native |
|--------|--------------|---------|-------------------------|
| `qmv_fast` (M=1, the decode hot path) | **SHIPPED** (commit 2/6) | single-stream decode | **1.65× geomean** (1.33–2.09×) |
| `qmm_n` (general small M)             | TODO | multi-stream decode | — |
| `qmm_t_aligned_n_b` (tiled GEMM)      | TODO | prefill | — |
| `gather_qmm` (MoE)                    | TODO | future MoE | — |
| `affine_quantize` / `affine_dequantize` | not needed (we have our own) | load-time quantization | — |

### qmv_fast bench (2026-04-18, M-series, bf16, group_size=64, M=1)

| Shape (N × K)        | Velox naive | MLX-ported | Speedup |
|----------------------|-------------|------------|---------|
| 1024 × 1024          | 338 µs      | 255 µs     | 1.33×   |
| 3072 × 1024          | 356 µs      | 243 µs     | 1.46×   |
| 3584 × 3584 (Q3-7 q) | 586 µs      | 292 µs     | **2.00×** |
| 4096 × 4096 (L8 q)   | 579 µs      | 294 µs     | **1.97×** |
| 14336 × 4096 (L8 up) | 657 µs      | 450 µs     | 1.46×   |
| 3072 × 3072 (Phi3 q) | 572 µs      | 273 µs     | **2.09×** |
| 8192 × 3072 (Phi3 up)| 621 µs      | 345 µs     | 1.80×   |

Reproduce with `cargo test --release --features candle-metal --test metal_qmv_mlx_bench -- --nocapture --test-threads=1`.

## Updating

```
git clone --depth 1 https://github.com/ml-explore/mlx /tmp/mlx-source
cp /tmp/mlx-source/mlx/backend/metal/kernels/quantized.{h,metal} kernels/
cp /tmp/mlx-source/mlx/backend/metal/kernels/quantized_utils.h kernels/
cp /tmp/mlx-source/LICENSE LICENSE
```

Then bump the pinned commit in this README and re-run the parity tests:

```
cargo test --release -p velox metal_qmm
```

## Attribution in shipped binaries

The Velox CLI surfaces this attribution via `velox --licenses`. Do not
remove the LICENSE file from this directory.
