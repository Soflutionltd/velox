//! Microbench: MLX-ported `qmv_fast` vs Velox naive `qmm_4bit` (M=1).
//!
//! Both backends produce identical output (covered by
//! `metal_qmv_mlx_parity`). This bench measures wall-clock time for a
//! single-token decode (M=1, gs=64) across realistic projection shapes.
//! Run with:
//!
//!   cargo test --release --features candle-metal \
//!       --test metal_qmv_mlx_bench -- --nocapture --test-threads=1
//!
//! Output is informational; never fails CI.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use candle_core::{DType, Device, Tensor};
use std::time::Instant;
use velox::paged::metal_kernels::{qmm_4bit, qmv_fast_mlx_g64};

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn quantize(w: &[f32], n: usize, k: usize, g: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let kg = k / g;
    let kp = k / 8;
    let mut qw = vec![0u32; n * kp];
    let mut sc = vec![0f32; n * kg];
    let mut bi = vec![0f32; n * kg];
    for ni in 0..n {
        for gi in 0..kg {
            let off = ni * k + gi * g;
            let group = &w[off..off + g];
            let wmin = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let wmax = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((wmax - wmin) / 15.0).max(1e-12);
            sc[ni * kg + gi] = scale;
            bi[ni * kg + gi] = wmin;
            for (i, &v) in group.iter().enumerate() {
                let q = (((v - wmin) / scale).round() as i32).clamp(0, 15) as u32;
                let pkg_idx = ni * kp + (gi * g + i) / 8;
                qw[pkg_idx] |= q << (((gi * g + i) % 8) * 4);
            }
        }
    }
    (qw, sc, bi)
}

fn bench_one(
    label: &str,
    use_mlx: bool,
    dev: &Device,
    n: usize,
    k: usize,
    g: usize,
    iters: usize,
) -> f64 {
    let dtype = DType::BF16;
    let m = 1;
    let x = Tensor::from_vec(rand_vec(m * k, 1), (m, k), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let w = rand_vec(n * k, 2);
    let (qw, sc, bi) = quantize(&w, n, k, g);
    let qw_t = Tensor::from_vec(qw, (n, k / 8), dev).unwrap();
    let sc_t = Tensor::from_vec(sc, (n, k / g), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bi_t = Tensor::from_vec(bi, (n, k / g), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    // Warmup.
    for _ in 0..5 {
        let y = if use_mlx {
            qmv_fast_mlx_g64(&x, &qw_t, &sc_t, &bi_t, g).unwrap()
        } else {
            qmm_4bit(&x, &qw_t, &sc_t, &bi_t, None, g).unwrap()
        };
        let _ = y.to_device(&Device::Cpu).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let y = if use_mlx {
            qmv_fast_mlx_g64(&x, &qw_t, &sc_t, &bi_t, g).unwrap()
        } else {
            qmm_4bit(&x, &qw_t, &sc_t, &bi_t, None, g).unwrap()
        };
        let _ = y.to_device(&Device::Cpu).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let us = elapsed * 1_000_000.0 / iters as f64;
    println!("  {label:<14}  N={n:<6} K={k:<5}  {us:>8.1} µs/iter");
    us
}

#[test]
fn bench_qmv_velox_vs_mlx_decode_shapes() {
    let dev = Device::new_metal(0).unwrap();
    let g = 64;
    let iters = 200;

    // Real M=1 decode-step shapes from supported models.
    // (N, K) per model:
    //   Qwen3 0.6B   hidden=1024, intermediate=3072
    //   Qwen3 7B     hidden=3584, intermediate=18944
    //   Llama 3.1 8B hidden=4096, intermediate=14336
    //   Phi-3 mini   hidden=3072, intermediate=8192
    //   Mistral 7B   hidden=4096, intermediate=14336
    let cases: &[(&str, usize, usize)] = &[
        ("Q3-0.6 q_proj", 1024, 1024),
        ("Q3-0.6 up_proj", 3072, 1024),
        ("Q3-7   q_proj", 3584, 3584),
        ("Q3-7   up_proj", 18944, 3584),
        ("L8     q_proj", 4096, 4096),
        ("L8     up_proj", 14336, 4096),
        ("Phi3   q_proj", 3072, 3072),
        ("Phi3   up_proj", 8192, 3072),
        ("M7     up_proj", 14336, 4096),
    ];

    println!("\nqmv (M=1) Velox naive vs MLX-ported qmv_fast — bf16, group_size=64");
    let mut total_velox = 0.0f64;
    let mut total_mlx = 0.0f64;
    for &(name, n, k) in cases {
        let velox = bench_one(&format!("velox {name}"), false, &dev, n, k, g, iters);
        let mlx = bench_one(&format!("mlx   {name}"), true, &dev, n, k, g, iters);
        let speedup = velox / mlx;
        println!(
            "  -> {name:<16}  N={n:<6} K={k:<5}  speedup mlx/velox = {speedup:.2}x\n"
        );
        total_velox += velox;
        total_mlx += mlx;
    }
    println!(
        "  TOTAL across {} cases: velox={:.1} µs   mlx={:.1} µs   geomean speedup ≈ {:.2}x\n",
        cases.len(),
        total_velox,
        total_mlx,
        total_velox / total_mlx
    );
}
