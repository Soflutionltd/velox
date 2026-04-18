//! Microbench: qmm_4bit naive (v1) vs multi-SIMD shared-X (v2).
//!
//! Forces each path via `VELOX_QMM_V2={0,1}` and measures GPU wall time
//! over a few hot iterations on Qwen3-shape projections. Run with:
//!
//!   cargo test --release --features candle-metal \
//!       --test metal_qmm_v2_bench -- --nocapture --test-threads=1
//!
//! Prints a table; not gated on a perf threshold so it never fails CI.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use candle_core::{DType, Device, Tensor};
use std::time::Instant;
use velox::paged::metal_kernels::qmm_4bit;

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
    assert_eq!(w.len(), n * k);
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

fn time_qmm(
    label: &str,
    use_v2: bool,
    dev: &Device,
    m: usize,
    n: usize,
    k: usize,
    g: usize,
    iters: usize,
) -> f64 {
    if use_v2 {
        std::env::set_var("VELOX_QMM_V2", "1");
    } else {
        std::env::set_var("VELOX_QMM_V2", "0");
    }
    let dtype = DType::BF16;
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
    for _ in 0..3 {
        let y = qmm_4bit(&x, &qw_t, &sc_t, &bi_t, None, g).unwrap();
        let _ = y.to_device(&Device::Cpu).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let y = qmm_4bit(&x, &qw_t, &sc_t, &bi_t, None, g).unwrap();
        let _ = y.to_device(&Device::Cpu).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_ms = elapsed * 1000.0 / iters as f64;
    println!("  {label:<10}  M={m:<4} N={n:<5} K={k:<5}  {avg_ms:>8.3} ms/iter");
    avg_ms
}

#[test]
fn bench_qmm_v1_vs_v2_qwen3_shapes() {
    let dev = Device::new_metal(0).unwrap();
    let g = 64;
    let iters = 20;

    // Qwen3 0.6B / Llama-style projection shapes (hidden=1024,
    // intermediate=3072 typical). Sweep M to see where v2 takes over.
    // v2 kernel requires M to be a multiple of BM_V2 = 32, so the bench
    // sticks to those shapes. (At M < 32 the host falls back to naive
    // and there is nothing to compare.)
    let cases: &[(usize, usize, usize)] = &[
        (32, 3072, 1024),   // medium batch decode, MLP up
        (32, 1024, 1024),   // attn projection batch
        (64, 3072, 1024),   // larger batch
        (128, 3072, 1024),  // prefill chunk, MLP up
        (128, 1024, 1024),  // prefill chunk, attn proj
        (256, 3072, 1024),  // big prefill chunk
        (256, 1024, 4096),  // narrow prefill (gate proj for 0.6B variant)
    ];

    println!("\nqmm_4bit v1 (naive) vs v2 (multi-SIMD + shared X)");
    for &(m, n, k) in cases {
        let v1 = time_qmm("v1.naive", false, &dev, m, n, k, g, iters);
        let v2 = time_qmm("v2.tiled", true, &dev, m, n, k, g, iters);
        let speedup = v1 / v2;
        println!(
            "  -> M={m:<4} N={n:<5} K={k:<5}  speedup v2/v1 = {:.2}x\n",
            speedup
        );
    }
}
