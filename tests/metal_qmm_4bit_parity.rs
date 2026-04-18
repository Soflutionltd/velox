//! Parity tests for the qmm_4bit Metal kernel against the Rust CPU
//! reference, plus a sanity check against a fully-dequantized FP matmul
//! to confirm the math matches MLX's quantize/dequantize semantics.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use candle_core::{DType, Device, Tensor};
use velox::paged::metal_kernels::{qmm_4bit, qmm_4bit_cpu};

fn metal_device() -> Device {
    Device::new_metal(0).unwrap()
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Quantize an [N, K] FP weight into MLX 4-bit format.
/// Returns (qweight [N, K/8] u32, scales [N, K/g] f32, biases [N, K/g] f32).
fn quantize_mlx_4bit(w: &[f32], n: usize, k: usize, group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    assert_eq!(w.len(), n * k);
    assert_eq!(k % group_size, 0);
    assert_eq!(group_size % 8, 0);
    let k_groups = k / group_size;
    let k_packed = k / 8;
    let mut qweight = vec![0u32; n * k_packed];
    let mut scales = vec![0f32; n * k_groups];
    let mut biases = vec![0f32; n * k_groups];
    for ni in 0..n {
        for kg in 0..k_groups {
            let off = ni * k + kg * group_size;
            let group = &w[off..off + group_size];
            let mut wmin = f32::INFINITY;
            let mut wmax = f32::NEG_INFINITY;
            for &v in group {
                if v < wmin { wmin = v; }
                if v > wmax { wmax = v; }
            }
            let scale = ((wmax - wmin) / 15.0).max(1e-12);
            let bias = wmin;
            scales[ni * k_groups + kg] = scale;
            biases[ni * k_groups + kg] = bias;
            for (i, &v) in group.iter().enumerate() {
                let q_int = (((v - bias) / scale).round() as i32).clamp(0, 15) as u32;
                let pkg_idx = ni * k_packed + (kg * group_size + i) / 8;
                let shift = ((kg * group_size + i) % 8) * 4;
                qweight[pkg_idx] |= q_int << shift;
            }
        }
    }
    (qweight, scales, biases)
}

/// Reconstruct W_dequant from the MLX-quant triplets.
fn dequant_mlx_4bit(qweight: &[u32], scales: &[f32], biases: &[f32], n: usize, k: usize, group_size: usize) -> Vec<f32> {
    let k_packed = k / 8;
    let k_groups = k / group_size;
    let mut out = vec![0f32; n * k];
    for ni in 0..n {
        for kg in 0..k_groups {
            let scale = scales[ni * k_groups + kg];
            let bias = biases[ni * k_groups + kg];
            for kp in 0..(group_size / 8) {
                let pkg = qweight[ni * k_packed + kg * (group_size / 8) + kp];
                for b in 0..8 {
                    let q = ((pkg >> (b * 4)) & 0xF) as f32;
                    out[ni * k + kg * group_size + kp * 8 + b] = q * scale + bias;
                }
            }
        }
    }
    out
}

fn assert_close(label: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_err = 0f32;
    let mut max_idx = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_err {
            max_err = d;
            max_idx = i;
        }
    }
    let tol = atol + rtol * b[max_idx].abs();
    assert!(
        max_err <= tol,
        "{label}: max_err={max_err} (idx={max_idx}: a={} b={}) > tol={tol}",
        a[max_idx],
        b[max_idx],
    );
}

// =====================================================================
// 1. CPU reference matches a naive Y = X @ Wdq^T (sanity)
// =====================================================================

#[test]
fn qmm_4bit_cpu_matches_dense_matmul() {
    let m = 3;
    let n = 64;
    let k = 128;
    let g = 64;

    let x = rand_vec(m * k, 1);
    let w = rand_vec(n * k, 2);
    let (qw, sc, bi) = quantize_mlx_4bit(&w, n, k, g);
    let w_dq = dequant_mlx_4bit(&qw, &sc, &bi, n, k, g);

    let y_cpu = qmm_4bit_cpu(&x, &qw, &sc, &bi, None, m, n, k, g);

    let mut y_ref = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += x[mi * k + ki] * w_dq[ni * k + ki];
            }
            y_ref[mi * n + ni] = acc;
        }
    }

    assert_close("cpu vs dense", &y_cpu, &y_ref, 1e-3, 1e-3);
}

// =====================================================================
// 2. Metal kernel vs CPU reference (no bias)
// =====================================================================

#[test]
fn qmm_4bit_metal_vs_cpu_bf16() {
    let dev = metal_device();
    let m = 4;
    let n = 256;
    let k = 1024;
    let g = 64;

    let x = rand_vec(m * k, 11);
    let w = rand_vec(n * k, 12);
    let (qw, sc, bi) = quantize_mlx_4bit(&w, n, k, g);

    let cpu = qmm_4bit_cpu(&x, &qw, &sc, &bi, None, m, n, k, g);

    let dtype = DType::BF16;
    let x_t = Tensor::from_vec(x, (m, k), &dev).unwrap().to_dtype(dtype).unwrap();
    let qw_t = Tensor::from_vec(qw, (n, k / 8), &dev).unwrap();
    let sc_t = Tensor::from_vec(sc, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();
    let bi_t = Tensor::from_vec(bi, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();

    let y = qmm_4bit(&x_t, &qw_t, &sc_t, &bi_t, None, g).unwrap();
    let gpu = y
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_close("metal bf16 vs cpu", &gpu, &cpu, 5e-2, 5e-2);
}

#[test]
fn qmm_4bit_metal_vs_cpu_f16() {
    let dev = metal_device();
    let m = 2;
    let n = 128;
    let k = 512;
    let g = 64;

    let x = rand_vec(m * k, 21);
    let w = rand_vec(n * k, 22);
    let (qw, sc, bi) = quantize_mlx_4bit(&w, n, k, g);

    let cpu = qmm_4bit_cpu(&x, &qw, &sc, &bi, None, m, n, k, g);

    let dtype = DType::F16;
    let x_t = Tensor::from_vec(x, (m, k), &dev).unwrap().to_dtype(dtype).unwrap();
    let qw_t = Tensor::from_vec(qw, (n, k / 8), &dev).unwrap();
    let sc_t = Tensor::from_vec(sc, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();
    let bi_t = Tensor::from_vec(bi, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();

    let y = qmm_4bit(&x_t, &qw_t, &sc_t, &bi_t, None, g).unwrap();
    let gpu = y
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_close("metal f16 vs cpu", &gpu, &cpu, 1e-2, 1e-2);
}

#[test]
fn qmm_4bit_metal_vs_cpu_f32() {
    let dev = metal_device();
    let m = 1;
    let n = 64;
    let k = 128;
    let g = 64;

    let x = rand_vec(m * k, 31);
    let w = rand_vec(n * k, 32);
    let (qw, sc, bi) = quantize_mlx_4bit(&w, n, k, g);

    let cpu = qmm_4bit_cpu(&x, &qw, &sc, &bi, None, m, n, k, g);

    let dtype = DType::F32;
    let x_t = Tensor::from_vec(x, (m, k), &dev).unwrap();
    let qw_t = Tensor::from_vec(qw, (n, k / 8), &dev).unwrap();
    let sc_t = Tensor::from_vec(sc, (n, k / g), &dev).unwrap();
    let bi_t = Tensor::from_vec(bi, (n, k / g), &dev).unwrap();

    let y = qmm_4bit(&x_t, &qw_t, &sc_t, &bi_t, None, g).unwrap();
    let gpu = y.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_close("metal f32 vs cpu", &gpu, &cpu, 1e-4, 1e-4);
}

// =====================================================================
// 3. With bias vector
// =====================================================================

#[test]
fn qmm_4bit_metal_with_bias_bf16() {
    let dev = metal_device();
    let m = 2;
    let n = 128;
    let k = 512;
    let g = 64;

    let x = rand_vec(m * k, 41);
    let w = rand_vec(n * k, 42);
    let bias = rand_vec(n, 43);
    let (qw, sc, bi) = quantize_mlx_4bit(&w, n, k, g);

    let cpu = qmm_4bit_cpu(&x, &qw, &sc, &bi, Some(&bias), m, n, k, g);

    let dtype = DType::BF16;
    let x_t = Tensor::from_vec(x, (m, k), &dev).unwrap().to_dtype(dtype).unwrap();
    let qw_t = Tensor::from_vec(qw, (n, k / 8), &dev).unwrap();
    let sc_t = Tensor::from_vec(sc, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();
    let bi_t = Tensor::from_vec(bi, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();
    let b_t = Tensor::from_vec(bias, n, &dev).unwrap().to_dtype(dtype).unwrap();

    let y = qmm_4bit(&x_t, &qw_t, &sc_t, &bi_t, Some(&b_t), g).unwrap();
    let gpu = y
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_close("metal bf16 + bias", &gpu, &cpu, 5e-2, 5e-2);
}

// =====================================================================
// 4. Qwen3-shaped matmul (hidden=1024, intermediate=3072)
// =====================================================================

#[test]
fn qmm_4bit_qwen3_proj_shape() {
    let dev = metal_device();
    let m = 8; // batch
    let n = 3072; // intermediate
    let k = 1024; // hidden
    let g = 64;

    let x = rand_vec(m * k, 51);
    let w = rand_vec(n * k, 52);
    let (qw, sc, bi) = quantize_mlx_4bit(&w, n, k, g);

    let cpu = qmm_4bit_cpu(&x, &qw, &sc, &bi, None, m, n, k, g);

    let dtype = DType::BF16;
    let x_t = Tensor::from_vec(x, (m, k), &dev).unwrap().to_dtype(dtype).unwrap();
    let qw_t = Tensor::from_vec(qw, (n, k / 8), &dev).unwrap();
    let sc_t = Tensor::from_vec(sc, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();
    let bi_t = Tensor::from_vec(bi, (n, k / g), &dev).unwrap().to_dtype(dtype).unwrap();

    let y = qmm_4bit(&x_t, &qw_t, &sc_t, &bi_t, None, g).unwrap();
    let gpu = y
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Tolerances scale with K (accumulator drift in bf16)
    assert_close("metal qwen3 shape bf16", &gpu, &cpu, 0.5, 0.05);
}
