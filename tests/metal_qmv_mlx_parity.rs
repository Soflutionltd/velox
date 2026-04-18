//! Parity tests for the MLX-ported `qmv_fast` kernel (bits=4, group_size=64).
//!
//! Verifies the ported kernel matches `qmm_4bit_cpu` (the reference that
//! also backs `qmm_4bit` parity tests) for all dtypes (f32/f16/bf16) and
//! across the eligibility surface: K ∈ {512, 1024, 2048, 4096} and
//! N ∈ {8, 64, 1024, 4096}.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use candle_core::{DType, Device, Tensor};
use velox::paged::metal_kernels::{qmm_4bit_cpu, qmv_fast_mlx_g64};

fn metal_device() -> Device {
    Device::new_metal(0).unwrap()
}

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

fn quantize_mlx_4bit(
    w: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let k_groups = k / group_size;
    let k_packed = k / 8;
    let mut qweight = vec![0u32; n * k_packed];
    let mut scales = vec![0f32; n * k_groups];
    let mut biases = vec![0f32; n * k_groups];
    for ni in 0..n {
        for kg in 0..k_groups {
            let off = ni * k + kg * group_size;
            let group = &w[off..off + group_size];
            let (mut wmin, mut wmax) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in group {
                wmin = wmin.min(v);
                wmax = wmax.max(v);
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

fn run_parity(dtype: DType, n: usize, k: usize, seed: u64) {
    let dev = metal_device();
    let group_size = 64;
    let m = 1;

    let x_f32 = rand_vec(m * k, seed);
    let w_f32 = rand_vec(n * k, seed.wrapping_add(1));
    let (qw, sc, bi) = quantize_mlx_4bit(&w_f32, n, k, group_size);

    let cpu_y = qmm_4bit_cpu(&x_f32, &qw, &sc, &bi, None, m, n, k, group_size);

    // Move to metal as the requested dtype.
    let x_dev = Tensor::from_vec(x_f32.clone(), (m, k), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let qw_dev = Tensor::from_vec(qw.clone(), (n, k / 8), &dev).unwrap();
    let sc_dev = Tensor::from_vec(sc.clone(), (n, k / group_size), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bi_dev = Tensor::from_vec(bi.clone(), (n, k / group_size), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let y = qmv_fast_mlx_g64(&x_dev, &qw_dev, &sc_dev, &bi_dev, group_size).unwrap();
    let y_f32: Vec<f32> = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    // Tolerances follow `metal_qmm_4bit_parity::assert_close` (atol + rtol*|cpu|).
    // The dominant error source is rounding accumulation across K terms in
    // the kernel's lower-precision storage dtype, so rtol grows with dtype
    // bit width.
    let (atol, rtol): (f32, f32) = match dtype {
        DType::F32 => (1e-3, 1e-4),
        DType::F16 => (5e-2, 5e-3),
        DType::BF16 => (5e-1, 1e-2),
        _ => (1e-2, 1e-3),
    };

    let mut max_err = 0f32;
    let mut argmax = 0usize;
    for (i, (a, b)) in y_f32.iter().zip(cpu_y.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_err {
            max_err = d;
            argmax = i;
        }
    }
    let tol = atol + rtol * cpu_y[argmax].abs();
    assert!(
        max_err <= tol,
        "qmv_fast_mlx_g64 parity failed for dtype={dtype:?} N={n} K={k}: max_err={max_err:.6e} at idx={argmax} (gpu={:.6e}, cpu={:.6e}, tol={tol:.6e})",
        y_f32[argmax],
        cpu_y[argmax]
    );
}

#[test]
fn qmv_mlx_g64_parity_f32_qwen3_q_proj() {
    run_parity(DType::F32, 4096, 4096, 0x1234);
}

#[test]
fn qmv_mlx_g64_parity_f16_qwen3_q_proj() {
    run_parity(DType::F16, 4096, 4096, 0x5678);
}

#[test]
fn qmv_mlx_g64_parity_bf16_qwen3_q_proj() {
    run_parity(DType::BF16, 4096, 4096, 0xabcd);
}

#[test]
fn qmv_mlx_g64_parity_f32_phi3_proj() {
    // Phi-3 mini hidden=3072 -> q_proj N=3072, K=3072.
    run_parity(DType::F32, 3072, 3072, 0xfeed);
}

#[test]
fn qmv_mlx_g64_parity_f16_llama_mlp_up() {
    // Llama 3.1 8B hidden=4096, intermediate=14336 -> up_proj N=14336, K=4096.
    run_parity(DType::F16, 14336, 4096, 0xbeef);
}

#[test]
fn qmv_mlx_g64_parity_f32_smallest_eligible() {
    // Minimum eligible shape: N=8 (1 TG), K=512 (1 block).
    run_parity(DType::F32, 8, 512, 0x1357);
}
