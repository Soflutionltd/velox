//! End-to-end check: dynamic-quant a BF16 weight, run qmm_4bit on
//! Metal, compare against plain BF16 matmul. Anything within ~5%
//! relative error per output cell is a passing reconstruction.

#![cfg(all(feature = "candle-metal", target_os = "macos"))]

use candle_core::{DType, Device, Tensor};
use velox::paged::dyn_quant::quantize_to_int4_groupwise;
use velox::paged::metal_kernels;

#[test]
fn dyn_quant_roundtrip_through_qmm_4bit_kernel() {
    let device = Device::new_metal(0).unwrap();
    let n = 256usize;
    let k = 1024usize;
    let m = 4usize;
    let group_size = 64usize;

    // Random-ish weight matrix: deterministic but varied.
    let mut data = Vec::with_capacity(n * k);
    for r in 0..n {
        for c in 0..k {
            // Mix a few periodic components so groups have non-trivial range.
            let f = ((r as f32 * 0.013).sin()
                + (c as f32 * 0.027).cos()
                + ((r * c) as f32 * 0.001).sin())
                / 3.0;
            data.push(f);
        }
    }
    let w_f32 = Tensor::from_vec(data, (n, k), &device).unwrap();
    let w_bf16 = w_f32.to_dtype(DType::BF16).unwrap();

    // Activations: simple ramp, again BF16 to match the quant kernel's T.
    let mut x_data = Vec::with_capacity(m * k);
    for r in 0..m {
        for c in 0..k {
            x_data.push(((r * k + c) as f32 / (m * k) as f32) - 0.5);
        }
    }
    let x_f32 = Tensor::from_vec(x_data, (m, k), &device).unwrap();
    let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();

    // Reference: y_ref = x @ w^T in BF16 (cast to F32 for comparison).
    let y_ref = x_bf16
        .matmul(&w_bf16.t().unwrap().contiguous().unwrap())
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Dynamic quant the weight, then call qmm_4bit on Metal.
    let (qw, sc, bi) = quantize_to_int4_groupwise(&w_bf16, group_size).unwrap();
    let y_q = metal_kernels::qmm_4bit(&x_bf16, &qw, &sc, &bi, None, group_size)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Compare. With group_size=64 and BF16 storage we expect mean
    // relative error well under 5%.
    let y_ref_v: Vec<f32> = y_ref.flatten_all().unwrap().to_vec1().unwrap();
    let y_q_v: Vec<f32> = y_q.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_abs = 0f32;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    for (a, b) in y_ref_v.iter().zip(y_q_v.iter()) {
        let e = (a - b).abs();
        if e > max_abs {
            max_abs = e;
        }
        sum_sq_err += (e as f64).powi(2);
        sum_sq_ref += (*a as f64).powi(2);
    }
    let rrmse = (sum_sq_err / sum_sq_ref).sqrt();
    eprintln!("dyn_quant qmm_4bit: max_abs={max_abs:.4}, rrmse={rrmse:.4}");
    assert!(
        rrmse < 0.05,
        "relative RMSE {rrmse:.4} exceeds 5%; quantization or kernel diverged"
    );
}
