//! Dynamic quantization at load time.
//!
//! Takes a BF16/F16/F32 weight matrix `[N, K]` and produces the same
//! three buffers an MLX-Int4 checkpoint would carry:
//!
//!   qweight : [N, K/8]            U32, 8 packed Int4 per u32 (little-endian)
//!   scales  : [N, K/group_size]   same dtype as input
//!   biases  : [N, K/group_size]   same dtype as input (== group min)
//!
//! Group-wise affine quant — matches `mlx.core.quantize`:
//!
//!   scale = (w_max - w_min) / 15
//!   bias  = w_min
//!   q     = round((w - bias) / scale).clamp(0, 15) as u4
//!
//! Dequant (used by our `qmm_4bit` kernel):
//!
//!   w = q * scale + bias
//!
//! The point of doing this at load time (rather than only loading
//! pre-quantized MLX checkpoints) is twofold:
//!
//! 1. Coverage. Lots of useful checkpoints on the Hub ship only in
//!    BF16 or F16 (e.g. the original Qwen3 / Llama / Mistral repos
//!    rather than the `mlx-community/...-4bit` mirrors).
//! 2. Memory. A 7B model in BF16 is ~14 GB; quantized to int4 it's
//!    ~3.5 GB plus group metadata. On a 16 GB Mac this is the
//!    difference between OOM and serving.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};

/// Group-wise affine quantize a 2D weight matrix to MLX Int4 layout.
///
/// `weight`: `[out_features, in_features]`, must be on the same device
/// the resulting tensors will live on. Any of F32/F16/BF16 accepted.
///
/// `group_size`: MLX default is 64. Must divide `in_features` AND be
/// a multiple of 8 (the qmm_4bit kernel's u32 unpack chunk).
///
/// Returns `(qweight, scales, biases)`:
///   * `qweight`: `[out_features, in_features/8]` U32
///   * `scales` : `[out_features, in_features/group_size]` same dtype as input
///   * `biases` : `[out_features, in_features/group_size]` same dtype as input
pub fn quantize_to_int4_groupwise(
    weight: &Tensor,
    group_size: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = weight.dims();
    if dims.len() != 2 {
        return Err(anyhow!(
            "expected 2D weight, got shape {dims:?}"
        ));
    }
    let (n, k) = (dims[0], dims[1]);
    if k % group_size != 0 {
        return Err(anyhow!(
            "in_features {k} not divisible by group_size {group_size}"
        ));
    }
    if group_size % 8 != 0 {
        return Err(anyhow!(
            "group_size {group_size} must be a multiple of 8 (kernel unpack chunk)"
        ));
    }

    let device = weight.device().clone();
    let out_dtype = weight.dtype();
    let n_groups = k / group_size;

    // Compute everything in F32 for numerical accuracy on the
    // min/max + clamp pass, then cast scales/biases back to the
    // weight's storage dtype to match the qmm_4bit kernel.
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_g = w_f32.reshape((n, n_groups, group_size))?;

    // Per-group min/max along the last axis.
    let w_min = w_g.min_keepdim(2)?; // [N, G, 1]
    let w_max = w_g.max_keepdim(2)?; // [N, G, 1]
    let range = (&w_max - &w_min)?;
    // scale = range / 15. Replace zero-range groups with 1.0 so
    // we don't divide by zero — those groups will quantize to 0
    // anyway since w == w_min everywhere.
    let fifteen = Tensor::new(15f32, &device)?;
    let raw_scale = range.broadcast_div(&fifteen)?;
    let zero = Tensor::new(0f32, &device)?;
    let one = Tensor::new(1f32, &device)?;
    let safe_scale = {
        let mask = raw_scale.broadcast_eq(&zero)?;
        // Where scale==0 use 1.0, else raw_scale.
        mask.where_cond(&one.broadcast_as(raw_scale.shape())?, &raw_scale)?
    };

    // q = round((w - w_min) / scale).clamp(0, 15)
    let centered = w_g.broadcast_sub(&w_min)?;
    let normalized = centered.broadcast_div(&safe_scale)?;
    let q_round = (normalized + 0.5f64)?.floor()?; // Rust round-half-up
    let zero_v = Tensor::new(0f32, &device)?;
    let fifteen_v = Tensor::new(15f32, &device)?;
    let q_clamped = q_round
        .broadcast_maximum(&zero_v)?
        .broadcast_minimum(&fifteen_v)?;
    // Cast to u32 for packing. We need values in [0, 15] which fit.
    let q_u32 = q_clamped.to_dtype(DType::U32)?; // [N, G, group_size]

    // Reshape to [N, K] then pack 8 nibbles per u32 little-endian.
    let q_flat = q_u32.reshape((n, k))?; // [N, K]
    let qweight = pack_int4_into_u32(&q_flat)?; // [N, K/8] U32

    // scales / biases back to original dtype, shape [N, G].
    let scales = safe_scale
        .reshape((n, n_groups))?
        .to_dtype(out_dtype)?;
    let biases = w_min
        .reshape((n, n_groups))?
        .to_dtype(out_dtype)?;

    Ok((qweight, scales, biases))
}

/// Pack an `[N, K]` U32 tensor of values in `[0, 15]` into an
/// `[N, K/8]` U32 tensor where each output element holds 8 nibbles
/// little-endian. Implemented on the host because Candle doesn't
/// have a `pack_int4` op and this only runs once at load time.
fn pack_int4_into_u32(q: &Tensor) -> Result<Tensor> {
    let dims = q.dims();
    let (n, k) = (dims[0], dims[1]);
    if k % 8 != 0 {
        return Err(anyhow!("pack: k {k} not multiple of 8"));
    }
    let device = q.device().clone();
    let host: Vec<u32> = q.flatten_all()?.to_vec1::<u32>()?;
    let mut packed = Vec::with_capacity(n * k / 8);
    for row in 0..n {
        let row_off = row * k;
        let mut col = 0;
        while col < k {
            let p = (host[row_off + col] & 0xF)
                | ((host[row_off + col + 1] & 0xF) << 4)
                | ((host[row_off + col + 2] & 0xF) << 8)
                | ((host[row_off + col + 3] & 0xF) << 12)
                | ((host[row_off + col + 4] & 0xF) << 16)
                | ((host[row_off + col + 5] & 0xF) << 20)
                | ((host[row_off + col + 6] & 0xF) << 24)
                | ((host[row_off + col + 7] & 0xF) << 28);
            packed.push(p);
            col += 8;
        }
    }
    Ok(Tensor::from_vec(packed, (n, k / 8), &device)?)
}

/// Sanity-check helper: dequantize an int4 weight back to the input
/// dtype using the same formula as the Metal kernel. Useful for
/// numerical-accuracy tests in CPU-only contexts where we can't
/// compare against the kernel directly.
#[cfg(test)]
pub fn dequantize_int4_groupwise(
    qweight: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let dims = qweight.dims();
    let (n, k_packed) = (dims[0], dims[1]);
    let k = k_packed * 8;
    let device = qweight.device().clone();
    let dtype = scales.dtype();

    let q_host: Vec<u32> = qweight.flatten_all()?.to_vec1::<u32>()?;
    let mut unpacked = Vec::with_capacity(n * k);
    for v in q_host {
        unpacked.push((v) & 0xF);
        unpacked.push((v >> 4) & 0xF);
        unpacked.push((v >> 8) & 0xF);
        unpacked.push((v >> 12) & 0xF);
        unpacked.push((v >> 16) & 0xF);
        unpacked.push((v >> 20) & 0xF);
        unpacked.push((v >> 24) & 0xF);
        unpacked.push((v >> 28) & 0xF);
    }
    let q_t = Tensor::from_vec(unpacked, (n, k), &device)?
        .to_dtype(DType::F32)?;
    let s_f32 = scales.to_dtype(DType::F32)?;
    let b_f32 = biases.to_dtype(DType::F32)?;
    let s_b = s_f32
        .unsqueeze(2)?
        .broadcast_as((n, k / group_size, group_size))?
        .reshape((n, k))?;
    let b_b = b_f32
        .unsqueeze(2)?
        .broadcast_as((n, k / group_size, group_size))?
        .reshape((n, k))?;
    Ok((q_t * s_b)?.add(&b_b)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let device = cpu();
        // 1×16 ramp 0..15 should pack into 0xFEDCBA9876543210
        let q = Tensor::from_vec(
            (0u32..16).collect::<Vec<_>>(),
            (1, 16),
            &device,
        )
        .unwrap();
        let p = pack_int4_into_u32(&q).unwrap();
        let host: Vec<u32> = p.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(host, vec![0x76543210, 0xFEDCBA98]);
    }

    #[test]
    fn quantize_then_dequantize_recovers_signal() {
        let device = cpu();
        // 2 rows of 64 values each (1 group per row at gs=64).
        // Row 0: linear ramp -1..1.
        // Row 1: bell shape.
        let n = 2usize;
        let k = 64usize;
        let mut data = Vec::with_capacity(n * k);
        for i in 0..k {
            let x = (i as f32 / (k - 1) as f32) * 2.0 - 1.0;
            data.push(x);
        }
        for i in 0..k {
            let x = ((i as f32 - 32.0) / 16.0).powi(2);
            data.push((-x).exp());
        }
        let w = Tensor::from_vec(data.clone(), (n, k), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let (qw, sc, bi) = quantize_to_int4_groupwise(&w, 64).unwrap();
        assert_eq!(qw.dims(), &[2, 8]);
        assert_eq!(sc.dims(), &[2, 1]);
        assert_eq!(bi.dims(), &[2, 1]);
        let recovered = dequantize_int4_groupwise(&qw, &sc, &bi, 64).unwrap();
        let recovered_v: Vec<f32> = recovered.flatten_all().unwrap().to_vec1().unwrap();
        // Per-element max abs error should be below scale (= range/15).
        // For the ramp, range = 2.0 → scale = 0.133. Error bound 0.07.
        let mut max_err = 0f32;
        for (a, b) in data.iter().zip(recovered_v.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 0.07, "max err too large: {max_err}");
    }

    #[test]
    fn flat_group_yields_zero_quants_no_nans() {
        let device = cpu();
        // A row that's entirely 3.14 → degenerate case (range==0).
        let w = Tensor::from_vec(vec![3.14f32; 64], (1, 64), &device)
            .unwrap();
        let (qw, sc, bi) = quantize_to_int4_groupwise(&w, 64).unwrap();
        let recovered = dequantize_int4_groupwise(&qw, &sc, &bi, 64).unwrap();
        let v: Vec<f32> = recovered.flatten_all().unwrap().to_vec1().unwrap();
        for x in v {
            assert!((x - 3.14).abs() < 1e-3, "got {x}");
        }
    }
}
