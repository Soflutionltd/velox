//! Parity tests for batched_rope_decode, batched_scatter, and
//! paged_prefill_attention kernels.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use velox::paged::metal_kernels::{
    batched_rope_decode, batched_scatter, paged_decode_attention_cpu, paged_prefill_attention,
};

fn metal_device() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn assert_close(label: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_abs = 0f32;
    let mut max_idx = 0usize;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_abs {
            max_abs = diff;
            max_idx = i;
        }
    }
    let tol = atol + rtol * b[max_idx].abs();
    assert!(
        max_abs <= tol,
        "{label}: max_abs={max_abs} (idx {max_idx}: a={} b={}) > tol={tol}",
        a[max_idx],
        b[max_idx],
    );
}

// =====================================================================
// batched_rope_decode parity vs candle_nn::rotary_emb::rope per-seq
// =====================================================================

fn ref_rope_per_seq(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    offsets: &[u32],
    n: usize,
    h: usize,
    d: usize,
    half_d: usize,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for ni in 0..n {
        let off = offsets[ni] as usize;
        for hi in 0..h {
            let base = (ni * h + hi) * d;
            for dp in 0..half_d {
                let c = cos[off * half_d + dp];
                let s = sin[off * half_d + dp];
                let x1 = x[base + dp];
                let x2 = x[base + dp + half_d];
                out[base + dp] = x1 * c - x2 * s;
                out[base + dp + half_d] = x1 * s + x2 * c;
            }
        }
    }
    out
}

#[test]
fn rope_parity_bf16_qwen3() {
    let device = metal_device().unwrap();
    let n = 4;
    let h = 16;
    let d = 128;
    let half_d = 64;
    let lmax = 1024;
    let dtype = DType::BF16;

    let x_f32 = rand_vec(n * h * d, 1);
    let cos_f32 = rand_vec(lmax * half_d, 2);
    let sin_f32 = rand_vec(lmax * half_d, 3);
    let offsets: Vec<u32> = vec![0, 5, 100, 511];

    let x = Tensor::from_vec(x_f32.clone(), (n, h, d), &device).unwrap().to_dtype(dtype).unwrap();
    let cos = Tensor::from_vec(cos_f32.clone(), (lmax, half_d), &device).unwrap().to_dtype(dtype).unwrap();
    let sin = Tensor::from_vec(sin_f32.clone(), (lmax, half_d), &device).unwrap().to_dtype(dtype).unwrap();
    let off_t = Tensor::from_vec(offsets.clone(), n, &device).unwrap();

    batched_rope_decode(&x, &cos, &sin, &off_t).unwrap();

    let gpu_out = x.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Round-trip the inputs through bf16 for fair comparison.
    let x_in = Tensor::from_vec(x_f32, (n, h, d), &device).unwrap().to_dtype(dtype).unwrap()
        .to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cos_in = Tensor::from_vec(cos_f32, (lmax, half_d), &device).unwrap().to_dtype(dtype).unwrap()
        .to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let sin_in = Tensor::from_vec(sin_f32, (lmax, half_d), &device).unwrap().to_dtype(dtype).unwrap()
        .to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cpu_out = ref_rope_per_seq(&x_in, &cos_in, &sin_in, &offsets, n, h, d, half_d);

    assert_close("rope bf16 qwen3", &gpu_out, &cpu_out, 5e-2, 5e-2);
}

#[test]
fn rope_parity_f32() {
    let device = metal_device().unwrap();
    let n = 2;
    let h = 4;
    let d = 64;
    let half_d = 32;
    let lmax = 256;
    let dtype = DType::F32;

    let x_f32 = rand_vec(n * h * d, 5);
    let cos_f32 = rand_vec(lmax * half_d, 6);
    let sin_f32 = rand_vec(lmax * half_d, 7);
    let offsets: Vec<u32> = vec![3, 200];

    let x = Tensor::from_vec(x_f32.clone(), (n, h, d), &device).unwrap().to_dtype(dtype).unwrap();
    let cos = Tensor::from_vec(cos_f32.clone(), (lmax, half_d), &device).unwrap();
    let sin = Tensor::from_vec(sin_f32.clone(), (lmax, half_d), &device).unwrap();
    let off_t = Tensor::from_vec(offsets.clone(), n, &device).unwrap();

    batched_rope_decode(&x, &cos, &sin, &off_t).unwrap();

    let gpu = x.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cpu = ref_rope_per_seq(&x_f32, &cos_f32, &sin_f32, &offsets, n, h, d, half_d);
    assert_close("rope f32", &gpu, &cpu, 1e-5, 1e-5);
}

// =====================================================================
// batched_scatter parity
// =====================================================================

#[test]
fn batched_scatter_parity_bf16() {
    let device = metal_device().unwrap();
    let p = 8;
    let h = 8;
    let s = 16;
    let d = 128;
    let n = 4;
    let dtype = DType::BF16;

    let pool_f32 = rand_vec(p * h * s * d, 11);
    let val_f32 = rand_vec(n * h * d, 13);
    let page_ids: Vec<u32> = vec![0, 3, 5, 7];
    let slots: Vec<u32> = vec![0, 7, 15, 5];

    let pool = Tensor::from_vec(pool_f32.clone(), (p, h, s, d), &device).unwrap().to_dtype(dtype).unwrap();
    let values = Tensor::from_vec(val_f32.clone(), (n, h, d), &device).unwrap().to_dtype(dtype).unwrap();
    let pid = Tensor::from_vec(page_ids.clone(), n, &device).unwrap();
    let slot_t = Tensor::from_vec(slots.clone(), n, &device).unwrap();

    batched_scatter(&pool, &values, &pid, &slot_t).unwrap();

    // Reference: do the scatter on a fresh CPU-side copy of the pool.
    let pool_dtype_in = Tensor::from_vec(pool_f32.clone(), (p, h, s, d), &device).unwrap().to_dtype(dtype).unwrap()
        .to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let val_in = Tensor::from_vec(val_f32, (n, h, d), &device).unwrap().to_dtype(dtype).unwrap()
        .to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut cpu = pool_dtype_in.clone();
    for ni in 0..n {
        let pid = page_ids[ni] as usize;
        let slot = slots[ni] as usize;
        for hi in 0..h {
            for di in 0..d {
                let dst = pid * (h * s * d) + hi * (s * d) + slot * d + di;
                let src = (ni * h + hi) * d + di;
                cpu[dst] = val_in[src];
            }
        }
    }

    let gpu = pool.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(gpu.len(), cpu.len());
    for i in 0..gpu.len() {
        assert_eq!(gpu[i], cpu[i], "batched_scatter bf16 mismatch at {i}: gpu={} cpu={}", gpu[i], cpu[i]);
    }
}

// =====================================================================
// paged_prefill_attention parity vs CPU decode-style ref
// =====================================================================

/// Pure-Rust reference for prefill: same online-softmax math as the
/// kernel. Computes one query at a time.
fn ref_prefill(
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_table: &[u32],
    cu_seqlens: &[u32],
    seq_id_per_q: &[u32],
    kv_offsets: &[u32],
    total_q: usize,
    h_q: usize,
    h_kv: usize,
    head_dim: usize,
    page_size: usize,
    max_blocks: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; total_q * h_q * head_dim];
    for q_row in 0..total_q {
        let seq = seq_id_per_q[q_row] as usize;
        let q_pos = q_row - cu_seqlens[seq] as usize;
        let abs_pos = kv_offsets[seq] as usize + q_pos;
        let kv_len = abs_pos + 1;

        // Build a single-row Q for this q_row, then reuse the decode CPU ref.
        let q_off = q_row * h_q * head_dim;
        let q_one = q[q_off..q_off + h_q * head_dim].to_vec();
        // Wrap the seq's block_table into a single-row index.
        let bt_one = block_table[seq * max_blocks..seq * max_blocks + max_blocks].to_vec();
        let kv_lens_one = vec![kv_len as u32];
        let out_one = paged_decode_attention_cpu(
            &q_one,
            k_pool,
            v_pool,
            &bt_one,
            &kv_lens_one,
            1,
            h_q,
            h_kv,
            head_dim,
            page_size,
            max_blocks,
            scale,
            /*sliding_window=*/ 0,
        );
        out[q_off..q_off + h_q * head_dim].copy_from_slice(&out_one);
    }
    out
}

#[test]
fn prefill_parity_bf16_one_seq() {
    let device = metal_device().unwrap();
    let dtype = DType::BF16;
    let h_q = 16;
    let h_kv = 8;
    let head_dim = 128;
    let page_size = 16;
    let max_blocks = 4;
    let p = 8;

    // One seq with 12 new queries starting at kv_offset=0 (cold prefill).
    let new_lens = [12u32];
    let cu_seqlens: Vec<u32> = std::iter::once(0).chain(new_lens.iter().scan(0u32, |a, &x| { *a += x; Some(*a) })).collect();
    let total_q = *cu_seqlens.last().unwrap() as usize;
    let seq_id_per_q: Vec<u32> = vec![0u32; total_q];
    let kv_offsets: Vec<u32> = vec![0];

    let block_table: Vec<u32> = vec![0, 0, 0, 0]; // page 0, only 12 < page_size=16

    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_f32 = rand_vec(total_q * h_q * head_dim, 21);
    let k_f32 = rand_vec(p * h_kv * page_size * head_dim, 22);
    let v_f32 = rand_vec(p * h_kv * page_size * head_dim, 23);

    let q = Tensor::from_vec(q_f32.clone(), (total_q, h_q, head_dim), &device).unwrap().to_dtype(dtype).unwrap();
    let k = Tensor::from_vec(k_f32.clone(), (p, h_kv, page_size, head_dim), &device).unwrap().to_dtype(dtype).unwrap();
    let v = Tensor::from_vec(v_f32.clone(), (p, h_kv, page_size, head_dim), &device).unwrap().to_dtype(dtype).unwrap();
    let bt = Tensor::from_vec(block_table.clone(), (1, max_blocks), &device).unwrap();
    let cu = Tensor::from_vec(cu_seqlens.clone(), 2, &device).unwrap();
    let sid = Tensor::from_vec(seq_id_per_q.clone(), total_q, &device).unwrap();
    let kvo = Tensor::from_vec(kv_offsets.clone(), 1, &device).unwrap();

    let out = paged_prefill_attention(
        &q, &k, &v, &bt, &cu, &sid, &kvo, scale, /*sliding_window=*/ 0,
    )
    .unwrap();
    let gpu = out.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let q_in = q.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let k_in = k.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_in = v.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cpu = ref_prefill(
        &q_in, &k_in, &v_in, &block_table, &cu_seqlens, &seq_id_per_q, &kv_offsets,
        total_q, h_q, h_kv, head_dim, page_size, max_blocks, scale,
    );

    assert_close("prefill bf16 one seq", &gpu, &cpu, 3e-2, 3e-2);
}

#[test]
fn prefill_parity_bf16_mixed_batch() {
    // Three seqs: cold prefill of 8 tokens, warm continuation (kv_offset=20, 4 new),
    // and a single-token decode-style query (kv_offset=10, 1 new).
    let device = metal_device().unwrap();
    let dtype = DType::BF16;
    let h_q = 16;
    let h_kv = 8;
    let head_dim = 128;
    let page_size = 16;
    let max_blocks = 4;
    let p = 16;

    let new_lens = [8u32, 4u32, 1u32];
    let mut cu_seqlens = vec![0u32];
    let mut acc = 0u32;
    for &l in &new_lens {
        acc += l;
        cu_seqlens.push(acc);
    }
    let total_q = *cu_seqlens.last().unwrap() as usize;
    let mut seq_id_per_q = Vec::with_capacity(total_q);
    for (i, &l) in new_lens.iter().enumerate() {
        for _ in 0..l {
            seq_id_per_q.push(i as u32);
        }
    }
    let kv_offsets: Vec<u32> = vec![0, 20, 10];
    let block_table: Vec<u32> = vec![
        0, 0, 0, 0, // seq 0: 8 < 16, page 0
        2, 3, 0, 0, // seq 1: kv up to 24 → pages 2,3
        4, 0, 0, 0, // seq 2: kv up to 11 → page 4
    ];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_f32 = rand_vec(total_q * h_q * head_dim, 31);
    let k_f32 = rand_vec(p * h_kv * page_size * head_dim, 32);
    let v_f32 = rand_vec(p * h_kv * page_size * head_dim, 33);

    let q = Tensor::from_vec(q_f32, (total_q, h_q, head_dim), &device).unwrap().to_dtype(dtype).unwrap();
    let k = Tensor::from_vec(k_f32, (p, h_kv, page_size, head_dim), &device).unwrap().to_dtype(dtype).unwrap();
    let v = Tensor::from_vec(v_f32, (p, h_kv, page_size, head_dim), &device).unwrap().to_dtype(dtype).unwrap();
    let bt = Tensor::from_vec(block_table.clone(), (3, max_blocks), &device).unwrap();
    let cu = Tensor::from_vec(cu_seqlens.clone(), 4, &device).unwrap();
    let sid = Tensor::from_vec(seq_id_per_q.clone(), total_q, &device).unwrap();
    let kvo = Tensor::from_vec(kv_offsets.clone(), 3, &device).unwrap();

    let out = paged_prefill_attention(
        &q, &k, &v, &bt, &cu, &sid, &kvo, scale, /*sliding_window=*/ 0,
    )
    .unwrap();
    let gpu = out.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let q_in = q.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let k_in = k.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_in = v.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cpu = ref_prefill(
        &q_in, &k_in, &v_in, &block_table, &cu_seqlens, &seq_id_per_q, &kv_offsets,
        total_q, h_q, h_kv, head_dim, page_size, max_blocks, scale,
    );

    assert_close("prefill bf16 mixed batch", &gpu, &cpu, 3e-2, 3e-2);
}
