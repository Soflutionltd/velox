//! Parity test: fused Metal `paged_decode_attention` vs the pure-Rust
//! reference implementation. Both consume the same bit-identical inputs
//! (we cast to BF16 once on the GPU side and once on the CPU side via a
//! round-trip) so any diff comes from the kernel's online-softmax math
//! itself.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use velox::paged::metal_kernels::{paged_decode_attention, paged_decode_attention_cpu};

fn metal_device() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    // tiny LCG so we don't pull rand into dev-deps
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], atol: f32, rtol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_idx = 0usize;
    for i in 0..gpu.len() {
        let diff = (gpu[i] - cpu[i]).abs();
        let denom = cpu[i].abs().max(1e-3);
        let rel = diff / denom;
        if diff > max_abs {
            max_abs = diff;
            max_idx = i;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let tol = atol + rtol * cpu[max_idx].abs();
    assert!(
        max_abs <= tol,
        "{label}: max_abs={max_abs} (idx {max_idx}: gpu={} cpu={}) > tol={tol}, max_rel={max_rel}",
        gpu[max_idx],
        cpu[max_idx],
    );
}

fn run_parity(
    dtype: DType,
    n: usize,
    h_q: usize,
    h_kv: usize,
    head_dim: usize,
    p: usize,
    page_size: usize,
    max_blocks: usize,
    kv_lens: Vec<u32>,
    block_table: Vec<u32>,
    seed: u64,
) -> Result<()> {
    assert_eq!(kv_lens.len(), n);
    assert_eq!(block_table.len(), n * max_blocks);

    let device = metal_device()?;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Build inputs in F32 first so the CPU and GPU paths see the same
    // values after a single dtype cast.
    let q_f32 = rand_vec(n * h_q * head_dim, seed);
    let k_f32 = rand_vec(p * h_kv * page_size * head_dim, seed.wrapping_add(1));
    let v_f32 = rand_vec(p * h_kv * page_size * head_dim, seed.wrapping_add(2));

    let q_dev = Tensor::from_vec(q_f32.clone(), (n, h_q, head_dim), &device)?.to_dtype(dtype)?;
    let k_dev =
        Tensor::from_vec(k_f32.clone(), (p, h_kv, page_size, head_dim), &device)?.to_dtype(dtype)?;
    let v_dev =
        Tensor::from_vec(v_f32.clone(), (p, h_kv, page_size, head_dim), &device)?.to_dtype(dtype)?;
    let bt_dev = Tensor::from_vec(block_table.clone(), (n, max_blocks), &device)?;
    let kvl_dev = Tensor::from_vec(kv_lens.clone(), n, &device)?;

    let out_metal = paged_decode_attention(
        &q_dev, &k_dev, &v_dev, &bt_dev, &kvl_dev, scale, /*sliding_window=*/ 0,
    )?;
    let out_metal_f32 = out_metal
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Round-trip the inputs through the same dtype so the CPU reference
    // sees the same quantisation as the GPU side. This is what we actually
    // want to validate: the kernel maths, modulo dtype rounding.
    let q_in = q_dev
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let k_in = k_dev
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let v_in = v_dev
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let out_cpu = paged_decode_attention_cpu(
        &q_in,
        &k_in,
        &v_in,
        &block_table,
        &kv_lens,
        n,
        h_q,
        h_kv,
        head_dim,
        page_size,
        max_blocks,
        scale,
        /*sliding_window=*/ 0,
    );

    let (atol, rtol) = match dtype {
        DType::F32 => (1e-4, 1e-4),
        DType::F16 => (5e-3, 5e-3),
        DType::BF16 => (3e-2, 3e-2),
        _ => unreachable!(),
    };
    assert_close(
        &format!(
            "{:?} N={n} H_q={h_q} H_kv={h_kv} D={head_dim} P={p} S={page_size}",
            dtype
        ),
        &out_metal_f32,
        &out_cpu,
        atol,
        rtol,
    );
    Ok(())
}

#[test]
fn decode_parity_f32_single_seq_short() {
    run_parity(
        DType::F32,
        /*n=*/ 1,
        /*h_q=*/ 4,
        /*h_kv=*/ 2,
        /*head_dim=*/ 32,
        /*p=*/ 4,
        /*page_size=*/ 4,
        /*max_blocks=*/ 2,
        /*kv_lens=*/ vec![5],
        /*block_table=*/ vec![0, 1],
        42,
    )
    .unwrap();
}

#[test]
fn decode_parity_f16_two_seqs_partial_pages() {
    run_parity(
        DType::F16,
        2,
        8,
        2,
        64,
        8,
        16,
        3,
        vec![10, 21],
        // seq 0 → pages [3,5,_]; seq 1 → pages [0,1,2]
        vec![3, 5, 0, 0, 1, 2],
        7,
    )
    .unwrap();
}

#[test]
fn decode_parity_bf16_qwen3_like() {
    // Qwen3-0.6B: H_q=16 H_kv=8 D=128. Use page_size=16, kv_len ~ 64.
    run_parity(
        DType::BF16,
        4,
        16,
        8,
        128,
        16,
        16,
        5,
        vec![16, 32, 48, 64],
        vec![
            0, 1, 0, 0, 0, // seq0 (1 page)
            2, 3, 0, 0, 0, // seq1 (2 pages)
            4, 5, 6, 0, 0, // seq2 (3 pages)
            7, 8, 9, 10, 0, // seq3 (4 pages)
        ],
        13,
    )
    .unwrap();
}

#[test]
fn decode_parity_bf16_decoded_long_context() {
    // Mimics steady state: 8 seqs each with kv_len ~ 256 across 16 pages.
    let n = 8;
    let max_blocks = 16;
    let mut block_table = vec![0u32; n * max_blocks];
    for s in 0..n {
        for b in 0..max_blocks {
            block_table[s * max_blocks + b] = ((s * max_blocks + b) % 64) as u32;
        }
    }
    let kv_lens = vec![240u32; n];
    run_parity(
        DType::BF16,
        n,
        16,
        8,
        128,
        64,
        16,
        max_blocks,
        kv_lens,
        block_table,
        99,
    )
    .unwrap();
}

#[test]
fn decode_parity_bf16_kv_len_one() {
    // Edge case: only 1 KV position (the very first decode token).
    run_parity(
        DType::BF16,
        2,
        16,
        8,
        128,
        4,
        16,
        1,
        vec![1, 1],
        vec![0, 1],
        3,
    )
    .unwrap();
}
