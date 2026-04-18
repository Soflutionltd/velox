//! Sliding-window attention parity:
//!   * Verify the GPU kernel respects the window (matches the CPU ref).
//!   * Verify the window actually changes the result (not a silent no-op).
//!
//! The second check is the critical one — earlier kernels accepted a
//! `sliding_window` arg that was wired but ignored; the assertion below
//! makes sure the loop start is honored end-to-end.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use candle_core::{DType, Device, Tensor};
use velox::paged::metal_kernels::{
    paged_decode_attention, paged_decode_attention_cpu, paged_prefill_attention,
};

fn metal_device() -> anyhow::Result<Device> {
    Ok(Device::new_metal(0)?)
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761) ^ 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as f64 / u64::MAX as f64) - 0.5) as f32
        })
        .collect()
}

fn assert_close(label: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        let denom = x.abs().max(1e-6);
        max_rel = max_rel.max(d / denom);
    }
    if max_abs > atol && max_rel > rtol {
        panic!("{label}: max_abs={max_abs}, max_rel={max_rel} (atol={atol}, rtol={rtol})");
    }
}

#[test]
fn decode_swa_matches_cpu_ref_and_differs_from_global() {
    let device = match metal_device() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };

    let dtype = DType::BF16;
    let n = 1;
    let h_q = 8;
    let h_kv = 4;
    let head_dim = 64;
    let page_size = 16;
    let max_blocks = 8; // up to 128 KV positions
    let p = 8;
    let kv_len = 96u32; // exceeds the window so SWA actually clips
    let window = 32u32; // visible: positions 64..96

    let block_table: Vec<u32> = (0..max_blocks as u32).collect();
    let kv_lens = vec![kv_len];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_f32 = rand_vec(n * h_q * head_dim, 11);
    let k_f32 = rand_vec(p * h_kv * page_size * head_dim, 12);
    let v_f32 = rand_vec(p * h_kv * page_size * head_dim, 13);

    let q = Tensor::from_vec(q_f32.clone(), (n, h_q, head_dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let k = Tensor::from_vec(k_f32.clone(), (p, h_kv, page_size, head_dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let v = Tensor::from_vec(v_f32.clone(), (p, h_kv, page_size, head_dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bt = Tensor::from_vec(block_table.clone(), (n, max_blocks), &device).unwrap();
    let kvl = Tensor::from_vec(kv_lens.clone(), n, &device).unwrap();

    let out_global = paged_decode_attention(&q, &k, &v, &bt, &kvl, scale, 0)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let out_swa = paged_decode_attention(&q, &k, &v, &bt, &kvl, scale, window)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let q_in = q
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let k_in = k
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let v_in = v
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cpu_swa = paged_decode_attention_cpu(
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
        window,
    );

    // 1) GPU(SWA) must match CPU(SWA).
    assert_close("decode swa gpu vs cpu", &out_swa, &cpu_swa, 3e-2, 3e-2);

    // 2) SWA must actually change the result vs global (otherwise the
    //    loop-start arg would be silently ignored).
    let mut diff = 0f32;
    for (a, b) in out_global.iter().zip(out_swa.iter()) {
        diff = diff.max((a - b).abs());
    }
    assert!(
        diff > 1e-2,
        "sliding_window={} did not change attention output (max abs diff = {diff}) — kernel might be ignoring the arg",
        window
    );
}

#[test]
fn prefill_swa_clips_history_for_late_query() {
    let device = match metal_device() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping: no Metal device");
            return;
        }
    };

    let dtype = DType::BF16;
    let h_q = 8;
    let h_kv = 4;
    let head_dim = 64;
    let page_size = 16;
    let max_blocks = 8;
    let p = 8;

    // One seq, 4 new queries, kv_offset=80 → abs positions 80..84.
    // With window=16, positions 80..84 each see only the last 16
    // tokens from their own causal history (i.e. 64..81, 65..82, ...).
    let new_lens = [4u32];
    let cu_seqlens: Vec<u32> = std::iter::once(0)
        .chain(new_lens.iter().scan(0u32, |a, &x| {
            *a += x;
            Some(*a)
        }))
        .collect();
    let total_q = *cu_seqlens.last().unwrap() as usize;
    let seq_id_per_q: Vec<u32> = vec![0u32; total_q];
    let kv_offsets: Vec<u32> = vec![80];
    let block_table: Vec<u32> = (0..max_blocks as u32).collect();

    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_f32 = rand_vec(total_q * h_q * head_dim, 41);
    let k_f32 = rand_vec(p * h_kv * page_size * head_dim, 42);
    let v_f32 = rand_vec(p * h_kv * page_size * head_dim, 43);

    let q = Tensor::from_vec(q_f32, (total_q, h_q, head_dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let k = Tensor::from_vec(k_f32, (p, h_kv, page_size, head_dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let v = Tensor::from_vec(v_f32, (p, h_kv, page_size, head_dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bt = Tensor::from_vec(block_table.clone(), (1, max_blocks), &device).unwrap();
    let cu = Tensor::from_vec(cu_seqlens.clone(), 2, &device).unwrap();
    let sid = Tensor::from_vec(seq_id_per_q.clone(), total_q, &device).unwrap();
    let kvo = Tensor::from_vec(kv_offsets.clone(), 1, &device).unwrap();

    let out_global = paged_prefill_attention(&q, &k, &v, &bt, &cu, &sid, &kvo, scale, 0)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let out_swa = paged_prefill_attention(&q, &k, &v, &bt, &cu, &sid, &kvo, scale, 16)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut diff = 0f32;
    for (a, b) in out_global.iter().zip(out_swa.iter()) {
        diff = diff.max((a - b).abs());
    }
    assert!(
        diff > 1e-2,
        "prefill sliding_window=16 did not change output (max abs diff={diff})"
    );
}
