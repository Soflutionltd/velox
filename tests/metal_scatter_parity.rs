//! Byte-exact parity test for the custom Metal `scatter_slot` kernel
//! against the reference cat-based scatter on a single page.
//!
//! The pool layout is `[num_pages, num_kv_heads, page_size, head_dim]`;
//! we exercise multiple `(page_id, slot)` combinations for each shape.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use velox::paged::metal_kernels::ScatterSlot;

fn metal_device() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

fn make_pool(p: usize, h: usize, s: usize, d: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let n = p * h * s * d;
    let raw: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
    Tensor::from_vec(raw, (p, h, s, d), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn make_value(h: usize, d: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let n = h * d;
    let raw: Vec<f32> = (0..n).map(|i| 100.0 + (i as f32) * 0.5).collect();
    Tensor::from_vec(raw, (h, d), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

/// Reference: build a NEW pool by replacing only one (page_id, slot)
/// with `value`, all other entries unchanged. Done with index_select +
/// cat — pure Candle ops.
fn ref_scatter_pool(pool: &Tensor, value: &Tensor, page_id: usize, slot: usize) -> Result<Tensor> {
    let (_p, h, s, d) = pool.dims4()?;

    let page = pool.narrow(0, page_id, 1)?.squeeze(0)?; // [H, S, D]
    let middle = value.reshape((h, 1, d))?; // [H, 1, D]
    let new_page = if s == 1 {
        middle
    } else {
        let mut parts = Vec::with_capacity(3);
        if slot > 0 {
            parts.push(page.narrow(1, 0, slot)?);
        }
        parts.push(middle);
        if slot + 1 < s {
            parts.push(page.narrow(1, slot + 1, s - slot - 1)?);
        }
        Tensor::cat(&parts, 1)?
    };
    let new_page = new_page.unsqueeze(0)?; // [1, H, S, D]

    let p = pool.dims()[0];
    let mut parts = Vec::with_capacity(3);
    if page_id > 0 {
        parts.push(pool.narrow(0, 0, page_id)?);
    }
    parts.push(new_page);
    if page_id + 1 < p {
        parts.push(pool.narrow(0, page_id + 1, p - page_id - 1)?);
    }
    Tensor::cat(&parts, 0).map_err(Into::into)
}

fn assert_eq_tensor(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let a = a.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
    let av = a.to_vec1::<f32>()?;
    let bv = b.to_vec1::<f32>()?;
    assert_eq!(av.len(), bv.len(), "{label}: length mismatch");
    for i in 0..av.len() {
        assert_eq!(av[i], bv[i], "{label}: element {i} differs ({} vs {})", av[i], bv[i]);
    }
    Ok(())
}

fn parity_for(dtype: DType, p: usize, h: usize, s: usize, d: usize, page_id: usize, slot: usize) -> Result<()> {
    let device = metal_device()?;
    let pool_metal = make_pool(p, h, s, d, dtype, &device)?;
    let value = make_value(h, d, dtype, &device)?;

    let expected = ref_scatter_pool(&pool_metal, &value, page_id, slot)?;
    pool_metal.inplace_op2(
        &value,
        &ScatterSlot {
            page_id: page_id as u32,
            slot: slot as u32,
        },
    )?;

    assert_eq_tensor(
        &format!("{:?} p={p} h={h} s={s} d={d} page={page_id} slot={slot}", dtype),
        &pool_metal,
        &expected,
    )?;
    Ok(())
}

#[test]
fn scatter_parity_bf16_qwen3_shapes() {
    parity_for(DType::BF16, 4, 8, 16, 128, 0, 0).unwrap();
    parity_for(DType::BF16, 4, 8, 16, 128, 0, 7).unwrap();
    parity_for(DType::BF16, 4, 8, 16, 128, 0, 15).unwrap();
    parity_for(DType::BF16, 4, 8, 16, 128, 2, 5).unwrap();
    parity_for(DType::BF16, 4, 8, 16, 128, 3, 15).unwrap();
}

#[test]
fn scatter_parity_f16_misc() {
    parity_for(DType::F16, 2, 4, 8, 64, 0, 0).unwrap();
    parity_for(DType::F16, 2, 4, 8, 64, 1, 4).unwrap();
    parity_for(DType::F16, 2, 4, 8, 64, 1, 7).unwrap();
}

#[test]
fn scatter_parity_f32_small() {
    parity_for(DType::F32, 3, 2, 4, 8, 0, 0).unwrap();
    parity_for(DType::F32, 3, 2, 4, 8, 1, 1).unwrap();
    parity_for(DType::F32, 3, 2, 4, 8, 2, 3).unwrap();
}

#[test]
fn scatter_parity_single_slot_page() {
    parity_for(DType::BF16, 4, 8, 1, 128, 1, 0).unwrap();
}
