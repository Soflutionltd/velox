//! Byte-exact parity test between the custom Metal `scatter_slot` kernel
//! and the previous Candle `cat(pre, mid, post)` implementation.
//!
//! We build a known page tensor, run BOTH paths, then compare the results
//! element-by-element. A single mismatch fails the test.
//!
//! This guards against silent numerical drift from the kernel — the
//! decode loop calls `scatter` thousands of times per request, so even a
//! tiny rounding difference would compound.

#![cfg(all(target_os = "macos", feature = "candle-metal"))]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use velox::paged::metal_kernels::ScatterSlot;

fn devices() -> Result<(Device, Device)> {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
    Ok((cpu, metal))
}

/// Build a deterministic `[H, S, D]` page filled with strided integers.
fn make_page(h: usize, s: usize, d: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let n = h * s * d;
    let raw: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    Tensor::from_vec(raw, (h, s, d), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

/// Build a deterministic `[H, D]` value tensor.
fn make_value(h: usize, d: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let n = h * d;
    let raw: Vec<f32> = (0..n).map(|i| 100.0 + (i as f32) * 0.5).collect();
    Tensor::from_vec(raw, (h, d), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

/// Reference implementation: the old cat-based scatter.
fn ref_scatter(page: &Tensor, value: &Tensor, slot: usize) -> Result<Tensor> {
    let (h, s, d) = page.dims3()?;
    let middle = value.reshape((h, 1, d))?;
    if s == 1 {
        return Ok(middle);
    }
    let mut parts = Vec::with_capacity(3);
    if slot > 0 {
        parts.push(page.narrow(1, 0, slot)?);
    }
    parts.push(middle);
    if slot + 1 < s {
        parts.push(page.narrow(1, slot + 1, s - slot - 1)?);
    }
    Tensor::cat(&parts, 1).map_err(Into::into)
}

fn assert_eq_tensor(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let a = a.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
    let av = a.to_vec1::<f32>()?;
    let bv = b.to_vec1::<f32>()?;
    assert_eq!(
        av.len(),
        bv.len(),
        "{label}: length mismatch {} vs {}",
        av.len(),
        bv.len()
    );
    for i in 0..av.len() {
        assert_eq!(av[i], bv[i], "{label}: element {i} differs ({} vs {})", av[i], bv[i]);
    }
    Ok(())
}

fn parity_for(dtype: DType, h: usize, s: usize, d: usize, slot: usize) -> Result<()> {
    let (cpu, metal) = devices()?;

    let mut metal_page = make_page(h, s, d, dtype, &metal)?;
    let value_metal = make_value(h, d, dtype, &metal)?;

    let cpu_page = make_page(h, s, d, dtype, &cpu)?;
    let value_cpu = make_value(h, d, dtype, &cpu)?;

    metal_page.inplace_op2(&value_metal, &ScatterSlot { slot: slot as u32 })?;

    let cpu_result = ref_scatter(&cpu_page, &value_cpu, slot)?;

    assert_eq_tensor(
        &format!("{:?} h={h} s={s} d={d} slot={slot}", dtype),
        &metal_page,
        &cpu_result,
    )?;
    Ok(())
}

#[test]
fn scatter_parity_bf16_qwen3_shapes() {
    // Real shapes used by the Qwen3-0.6B paged backend.
    parity_for(DType::BF16, 8, 16, 128, 0).unwrap();
    parity_for(DType::BF16, 8, 16, 128, 7).unwrap();
    parity_for(DType::BF16, 8, 16, 128, 15).unwrap();
}

#[test]
fn scatter_parity_f16_misc_shapes() {
    parity_for(DType::F16, 4, 8, 64, 0).unwrap();
    parity_for(DType::F16, 4, 8, 64, 4).unwrap();
    parity_for(DType::F16, 4, 8, 64, 7).unwrap();
}

#[test]
fn scatter_parity_f32_small() {
    parity_for(DType::F32, 2, 4, 8, 0).unwrap();
    parity_for(DType::F32, 2, 4, 8, 1).unwrap();
    parity_for(DType::F32, 2, 4, 8, 3).unwrap();
}

#[test]
fn scatter_parity_single_slot_page() {
    // Edge case: page_size == 1.
    parity_for(DType::BF16, 8, 1, 128, 0).unwrap();
}
