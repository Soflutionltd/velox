//! Custom Metal kernels for the paged KV cache.
//!
//! v1 ships ONE kernel: `scatter_slot`, an in-place per-page write that
//! replaces a single `[H_kv, D]` slot in a `[H_kv, page_size, D]` page
//! tensor. This is the hottest write path of the paged backend (one call
//! per layer × per new token × 2 for K and V), and going from a Candle
//! `cat(pre, mid, post)` (which allocates and rebuilds the whole page) to
//! a direct GPU memcpy of `H_kv * D` elements is the single biggest win
//! for decode throughput on Apple Silicon.
//!
//! The `ScatterSlot` op implements `InplaceOp2`, so it integrates cleanly
//! with Candle's `Tensor::inplace_op2` and we keep the rest of the model
//! plain Candle.

use anyhow::{anyhow, Result as AnyResult};
use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, DType, InplaceOp2, Layout, MetalStorage, Result as CandleResult, WithDType,
};
use candle_metal_kernels::metal::{ComputePipeline, Library};
use objc2_metal::MTLSize;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Metal Shading Language source for the paged-cache kernels.
///
/// We compile one library per device, then ask for one pipeline per
/// (kernel name = dtype) on first use. Both are cached forever — the
/// library is ~a few hundred bytes of MSL and the pipeline is the
/// JIT-compiled version of one function.
const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// scatter_slot_TYPE
//
// page  : device pointer to a [H, S, D] contiguous tensor.
// value : device pointer to a [H, D] contiguous tensor.
// slot  : write position along the S dim (0..S-1).
// h_dim, s_dim, d_dim: shape of `page`.
//
// dst[h, slot, d] = value[h, d] for all h in [0,H), d in [0,D)
//
// Grid: (D, H). One thread = one element copy. We use `dispatch_threads`
// so the runtime computes the threadgroup tiling for us.

#define SCATTER_SLOT(NAME, T)                                                  \
kernel void NAME(                                                              \
    device T *page              [[buffer(0)]],                                 \
    device const T *value       [[buffer(1)]],                                 \
    constant uint &slot         [[buffer(2)]],                                 \
    constant uint &h_dim        [[buffer(3)]],                                 \
    constant uint &s_dim        [[buffer(4)]],                                 \
    constant uint &d_dim        [[buffer(5)]],                                 \
    uint2 tid                   [[thread_position_in_grid]]                    \
) {                                                                            \
    uint d = tid.x;                                                            \
    uint h = tid.y;                                                            \
    if (d >= d_dim || h >= h_dim) { return; }                                  \
    uint dst = h * (s_dim * d_dim) + slot * d_dim + d;                         \
    uint src = h * d_dim + d;                                                  \
    page[dst] = value[src];                                                    \
}

SCATTER_SLOT(scatter_slot_f32,  float)
SCATTER_SLOT(scatter_slot_f16,  half)
SCATTER_SLOT(scatter_slot_bf16, bfloat)
"#;

/// Per-process cache of compiled MSL libraries / pipelines, keyed by the
/// raw Metal `DeviceId` (since Candle wraps the device but we can fish out
/// the registry id).
///
/// Cache is `OnceLock<Mutex<HashMap<...>>>`: lazily initialised, then
/// guarded by a fast lock for inserts. Reads after the first compile per
/// (device, dtype) are O(1) hash lookups.
struct PipelineCache {
    libs: HashMap<u64, Library>,
    pipelines: HashMap<(u64, &'static str), ComputePipeline>,
}

static CACHE: OnceLock<Mutex<PipelineCache>> = OnceLock::new();

fn cache() -> &'static Mutex<PipelineCache> {
    CACHE.get_or_init(|| {
        Mutex::new(PipelineCache {
            libs: HashMap::new(),
            pipelines: HashMap::new(),
        })
    })
}

fn ensure_pipeline(
    dev: &candle_core::MetalDevice,
    fn_name: &'static str,
) -> AnyResult<ComputePipeline> {
    let registry = dev.metal_device().registry_id();
    let mut guard = cache().lock();

    if let Some(pl) = guard.pipelines.get(&(registry, fn_name)) {
        return Ok(pl.clone());
    }

    let lib = if let Some(lib) = guard.libs.get(&registry) {
        lib.clone()
    } else {
        let lib = dev
            .metal_device()
            .new_library_with_source(MSL_SOURCE, None)
            .map_err(|e| anyhow!("compile paged MSL: {e:?}"))?;
        guard.libs.insert(registry, lib.clone());
        lib
    };

    let func = lib
        .get_function(fn_name, None)
        .map_err(|e| anyhow!("get function {fn_name}: {e:?}"))?;
    let pipeline = dev
        .metal_device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow!("compile pipeline {fn_name}: {e:?}"))?;
    guard.pipelines.insert((registry, fn_name), pipeline.clone());
    Ok(pipeline)
}

fn dtype_kernel_name(dtype: DType) -> CandleResult<&'static str> {
    match dtype {
        DType::F32 => Ok("scatter_slot_f32"),
        DType::F16 => Ok("scatter_slot_f16"),
        DType::BF16 => Ok("scatter_slot_bf16"),
        other => Err(candle_core::Error::Msg(format!(
            "paged scatter_slot: unsupported dtype {other:?}"
        ))),
    }
}

/// In-place scatter of a single `[H, D]` slot into a `[H, S, D]` page.
///
/// `self` is the page and `rhs` is the value when used via
/// `Tensor::inplace_op2(value, &ScatterSlot { slot })`.
///
/// Both tensors MUST be contiguous and the page MUST be sized
/// `[h_dim, s_dim, d_dim]` with `slot < s_dim`. The cpu fallback uses
/// the layouts directly (no contiguity assumption) so it's safe to call
/// from tests with sliced views.
pub struct ScatterSlot {
    pub slot: u32,
}

impl InplaceOp2 for ScatterSlot {
    fn name(&self) -> &'static str {
        "paged_scatter_slot"
    }

    fn cpu_fwd(
        &self,
        page: &mut CpuStorage,
        page_l: &Layout,
        value: &CpuStorage,
        value_l: &Layout,
    ) -> CandleResult<()> {
        let (h, s, d) = three_dims(page_l, "page")?;
        let (vh, vd) = two_dims(value_l, "value")?;
        if h != vh || d != vd {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter: page=[{h},{s},{d}] value=[{vh},{vd}] mismatch"
            )));
        }
        let slot = self.slot as usize;
        if slot >= s {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter: slot {slot} >= s_dim {s}"
            )));
        }

        match (page, value) {
            (CpuStorage::F32(p), CpuStorage::F32(v)) => scatter_cpu(p, page_l, v, value_l, slot, h, s, d),
            (CpuStorage::F16(p), CpuStorage::F16(v)) => scatter_cpu(p, page_l, v, value_l, slot, h, s, d),
            (CpuStorage::BF16(p), CpuStorage::BF16(v)) => scatter_cpu(p, page_l, v, value_l, slot, h, s, d),
            (p, v) => Err(candle_core::Error::Msg(format!(
                "paged scatter: dtype mismatch page={:?} value={:?}",
                p.dtype(),
                v.dtype()
            ))),
        }
    }

    fn metal_fwd(
        &self,
        page: &mut MetalStorage,
        page_l: &Layout,
        value: &MetalStorage,
        value_l: &Layout,
    ) -> CandleResult<()> {
        let (h, s, d) = three_dims(page_l, "page")?;
        let (vh, vd) = two_dims(value_l, "value")?;
        if h != vh || d != vd {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter (metal): page=[{h},{s},{d}] value=[{vh},{vd}] mismatch"
            )));
        }
        if !page_l.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "paged scatter (metal): page must be contiguous".into(),
            ));
        }
        if !value_l.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "paged scatter (metal): value must be contiguous".into(),
            ));
        }
        let slot = self.slot;
        if (slot as usize) >= s {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter (metal): slot {slot} >= s_dim {s}"
            )));
        }
        if page.dtype() != value.dtype() {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter (metal): dtype mismatch page={:?} value={:?}",
                page.dtype(),
                value.dtype()
            )));
        }

        let dtype = page.dtype();
        let fn_name = dtype_kernel_name(dtype)?;
        let elem_bytes = dtype.size_in_bytes();

        let device = page.device().clone();
        let pipeline = ensure_pipeline(&device, fn_name)
            .map_err(|e| candle_core::Error::Msg(format!("paged scatter pipeline: {e}")))?;

        let page_offset = page_l.start_offset() * elem_bytes;
        let value_offset = value_l.start_offset() * elem_bytes;

        let h_dim = h as u32;
        let s_dim = s as u32;
        let d_dim = d as u32;

        let encoder = device
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("paged scatter encoder: {e}")))?;
        encoder.set_label("velox.paged.scatter_slot");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(page.buffer()), page_offset);
        encoder.set_buffer(1, Some(value.buffer()), value_offset);
        encoder.set_bytes(2, &slot);
        encoder.set_bytes(3, &h_dim);
        encoder.set_bytes(4, &s_dim);
        encoder.set_bytes(5, &d_dim);

        // dispatch_threads lets Metal pick the threadgroup tiling. Grid =
        // (D, H, 1), one thread per element to copy.
        let grid = MTLSize {
            width: d,
            height: h,
            depth: 1,
        };
        let tg_w = std::cmp::min(d, 32);
        let tg_h = std::cmp::min(h, 8);
        let threadgroup = MTLSize {
            width: tg_w.max(1),
            height: tg_h.max(1),
            depth: 1,
        };
        encoder.dispatch_threads(grid, threadgroup);
        // Note: we MUST NOT call `encoder.end_encoding()` ourselves — the
        // encoder's Drop impl does it. Calling it twice triggers an
        // assertion failure inside Metal.
        drop(encoder);
        Ok(())
    }
}

fn three_dims(l: &Layout, label: &'static str) -> CandleResult<(usize, usize, usize)> {
    let dims = l.dims();
    if dims.len() != 3 {
        return Err(candle_core::Error::Msg(format!(
            "paged scatter: {label} must be 3-D, got {dims:?}"
        )));
    }
    Ok((dims[0], dims[1], dims[2]))
}

fn two_dims(l: &Layout, label: &'static str) -> CandleResult<(usize, usize)> {
    let dims = l.dims();
    if dims.len() != 2 {
        return Err(candle_core::Error::Msg(format!(
            "paged scatter: {label} must be 2-D, got {dims:?}"
        )));
    }
    Ok((dims[0], dims[1]))
}

fn scatter_cpu<T: WithDType + Copy>(
    page: &mut [T],
    page_l: &Layout,
    value: &[T],
    value_l: &Layout,
    slot: usize,
    h: usize,
    s: usize,
    d: usize,
) -> CandleResult<()> {
    let p_strides = page_l.stride();
    let v_strides = value_l.stride();
    let p_off = page_l.start_offset();
    let v_off = value_l.start_offset();
    if p_strides.len() != 3 || v_strides.len() != 3.min(v_strides.len()) {
        // Defensive: layouts always match dims().len(), this is just to make
        // sure we don't go out of bounds in case Candle ever changes the
        // invariant.
        return Err(candle_core::Error::Msg(
            "paged scatter cpu: bad strides".into(),
        ));
    }
    for hh in 0..h {
        for dd in 0..d {
            let dst = p_off + hh * p_strides[0] + slot * p_strides[1] + dd * p_strides[2];
            let src = v_off + hh * v_strides[0] + dd * v_strides[1];
            page[dst] = value[src];
        }
    }
    let _ = s;
    Ok(())
}
