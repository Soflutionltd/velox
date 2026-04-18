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

use anyhow::{anyhow, bail, Result as AnyResult};
use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, DType, Device, InplaceOp2, Layout, MetalStorage, Result as CandleResult, Storage,
    Tensor, WithDType,
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
// pool   : device pointer to a [P, H, S, D] contiguous tensor (full
//          per-layer KV pool).
// value  : device pointer to a [H, D] contiguous tensor.
// page_id: which page in the pool to write.
// slot   : write position along the S dim (0..S-1).
// h_dim, s_dim, d_dim: per-page shape.
//
// dst[page_id, h, slot, d] = value[h, d] for all h in [0,H), d in [0,D)
//
// Grid: (D, H). One thread = one element copy.

#define SCATTER_SLOT(NAME, T)                                                  \
kernel void NAME(                                                              \
    device T *pool              [[buffer(0)]],                                 \
    device const T *value       [[buffer(1)]],                                 \
    constant uint &page_id      [[buffer(2)]],                                 \
    constant uint &slot         [[buffer(3)]],                                 \
    constant uint &h_dim        [[buffer(4)]],                                 \
    constant uint &s_dim        [[buffer(5)]],                                 \
    constant uint &d_dim        [[buffer(6)]],                                 \
    uint2 tid                   [[thread_position_in_grid]]                    \
) {                                                                            \
    uint d = tid.x;                                                            \
    uint h = tid.y;                                                            \
    if (d >= d_dim || h >= h_dim) { return; }                                  \
    uint dst = page_id * (h_dim * s_dim * d_dim)                               \
             + h * (s_dim * d_dim)                                             \
             + slot * d_dim                                                    \
             + d;                                                              \
    uint src = h * d_dim + d;                                                  \
    pool[dst] = value[src];                                                    \
}

SCATTER_SLOT(scatter_slot_f32,  float)
SCATTER_SLOT(scatter_slot_f16,  half)
SCATTER_SLOT(scatter_slot_bf16, bfloat)

// ----- paged_decode_attention -----------------------------------------
//
// One fused kernel that does, for a batch of N decoding sequences, the
// full per-token attention against the layer's paged KV pool:
//
//   Out[seq, h_q, :] =
//       softmax( scale * Q[seq, h_q, :] @ gather_K(seq, h_kv, :, :)^T )
//       @ gather_V(seq, h_kv, :, :)
//
// where `gather_K(seq, h_kv, :, :)` walks the seq's block_table,
// concatenating the live KV positions across pages.
//
// Tensor layouts (all contiguous, all same dtype T except the integer
// tables which are u32):
//   Q             : [N, H_q, D]
//   K, V          : [P, H_kv, S, D]
//   block_tables  : [N, MB]            uint
//   kv_lens       : [N]                uint        (number of valid KV positions per seq)
//   Out           : [N, H_q, D]
//
// Threadgroup grid = (N, H_q, 1). Threads per group = D (we currently
// require D <= 256 and D divisible by 32 — fits Qwen3, Llama, Mistral,
// Phi, Gemma, Qwen2). Each thread owns one element of the output along
// the head dim.
//
// The softmax is computed online (Flash-style): we maintain a running
// max `m`, exp-sum `l`, and accumulator `out_acc[d]` per kv position,
// updating them with `alpha = exp(m_old - m_new)` and
// `beta = exp(s - m_new)`. This avoids materialising the full kv_len-
// long score vector in threadgroup memory.

#define D_MAX 256

#define PAGED_DECODE_ATTN(NAME, T)                                             \
kernel void NAME(                                                              \
    device const T    *Q              [[buffer(0)]],                           \
    device const T    *K              [[buffer(1)]],                           \
    device const T    *V              [[buffer(2)]],                           \
    device const uint *block_tables   [[buffer(3)]],                           \
    device const uint *kv_lens        [[buffer(4)]],                           \
    device       T    *Out            [[buffer(5)]],                           \
    constant float &scale             [[buffer(6)]],                           \
    constant uint  &num_q_heads       [[buffer(7)]],                           \
    constant uint  &num_kv_heads      [[buffer(8)]],                           \
    constant uint  &head_dim          [[buffer(9)]],                           \
    constant uint  &page_size         [[buffer(10)]],                          \
    constant uint  &max_blocks        [[buffer(11)]],                          \
    uint3 tg_id                       [[threadgroup_position_in_grid]],        \
    uint3 tid_v                       [[thread_position_in_threadgroup]]       \
) {                                                                            \
    uint seq_id = tg_id.x;                                                     \
    uint h_q    = tg_id.y;                                                     \
    uint tid    = tid_v.x;                                                     \
    uint kv_groups = num_q_heads / num_kv_heads;                               \
    uint h_kv   = h_q / kv_groups;                                             \
    uint kv_len = kv_lens[seq_id];                                             \
                                                                               \
    threadgroup float q_shared[D_MAX];                                         \
    threadgroup float scratch[D_MAX];                                          \
                                                                               \
    if (tid < head_dim) {                                                      \
        q_shared[tid] = (float)Q[(seq_id * num_q_heads + h_q) * head_dim       \
                                 + tid];                                       \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
                                                                               \
    float out_acc = 0.0;                                                       \
    float m = -INFINITY;                                                       \
    float l = 0.0;                                                             \
                                                                               \
    for (uint pos = 0; pos < kv_len; pos++) {                                  \
        uint page_idx = pos / page_size;                                       \
        uint slot     = pos - page_idx * page_size;                            \
        uint page_id  = block_tables[seq_id * max_blocks + page_idx];          \
        uint base = page_id * (num_kv_heads * page_size * head_dim)            \
                  + h_kv    * (page_size * head_dim)                           \
                  + slot    * head_dim;                                        \
                                                                               \
        scratch[tid] = (tid < head_dim)                                        \
            ? q_shared[tid] * (float)K[base + tid]                             \
            : 0.0;                                                             \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                               \
        for (uint stride = head_dim / 2; stride > 0; stride /= 2) {            \
            if (tid < stride) {                                                \
                scratch[tid] += scratch[tid + stride];                         \
            }                                                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        }                                                                      \
        float s = scratch[0] * scale;                                          \
                                                                               \
        float new_m = max(m, s);                                               \
        float alpha = exp(m - new_m);                                          \
        float beta  = exp(s - new_m);                                          \
        if (tid < head_dim) {                                                  \
            float v_val = (float)V[base + tid];                                \
            out_acc = out_acc * alpha + beta * v_val;                          \
        }                                                                      \
        l = l * alpha + beta;                                                  \
        m = new_m;                                                             \
    }                                                                          \
                                                                               \
    if (tid < head_dim) {                                                      \
        Out[(seq_id * num_q_heads + h_q) * head_dim + tid] = (T)(out_acc / l); \
    }                                                                          \
}

PAGED_DECODE_ATTN(paged_decode_attention_f32,  float)
PAGED_DECODE_ATTN(paged_decode_attention_f16,  half)
PAGED_DECODE_ATTN(paged_decode_attention_bf16, bfloat)

// ----- batched_rope_decode --------------------------------------------
//
// Apply RoPE in-place on a packed [N, H, D] tensor where each row n has
// its own absolute position offsets[n]. Replaces the per-seq RoPE loop
// in the decode fast path with a single dispatch.
//
// Convention matches `candle_nn::rotary_emb::rope`:
//   y[..,..,d]       = x[..,..,d]       * cos[off,d] - x[..,..,d+D/2] * sin[off,d]
//   y[..,..,d+D/2]   = x[..,..,d]       * sin[off,d] + x[..,..,d+D/2] * cos[off,d]
// for d in [0, D/2).
//
// Each thread handles ONE (n, h, d_pair) triple. It reads both x1 and x2
// before writing either, which makes the in-place rewrite safe.
//
// Grid: (D/2, H, N). Threads per group: tuned by Metal via dispatch_threads.

#define BATCHED_ROPE_DECODE(NAME, T)                                           \
kernel void NAME(                                                              \
    device       T    *X              [[buffer(0)]],                           \
    device const T    *cos_tab        [[buffer(1)]],                           \
    device const T    *sin_tab        [[buffer(2)]],                           \
    device const uint *offsets        [[buffer(3)]],                           \
    constant uint &h_dim              [[buffer(4)]],                           \
    constant uint &half_d             [[buffer(5)]],                           \
    constant uint &cos_stride         [[buffer(6)]],                           \
    uint3 tid                         [[thread_position_in_grid]]              \
) {                                                                            \
    uint d_pair = tid.x;                                                       \
    uint h      = tid.y;                                                       \
    uint n      = tid.z;                                                       \
    if (d_pair >= half_d || h >= h_dim) { return; }                            \
                                                                               \
    uint off = offsets[n];                                                     \
    uint base = (n * h_dim + h) * (2u * half_d);                               \
    uint i1 = base + d_pair;                                                   \
    uint i2 = base + d_pair + half_d;                                          \
                                                                               \
    float x1 = (float)X[i1];                                                   \
    float x2 = (float)X[i2];                                                   \
    float c  = (float)cos_tab[off * cos_stride + d_pair];                      \
    float s  = (float)sin_tab[off * cos_stride + d_pair];                      \
                                                                               \
    X[i1] = (T)(x1 * c - x2 * s);                                              \
    X[i2] = (T)(x1 * s + x2 * c);                                              \
}

BATCHED_ROPE_DECODE(batched_rope_decode_f32,  float)
BATCHED_ROPE_DECODE(batched_rope_decode_f16,  half)
BATCHED_ROPE_DECODE(batched_rope_decode_bf16, bfloat)

// ----- batched_scatter -------------------------------------------------
//
// Scatter N (H, D)-shaped values into the layer KV pool [P, H, S, D] at
// per-row (page_id, slot) destinations. Replaces N inplace_op2 calls
// per layer (in the decode fast path) with a single dispatch.
//
// Each thread copies one element. Grid: (D, H, N).

#define BATCHED_SCATTER(NAME, T)                                               \
kernel void NAME(                                                              \
    device       T    *pool           [[buffer(0)]],                           \
    device const T    *values         [[buffer(1)]],                           \
    device const uint *page_ids       [[buffer(2)]],                           \
    device const uint *slots          [[buffer(3)]],                           \
    constant uint &h_dim              [[buffer(4)]],                           \
    constant uint &s_dim              [[buffer(5)]],                           \
    constant uint &d_dim              [[buffer(6)]],                           \
    uint3 tid                         [[thread_position_in_grid]]              \
) {                                                                            \
    uint d = tid.x;                                                            \
    uint h = tid.y;                                                            \
    uint n = tid.z;                                                            \
    if (d >= d_dim || h >= h_dim) { return; }                                  \
                                                                               \
    uint page_id = page_ids[n];                                                \
    uint slot    = slots[n];                                                   \
    uint dst = page_id * (h_dim * s_dim * d_dim)                               \
             + h * (s_dim * d_dim)                                             \
             + slot * d_dim                                                    \
             + d;                                                              \
    uint src = (n * h_dim + h) * d_dim + d;                                    \
    pool[dst] = values[src];                                                   \
}

BATCHED_SCATTER(batched_scatter_f32,  float)
BATCHED_SCATTER(batched_scatter_f16,  half)
BATCHED_SCATTER(batched_scatter_bf16, bfloat)

// ----- paged_prefill_attention -----------------------------------------
//
// Attention for chunked-prefill batches where each seq has a variable
// number of new query tokens (n_new ≥ 1). Queries are packed into a
// single [total_q, H_q, D] tensor and we use a CSR-like layout to know
// which seq each row belongs to.
//
//   Q              : [total_q, H_q, D]
//   K, V           : [P, H_kv, S, D]
//   block_tables   : [N, MB]          uint
//   cu_seqlens     : [N + 1]          uint   cumulative new_lens
//                                            (cu_seqlens[N] == total_q)
//   seq_id_per_q   : [total_q]        uint
//   kv_offsets     : [N]              uint   absolute position of seq i's
//                                            FIRST new query
//   Out            : [total_q, H_q, D]
//
// For each (q_row, h_q) pair:
//   seq      = seq_id_per_q[q_row]
//   q_pos    = q_row - cu_seqlens[seq]                   (0-indexed in seq)
//   abs_pos  = kv_offsets[seq] + q_pos
//   kv_len   = abs_pos + 1                               (causal)
//   walk block_tables[seq, :], same online softmax loop as decode kernel.
//
// Threadgroup grid: (total_q, H_q, 1). Threads per group: head_dim.

#define PAGED_PREFILL_ATTN(NAME, T)                                            \
kernel void NAME(                                                              \
    device const T    *Q              [[buffer(0)]],                           \
    device const T    *K              [[buffer(1)]],                           \
    device const T    *V              [[buffer(2)]],                           \
    device const uint *block_tables   [[buffer(3)]],                           \
    device const uint *cu_seqlens     [[buffer(4)]],                           \
    device const uint *seq_id_per_q   [[buffer(5)]],                           \
    device const uint *kv_offsets     [[buffer(6)]],                           \
    device       T    *Out            [[buffer(7)]],                           \
    constant float &scale             [[buffer(8)]],                           \
    constant uint  &num_q_heads       [[buffer(9)]],                           \
    constant uint  &num_kv_heads      [[buffer(10)]],                          \
    constant uint  &head_dim          [[buffer(11)]],                          \
    constant uint  &page_size         [[buffer(12)]],                          \
    constant uint  &max_blocks        [[buffer(13)]],                          \
    uint3 tg_id                       [[threadgroup_position_in_grid]],        \
    uint3 tid_v                       [[thread_position_in_threadgroup]]       \
) {                                                                            \
    uint q_row  = tg_id.x;                                                     \
    uint h_q    = tg_id.y;                                                     \
    uint tid    = tid_v.x;                                                     \
    uint kv_groups = num_q_heads / num_kv_heads;                               \
    uint h_kv   = h_q / kv_groups;                                             \
                                                                               \
    uint seq    = seq_id_per_q[q_row];                                         \
    uint q_pos  = q_row - cu_seqlens[seq];                                     \
    uint abs_pos = kv_offsets[seq] + q_pos;                                    \
    uint kv_len = abs_pos + 1u;                                                \
                                                                               \
    threadgroup float q_shared[D_MAX];                                         \
    threadgroup float scratch[D_MAX];                                          \
                                                                               \
    if (tid < head_dim) {                                                      \
        q_shared[tid] = (float)Q[(q_row * num_q_heads + h_q) * head_dim        \
                                 + tid];                                       \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
                                                                               \
    float out_acc = 0.0;                                                       \
    float m = -INFINITY;                                                       \
    float l = 0.0;                                                             \
                                                                               \
    for (uint pos = 0; pos < kv_len; pos++) {                                  \
        uint page_idx = pos / page_size;                                       \
        uint slot     = pos - page_idx * page_size;                            \
        uint page_id  = block_tables[seq * max_blocks + page_idx];             \
        uint base = page_id * (num_kv_heads * page_size * head_dim)            \
                  + h_kv    * (page_size * head_dim)                           \
                  + slot    * head_dim;                                        \
                                                                               \
        scratch[tid] = (tid < head_dim)                                        \
            ? q_shared[tid] * (float)K[base + tid]                             \
            : 0.0;                                                             \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                               \
        for (uint stride = head_dim / 2; stride > 0; stride /= 2) {            \
            if (tid < stride) {                                                \
                scratch[tid] += scratch[tid + stride];                         \
            }                                                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        }                                                                      \
        float s = scratch[0] * scale;                                          \
                                                                               \
        float new_m = max(m, s);                                               \
        float alpha = exp(m - new_m);                                          \
        float beta  = exp(s - new_m);                                          \
        if (tid < head_dim) {                                                  \
            float v_val = (float)V[base + tid];                                \
            out_acc = out_acc * alpha + beta * v_val;                          \
        }                                                                      \
        l = l * alpha + beta;                                                  \
        m = new_m;                                                             \
    }                                                                          \
                                                                               \
    if (tid < head_dim) {                                                      \
        Out[(q_row * num_q_heads + h_q) * head_dim + tid] = (T)(out_acc / l);  \
    }                                                                          \
}

PAGED_PREFILL_ATTN(paged_prefill_attention_f32,  float)
PAGED_PREFILL_ATTN(paged_prefill_attention_f16,  half)
PAGED_PREFILL_ATTN(paged_prefill_attention_bf16, bfloat)
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

/// In-place scatter of a single `[H, D]` slot into a `[P, H, S, D]` pool
/// at `(page_id, slot)`.
///
/// `self` is the pool and `rhs` is the value when used via
/// `Tensor::inplace_op2(value, &ScatterSlot { page_id, slot })`.
///
/// Both tensors MUST be contiguous on Metal. The cpu fallback uses the
/// layouts directly (no contiguity assumption) so it's safe to call from
/// tests with sliced views.
pub struct ScatterSlot {
    pub page_id: u32,
    pub slot: u32,
}

impl InplaceOp2 for ScatterSlot {
    fn name(&self) -> &'static str {
        "paged_scatter_slot"
    }

    fn cpu_fwd(
        &self,
        pool: &mut CpuStorage,
        pool_l: &Layout,
        value: &CpuStorage,
        value_l: &Layout,
    ) -> CandleResult<()> {
        let (p, h, s, d) = four_dims(pool_l, "pool")?;
        let (vh, vd) = two_dims(value_l, "value")?;
        if h != vh || d != vd {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter: pool=[{p},{h},{s},{d}] value=[{vh},{vd}] mismatch"
            )));
        }
        let slot = self.slot as usize;
        let page_id = self.page_id as usize;
        if slot >= s {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter: slot {slot} >= s_dim {s}"
            )));
        }
        if page_id >= p {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter: page_id {page_id} >= num_pages {p}"
            )));
        }

        match (pool, value) {
            (CpuStorage::F32(b), CpuStorage::F32(v)) => {
                scatter_cpu(b, pool_l, v, value_l, page_id, slot, h, s, d)
            }
            (CpuStorage::F16(b), CpuStorage::F16(v)) => {
                scatter_cpu(b, pool_l, v, value_l, page_id, slot, h, s, d)
            }
            (CpuStorage::BF16(b), CpuStorage::BF16(v)) => {
                scatter_cpu(b, pool_l, v, value_l, page_id, slot, h, s, d)
            }
            (b, v) => Err(candle_core::Error::Msg(format!(
                "paged scatter: dtype mismatch pool={:?} value={:?}",
                b.dtype(),
                v.dtype()
            ))),
        }
    }

    fn metal_fwd(
        &self,
        pool: &mut MetalStorage,
        pool_l: &Layout,
        value: &MetalStorage,
        value_l: &Layout,
    ) -> CandleResult<()> {
        let (p, h, s, d) = four_dims(pool_l, "pool")?;
        let (vh, vd) = two_dims(value_l, "value")?;
        if h != vh || d != vd {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter (metal): pool=[{p},{h},{s},{d}] value=[{vh},{vd}] mismatch"
            )));
        }
        if !pool_l.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "paged scatter (metal): pool must be contiguous".into(),
            ));
        }
        if !value_l.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "paged scatter (metal): value must be contiguous".into(),
            ));
        }
        let slot = self.slot;
        let page_id = self.page_id;
        if (slot as usize) >= s {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter (metal): slot {slot} >= s_dim {s}"
            )));
        }
        if (page_id as usize) >= p {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter (metal): page_id {page_id} >= num_pages {p}"
            )));
        }
        if pool.dtype() != value.dtype() {
            return Err(candle_core::Error::Msg(format!(
                "paged scatter (metal): dtype mismatch pool={:?} value={:?}",
                pool.dtype(),
                value.dtype()
            )));
        }

        let dtype = pool.dtype();
        let fn_name = dtype_kernel_name(dtype)?;
        let elem_bytes = dtype.size_in_bytes();

        let device = pool.device().clone();
        let pipeline = ensure_pipeline(&device, fn_name)
            .map_err(|e| candle_core::Error::Msg(format!("paged scatter pipeline: {e}")))?;

        let pool_offset = pool_l.start_offset() * elem_bytes;
        let value_offset = value_l.start_offset() * elem_bytes;

        let h_dim = h as u32;
        let s_dim = s as u32;
        let d_dim = d as u32;

        let encoder = device
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("paged scatter encoder: {e}")))?;
        encoder.set_label("velox.paged.scatter_slot");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(pool.buffer()), pool_offset);
        encoder.set_buffer(1, Some(value.buffer()), value_offset);
        encoder.set_bytes(2, &page_id);
        encoder.set_bytes(3, &slot);
        encoder.set_bytes(4, &h_dim);
        encoder.set_bytes(5, &s_dim);
        encoder.set_bytes(6, &d_dim);

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
        // Encoder Drop calls end_encoding for us; calling it twice asserts.
        drop(encoder);
        Ok(())
    }
}

fn four_dims(
    l: &Layout,
    label: &'static str,
) -> CandleResult<(usize, usize, usize, usize)> {
    let dims = l.dims();
    if dims.len() != 4 {
        return Err(candle_core::Error::Msg(format!(
            "paged scatter: {label} must be 4-D, got {dims:?}"
        )));
    }
    Ok((dims[0], dims[1], dims[2], dims[3]))
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
    pool: &mut [T],
    pool_l: &Layout,
    value: &[T],
    value_l: &Layout,
    page_id: usize,
    slot: usize,
    h: usize,
    s: usize,
    d: usize,
) -> CandleResult<()> {
    let p_strides = pool_l.stride();
    let v_strides = value_l.stride();
    let p_off = pool_l.start_offset();
    let v_off = value_l.start_offset();
    for hh in 0..h {
        for dd in 0..d {
            let dst = p_off
                + page_id * p_strides[0]
                + hh * p_strides[1]
                + slot * p_strides[2]
                + dd * p_strides[3];
            let src = v_off + hh * v_strides[0] + dd * v_strides[1];
            pool[dst] = value[src];
        }
    }
    let _ = s;
    Ok(())
}

// =====================================================================
//   paged_decode_attention — fused decode kernel + CPU reference
// =====================================================================

fn decode_kernel_name(dtype: DType) -> AnyResult<&'static str> {
    Ok(match dtype {
        DType::F32 => "paged_decode_attention_f32",
        DType::F16 => "paged_decode_attention_f16",
        DType::BF16 => "paged_decode_attention_bf16",
        other => bail!("paged_decode_attention: unsupported dtype {other:?}"),
    })
}

const D_MAX: usize = 256;

/// Fused per-decode-token attention against a paged KV pool.
///
/// All inputs must live on the same Metal device; the output is allocated
/// on that device. `q`, `k_pool`, `v_pool`, `out` share dtype (BF16/F16/F32);
/// `block_table` and `kv_lens` are `u32`.
///
/// Shapes:
///   q            : [N, H_q, D]
///   k_pool       : [P, H_kv, S, D]
///   v_pool       : [P, H_kv, S, D]
///   block_table  : [N, MB]   u32
///   kv_lens      : [N]       u32
///
/// Constraints (current v1):
///   * D divisible by 32 and D <= D_MAX (256)
///   * H_q % H_kv == 0 (GQA)
///   * All tensors contiguous
///
/// Returns the attention output tensor of shape `[N, H_q, D]`.
#[cfg(all(target_os = "macos", feature = "candle-metal"))]
pub fn paged_decode_attention(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    block_table: &Tensor,
    kv_lens: &Tensor,
    scale: f32,
) -> AnyResult<Tensor> {
    let device = q.device().clone();
    let metal_dev = match &device {
        Device::Metal(d) => d.clone(),
        _ => bail!("paged_decode_attention requires a Metal device"),
    };

    let (n, h_q, head_dim) = q.dims3()?;
    let (_p, h_kv, page_size, kd) = k_pool.dims4()?;
    let (_pv, h_kv_v, ps_v, kd_v) = v_pool.dims4()?;
    if h_kv != h_kv_v || page_size != ps_v || head_dim != kd || head_dim != kd_v {
        bail!("paged_decode_attention: K/V pool shapes must match Q head_dim");
    }
    if h_q % h_kv != 0 {
        bail!(
            "paged_decode_attention: H_q={h_q} must be divisible by H_kv={h_kv}"
        );
    }
    if head_dim > D_MAX {
        bail!("paged_decode_attention: head_dim {head_dim} > D_MAX {D_MAX}");
    }
    if head_dim % 32 != 0 {
        bail!(
            "paged_decode_attention: head_dim {head_dim} not divisible by 32 (kernel requires power-of-2 head dims)"
        );
    }

    let (n_bt, max_blocks) = block_table.dims2()?;
    if n_bt != n {
        bail!(
            "paged_decode_attention: block_table N={n_bt} mismatches Q N={n}"
        );
    }
    let (n_kv,) = kv_lens.dims1().map(|x| (x,))?;
    if n_kv != n {
        bail!(
            "paged_decode_attention: kv_lens N={n_kv} mismatches Q N={n}"
        );
    }

    let dtype = q.dtype();
    if k_pool.dtype() != dtype || v_pool.dtype() != dtype {
        bail!("paged_decode_attention: Q/K/V dtype mismatch");
    }
    if block_table.dtype() != DType::U32 || kv_lens.dtype() != DType::U32 {
        bail!("paged_decode_attention: block_table and kv_lens must be U32");
    }
    for (label, t) in [
        ("q", q),
        ("k_pool", k_pool),
        ("v_pool", v_pool),
        ("block_table", block_table),
        ("kv_lens", kv_lens),
    ] {
        if !t.is_contiguous() {
            bail!("paged_decode_attention: {label} must be contiguous");
        }
    }

    // Allocate output
    let out = Tensor::zeros((n, h_q, head_dim), dtype, &device)?;

    let fn_name = decode_kernel_name(dtype)?;
    let pipeline = ensure_pipeline(&metal_dev, fn_name)?;

    let elem_bytes = dtype.size_in_bytes();

    let h_q_u = h_q as u32;
    let h_kv_u = h_kv as u32;
    let head_dim_u = head_dim as u32;
    let page_size_u = page_size as u32;
    let max_blocks_u = max_blocks as u32;

    {
        // Scope all the storage read locks so they are released before we
        // return `out` to the caller.
        let (q_st, q_l) = q.storage_and_layout();
        let q_st = match &*q_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("Q not on metal"),
        };
        let (k_st, k_l) = k_pool.storage_and_layout();
        let k_st = match &*k_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("K not on metal"),
        };
        let (v_st, v_l) = v_pool.storage_and_layout();
        let v_st = match &*v_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("V not on metal"),
        };
        let (bt_st, bt_l) = block_table.storage_and_layout();
        let bt_st = match &*bt_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("block_table not on metal"),
        };
        let (kvl_st, kvl_l) = kv_lens.storage_and_layout();
        let kvl_st = match &*kvl_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("kv_lens not on metal"),
        };
        let (out_st, out_l) = out.storage_and_layout();
        let out_st = match &*out_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("out not on metal"),
        };

        let q_off = q_l.start_offset() * elem_bytes;
        let k_off = k_l.start_offset() * elem_bytes;
        let v_off = v_l.start_offset() * elem_bytes;
        let bt_off = bt_l.start_offset() * 4; // u32
        let kvl_off = kvl_l.start_offset() * 4;
        let out_off = out_l.start_offset() * elem_bytes;

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| anyhow!("paged_decode_attention encoder: {e}"))?;
        encoder.set_label("velox.paged.decode_attention");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q_st.buffer()), q_off);
        encoder.set_buffer(1, Some(k_st.buffer()), k_off);
        encoder.set_buffer(2, Some(v_st.buffer()), v_off);
        encoder.set_buffer(3, Some(bt_st.buffer()), bt_off);
        encoder.set_buffer(4, Some(kvl_st.buffer()), kvl_off);
        encoder.set_buffer(5, Some(out_st.buffer()), out_off);
        encoder.set_bytes(6, &scale);
        encoder.set_bytes(7, &h_q_u);
        encoder.set_bytes(8, &h_kv_u);
        encoder.set_bytes(9, &head_dim_u);
        encoder.set_bytes(10, &page_size_u);
        encoder.set_bytes(11, &max_blocks_u);

        let groups = MTLSize {
            width: n,
            height: h_q,
            depth: 1,
        };
        let tg = MTLSize {
            width: head_dim,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(groups, tg);
        drop(encoder);
    }

    Ok(out)
}

#[cfg(not(all(target_os = "macos", feature = "candle-metal")))]
pub fn paged_decode_attention(
    _q: &Tensor,
    _k_pool: &Tensor,
    _v_pool: &Tensor,
    _block_table: &Tensor,
    _kv_lens: &Tensor,
    _scale: f32,
) -> AnyResult<Tensor> {
    bail!("paged_decode_attention is only available on macOS with the candle-metal feature")
}

// =====================================================================
//   batched_rope_decode (in-place)
// =====================================================================

fn rope_kernel_name(dtype: DType) -> AnyResult<&'static str> {
    Ok(match dtype {
        DType::F32 => "batched_rope_decode_f32",
        DType::F16 => "batched_rope_decode_f16",
        DType::BF16 => "batched_rope_decode_bf16",
        other => bail!("batched_rope_decode: unsupported dtype {other:?}"),
    })
}

/// Apply RoPE in-place on `x` of shape `[N, H, D]` using per-row
/// `offsets[N]`. `cos`/`sin` are `[Lmax, D/2]`, same dtype as `x`.
#[cfg(all(target_os = "macos", feature = "candle-metal"))]
pub fn batched_rope_decode(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    offsets: &Tensor,
) -> AnyResult<()> {
    let device = x.device().clone();
    let metal_dev = match &device {
        Device::Metal(d) => d.clone(),
        _ => bail!("batched_rope_decode requires Metal device"),
    };

    let (n, h, d) = x.dims3()?;
    let (_lmax, half_d) = cos.dims2()?;
    let (_lmax2, half_d2) = sin.dims2()?;
    if half_d != half_d2 || half_d * 2 != d {
        bail!(
            "batched_rope_decode: cos/sin shape [Lmax, {half_d}] mismatches D={d}"
        );
    }
    let (n_off,) = offsets.dims1().map(|x| (x,))?;
    if n_off != n {
        bail!("batched_rope_decode: offsets N={n_off} mismatches X N={n}");
    }
    let dtype = x.dtype();
    if cos.dtype() != dtype || sin.dtype() != dtype {
        bail!("batched_rope_decode: cos/sin dtype must match X");
    }
    if offsets.dtype() != DType::U32 {
        bail!("batched_rope_decode: offsets must be U32");
    }
    for (label, t) in [("x", x), ("cos", cos), ("sin", sin), ("offsets", offsets)] {
        if !t.is_contiguous() {
            bail!("batched_rope_decode: {label} must be contiguous");
        }
    }

    let fn_name = rope_kernel_name(dtype)?;
    let pipeline = ensure_pipeline(&metal_dev, fn_name)?;
    let elem_bytes = dtype.size_in_bytes();

    let h_dim_u = h as u32;
    let half_d_u = half_d as u32;
    let cos_stride_u = half_d as u32; // [Lmax, D/2] row-major

    {
        let (x_st, x_l) = x.storage_and_layout();
        let x_st = match &*x_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("X not metal"),
        };
        let (c_st, c_l) = cos.storage_and_layout();
        let c_st = match &*c_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("cos not metal"),
        };
        let (s_st, s_l) = sin.storage_and_layout();
        let s_st = match &*s_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("sin not metal"),
        };
        let (o_st, o_l) = offsets.storage_and_layout();
        let o_st = match &*o_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("offsets not metal"),
        };

        let x_off = x_l.start_offset() * elem_bytes;
        let c_off = c_l.start_offset() * elem_bytes;
        let s_off = s_l.start_offset() * elem_bytes;
        let o_off = o_l.start_offset() * 4;

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| anyhow!("batched_rope_decode encoder: {e}"))?;
        encoder.set_label("velox.paged.batched_rope_decode");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_st.buffer()), x_off);
        encoder.set_buffer(1, Some(c_st.buffer()), c_off);
        encoder.set_buffer(2, Some(s_st.buffer()), s_off);
        encoder.set_buffer(3, Some(o_st.buffer()), o_off);
        encoder.set_bytes(4, &h_dim_u);
        encoder.set_bytes(5, &half_d_u);
        encoder.set_bytes(6, &cos_stride_u);

        let grid = MTLSize {
            width: half_d,
            height: h,
            depth: n,
        };
        let tg_w = std::cmp::min(half_d, 32);
        let tg_h = std::cmp::min(h, 8);
        let tg = MTLSize {
            width: tg_w.max(1),
            height: tg_h.max(1),
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        drop(encoder);
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "candle-metal")))]
pub fn batched_rope_decode(
    _x: &Tensor,
    _cos: &Tensor,
    _sin: &Tensor,
    _offsets: &Tensor,
) -> AnyResult<()> {
    bail!("batched_rope_decode is only available on macOS with candle-metal")
}

// =====================================================================
//   batched_scatter (one dispatch per layer × {K|V})
// =====================================================================

fn batched_scatter_kernel_name(dtype: DType) -> AnyResult<&'static str> {
    Ok(match dtype {
        DType::F32 => "batched_scatter_f32",
        DType::F16 => "batched_scatter_f16",
        DType::BF16 => "batched_scatter_bf16",
        other => bail!("batched_scatter: unsupported dtype {other:?}"),
    })
}

/// Scatter N rows of shape [H, D] from `values` [N, H, D] into `pool`
/// [P, H, S, D] at per-row (page_ids[n], slots[n]) destinations.
#[cfg(all(target_os = "macos", feature = "candle-metal"))]
pub fn batched_scatter(
    pool: &Tensor,
    values: &Tensor,
    page_ids: &Tensor,
    slots: &Tensor,
) -> AnyResult<()> {
    let device = pool.device().clone();
    let metal_dev = match &device {
        Device::Metal(d) => d.clone(),
        _ => bail!("batched_scatter requires Metal"),
    };

    let (_p, h, s, d) = pool.dims4()?;
    let (n, vh, vd) = values.dims3()?;
    if vh != h || vd != d {
        bail!(
            "batched_scatter: values [{n},{vh},{vd}] mismatches pool [_,{h},_,{d}]"
        );
    }
    let (n_pg,) = page_ids.dims1().map(|x| (x,))?;
    let (n_sl,) = slots.dims1().map(|x| (x,))?;
    if n_pg != n || n_sl != n {
        bail!("batched_scatter: page_ids/slots N mismatches values N");
    }
    if pool.dtype() != values.dtype() {
        bail!("batched_scatter: pool/values dtype mismatch");
    }
    if page_ids.dtype() != DType::U32 || slots.dtype() != DType::U32 {
        bail!("batched_scatter: page_ids and slots must be U32");
    }
    for (l, t) in [("pool", pool), ("values", values), ("page_ids", page_ids), ("slots", slots)] {
        if !t.is_contiguous() {
            bail!("batched_scatter: {l} must be contiguous");
        }
    }

    let dtype = pool.dtype();
    let fn_name = batched_scatter_kernel_name(dtype)?;
    let pipeline = ensure_pipeline(&metal_dev, fn_name)?;
    let elem_bytes = dtype.size_in_bytes();

    let h_dim_u = h as u32;
    let s_dim_u = s as u32;
    let d_dim_u = d as u32;

    {
        let (po_st, po_l) = pool.storage_and_layout();
        let po_st = match &*po_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("pool not metal"),
        };
        let (va_st, va_l) = values.storage_and_layout();
        let va_st = match &*va_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("values not metal"),
        };
        let (pg_st, pg_l) = page_ids.storage_and_layout();
        let pg_st = match &*pg_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("page_ids not metal"),
        };
        let (sl_st, sl_l) = slots.storage_and_layout();
        let sl_st = match &*sl_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("slots not metal"),
        };

        let po_off = po_l.start_offset() * elem_bytes;
        let va_off = va_l.start_offset() * elem_bytes;
        let pg_off = pg_l.start_offset() * 4;
        let sl_off = sl_l.start_offset() * 4;

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| anyhow!("batched_scatter encoder: {e}"))?;
        encoder.set_label("velox.paged.batched_scatter");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(po_st.buffer()), po_off);
        encoder.set_buffer(1, Some(va_st.buffer()), va_off);
        encoder.set_buffer(2, Some(pg_st.buffer()), pg_off);
        encoder.set_buffer(3, Some(sl_st.buffer()), sl_off);
        encoder.set_bytes(4, &h_dim_u);
        encoder.set_bytes(5, &s_dim_u);
        encoder.set_bytes(6, &d_dim_u);

        let grid = MTLSize {
            width: d,
            height: h,
            depth: n,
        };
        let tg_w = std::cmp::min(d, 32);
        let tg_h = std::cmp::min(h, 8);
        let tg = MTLSize {
            width: tg_w.max(1),
            height: tg_h.max(1),
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        drop(encoder);
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "candle-metal")))]
pub fn batched_scatter(
    _pool: &Tensor,
    _values: &Tensor,
    _page_ids: &Tensor,
    _slots: &Tensor,
) -> AnyResult<()> {
    bail!("batched_scatter is only available on macOS with candle-metal")
}

// =====================================================================
//   paged_prefill_attention
// =====================================================================

fn prefill_kernel_name(dtype: DType) -> AnyResult<&'static str> {
    Ok(match dtype {
        DType::F32 => "paged_prefill_attention_f32",
        DType::F16 => "paged_prefill_attention_f16",
        DType::BF16 => "paged_prefill_attention_bf16",
        other => bail!("paged_prefill_attention: unsupported dtype {other:?}"),
    })
}

/// Variable-length prefill attention against a paged KV pool.
///
///   q             : [total_q, H_q, D]
///   k_pool, v_pool: [P, H_kv, S, D]
///   block_table   : [N, MB]            u32
///   cu_seqlens    : [N + 1]            u32   (cumulative new_lens)
///   seq_id_per_q  : [total_q]          u32
///   kv_offsets    : [N]                u32   (abs pos of FIRST new query per seq)
///
/// Returns: [total_q, H_q, D]
#[cfg(all(target_os = "macos", feature = "candle-metal"))]
pub fn paged_prefill_attention(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    block_table: &Tensor,
    cu_seqlens: &Tensor,
    seq_id_per_q: &Tensor,
    kv_offsets: &Tensor,
    scale: f32,
) -> AnyResult<Tensor> {
    let device = q.device().clone();
    let metal_dev = match &device {
        Device::Metal(d) => d.clone(),
        _ => bail!("paged_prefill_attention requires Metal"),
    };

    let (total_q, h_q, head_dim) = q.dims3()?;
    let (_p, h_kv, page_size, kd) = k_pool.dims4()?;
    if head_dim != kd {
        bail!("prefill: head_dim mismatch Q vs K pool");
    }
    if h_q % h_kv != 0 {
        bail!("prefill: H_q must be divisible by H_kv");
    }
    if head_dim > D_MAX {
        bail!("prefill: head_dim {head_dim} > D_MAX {D_MAX}");
    }
    if head_dim % 32 != 0 {
        bail!("prefill: head_dim {head_dim} must be a multiple of 32");
    }

    let (n_blk, max_blocks) = block_table.dims2()?;
    let (n_plus_1,) = cu_seqlens.dims1().map(|x| (x,))?;
    let (tq2,) = seq_id_per_q.dims1().map(|x| (x,))?;
    let (n_off,) = kv_offsets.dims1().map(|x| (x,))?;
    if n_plus_1 != n_blk + 1 || tq2 != total_q || n_off != n_blk {
        bail!(
            "prefill: shape mismatch (block_table N={n_blk}, cu_seqlens={n_plus_1}, seq_id_per_q={tq2}, kv_offsets={n_off})"
        );
    }

    let dtype = q.dtype();
    if k_pool.dtype() != dtype || v_pool.dtype() != dtype {
        bail!("prefill: Q/K/V dtype mismatch");
    }
    for (l, t) in [
        ("q", q),
        ("k_pool", k_pool),
        ("v_pool", v_pool),
        ("block_table", block_table),
        ("cu_seqlens", cu_seqlens),
        ("seq_id_per_q", seq_id_per_q),
        ("kv_offsets", kv_offsets),
    ] {
        if !t.is_contiguous() {
            bail!("prefill: {l} must be contiguous");
        }
    }
    if block_table.dtype() != DType::U32
        || cu_seqlens.dtype() != DType::U32
        || seq_id_per_q.dtype() != DType::U32
        || kv_offsets.dtype() != DType::U32
    {
        bail!("prefill: index tensors must be U32");
    }

    let out = Tensor::zeros((total_q, h_q, head_dim), dtype, &device)?;
    let fn_name = prefill_kernel_name(dtype)?;
    let pipeline = ensure_pipeline(&metal_dev, fn_name)?;
    let elem_bytes = dtype.size_in_bytes();

    let h_q_u = h_q as u32;
    let h_kv_u = h_kv as u32;
    let head_dim_u = head_dim as u32;
    let page_size_u = page_size as u32;
    let max_blocks_u = max_blocks as u32;

    {
        let (q_st, q_l) = q.storage_and_layout();
        let q_st = match &*q_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("Q not metal"),
        };
        let (k_st, k_l) = k_pool.storage_and_layout();
        let k_st = match &*k_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("K not metal"),
        };
        let (v_st, v_l) = v_pool.storage_and_layout();
        let v_st = match &*v_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("V not metal"),
        };
        let (bt_st, bt_l) = block_table.storage_and_layout();
        let bt_st = match &*bt_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("block_table not metal"),
        };
        let (cu_st, cu_l) = cu_seqlens.storage_and_layout();
        let cu_st = match &*cu_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("cu_seqlens not metal"),
        };
        let (sq_st, sq_l) = seq_id_per_q.storage_and_layout();
        let sq_st = match &*sq_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("seq_id_per_q not metal"),
        };
        let (off_st, off_l) = kv_offsets.storage_and_layout();
        let off_st = match &*off_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("kv_offsets not metal"),
        };
        let (out_st, out_l) = out.storage_and_layout();
        let out_st = match &*out_st {
            Storage::Metal(s) => s.clone(),
            _ => bail!("out not metal"),
        };

        let q_off = q_l.start_offset() * elem_bytes;
        let k_off = k_l.start_offset() * elem_bytes;
        let v_off = v_l.start_offset() * elem_bytes;
        let bt_off = bt_l.start_offset() * 4;
        let cu_off = cu_l.start_offset() * 4;
        let sq_off = sq_l.start_offset() * 4;
        let off_off = off_l.start_offset() * 4;
        let out_off = out_l.start_offset() * elem_bytes;

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| anyhow!("prefill encoder: {e}"))?;
        encoder.set_label("velox.paged.prefill_attention");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q_st.buffer()), q_off);
        encoder.set_buffer(1, Some(k_st.buffer()), k_off);
        encoder.set_buffer(2, Some(v_st.buffer()), v_off);
        encoder.set_buffer(3, Some(bt_st.buffer()), bt_off);
        encoder.set_buffer(4, Some(cu_st.buffer()), cu_off);
        encoder.set_buffer(5, Some(sq_st.buffer()), sq_off);
        encoder.set_buffer(6, Some(off_st.buffer()), off_off);
        encoder.set_buffer(7, Some(out_st.buffer()), out_off);
        encoder.set_bytes(8, &scale);
        encoder.set_bytes(9, &h_q_u);
        encoder.set_bytes(10, &h_kv_u);
        encoder.set_bytes(11, &head_dim_u);
        encoder.set_bytes(12, &page_size_u);
        encoder.set_bytes(13, &max_blocks_u);

        let groups = MTLSize {
            width: total_q,
            height: h_q,
            depth: 1,
        };
        let tg = MTLSize {
            width: head_dim,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(groups, tg);
        drop(encoder);
    }

    Ok(out)
}

#[cfg(not(all(target_os = "macos", feature = "candle-metal")))]
pub fn paged_prefill_attention(
    _q: &Tensor,
    _k_pool: &Tensor,
    _v_pool: &Tensor,
    _block_table: &Tensor,
    _cu_seqlens: &Tensor,
    _seq_id_per_q: &Tensor,
    _kv_offsets: &Tensor,
    _scale: f32,
) -> AnyResult<Tensor> {
    bail!("paged_prefill_attention is only available on macOS with candle-metal")
}

/// Pure-Rust reference implementation of `paged_decode_attention` for
/// parity tests. All buffers laid out as in the Metal kernel.
pub fn paged_decode_attention_cpu(
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_table: &[u32],
    kv_lens: &[u32],
    n: usize,
    h_q: usize,
    h_kv: usize,
    head_dim: usize,
    page_size: usize,
    max_blocks: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; n * h_q * head_dim];
    let kv_groups = h_q / h_kv;
    for seq_id in 0..n {
        for hq in 0..h_q {
            let h_kv_idx = hq / kv_groups;
            let kv_len = kv_lens[seq_id] as usize;
            if kv_len == 0 {
                continue;
            }

            // Online softmax to mirror what the kernel does (so any
            // numerical differences are due to dtype rounding, not the
            // algorithm).
            let mut m = f32::NEG_INFINITY;
            let mut l = 0f32;
            let mut acc = vec![0f32; head_dim];
            let q_off = (seq_id * h_q + hq) * head_dim;

            for pos in 0..kv_len {
                let page_id = block_table[seq_id * max_blocks + pos / page_size] as usize;
                let slot = pos % page_size;
                let base =
                    page_id * (h_kv * page_size * head_dim) + h_kv_idx * (page_size * head_dim) + slot * head_dim;

                let mut s = 0f32;
                for d in 0..head_dim {
                    s += q[q_off + d] * k_pool[base + d];
                }
                s *= scale;

                let new_m = m.max(s);
                let alpha = (m - new_m).exp();
                let beta = (s - new_m).exp();
                for d in 0..head_dim {
                    acc[d] = acc[d] * alpha + beta * v_pool[base + d];
                }
                l = l * alpha + beta;
                m = new_m;
            }

            let out_off = (seq_id * h_q + hq) * head_dim;
            for d in 0..head_dim {
                out[out_off + d] = acc[d] / l;
            }
        }
    }
    out
}
