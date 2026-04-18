//! Qwen3 with paged KV attention, designed for continuous batching.
//!
//! This is a fork of [`candle_transformers::models::qwen3`] with the
//! attention layer rewritten so the K/V state lives inside an external
//! [`PagedKvCache`] instead of each layer's internal `ConcatKvCache`. The
//! rest of the model (embeddings, MLP, RMSNorm, RoPE) is unchanged.
//!
//! Forward pass shape: a single batch step sees N requests packed
//! end-to-end into one `[total_tokens]` 1-D input tensor. Per-request
//! offsets and block tables tell the attention where in the page pool
//! each request's history lives, and where to write the new K/V.
//!
//! For v1 we don't use a custom Metal kernel for paged attention. Instead
//! we materialise per-request K/V tensors via `index_select` from the page
//! pool and run standard SDPA. This is correct and gives us continuous
//! batching with paged storage; a custom kernel will come in a follow-up.

use super::pages::PagedKvCache;
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{linear_b, linear_no_bias, rms_norm, Activation, Embedding, Linear, RmsNorm, VarBuilder};
use std::sync::Arc;

#[cfg(all(target_os = "macos", feature = "candle-metal"))]
use super::metal_kernels::{paged_decode_attention, ScatterSlot};

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

/// One step of the batched forward pass. All requests in the batch share
/// the same `Tensor` of input ids — packed end-to-end — but each one has
/// its own seq slice, RoPE offset, and KV block table.
pub struct BatchStep<'a> {
    /// Concatenated token IDs for this batch step. Shape: `[total_tokens]`.
    pub input_ids: &'a Tensor,
    /// Per-request slice metadata, in the same order as `input_ids` packs them.
    pub seqs: &'a [SeqSlice<'a>],
}

/// One request's contribution to the current batch step.
#[derive(Debug, Clone)]
pub struct SeqSlice<'a> {
    /// Number of new tokens this step (1 for decode, prompt_len for prefill).
    pub new_tokens: usize,
    /// Where in the KV cache the FIRST new token lives (logical position).
    /// Equivalently: number of tokens already in the cache for this request.
    pub kv_offset: usize,
    /// Physical pages of this request's KV cache. Length must satisfy
    /// `block_table.len() * page_size >= kv_offset + new_tokens`.
    pub block_table: &'a [u32],
}

impl SeqSlice<'_> {
    pub fn total_kv_len(&self) -> usize {
        self.kv_offset + self.new_tokens
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE to q,k of shape `[H, L, D]` for tokens at positions
    /// `[offset .. offset+L)`.
    fn apply_at_offset(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _) = q.dims3()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        // candle_nn::rotary_emb::rope expects [B, H, L, D]; we have [H, L, D] so unsqueeze.
        let q = q.unsqueeze(0)?.contiguous()?;
        let k = k.unsqueeze(0)?.contiguous()?;
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?.squeeze(0)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?.squeeze(0)?;
        Ok((q, k))
    }
}

#[derive(Debug, Clone)]
struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct PagedQwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    layer_idx: usize,
}

impl PagedQwen3Attention {
    fn new(
        cfg: &Qwen3Config,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            anyhow::bail!("sliding window is not supported in paged backend");
        }
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = linear_b(cfg.hidden_size, num_heads * head_dim, cfg.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, cfg.hidden_size, cfg.attention_bias, vb.pp("o_proj"))?;
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let hidden_size = head_dim * cfg.num_attention_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary,
            layer_idx,
        })
    }

    /// Forward for one packed batch step.
    ///
    /// `x` shape: `[total_tokens, hidden_size]` (already projected from the
    /// embedding stage; we keep things 2-D to make per-seq slicing trivial).
    fn forward(
        &self,
        x: &Tensor,
        seqs: &[SeqSlice<'_>],
        pages: &PagedKvCache,
    ) -> Result<Tensor> {
        let total = x.dim(0)?;
        debug_assert_eq!(total, seqs.iter().map(|s| s.new_tokens).sum::<usize>());

        // 1. Project Q, K, V for the new tokens. Shapes: [total, n_h*d], [total, n_kv*d].
        let q_all = self.q_proj.forward(x)?;
        let k_all = self.k_proj.forward(x)?;
        let v_all = self.v_proj.forward(x)?;

        // 2. Per-head RMSNorm on Q and K.
        //    Reshape to [total, num_heads, head_dim] then norm over head_dim.
        let q_all = q_all
            .reshape((total, self.num_heads, self.head_dim))?
            .reshape((total * self.num_heads, self.head_dim))?;
        let k_all = k_all
            .reshape((total, self.num_kv_heads, self.head_dim))?
            .reshape((total * self.num_kv_heads, self.head_dim))?;
        let q_all = self.q_norm.forward(&q_all)?;
        let k_all = self.k_norm.forward(&k_all)?;
        let q_all = q_all.reshape((total, self.num_heads, self.head_dim))?;
        let k_all = k_all.reshape((total, self.num_kv_heads, self.head_dim))?;
        let v_all = v_all.reshape((total, self.num_kv_heads, self.head_dim))?;

        // ---- Fused decode fast path -------------------------------------
        //
        // When every sequence in the batch contributes exactly one new
        // token (the steady-state case under continuous batching), we can
        // collapse the per-seq attention loop — gather K, gather V,
        // matmul Q·Kᵀ, softmax, matmul ·V — into one fused Metal kernel.
        // This eliminates O(N · num_layers · ~5) GPU dispatches per step.
        #[cfg(all(target_os = "macos", feature = "candle-metal"))]
        {
            let pure_decode = !seqs.is_empty()
                && seqs.iter().all(|s| s.new_tokens == 1)
                && matches!(x.device(), Device::Metal(_))
                && self.head_dim <= 256
                && self.head_dim % 32 == 0;
            if pure_decode {
                return self.forward_decode_fused(&q_all, &k_all, &v_all, seqs, pages);
            }
        }
        // -----------------------------------------------------------------

        // 3. Per-request: apply RoPE at the right offset, write new K/V into
        //    the page pool, gather full K/V history, then SDPA.
        //
        //    Output is built request by request and concatenated at the end.
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seqs.len());
        let mut cursor = 0usize;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let dtype = x.dtype();
        let device = x.device();
        let page_size = pages.page_size();

        for seq in seqs {
            let n_new = seq.new_tokens;
            let kv_offset = seq.kv_offset;
            let total_kv = kv_offset + n_new;

            // 3a. Slice this request's Q,K,V from the packed batch.
            let q = q_all.narrow(0, cursor, n_new)?; // [n_new, H, D]
            let k_new = k_all.narrow(0, cursor, n_new)?; // [n_new, H_kv, D]
            let v_new = v_all.narrow(0, cursor, n_new)?; // [n_new, H_kv, D]
            cursor += n_new;

            // 3b. RoPE on Q and K for positions [kv_offset .. kv_offset+n_new).
            //     Need [H, L, D] order for rope helper.
            let q_for_rope = q.transpose(0, 1)?.contiguous()?; // [H, n_new, D]
            let k_for_rope = k_new.transpose(0, 1)?.contiguous()?; // [H_kv, n_new, D]
            let (q_rot, k_rot) = self
                .rotary
                .apply_at_offset(&q_for_rope, &k_for_rope, kv_offset)?;

            // 3c. Write the new K/V into the page pool.
            //     We need to scatter them into pages [block_table[kv_offset/page_size] ..]
            //     at the appropriate within-page offset.
            //     To keep this CPU-friendly we reshape and use slice_assign.
            self.write_new_kv_to_pool(&k_rot, &v_new.transpose(0, 1)?.contiguous()?, seq, pages)?;

            // 3d. Gather the full K/V history for this request from the page pool.
            //     Result shape: [H_kv, total_kv, D]
            let (k_hist, v_hist) = self.gather_full_kv(seq, total_kv, dtype, device, pages)?;

            // 3e. GQA: repeat K/V to match Q heads.
            let k_hist = repeat_kv_dim0(k_hist, self.num_kv_groups)?; // [H, total_kv, D]
            let v_hist = repeat_kv_dim0(v_hist, self.num_kv_groups)?;

            // 3f. SDPA. q_rot is [H, n_new, D], k_hist is [H, total_kv, D].
            //     scores = q @ k^T * scale → [H, n_new, total_kv]
            let scores = (q_rot.matmul(&k_hist.transpose(1, 2)?.contiguous()?)? * scale)?;

            // Causal mask: position (kv_offset + i) attends to j ≤ kv_offset + i.
            // For decode (n_new=1) the mask is all-allowed, no need to apply.
            let scores = if n_new > 1 {
                let mask = build_causal_mask(n_new, total_kv, kv_offset, dtype, device)?;
                scores.broadcast_add(&mask)?
            } else {
                scores
            };

            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v_hist)?; // [H, n_new, D]

            // 3g. Reshape back to [n_new, H*D] and project out.
            let ctx = ctx.transpose(0, 1)?.reshape((n_new, self.hidden_size))?;
            let out = self.o_proj.forward(&ctx)?;
            outputs.push(out);
        }

        // 4. Concat per-request outputs back into the packed batch.
        Tensor::cat(&outputs, 0).context("concat attention outputs").map_err(Into::into)
    }

    /// Fused-decode fast path. Pre-conditions checked by caller:
    ///   * every seq has `new_tokens == 1`
    ///   * device is Metal
    ///   * head_dim is 32-aligned and ≤ 256
    ///
    /// Inputs:
    ///   q_all : [N, H_q,  D]   (post-RMSNorm, pre-RoPE)
    ///   k_all : [N, H_kv, D]   (post-RMSNorm, pre-RoPE)
    ///   v_all : [N, H_kv, D]   (no norm, no RoPE)
    ///
    /// We:
    ///   1) Apply RoPE to Q and K per-seq at their kv_offset (cheap, single
    ///      token each), accumulate into a packed [N, H_q, D] / [N, H_kv, D].
    ///   2) Scatter the new K and V into the layer pool (one inplace_op2
    ///      per seq × (K|V), uses our custom MSL scatter kernel).
    ///   3) Build block_table [N, MB] u32 and kv_lens [N] u32 tensors.
    ///   4) Call paged_decode_attention → [N, H_q, D].
    ///   5) o_proj.
    #[cfg(all(target_os = "macos", feature = "candle-metal"))]
    fn forward_decode_fused(
        &self,
        q_all: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        seqs: &[SeqSlice<'_>],
        pages: &PagedKvCache,
    ) -> Result<Tensor> {
        let n = seqs.len();
        let h_q = self.num_heads;
        let h_kv = self.num_kv_heads;
        let d = self.head_dim;
        let device = q_all.device();
        let dtype = q_all.dtype();

        // 1) Per-seq RoPE then accumulate into packed buffers.
        //    We collect q_rot rows and k_rot rows then cat once, then scatter.
        let mut q_rows: Vec<Tensor> = Vec::with_capacity(n);
        let mut k_rows: Vec<Tensor> = Vec::with_capacity(n);
        let mut v_rows: Vec<Tensor> = Vec::with_capacity(n);

        for (i, seq) in seqs.iter().enumerate() {
            // Slice the i-th token from the packed batch.
            let q_i = q_all.narrow(0, i, 1)?; // [1, H_q,  D]
            let k_i = k_all.narrow(0, i, 1)?; // [1, H_kv, D]
            let v_i = v_all.narrow(0, i, 1)?;

            // RoPE wants [H, L=1, D]. transpose(0,1) flips the leading
            // batch and head dims, giving [H, 1, D] directly (NO squeeze).
            let q_for_rope = q_i.transpose(0, 1)?.contiguous()?; // [H_q, 1, D]
            let k_for_rope = k_i.transpose(0, 1)?.contiguous()?; // [H_kv, 1, D]
            let (q_rot, k_rot) = self
                .rotary
                .apply_at_offset(&q_for_rope, &k_for_rope, seq.kv_offset)?;
            // Back to [1, H, D]
            let q_rot = q_rot.transpose(0, 1)?.contiguous()?; // [1, H_q, D]
            let k_rot = k_rot.transpose(0, 1)?.contiguous()?; // [1, H_kv, D]

            q_rows.push(q_rot);
            k_rows.push(k_rot);
            v_rows.push(v_i);
        }

        let q_packed = Tensor::cat(&q_rows, 0)?.contiguous()?; // [N, H_q,  D]
        let k_packed = Tensor::cat(&k_rows, 0)?.contiguous()?; // [N, H_kv, D]
        let v_packed = Tensor::cat(&v_rows, 0)?.contiguous()?; // [N, H_kv, D]

        // 2) Scatter the new K/V into the layer pool, one slot per seq.
        let k_pool = pages.layer_k_pool(self.layer_idx);
        let v_pool = pages.layer_v_pool(self.layer_idx);
        let page_size = pages.page_size();

        for (i, seq) in seqs.iter().enumerate() {
            let logical_pos = seq.kv_offset; // n_new=1, write at kv_offset
            let page_block = logical_pos / page_size;
            let slot = (logical_pos % page_size) as u32;
            let page_id = *seq.block_table.get(page_block).ok_or_else(|| {
                anyhow!(
                    "fused decode: block_table too short: pos={} needs block {}, have {}",
                    logical_pos,
                    page_block,
                    seq.block_table.len()
                )
            })?;

            let k_t = k_packed.narrow(0, i, 1)?.squeeze(0)?; // [H_kv, D]
            let v_t = v_packed.narrow(0, i, 1)?.squeeze(0)?;
            scatter_into_pool(k_pool, &k_t, page_id, slot)?;
            scatter_into_pool(v_pool, &v_t, page_id, slot)?;
        }

        // 3) Build block_table and kv_lens tensors.
        //    kv_lens[i] = kv_offset[i] + 1  (the new token is now in the pool).
        let max_blocks = seqs.iter().map(|s| s.block_table.len()).max().unwrap_or(1);
        let mut bt = vec![0u32; n * max_blocks];
        for (i, seq) in seqs.iter().enumerate() {
            for (j, b) in seq.block_table.iter().enumerate() {
                bt[i * max_blocks + j] = *b;
            }
        }
        let block_table = Tensor::from_vec(bt, (n, max_blocks), device)?;
        let kv_lens: Vec<u32> = seqs.iter().map(|s| (s.kv_offset + 1) as u32).collect();
        let kv_lens_t = Tensor::from_vec(kv_lens, n, device)?;

        // 4) Fused attention.
        let scale = 1.0 / (d as f32).sqrt();
        let attn_out = paged_decode_attention(
            &q_packed,
            k_pool,
            v_pool,
            &block_table,
            &kv_lens_t,
            scale,
        )?; // [N, H_q, D]

        // 5) o_proj. Reshape [N, H_q, D] → [N, H_q*D] = [N, hidden].
        let attn_out = attn_out
            .reshape((n, h_q * d))?
            .to_dtype(dtype)?;
        let _ = h_kv;
        let out = self.o_proj.forward(&attn_out)?;
        Ok(out)
    }

    /// Write K/V for the `new_tokens` newly-arrived positions into the
    /// per-page pool at this request's logical offset.
    ///
    /// `k`, `v` shape: `[H_kv, n_new, D]`.
    ///
    /// We hold the per-layer page Vec lock once, group writes by destination
    /// page, and for each touched page rebuild only that single
    /// `[H_kv, page_size, D]` tensor (much cheaper than rewriting the whole
    /// `[num_pages, ...]` pool).
    fn write_new_kv_to_pool(
        &self,
        k: &Tensor,
        v: &Tensor,
        seq: &SeqSlice<'_>,
        pages: &PagedKvCache,
    ) -> Result<()> {
        let n_new = seq.new_tokens;
        let page_size = pages.page_size();
        let kv_offset = seq.kv_offset;

        let k_pool = pages.layer_k_pool(self.layer_idx);
        let v_pool = pages.layer_v_pool(self.layer_idx);

        for t in 0..n_new {
            let logical_pos = kv_offset + t;
            let page_block = logical_pos / page_size;
            let slot = logical_pos % page_size;
            let page_id = *seq.block_table.get(page_block).ok_or_else(|| {
                anyhow!(
                    "block_table too short: pos={} needs block {}, have {}",
                    logical_pos,
                    page_block,
                    seq.block_table.len()
                )
            })? as u32;

            let k_t = k.narrow(1, t, 1)?.squeeze(1)?; // [H_kv, D]
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;

            scatter_into_pool(k_pool, &k_t, page_id, slot as u32)?;
            scatter_into_pool(v_pool, &v_t, page_id, slot as u32)?;
        }

        Ok(())
    }

    /// Gather this request's K and V history for the first `total_kv` positions
    /// into compact `[H_kv, total_kv, D]` tensors.
    ///
    /// Pages are stored as a `Vec<Tensor>` of `[H_kv, page_size, D]`. We
    /// gather the relevant pages by cloning the references (cheap — Candle
    /// tensors are Arc-backed), concatenate them along the seq dim, then
    /// truncate to `total_kv`.
    fn gather_full_kv(
        &self,
        seq: &SeqSlice<'_>,
        total_kv: usize,
        _dtype: DType,
        device: &Device,
        pages: &PagedKvCache,
    ) -> Result<(Tensor, Tensor)> {
        let page_size = pages.page_size();
        let num_kv_heads = pages.num_kv_heads();
        let head_dim = pages.head_dim();
        let n_pages_needed = total_kv.div_ceil(page_size);

        let block_ids: Vec<u32> = seq.block_table[..n_pages_needed].to_vec();
        let idx = Tensor::from_vec(block_ids, n_pages_needed, device)?;

        let k_pool = pages.layer_k_pool(self.layer_idx); // [P, H_kv, S, D]
        let v_pool = pages.layer_v_pool(self.layer_idx);

        // index_select on dim 0: [n_pages, H_kv, S, D]
        let k = k_pool
            .index_select(&idx, 0)?
            .transpose(0, 1)? // [H_kv, n_pages, S, D]
            .contiguous()?
            .reshape((num_kv_heads, n_pages_needed * page_size, head_dim))?
            .narrow(1, 0, total_kv)?
            .contiguous()?;
        let v = v_pool
            .index_select(&idx, 0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((num_kv_heads, n_pages_needed * page_size, head_dim))?
            .narrow(1, 0, total_kv)?
            .contiguous()?;

        Ok((k, v))
    }
}

/// Repeat the head dim of an [H, L, D] tensor `n_rep` times along H.
fn repeat_kv_dim0(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (h, l, d) = x.dims3()?;
    let x = x
        .unsqueeze(1)? // [H, 1, L, D]
        .expand((h, n_rep, l, d))?
        .reshape((h * n_rep, l, d))?
        .contiguous()?;
    Ok(x)
}

fn build_causal_mask(
    new_tokens: usize,
    total_kv: usize,
    kv_offset: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..new_tokens)
        .flat_map(|i| {
            (0..total_kv).map(move |j| if j <= kv_offset + i { 0.0 } else { minf })
        })
        .collect();
    Tensor::from_slice(&mask, (1, new_tokens, total_kv), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

/// Write a single `[H_kv, D]` slot into the layer KV pool
/// `[P, H_kv, S, D]` at `(page_id, slot)`, in place when on Metal
/// (custom MSL kernel) or via a slow CPU loop fallback.
fn scatter_into_pool(pool: &Tensor, value: &Tensor, page_id: u32, slot: u32) -> Result<()> {
    let (_p, h, s, d) = pool.dims4()?;
    debug_assert!((slot as usize) < s);
    debug_assert_eq!(value.dims2()?, (h, d));

    #[cfg(all(target_os = "macos", feature = "candle-metal"))]
    {
        if matches!(pool.device(), Device::Metal(_)) {
            let value = value.contiguous()?;
            pool.inplace_op2(&value, &ScatterSlot { page_id, slot })?;
            return Ok(());
        }
    }

    // CPU fallback: route through the same InplaceOp2, which handles
    // arbitrary value strides via the layout. This keeps tests honest.
    let value = value.contiguous()?;
    pool.inplace_op2(&value, &ScatterSlot { page_id, slot })?;
    Ok(())
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: PagedQwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Qwen3Config, rotary: Arc<RotaryEmbedding>, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: PagedQwen3Attention::new(cfg, rotary, layer_idx, vb.pp("self_attn"))?,
            mlp: Qwen3MLP::new(cfg, vb.pp("mlp"))?,
            ln1: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, seqs: &[SeqSlice<'_>], pages: &PagedKvCache) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, seqs, pages)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        (x + h2).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct PagedQwen3 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg: Qwen3Config,
    pub device: Device,
    pub dtype: DType,
}

impl PagedQwen3 {
    pub fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), i, vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    /// Run one batched forward step. Returns logits for the LAST new token
    /// of EACH request, as a `[num_requests, vocab_size]` tensor.
    pub fn forward(&self, step: &BatchStep<'_>, pages: &PagedKvCache) -> Result<Tensor> {
        let total = step.input_ids.dim(0)?;
        debug_assert_eq!(total, step.seqs.iter().map(|s| s.new_tokens).sum::<usize>());

        let mut h = self.embed_tokens.forward(step.input_ids)?; // [total, hidden]

        for layer in &self.layers {
            h = layer.forward(&h, step.seqs, pages)?;
        }

        let h = self.norm.forward(&h)?;

        // Pick the last new token's hidden state per request, then project.
        let mut last_hs: Vec<Tensor> = Vec::with_capacity(step.seqs.len());
        let mut cursor = 0usize;
        for seq in step.seqs {
            let last_idx = cursor + seq.new_tokens - 1;
            last_hs.push(h.i(last_idx)?.unsqueeze(0)?);
            cursor += seq.new_tokens;
        }
        let last = Tensor::cat(&last_hs, 0)?; // [num_reqs, hidden]
        let logits = self.lm_head.forward(&last)?;
        Ok(logits)
    }
}
