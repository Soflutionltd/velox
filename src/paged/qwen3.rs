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
use super::metal_kernels::{
    batched_rope_decode, batched_scatter, paged_decode_attention, paged_prefill_attention,
    ScatterSlot,
};

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
    /// MLX-quant config, present iff the checkpoint is 4-bit quantized.
    /// MLX writes this both as `quantization` and (for HF compat)
    /// `quantization_config`. Both serialize to the same shape so we
    /// just read `quantization`; HF-only checkpoints would require a
    /// manual fallback (none in the wild for MLX-quant).
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
    #[serde(default, rename = "quantization_config")]
    _quantization_config_ignored: Option<serde_json::Value>,
}

/// MLX 4-bit quantization parameters (parsed from the model's
/// `config.json` `quantization_config` field, or `quantization` for
/// older mlx-community dumps).
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
pub struct QuantConfig {
    pub bits: usize,
    pub group_size: usize,
}

/// A linear layer that may be either a regular `Linear` or a 4-bit
/// MLX-quant `QLinear`. Same forward interface either way.
#[derive(Debug)]
pub enum MaybeQLinear {
    Plain(Linear),
    Q(QLinear),
}

impl MaybeQLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            MaybeQLinear::Plain(l) => Ok(l.forward(x)?),
            MaybeQLinear::Q(q) => q.forward(x),
        }
    }
}

/// 4-bit MLX-quantized linear layer.
///
/// Storage matches MLX exactly:
///   weight  : [N, K/8]            U32 packed (8 Int4 per u32, little-endian)
///   scales  : [N, K/group_size]   activation dtype
///   biases  : [N, K/group_size]   activation dtype  (= original w_min)
///   bias    : Option<[N]>         activation dtype  (Linear's add-bias)
#[derive(Debug)]
pub struct QLinear {
    pub qweight: Tensor,
    pub scales: Tensor,
    pub biases: Tensor,
    pub bias: Option<Tensor>,
    pub group_size: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl QLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // qmm_4bit expects 2D X. Flatten leading dims, restore after.
        let original_shape = x.dims().to_vec();
        let k = self.in_features;
        let leading: usize = original_shape[..original_shape.len() - 1].iter().product();
        let x2d = x.reshape((leading, k))?.contiguous()?;
        #[cfg(all(target_os = "macos", feature = "candle-metal"))]
        {
            let y = super::metal_kernels::qmm_4bit(
                &x2d,
                &self.qweight,
                &self.scales,
                &self.biases,
                self.bias.as_ref(),
                self.group_size,
            )?;
            let mut new_shape = original_shape.clone();
            *new_shape.last_mut().unwrap() = self.out_features;
            return Ok(y.reshape(new_shape)?);
        }
        #[cfg(not(all(target_os = "macos", feature = "candle-metal")))]
        {
            let _ = x2d;
            anyhow::bail!("QLinear::forward requires macOS + candle-metal feature");
        }
    }
}

/// Build a `MaybeQLinear` from a VarBuilder. If `q_cfg` is `Some` we
/// load `weight` (U32 packed), `scales`, `biases`, and optional `bias`.
/// Otherwise we fall back to `linear_b` / `linear_no_bias`.
fn maybe_qlinear(
    in_features: usize,
    out_features: usize,
    bias: bool,
    vb: VarBuilder,
    q_cfg: Option<&QuantConfig>,
) -> Result<MaybeQLinear> {
    if let Some(q) = q_cfg {
        if q.bits != 4 {
            anyhow::bail!("velox only supports 4-bit MLX quant (got bits={})", q.bits);
        }
        let g = q.group_size;
        if in_features % g != 0 {
            anyhow::bail!(
                "QLinear: in_features {in_features} not divisible by group_size {g}"
            );
        }
        if in_features % 8 != 0 {
            anyhow::bail!("QLinear: in_features {in_features} not multiple of 8");
        }
        let qweight = vb.get_with_hints_dtype(
            (out_features, in_features / 8),
            "weight",
            Default::default(),
            DType::U32,
        )?;
        let scales = vb.get((out_features, in_features / g), "scales")?;
        let biases = vb.get((out_features, in_features / g), "biases")?;
        let b = if bias {
            Some(vb.get(out_features, "bias")?)
        } else {
            None
        };
        Ok(MaybeQLinear::Q(QLinear {
            qweight,
            scales,
            biases,
            bias: b,
            group_size: g,
            in_features,
            out_features,
        }))
    } else if bias {
        Ok(MaybeQLinear::Plain(linear_b(
            in_features,
            out_features,
            true,
            vb,
        )?))
    } else {
        Ok(MaybeQLinear::Plain(linear_no_bias(
            in_features,
            out_features,
            vb,
        )?))
    }
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

#[derive(Debug)]
struct Qwen3MLP {
    gate_proj: MaybeQLinear,
    up_proj: MaybeQLinear,
    down_proj: MaybeQLinear,
    act_fn: Activation,
}

impl Qwen3MLP {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let q = cfg.quantization.as_ref();
        Ok(Self {
            gate_proj: maybe_qlinear(cfg.hidden_size, cfg.intermediate_size, false, vb.pp("gate_proj"), q)?,
            up_proj: maybe_qlinear(cfg.hidden_size, cfg.intermediate_size, false, vb.pp("up_proj"), q)?,
            down_proj: maybe_qlinear(cfg.intermediate_size, cfg.hidden_size, false, vb.pp("down_proj"), q)?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = self.gate_proj.forward(x).map_err(candle_core::Error::wrap)?
            .apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(x).map_err(candle_core::Error::wrap)?;
        let h = (lhs * rhs)?;
        self.down_proj.forward(&h).map_err(candle_core::Error::wrap)
    }
}

#[derive(Debug)]
struct PagedQwen3Attention {
    q_proj: MaybeQLinear,
    k_proj: MaybeQLinear,
    v_proj: MaybeQLinear,
    o_proj: MaybeQLinear,
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
        let q_cfg = cfg.quantization.as_ref();
        let q_proj = maybe_qlinear(cfg.hidden_size, num_heads * head_dim, cfg.attention_bias, vb.pp("q_proj"), q_cfg)?;
        let k_proj = maybe_qlinear(cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_bias, vb.pp("k_proj"), q_cfg)?;
        let v_proj = maybe_qlinear(cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_bias, vb.pp("v_proj"), q_cfg)?;
        let o_proj = maybe_qlinear(num_heads * head_dim, cfg.hidden_size, cfg.attention_bias, vb.pp("o_proj"), q_cfg)?;
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

        // ---- Fused fast paths -------------------------------------------
        //
        // We have two specialised Metal paths:
        //   * forward_decode_fused : every seq has exactly 1 new token
        //     (steady state under continuous batching).
        //   * forward_prefill_fused : at least one seq has >1 new tokens,
        //     i.e. we're admitting / chunked-prefilling. Both paths
        //     collapse the per-seq attention loop (gather K, gather V,
        //     Q·Kᵀ, softmax, ·V) into one Metal kernel dispatch per
        //     layer.
        #[cfg(all(target_os = "macos", feature = "candle-metal"))]
        {
            let metal_ok = matches!(x.device(), Device::Metal(_))
                && self.head_dim <= 256
                && self.head_dim % 32 == 0;
            if metal_ok && !seqs.is_empty() {
                let pure_decode = seqs.iter().all(|s| s.new_tokens == 1);
                if pure_decode {
                    return self.forward_decode_fused(&q_all, &k_all, &v_all, seqs, pages);
                }
                return self.forward_prefill_fused(&q_all, &k_all, &v_all, seqs, pages);
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

    /// Fused-decode fast path (pure decode: every seq has n_new=1).
    ///
    /// Pipeline (all on Metal, ~3 dispatches per layer instead of N+5):
    ///   1) batched_rope_decode on Q (in-place) using per-seq offsets
    ///   2) batched_rope_decode on K (in-place) using per-seq offsets
    ///   3) batched_scatter on K_pool with per-seq (page_id, slot)
    ///   4) batched_scatter on V_pool with per-seq (page_id, slot)
    ///   5) paged_decode_attention → [N, H_q, D]
    ///   6) o_proj
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
        let d = self.head_dim;
        let device = q_all.device();
        let dtype = q_all.dtype();
        let page_size = pages.page_size();

        // 1+2) Per-seq RoPE on Q and K via the batched in-place kernel.
        //      We need contiguous [N, H, D] tensors. q_all/k_all are already
        //      that shape; just .contiguous() to drop any view stride.
        let q_packed = q_all.contiguous()?;
        let k_packed = k_all.contiguous()?;
        let v_packed = v_all.contiguous()?;

        let offsets: Vec<u32> = seqs.iter().map(|s| s.kv_offset as u32).collect();
        let offsets_t = Tensor::from_vec(offsets, n, device)?;

        batched_rope_decode(&q_packed, &self.rotary.cos, &self.rotary.sin, &offsets_t)?;
        batched_rope_decode(&k_packed, &self.rotary.cos, &self.rotary.sin, &offsets_t)?;

        // 3+4) Build (page_id, slot) per seq and batch-scatter K/V.
        let mut page_ids = Vec::with_capacity(n);
        let mut slots = Vec::with_capacity(n);
        for seq in seqs {
            let logical_pos = seq.kv_offset;
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
            page_ids.push(page_id);
            slots.push(slot);
        }
        let page_ids_t = Tensor::from_vec(page_ids, n, device)?;
        let slots_t = Tensor::from_vec(slots, n, device)?;

        let k_pool = pages.layer_k_pool(self.layer_idx);
        let v_pool = pages.layer_v_pool(self.layer_idx);

        batched_scatter(k_pool, &k_packed, &page_ids_t, &slots_t)?;
        batched_scatter(v_pool, &v_packed, &page_ids_t, &slots_t)?;

        // 5) Block table + kv_lens for the fused attention kernel.
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

        let scale = 1.0 / (d as f32).sqrt();
        let attn_out = paged_decode_attention(
            &q_packed,
            k_pool,
            v_pool,
            &block_table,
            &kv_lens_t,
            scale,
        )?; // [N, H_q, D]

        // 6) o_proj.
        let attn_out = attn_out.reshape((n, h_q * d))?.to_dtype(dtype)?;
        let out = self.o_proj.forward(&attn_out)?;
        Ok(out)
    }

    /// Fused-prefill fast path (any batch with n_new ≥ 1, including
    /// chunked-prefill admission steps).
    ///
    /// Pipeline:
    ///   1) Build per-position absolute offsets, RoPE Q and K in place
    ///      using the batched kernel.
    ///   2) Build per-position (page_id, slot) and batch-scatter K and V.
    ///   3) Build cu_seqlens, seq_id_per_q, kv_offsets tensors.
    ///   4) paged_prefill_attention → [total_q, H_q, D].
    ///   5) o_proj.
    #[cfg(all(target_os = "macos", feature = "candle-metal"))]
    fn forward_prefill_fused(
        &self,
        q_all: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        seqs: &[SeqSlice<'_>],
        pages: &PagedKvCache,
    ) -> Result<Tensor> {
        let total_q = q_all.dim(0)?;
        let h_q = self.num_heads;
        let d = self.head_dim;
        let device = q_all.device();
        let dtype = q_all.dtype();
        let page_size = pages.page_size();

        // 1) Build per-token absolute RoPE offsets (one entry per packed
        //    query position, accounting for kv_offset + position-in-seq).
        let mut tok_offsets = Vec::with_capacity(total_q);
        let mut tok_page_ids = Vec::with_capacity(total_q);
        let mut tok_slots = Vec::with_capacity(total_q);
        for seq in seqs {
            for t in 0..seq.new_tokens {
                let abs_pos = seq.kv_offset + t;
                let page_block = abs_pos / page_size;
                let slot = (abs_pos % page_size) as u32;
                let page_id = *seq.block_table.get(page_block).ok_or_else(|| {
                    anyhow!(
                        "fused prefill: block_table too short: abs_pos={} needs block {}, have {}",
                        abs_pos,
                        page_block,
                        seq.block_table.len()
                    )
                })?;
                tok_offsets.push(abs_pos as u32);
                tok_page_ids.push(page_id);
                tok_slots.push(slot);
            }
        }
        debug_assert_eq!(tok_offsets.len(), total_q);

        let q_packed = q_all.contiguous()?;
        let k_packed = k_all.contiguous()?;
        let v_packed = v_all.contiguous()?;

        let offsets_t = Tensor::from_vec(tok_offsets, total_q, device)?;
        batched_rope_decode(&q_packed, &self.rotary.cos, &self.rotary.sin, &offsets_t)?;
        batched_rope_decode(&k_packed, &self.rotary.cos, &self.rotary.sin, &offsets_t)?;

        // 2) Batch-scatter K/V at (page_id, slot) per token.
        let page_ids_t = Tensor::from_vec(tok_page_ids, total_q, device)?;
        let slots_t = Tensor::from_vec(tok_slots, total_q, device)?;

        let k_pool = pages.layer_k_pool(self.layer_idx);
        let v_pool = pages.layer_v_pool(self.layer_idx);
        batched_scatter(k_pool, &k_packed, &page_ids_t, &slots_t)?;
        batched_scatter(v_pool, &v_packed, &page_ids_t, &slots_t)?;

        // 3) Build cu_seqlens, seq_id_per_q, kv_offsets for the kernel.
        let n = seqs.len();
        let mut cu_seqlens = Vec::with_capacity(n + 1);
        let mut seq_id_per_q = Vec::with_capacity(total_q);
        let mut kv_offsets = Vec::with_capacity(n);
        let mut acc = 0u32;
        cu_seqlens.push(0u32);
        for (i, seq) in seqs.iter().enumerate() {
            for _ in 0..seq.new_tokens {
                seq_id_per_q.push(i as u32);
            }
            acc += seq.new_tokens as u32;
            cu_seqlens.push(acc);
            kv_offsets.push(seq.kv_offset as u32);
        }

        let max_blocks = seqs.iter().map(|s| s.block_table.len()).max().unwrap_or(1);
        let mut bt = vec![0u32; n * max_blocks];
        for (i, seq) in seqs.iter().enumerate() {
            for (j, b) in seq.block_table.iter().enumerate() {
                bt[i * max_blocks + j] = *b;
            }
        }
        let block_table = Tensor::from_vec(bt, (n, max_blocks), device)?;
        let cu_t = Tensor::from_vec(cu_seqlens, n + 1, device)?;
        let sid_t = Tensor::from_vec(seq_id_per_q, total_q, device)?;
        let kvo_t = Tensor::from_vec(kv_offsets, n, device)?;

        // 4) Fused varlen prefill attention with built-in causal mask.
        let scale = 1.0 / (d as f32).sqrt();
        let attn_out = paged_prefill_attention(
            &q_packed,
            k_pool,
            v_pool,
            &block_table,
            &cu_t,
            &sid_t,
            &kvo_t,
            scale,
        )?; // [total_q, H_q, D]

        // 5) o_proj.
        let attn_out = attn_out.reshape((total_q, h_q * d))?.to_dtype(dtype)?;
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

#[derive(Debug)]
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

#[derive(Debug)]
pub struct PagedQwen3 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: MaybeQLinear,
    cfg: Qwen3Config,
    pub device: Device,
    pub dtype: DType,
}

impl PagedQwen3 {
    pub fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        // For MLX-quant checkpoints, embed_tokens is also stored as
        // qweight + scales + biases. We detect that and dequantize it
        // once at load time so the rest of the model stays unchanged.
        let embed_tokens = load_embedding(cfg, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), i, vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            MaybeQLinear::Plain(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            maybe_qlinear(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                vb.pp("lm_head"),
                cfg.quantization.as_ref(),
            )?
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

/// Load the embedding table. For MLX-quant checkpoints, the table is
/// stored as `weight` (U32 packed) + `scales` + `biases` and we
/// dequantize it once at load time. For plain checkpoints we just defer
/// to `candle_nn::embedding`.
fn load_embedding(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Embedding> {
    let qcfg = match &cfg.quantization {
        None => return Ok(candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb)?),
        Some(q) => q,
    };
    // Try the quant triplet first; if scales is missing, the embed table
    // was kept full-precision in this checkpoint.
    let n = cfg.vocab_size;
    let k = cfg.hidden_size;
    let g = qcfg.group_size;
    let dtype = vb.dtype();
    let dev = vb.device().clone();

    if k % g != 0 || k % 8 != 0 {
        return Ok(candle_nn::embedding(n, k, vb)?);
    }
    let scales_res = vb.get((n, k / g), "scales");
    if scales_res.is_err() {
        return Ok(candle_nn::embedding(n, k, vb)?);
    }

    let qweight = vb.get_with_hints_dtype(
        (n, k / 8),
        "weight",
        Default::default(),
        DType::U32,
    )?;
    let scales = scales_res?;
    let biases = vb.get((n, k / g), "biases")?;

    // Dequantize on CPU (this happens once, embed table is small).
    let qw_cpu: Vec<u32> = qweight.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
    let sc_cpu: Vec<f32> = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let bi_cpu: Vec<f32> = biases.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let k_packed = k / 8;
    let k_groups = k / g;
    let mut dq = vec![0f32; n * k];
    for ni in 0..n {
        for kg in 0..k_groups {
            let scale = sc_cpu[ni * k_groups + kg];
            let bias = bi_cpu[ni * k_groups + kg];
            for kp in 0..(g / 8) {
                let pkg = qw_cpu[ni * k_packed + kg * (g / 8) + kp];
                for b in 0..8 {
                    let q = ((pkg >> (b * 4)) & 0xF) as f32;
                    dq[ni * k + kg * g + kp * 8 + b] = q * scale + bias;
                }
            }
        }
    }
    let table = Tensor::from_vec(dq, (n, k), &dev)?.to_dtype(dtype)?;
    Ok(Embedding::new(table, k))
}
