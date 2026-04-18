//! Phi-3 paged backend.
//!
//! Phi-3 ships with two **fused** projections that the rest of our
//! paged engine doesn't know about:
//!
//!   * `self_attn.qkv_proj`   — out = (n_q + 2 * n_kv) * head_dim
//!   * `mlp.gate_up_proj`     — out = 2 * intermediate_size
//!
//! Rather than fork the whole [`PagedQwen3`] model, we wrap the inner
//! `SimpleBackend` so requests for the *unfused* names that PagedQwen3
//! emits (`q_proj.weight`, `gate_proj.weight`, …) are answered by
//! row-slicing the corresponding fused tensor. The slice happens at
//! VarBuilder load time, so the rest of the engine — Metal kernels,
//! scheduler, KV pool, prefix cache — is identical.
//!
//! 4-bit slicing notes
//! -------------------
//! MLX-Int4 stores quantized projections as three tensors:
//!   * `weight` : `[out, K/8]`           U32 packed (8 nibbles / u32)
//!   * `scales` : `[out, K/group_size]`  activation dtype
//!   * `biases` : `[out, K/group_size]`  activation dtype
//!
//! All three are row-major, so narrowing rows along dim 0 is
//! layout-preserving — the bit-packing happens along K (input dim),
//! never across the output rows we're splitting on. We `.contiguous()`
//! the slice anyway because downstream `qmm_4bit` requires it.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Activation, Init, VarBuilder};
use serde::Deserialize;
use std::path::Path;

use crate::paged::qwen3::{PagedQwen3, QuantConfig, Qwen3Config};

/// Subset of `config.json` we read for Phi-3. Field names match HF.
#[derive(Debug, Clone, Deserialize)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Phi-3 mini-128k uses LongRoPE; we don't support that yet and
    /// bail loudly so users aren't silently mis-served.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// Phi-3 mini sets sliding_window=2047. Plumbed straight to the
    /// kernel.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// MLX-quant config (group size + bits). Present iff the
    /// checkpoint is 4-bit quantized.
    #[serde(default, alias = "quantization_config")]
    pub quantization: Option<QuantConfig>,
}

fn default_act() -> Activation {
    Activation::Silu
}

impl Phi3Config {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn to_qwen3_internal(self) -> Result<Qwen3Config> {
        if let Some(scaling) = &self.rope_scaling {
            // Phi-3 mini-128k uses LongRoPE / SU. We don't yet implement
            // the per-layer frequency rescaling, so refuse rather than
            // silently degrade quality on long contexts.
            anyhow::bail!(
                "Phi-3 rope_scaling ({scaling}) is not supported yet — load the 4k variant instead"
            );
        }
        let head_dim = self.head_dim();
        let num_kv_heads = self.num_kv_heads();
        Ok(Qwen3Config::for_llama(
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            head_dim,
            num_kv_heads,
            self.max_position_embeddings,
            self.tie_word_embeddings,
            self.rope_theta,
            self.rms_norm_eps,
            self.hidden_act,
            self.quantization,
            self.sliding_window,
        ))
    }
}

/// Backend wrapper that synthesizes split q/k/v and gate/up tensors
/// from Phi-3's fused checkpoints. All other names pass through.
struct Phi3SplitBackend<'a> {
    inner: Box<dyn SimpleBackend + 'a>,
    /// Number of rows in each q/k/v slice along the qkv_proj output dim.
    q_rows: usize,
    kv_rows: usize,
    /// Intermediate size — gate_up_proj is split into 2 * intermediate.
    intermediate: usize,
}

impl<'a> Phi3SplitBackend<'a> {
    fn new(inner: Box<dyn SimpleBackend + 'a>, cfg: &Phi3Config) -> Self {
        let head_dim = cfg.head_dim();
        let num_kv = cfg.num_kv_heads();
        Self {
            inner,
            q_rows: cfg.num_attention_heads * head_dim,
            kv_rows: num_kv * head_dim,
            intermediate: cfg.intermediate_size,
        }
    }

    /// Look at `name` and decide whether it's a synthesized projection.
    /// Returns `Some((fused_name, start_row, num_rows))` if the tensor
    /// should be sliced from a fused parent, or `None` for pass-through.
    fn slice_spec(&self, name: &str) -> Option<(String, usize, usize)> {
        // attn fused: ".self_attn.{q,k,v}_proj.{weight,scales,biases,bias}"
        if let Some(rest) = strip_suffix_after(name, ".self_attn.") {
            // rest = "<head>.{weight|scales|biases|bias}"
            let (head, suffix) = split_last_dot(rest)?;
            let parent_path = name.strip_suffix(rest)?; // includes "self_attn."
            let (start, len) = match head {
                "q_proj" => (0, self.q_rows),
                "k_proj" => (self.q_rows, self.kv_rows),
                "v_proj" => (self.q_rows + self.kv_rows, self.kv_rows),
                _ => return None,
            };
            return Some((format!("{}qkv_proj.{}", parent_path, suffix), start, len));
        }
        // mlp fused: ".mlp.{gate,up}_proj.{weight,scales,biases,bias}"
        if let Some(rest) = strip_suffix_after(name, ".mlp.") {
            let (head, suffix) = split_last_dot(rest)?;
            let parent_path = name.strip_suffix(rest)?;
            let (start, len) = match head {
                "gate_proj" => (0, self.intermediate),
                "up_proj" => (self.intermediate, self.intermediate),
                _ => return None,
            };
            return Some((
                format!("{}gate_up_proj.{}", parent_path, suffix),
                start,
                len,
            ));
        }
        None
    }

    fn slice_along_rows(&self, fused: &Tensor, start: usize, len: usize) -> Result<Tensor> {
        let shape = fused.dims();
        if shape.is_empty() {
            return Err(anyhow!(
                "phi3 split: cannot slice a 0-D fused tensor (got shape={:?})",
                shape
            ));
        }
        if start + len > shape[0] {
            return Err(anyhow!(
                "phi3 split: range [{start}, {}) exceeds rows {} of fused tensor",
                start + len,
                shape[0]
            ));
        }
        Ok(fused.narrow(0, start, len)?.contiguous()?)
    }
}

impl<'a> SimpleBackend for Phi3SplitBackend<'a> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some((fused_name, start, len)) = self.slice_spec(name) {
            let fused = self.inner.get_unchecked(&fused_name, dtype, dev)?;
            let sliced = self
                .slice_along_rows(&fused, start, len)
                .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
            // Validate against the requested shape so any mismatch
            // explodes here, not deep inside qmm_4bit.
            if sliced.dims() != s.dims() {
                return Err(candle_core::Error::Msg(format!(
                    "phi3 split: requested {name} shape {:?} but sliced {:?}",
                    s.dims(),
                    sliced.dims()
                )));
            }
            return Ok(sliced);
        }
        self.inner.get(s, name, h, dtype, dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some((fused_name, start, len)) = self.slice_spec(name) {
            let fused = self.inner.get_unchecked(&fused_name, dtype, dev)?;
            return self
                .slice_along_rows(&fused, start, len)
                .map_err(|e| candle_core::Error::Msg(format!("{e:#}")));
        }
        self.inner.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if let Some((fused_name, _, _)) = self.slice_spec(name) {
            return self.inner.contains_tensor(&fused_name);
        }
        self.inner.contains_tensor(name)
    }
}

/// Returns the slice of `name` *after* `marker`, if `marker` appears.
/// Used to pull out the bit after `.self_attn.` / `.mlp.` etc.
fn strip_suffix_after<'a>(name: &'a str, marker: &str) -> Option<&'a str> {
    let idx = name.rfind(marker)?;
    Some(&name[idx + marker.len()..])
}

fn split_last_dot(s: &str) -> Option<(&str, &str)> {
    let idx = s.rfind('.')?;
    Some((&s[..idx], &s[idx + 1..]))
}

/// Load a Phi-3 checkpoint from a directory containing `config.json`
/// and one or more safetensors shards. Wraps the inner safetensors
/// backend with a [`Phi3SplitBackend`] then defers to [`PagedQwen3`]
/// for everything else (decoder layer, RoPE, attention kernels…).
///
/// The lifetime parameter `'a` on `vb` flows through into the wrapped
/// VarBuilder so that mmap-backed checkpoints (`VarBuilder<'static>`)
/// keep their full validity.
pub fn load_paged_phi3<'a>(model_dir: &Path, vb: VarBuilder<'a>) -> Result<PagedQwen3>
where
    'a: 'static,
{
    let cfg_path = model_dir.join("config.json");
    let cfg_raw = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("read {}", cfg_path.display()))?;
    let phi3_cfg: Phi3Config =
        serde_json::from_str(&cfg_raw).with_context(|| format!("parse {}", cfg_path.display()))?;

    let device = vb.device().clone();
    let dtype = vb.dtype();

    let inner: Box<dyn SimpleBackend + 'static> = Box::new(VarBuilderInnerAdapter { vb });
    let split_backend = Phi3SplitBackend::new(inner, &phi3_cfg);
    let split_vb = VarBuilder::from_backend(Box::new(split_backend), dtype, device);

    let qwen3_cfg = phi3_cfg.to_qwen3_internal()?;
    PagedQwen3::load(&qwen3_cfg, split_vb)
        .map_err(|e| anyhow!("PagedQwen3::load (phi3): {e:#}"))
}

/// Adapter so a [`VarBuilder`] (which holds a backend internally)
/// can itself be used as a [`SimpleBackend`]. We forward by name
/// using `VarBuilder::get_unchecked` for the typed get and
/// `VarBuilder::contains_tensor` for the existence check.
struct VarBuilderInnerAdapter<'a> {
    vb: VarBuilder<'a>,
}

impl<'a> SimpleBackend for VarBuilderInnerAdapter<'a> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        _dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Delegate to the underlying VarBuilder's `get` with default
        // hints. We re-validate the shape afterwards because the inner
        // backend's `get` enforces the requested shape, while
        // `get_unchecked` does not.
        let t = self.vb.get_unchecked_dtype(name, dtype)?;
        if t.dims() != s.dims() {
            return Err(candle_core::Error::Msg(format!(
                "VarBuilderInnerAdapter: tensor {name} has shape {:?}, requested {:?}",
                t.dims(),
                s.dims()
            )));
        }
        Ok(t)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        _dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.vb.get_unchecked_dtype(name, dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.vb.contains_tensor(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_spec_qkv() {
        let cfg = Phi3Config {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            rope_scaling: None,
            sliding_window: Some(2047),
            quantization: None,
        };
        let backend = Phi3SplitBackend::new(Box::new(DummyBackend), &cfg);

        // Phi-3 mini: head_dim=96, q_rows=32*96=3072, kv_rows=32*96=3072.
        let n = "model.layers.5.self_attn.q_proj.weight";
        assert_eq!(
            backend.slice_spec(n),
            Some(("model.layers.5.self_attn.qkv_proj.weight".into(), 0, 3072))
        );
        let n = "model.layers.5.self_attn.k_proj.scales";
        assert_eq!(
            backend.slice_spec(n),
            Some((
                "model.layers.5.self_attn.qkv_proj.scales".into(),
                3072,
                3072
            ))
        );
        let n = "model.layers.5.self_attn.v_proj.biases";
        assert_eq!(
            backend.slice_spec(n),
            Some((
                "model.layers.5.self_attn.qkv_proj.biases".into(),
                6144,
                3072
            ))
        );
    }

    #[test]
    fn slice_spec_gate_up() {
        let cfg = Phi3Config {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            rope_scaling: None,
            sliding_window: None,
            quantization: None,
        };
        let backend = Phi3SplitBackend::new(Box::new(DummyBackend), &cfg);

        let n = "model.layers.0.mlp.gate_proj.weight";
        assert_eq!(
            backend.slice_spec(n),
            Some(("model.layers.0.mlp.gate_up_proj.weight".into(), 0, 8192))
        );
        let n = "model.layers.0.mlp.up_proj.weight";
        assert_eq!(
            backend.slice_spec(n),
            Some((
                "model.layers.0.mlp.gate_up_proj.weight".into(),
                8192,
                8192
            ))
        );
    }

    #[test]
    fn slice_spec_passthrough() {
        let cfg = Phi3Config {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            rope_scaling: None,
            sliding_window: None,
            quantization: None,
        };
        let backend = Phi3SplitBackend::new(Box::new(DummyBackend), &cfg);

        // Names that aren't fused projections must pass through.
        assert_eq!(
            backend.slice_spec("model.layers.5.self_attn.o_proj.weight"),
            None
        );
        assert_eq!(
            backend.slice_spec("model.layers.5.mlp.down_proj.weight"),
            None
        );
        assert_eq!(backend.slice_spec("model.embed_tokens.weight"), None);
        assert_eq!(backend.slice_spec("model.norm.weight"), None);
    }

    struct DummyBackend;
    impl SimpleBackend for DummyBackend {
        fn get(
            &self,
            _s: Shape,
            name: &str,
            _h: Init,
            _dtype: DType,
            _dev: &Device,
        ) -> candle_core::Result<Tensor> {
            Err(candle_core::Error::Msg(format!("dummy: {name}")))
        }
        fn get_unchecked(
            &self,
            name: &str,
            _dtype: DType,
            _dev: &Device,
        ) -> candle_core::Result<Tensor> {
            Err(candle_core::Error::Msg(format!("dummy: {name}")))
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            false
        }
    }
}
