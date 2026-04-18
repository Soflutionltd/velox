//! Llama 3.x and Mistral support, via the shared `PagedQwen3` engine.
//!
//! Llama and Qwen3 share ~95% of their architecture (RMSNorm, GQA,
//! SwiGLU MLP, RoPE attention). The only structural difference Velox
//! needs to handle is that Qwen3 has per-head Q/K RMSNorms (a Qwen3
//! signature) which Llama does not. That's already handled in
//! [`super::qwen3::PagedQwen3Attention`] by making `q_norm`/`k_norm`
//! `Option`al and probing the checkpoint at load time.
//!
//! What this module does:
//!
//!   * Parses the Llama / Mistral `config.json` schema
//!     ([`LlamaConfig`]).
//!   * Maps it into the internal [`Qwen3Config`] shape (we keep one
//!     internal config struct rather than introducing a parallel
//!     hierarchy).
//!   * Calls [`PagedQwen3::load`] to build the actual model.
//!
//! ## Caveats (v1)
//!
//! * **RoPE scaling**: Llama 3.1+ uses a custom NTK-aware
//!   per-frequency-band scaling (`rope_type: "llama3"`). This module
//!   currently ignores `rope_scaling` and uses plain RoPE. That works
//!   correctly up to `original_max_position_embeddings` (8192 for
//!   Llama 3) but degrades beyond that. Long-context Llama 3.1 (128K)
//!   support is queued as a follow-up.
//! * **Sliding window** (Mistral): rejected at load time. Mistral 7B
//!   v0.3 and later use full attention so most checkpoints just work.
//!
//! Models tested with the paged backend through this module:
//!
//!   * `mlx-community/Llama-3.2-1B-Instruct[-4bit]`
//!   * `mlx-community/Llama-3.2-3B-Instruct[-4bit]`
//!   * `mlx-community/Llama-3.1-8B-Instruct[-4bit]`
//!   * `mlx-community/Mistral-7B-Instruct-v0.3[-4bit]`

use super::qwen3::{PagedQwen3, QuantConfig, Qwen3Config};
use anyhow::{anyhow, Context, Result};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;
use std::path::Path;

/// Llama / Mistral `config.json` schema. This is a strict subset of the
/// HuggingFace transformers Llama config — fields we don't need are
/// ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Optional in older Llama configs (computed as
    /// `hidden_size / num_attention_heads`). Llama 3.x and Mistral
    /// both write it explicitly.
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default = "default_silu")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Mistral uses `sliding_window`; we reject any non-null value.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Llama 3.1+ scaling: ignored in v1 (works up to original ctx len).
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// MLX-quantized checkpoints carry both `quantization` and
    /// `quantization_config` (HF-compat). They serialise identically.
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
    #[serde(default, rename = "quantization_config")]
    _quantization_config_ignored: Option<serde_json::Value>,
}

fn default_silu() -> Activation {
    Activation::Silu
}

impl LlamaConfig {
    /// Convert into the internal [`Qwen3Config`] shape used by
    /// [`PagedQwen3`]. Llama-specific gotchas are handled here:
    ///
    /// * Compute `head_dim` from `hidden_size / num_attention_heads`
    ///   if absent.
    /// * Default `num_key_value_heads` to `num_attention_heads` (MHA).
    /// * Reject sliding-window (not supported in the paged backend).
    /// * `attention_bias` is always `false` for Llama / Mistral.
    pub fn to_internal(self) -> Result<Qwen3Config> {
        if let Some(w) = self.sliding_window {
            // Mistral 7B v0.1/v0.2 used 4096; v0.3+ doesn't. Llama
            // never uses sliding window.
            anyhow::bail!(
                "sliding window attention (window={}) is not yet supported in the paged backend",
                w
            );
        }
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        let num_kv_heads = self
            .num_key_value_heads
            .unwrap_or(self.num_attention_heads);

        // Internal config carries Qwen3-specific fields we don't need
        // here; we just zero them out — the model loader skips them
        // when q_norm/k_norm aren't present in the checkpoint.
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
        ))
    }
}

/// Load a Llama / Mistral checkpoint from a directory containing
/// `config.json` and `model.safetensors`. Returns a fully-built
/// [`PagedQwen3`] ready to be wired into the scheduler.
pub fn load_paged_llama(model_dir: &Path, vb: VarBuilder) -> Result<PagedQwen3> {
    let cfg_path = model_dir.join("config.json");
    let cfg_raw =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))?;
    let llama_cfg: LlamaConfig =
        serde_json::from_str(&cfg_raw).with_context(|| format!("parse {}", cfg_path.display()))?;

    let arch = sniff_arch(&cfg_raw)
        .ok_or_else(|| anyhow!("config.json is missing `architectures` or `model_type`"))?;
    if !is_llama_family(&arch) {
        anyhow::bail!(
            "unsupported model architecture {:?}; only Llama / Mistral families are routed through this module",
            arch
        );
    }

    let internal = llama_cfg.to_internal()?;
    PagedQwen3::load(&internal, vb)
}

fn sniff_arch(cfg_raw: &str) -> Option<String> {
    // We accept either of the standard HF fields. `architectures` is
    // an array; we take the first entry. `model_type` is a single
    // string.
    let v: serde_json::Value = serde_json::from_str(cfg_raw).ok()?;
    if let Some(arr) = v.get("architectures").and_then(|x| x.as_array()) {
        if let Some(first) = arr.first().and_then(|s| s.as_str()) {
            return Some(first.to_string());
        }
    }
    v.get("model_type")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

fn is_llama_family(arch: &str) -> bool {
    matches!(
        arch,
        "LlamaForCausalLM"
            | "MistralForCausalLM"
            | "llama"
            | "mistral"
    )
}
