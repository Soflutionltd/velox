//! Apple MLX-derived Metal kernels (vendored from `vendor/mlx-quantized`).
//!
//! This module hosts kernels ported from MLX
//! (<https://github.com/ml-explore/mlx>, MIT licence — see
//! `vendor/mlx-quantized/LICENSE`). MLX kernels are the state of the art
//! for quantized inference on Apple Silicon, written and tuned by Apple's
//! own Metal team. Porting them lets Velox match MLX-LM single-stream
//! perf while keeping our continuous-batching / paged-cache server
//! architecture untouched.
//!
//! ## Port progression
//!
//! Kernels land here one at a time. Each landing must:
//!   1. Pass parity vs the existing `qmm_4bit_cpu` reference.
//!   2. Beat (or at least match) the current Velox kernel in
//!      `tests/metal_qmm_v2_bench.rs`.
//!   3. Be opt-in via `VELOX_QMM_BACKEND=mlx` until the dispatcher
//!      promotes it to default for its eligibility window.
//!
//! ## Status
//!
//! | Kernel              | Status     | Decode | Prefill | MoE |
//! |---------------------|------------|--------|---------|-----|
//! | `qmv_fast` (M=1)    | TODO       | ✓      |         |     |
//! | `qmm_n` (M ≤ 32)    | TODO       | ✓      | ✓       |     |
//! | `qmm_t_aligned_n_b` | TODO       |        | ✓       |     |
//! | `gather_qmm`        | TODO       |        |         | ✓   |
//!
//! Until at least one kernel ships, this module's only public surface is
//! the [`backend`] selector and the [`qmm_4bit_mlx`] stub, which lets the
//! dispatcher in `metal_kernels.rs` plumb the env var without breaking
//! the build.

use anyhow::{bail, Result as AnyResult};

#[cfg(all(target_os = "macos", feature = "candle-metal"))]
use candle_core::Tensor;

/// Which quantized-matmul backend the runtime selects.
///
/// Read from `VELOX_QMM_BACKEND` (case-insensitive). Defaults to
/// [`Backend::Velox`] until ported MLX kernels reach parity coverage of
/// the full eligibility set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Velox-native kernels (`qmm_4bit` naive + `qmm_4bit_v2` shared-X).
    Velox,
    /// MLX-ported kernels — opt-in. Falls back to Velox when a given
    /// shape has no ported kernel yet.
    Mlx,
}

impl Backend {
    /// Resolve the active backend from env. Unknown values silently fall
    /// back to [`Backend::Velox`] so a typo never breaks production.
    pub fn from_env() -> Self {
        match std::env::var("VELOX_QMM_BACKEND")
            .ok()
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("mlx") => Backend::Mlx,
            _ => Backend::Velox,
        }
    }
}

/// MLX-ported quantized matmul dispatcher.
///
/// Routes the call to the most appropriate ported kernel for the given
/// shape; `Err` if no port covers it yet. The dispatcher in
/// `metal_kernels::qmm_4bit` catches this `Err` and falls back to the
/// Velox-native kernel, so opt-in users never see a hard failure during
/// the progressive port.
///
/// ## Currently routed
///
/// | Shape                                          | Kernel                |
/// |------------------------------------------------|-----------------------|
/// | M=1, gs=64, K%512==0, N%8==0, no bias          | `qmv_fast_mlx_g64`    |
///
/// ## Not yet ported (falls back)
///
/// * M ≥ 2 (decode multi-stream, prefill)          → port `qmm_n` next
/// * group_size != 64                              → unroll the template
/// * bias additive vector                          → fold into post-pass
#[cfg(all(target_os = "macos", feature = "candle-metal"))]
pub fn qmm_4bit_mlx(
    x: &Tensor,
    qweight: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    bias: Option<&Tensor>,
    group_size: usize,
) -> AnyResult<Tensor> {
    use crate::paged::metal_kernels::{
        qmv_fast_mlx_g64, QMV_FAST_MLX_BLOCK_K, QMV_FAST_MLX_GROUP, QMV_FAST_MLX_ROWS_PER_TG,
    };

    let (m, k) = x.dims2()?;
    let (n, _) = qweight.dims2()?;

    if m == 1
        && bias.is_none()
        && group_size == QMV_FAST_MLX_GROUP
        && k % QMV_FAST_MLX_BLOCK_K == 0
        && n % QMV_FAST_MLX_ROWS_PER_TG == 0
    {
        return qmv_fast_mlx_g64(x, qweight, scales, biases, group_size);
    }

    bail!("mlx backend: no ported kernel covers this shape yet (M={m} N={n} K={k} gs={group_size} bias={})", bias.is_some())
}

/// Non-Metal stub so the module compiles cross-platform (ports never
/// reach Linux/Windows builds anyway, but keeping the signature here
/// prevents `#[cfg]` noise in the dispatcher).
#[cfg(not(all(target_os = "macos", feature = "candle-metal")))]
pub fn qmm_4bit_mlx<T>(
    _x: &T,
    _qweight: &T,
    _scales: &T,
    _biases: &T,
    _bias: Option<&T>,
    _group_size: usize,
) -> AnyResult<T> {
    bail!("mlx backend: Metal not available on this target")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_default_is_velox() {
        std::env::remove_var("VELOX_QMM_BACKEND");
        assert_eq!(Backend::from_env(), Backend::Velox);
    }

    #[test]
    fn backend_mlx_from_env() {
        std::env::set_var("VELOX_QMM_BACKEND", "mlx");
        assert_eq!(Backend::from_env(), Backend::Mlx);
        std::env::set_var("VELOX_QMM_BACKEND", "MLX");
        assert_eq!(Backend::from_env(), Backend::Mlx);
        std::env::remove_var("VELOX_QMM_BACKEND");
    }

    #[test]
    fn backend_unknown_falls_back_to_velox() {
        std::env::set_var("VELOX_QMM_BACKEND", "tensorrt");
        assert_eq!(Backend::from_env(), Backend::Velox);
        std::env::remove_var("VELOX_QMM_BACKEND");
    }
}
