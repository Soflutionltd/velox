//! Speculative decoding (Leviathan et al., ICML 2023).
//!
//! Speculative decoding speeds up autoregressive generation by using a
//! small **draft model** to propose `D` candidate tokens, then a single
//! batched forward of the **target model** to either accept or reject
//! them. With a good draft, you commit `~A + 1` tokens per target
//! forward instead of just 1 — a 1.5–3× single-stream speedup, with
//! **bit-identical output** to greedy target-only decoding.
//!
//! ## Rolling-bonus design
//!
//! The naive design feeds the bonus token to both KVs at the END of
//! each round, which costs an extra target forward per round and
//! often kills the speedup. We use a **rolling design**: the bonus is
//! stashed as `last_token` (NOT yet in KVs) and prepended to the next
//! round's verify batch. Per-round cost in steady state:
//!
//!   * `D` sequential draft single-token forwards.
//!   * `1` target multi-token forward (`D+1` inputs).
//!
//! Yielding `A+1` committed tokens (where `0 ≤ A ≤ D`).
//!
//! Speedup ≈ `(A+1) / (1 + D · draft_cost_ratio)`.
//!
//! ## Algorithm (per round)
//!
//! Inputs: state with `kv_len` (positions 0..kv_len consumed by both
//! KVs) and `last_token` (committed but NOT in any KV — it's either the
//! prompt's last token or the previous round's bonus).
//!
//!   1. **Draft**: feed `[last_token, c[0], …, c[D-2]]` to the draft
//!      one token at a time, generating `c = [c[0], …, c[D-1]]` by
//!      greedy argmax. Draft KV grows by `D`. (`c[D-1]` is generated
//!      but not fed back into draft.)
//!   2. **Verify**: feed `[last_token, c[0], …, c[D-1]]` (D+1 tokens)
//!      to the target in one shot. Returns logits `T[0..D]`.
//!         * `T[0]` predicts the token after `last_token` (= c[0]).
//!         * `T[i]` for `i ≥ 1` predicts the token after `c[i-1]`
//!           (= c[i]).
//!         * `T[D]` predicts the token after `c[D-1]` (= next bonus).
//!   3. **Accept**: walk `i = 0..D`, accept `c[i]` iff
//!      `argmax(T[i]) == c[i]`. Stop at first mismatch. Let `A` be the
//!      accepted count.
//!   4. **Bonus**: `new_bonus = argmax(T[A])`. Always defined since
//!      `T` has `D+1` rows.
//!   5. **Commit** `c[0..A] ++ [new_bonus]` (`A+1` tokens). Set
//!      `kv_len += A + 1` and `last_token = new_bonus`. KV pages stay
//!      allocated; the stale slots will be overwritten in place by
//!      the next round's draft / verify forwards.

use super::pages::PagedKvCache;
use super::qwen3::{BatchStep, PagedQwen3, SeqSlice};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

/// One speculative-decode-capable engine: a (small) draft model + a
/// (large) target model, each with its own paged KV cache.
pub struct SpecEngine {
    pub target: Arc<PagedQwen3>,
    pub target_pages: Arc<PagedKvCache>,
    pub draft: Arc<PagedQwen3>,
    pub draft_pages: Arc<PagedKvCache>,
    pub device: Device,
    /// Vocab size shared by both models (must match).
    pub vocab_size: usize,
}

impl SpecEngine {
    pub fn new(
        target: Arc<PagedQwen3>,
        target_pages: Arc<PagedKvCache>,
        draft: Arc<PagedQwen3>,
        draft_pages: Arc<PagedKvCache>,
    ) -> Result<Self> {
        let tcfg = target.config();
        let dcfg = draft.config();
        if tcfg.vocab_size != dcfg.vocab_size {
            bail!(
                "draft vocab_size ({}) != target vocab_size ({}); spec decoding requires matched tokenizers",
                dcfg.vocab_size,
                tcfg.vocab_size
            );
        }
        let device = target_pages.device().clone();
        Ok(Self {
            vocab_size: tcfg.vocab_size,
            target,
            target_pages,
            draft,
            draft_pages,
            device,
        })
    }

    fn page_size_or_err(&self) -> Result<usize> {
        let ps = self.target_pages.page_size();
        if self.draft_pages.page_size() != ps {
            bail!(
                "draft page_size ({}) != target page_size ({})",
                self.draft_pages.page_size(),
                ps
            );
        }
        Ok(ps)
    }
}

/// One in-flight speculative request.
pub struct SpecState {
    /// Pages owned by this request in the target KV cache.
    pub target_blocks: Vec<u32>,
    /// Pages owned by this request in the draft KV cache.
    pub draft_blocks: Vec<u32>,
    /// Logical KV length consumed by both KVs (kept in lockstep).
    pub kv_len: usize,
    /// Last committed token, NOT yet fed into either KV. Either the
    /// prompt's final token (right after prefill) or the previous
    /// round's bonus token.
    pub last_token: u32,
    /// Concatenated committed token IDs: prompt + every accepted /
    /// bonus token.
    pub tokens: Vec<u32>,
}

impl SpecEngine {
    /// Prefill both models with `prompt[..len-1]`. The final prompt
    /// token is stashed in `state.last_token` so the first round can
    /// uniformly feed `[last_token, c[0..D-1]]` like every subsequent
    /// round.
    pub fn prefill(&self, prompt: &[u32]) -> Result<SpecState> {
        if prompt.len() < 2 {
            bail!("speculative prefill requires a prompt with at least 2 tokens");
        }
        let ps = self.page_size_or_err()?;
        let head_len = prompt.len() - 1;
        let pages_needed = (head_len + 1 + ps - 1) / ps; // headroom for at least 1 more token

        let target_blocks = self
            .target_pages
            .alloc(pages_needed)
            .context("alloc target prefill pages")?;
        let draft_blocks = self
            .draft_pages
            .alloc(pages_needed)
            .context("alloc draft prefill pages")?;

        let head = &prompt[..head_len];
        let input = Tensor::from_vec(head.to_vec(), (head.len(),), &self.device)?;
        let target_seqs = vec![SeqSlice {
            new_tokens: head.len(),
            kv_offset: 0,
            block_table: &target_blocks,
        }];
        let draft_seqs = vec![SeqSlice {
            new_tokens: head.len(),
            kv_offset: 0,
            block_table: &draft_blocks,
        }];
        let target_step = BatchStep {
            input_ids: &input,
            seqs: &target_seqs,
        };
        let draft_step = BatchStep {
            input_ids: &input,
            seqs: &draft_seqs,
        };

        // Run prefill on both. We discard the returned logits; the
        // first round's verify will produce the prediction for the
        // first generated token using `last_token` as its first input.
        let _ = self.target.forward(&target_step, &self.target_pages)?;
        let _ = self.draft.forward(&draft_step, &self.draft_pages)?;

        Ok(SpecState {
            target_blocks,
            draft_blocks,
            kv_len: head_len,
            last_token: prompt[head_len],
            tokens: prompt.to_vec(),
        })
    }

    /// Run one rolling speculative round. See module docs for the
    /// per-round protocol and KV bookkeeping rules.
    pub fn step(
        &self,
        state: &mut SpecState,
        draft_k: usize,
        eos_id: Option<u32>,
    ) -> Result<RoundOutcome> {
        if draft_k == 0 {
            bail!("draft_k must be ≥ 1");
        }
        let d = draft_k;
        let ps = self.page_size_or_err()?;

        // Make sure both KVs have enough page slots for this round.
        // Target consumes (d + 1) more positions, draft consumes d.
        let needed_target = (state.kv_len + d + 1 + ps - 1) / ps;
        let needed_draft = (state.kv_len + d + ps - 1) / ps;
        ensure_capacity(&self.target_pages, &mut state.target_blocks, needed_target)
            .context("grow target block table")?;
        ensure_capacity(&self.draft_pages, &mut state.draft_blocks, needed_draft)
            .context("grow draft block table")?;

        // ---------- Draft phase ----------
        //
        // Feed [last_token, c[0], …, c[D-2]] to the draft sequentially.
        // Each forward returns one logit row used to pick the next
        // candidate (greedy argmax on GPU). `c[D-1]` is generated but
        // NOT fed back into draft.
        //
        // Critical perf detail: we keep the `next_input` tensor on the
        // GPU between iterations (no host sync) by using GPU-side
        // argmax. Then we sync the entire candidate vector to host in
        // ONE shot at the end of the draft phase, instead of D times.
        // Saves D-1 round-trip syncs per round (~1-2 ms each on M-class).
        use candle_core::D as DimD;
        let mut candidate_tensors: Vec<Tensor> = Vec::with_capacity(d);
        let mut next_input_t: Tensor =
            Tensor::from_vec(vec![state.last_token], (1,), &self.device)?;
        for i in 0..d {
            let seqs = vec![SeqSlice {
                new_tokens: 1,
                kv_offset: state.kv_len + i,
                block_table: &state.draft_blocks,
            }];
            let step = BatchStep {
                input_ids: &next_input_t,
                seqs: &seqs,
            };
            let logit = self.draft.forward(&step, &self.draft_pages)?; // [1, vocab]
            // GPU argmax → [1] u32 tensor on the same device.
            let next_tok_t = logit.argmax_keepdim(DimD::Minus1)?.to_dtype(DType::U32)?;
            // Reshape from [1,1] to [1] for next iter's input_ids.
            let next_tok_t = next_tok_t.squeeze(1)?;
            candidate_tensors.push(next_tok_t.clone());
            next_input_t = next_tok_t;
        }
        // Materialise all candidates to host in one sync.
        let candidates_pack = Tensor::cat(&candidate_tensors, 0)?; // [D] u32
        let candidates: Vec<u32> = candidates_pack.to_device(&Device::Cpu)?.to_vec1()?;
        // Draft KV now holds [last_token, c[0..D-2]] = D extra positions.

        // ---------- Verify phase ----------
        //
        // Feed [last_token, c[0..D-1]] = D+1 tokens to target.
        let mut verify_inputs: Vec<u32> = Vec::with_capacity(d + 1);
        verify_inputs.push(state.last_token);
        verify_inputs.extend_from_slice(&candidates);
        let verify_input = Tensor::from_vec(verify_inputs, (d + 1,), &self.device)?;
        let verify_seqs = vec![SeqSlice {
            new_tokens: d + 1,
            kv_offset: state.kv_len,
            block_table: &state.target_blocks,
        }];
        let verify_step = BatchStep {
            input_ids: &verify_input,
            seqs: &verify_seqs,
        };
        let target_logits = self
            .target
            .forward_all(&verify_step, &self.target_pages)?; // [D+1, vocab]

        // Pull all D+1 target argmaxes onto host once. D is small (≤8).
        let target_logits_cpu = target_logits
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let flat: Vec<f32> = target_logits_cpu.flatten_all()?.to_vec1()?;
        let vocab = self.vocab_size;
        let mut target_argmaxes: Vec<u32> = Vec::with_capacity(d + 1);
        for i in 0..(d + 1) {
            let row = &flat[i * vocab..(i + 1) * vocab];
            target_argmaxes.push(argmax_slice(row));
        }

        // ---------- Acceptance ----------
        //
        // T[i] predicts the (i+1)-th input of [last_token, c[0..D-1]].
        // So T[0] predicts c[0], T[1] predicts c[1], …, T[D-1]
        // predicts c[D-1], and T[D] is the bonus prediction for the
        // next round.
        let mut accepted = 0usize;
        for i in 0..d {
            if candidates[i] == target_argmaxes[i] {
                accepted += 1;
            } else {
                break;
            }
        }
        let new_bonus = target_argmaxes[accepted];

        // ---------- Commit ----------
        let mut committed: Vec<u32> = Vec::with_capacity(accepted + 1);
        committed.extend(&candidates[..accepted]);
        committed.push(new_bonus);

        state.kv_len += accepted + 1;
        state.last_token = new_bonus;
        state.tokens.extend(&committed);
        // KV pages: any stale slots beyond `state.kv_len` are simply
        // ignored by the next forward (`kv_offset = state.kv_len`)
        // and overwritten in place when we extend further.

        let stop = match eos_id {
            Some(eos) => committed.iter().any(|&t| t == eos),
            None => false,
        };

        Ok(RoundOutcome {
            committed_tokens: committed,
            accepted,
            stop,
        })
    }

    /// Free both block tables. Call exactly once after generation
    /// completes.
    pub fn free(&self, state: SpecState) {
        self.target_pages.free_pages(state.target_blocks);
        self.draft_pages.free_pages(state.draft_blocks);
    }
}

/// Result of one speculative round.
#[derive(Debug)]
pub struct RoundOutcome {
    /// Tokens committed this round (length `accepted + 1`).
    pub committed_tokens: Vec<u32>,
    /// Candidates accepted by the target (`0 ≤ accepted ≤ draft_k`).
    pub accepted: usize,
    /// True if EOS was committed and the caller should stop.
    pub stop: bool,
}

fn ensure_capacity(
    pages: &Arc<PagedKvCache>,
    block_table: &mut Vec<u32>,
    pages_needed: usize,
) -> Result<()> {
    if block_table.len() >= pages_needed {
        return Ok(());
    }
    let extra = pages_needed - block_table.len();
    let new = pages.alloc(extra).context("alloc extra pages")?;
    block_table.extend(new);
    Ok(())
}

fn argmax_u32(t: &Tensor) -> Result<u32> {
    let v: Vec<f32> = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1()?;
    Ok(argmax_slice(&v))
}

fn argmax_slice(v: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, x) in v.iter().enumerate() {
        if *x > best_v {
            best_v = *x;
            best_i = i as u32;
        }
    }
    best_i
}
