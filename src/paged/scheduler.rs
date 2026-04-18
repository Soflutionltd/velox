//! Continuous batching scheduler.
//!
//! Owns a single [`PagedQwen3`] instance + its [`PagedKvCache`] and runs an
//! infinite step loop:
//!
//!   1. Admit as many waiting requests as KV pages allow.
//!   2. Build one packed batch step from all running requests:
//!        * prefill requests contribute `prompt_len - kv_offset` tokens
//!          (chunked so total batch tokens stay below `max_batch_tokens`)
//!        * decode requests contribute exactly 1 token each
//!   3. Run the model forward.
//!   4. Sample one new token per request, push it into KV pool, send to
//!      its SSE channel, check for EOS / stop / max_tokens.
//!   5. Finished requests release their pages back to the pool.
//!
//! The whole loop runs inside a dedicated `spawn_blocking` worker because
//! Candle is sync and GPU/CPU bound. HTTP handlers communicate with the
//! scheduler via an mpsc channel of [`SubmitRequest`]s.

use super::pages::PagedKvCache;
use super::prefix_cache::PrefixCache;
use super::qwen3::{BatchStep, PagedQwen3, SeqSlice};
use super::request::{safe_text_delta, Request, RequestId, RequestStatus};
use crate::backend::traits::{ChatMessage, StreamChunk};
use anyhow::{anyhow, Context, Result};
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Hard cap on requests that can be in `Running` state at once.
    pub max_running_requests: usize,
    /// Soft cap on total NEW tokens processed in a single forward step
    /// (sum across all running requests). Used to chunk long prefills so
    /// they don't starve decode requests.
    pub max_batch_tokens: usize,
    /// Idle sleep when the run queue is empty and no work to do.
    pub idle_sleep: Duration,
    /// Max number of prefix-cached pages to keep alive across requests.
    /// 0 disables the prefix cache entirely.
    pub prefix_cache_capacity: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_running_requests: 64,
            max_batch_tokens: 512,
            idle_sleep: Duration::from_millis(2),
            prefix_cache_capacity: 256,
        }
    }
}

/// Submission payload sent by HTTP handlers into the scheduler.
pub struct SubmitRequest {
    /// Either pre-tokenised prompt OR a list of chat messages (we'll apply
    /// the chat template inside the scheduler).
    pub messages: Vec<ChatMessage>,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    pub stop_sequences: Vec<String>,
    pub tx: mpsc::Sender<StreamChunk>,
}

pub struct BatchScheduler {
    model: Arc<PagedQwen3>,
    pages: Arc<PagedKvCache>,
    tokenizer: Arc<Tokenizer>,
    chat_template: String,
    eos_token_ids: Vec<u32>,
    cfg: SchedulerConfig,

    /// Wait queue (FCFS).
    waiting: Mutex<VecDeque<Request>>,
    /// Currently-running requests. Indexed by RequestId for quick removal.
    running: Mutex<HashMap<RequestId, Request>>,
    /// Page-level prefix cache. None when capacity == 0.
    prefix_cache: Option<Mutex<PrefixCache>>,

    shutdown: AtomicBool,
    stats: Mutex<SchedulerStats>,
}

#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    pub steps: u64,
    pub tokens_generated: u64,
    pub requests_completed: u64,
    pub requests_aborted: u64,
    pub last_batch_size: usize,
    pub last_step_us: u64,
    pub prefix_cache_hits: u64,
    pub prefix_cache_misses: u64,
    pub prefix_tokens_skipped: u64,
}

impl BatchScheduler {
    pub fn new(
        model: Arc<PagedQwen3>,
        pages: Arc<PagedKvCache>,
        tokenizer: Arc<Tokenizer>,
        chat_template: String,
        eos_token_ids: Vec<u32>,
        cfg: SchedulerConfig,
    ) -> Self {
        let prefix_cache = if cfg.prefix_cache_capacity > 0 {
            Some(Mutex::new(PrefixCache::new(
                pages.clone(),
                cfg.prefix_cache_capacity,
            )))
        } else {
            None
        };
        Self {
            model,
            pages,
            tokenizer,
            chat_template,
            eos_token_ids,
            cfg,
            waiting: Mutex::new(VecDeque::new()),
            running: Mutex::new(HashMap::new()),
            prefix_cache,
            shutdown: AtomicBool::new(false),
            stats: Mutex::new(SchedulerStats::default()),
        }
    }

    pub fn stats(&self) -> SchedulerStats {
        self.stats.lock().clone()
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Push a new request into the wait queue. Tokenisation + chat template
    /// happen here (cheap, on the caller's thread).
    pub fn submit(&self, sub: SubmitRequest) -> Result<RequestId> {
        let prompt_tokens = if !sub.messages.is_empty() {
            let prompt_text = render_chat_template(&self.chat_template, &sub.messages)?;
            self.tokenizer
                .encode(prompt_text, true)
                .map_err(|e| anyhow!("tokenizer encode: {e}"))?
                .get_ids()
                .to_vec()
        } else {
            sub.prompt_tokens
        };

        if prompt_tokens.is_empty() {
            return Err(anyhow!("Empty prompt"));
        }

        let req = Request {
            id: RequestId::new(),
            prompt_tokens,
            generated_tokens: Vec::new(),
            seq_len: 0,
            temperature: sub.temperature,
            top_p: sub.top_p,
            max_new_tokens: sub.max_tokens.max(1) as usize,
            stop_sequences: sub.stop_sequences,
            eos_token_ids: self.eos_token_ids.clone(),
            block_table: Vec::new(),
            status: RequestStatus::Waiting,
            created_at: Instant::now(),
            admitted_at: None,
            cached_prefix_pages: 0,
            prefix_chain_hash: 0,
            tx: sub.tx,
            decoded_text: String::new(),
        };
        let id = req.id;
        self.waiting.lock().push_back(req);
        Ok(id)
    }

    /// Run the scheduler loop. Blocks the calling thread until `shutdown()`
    /// is called or no more work can ever arrive.
    pub fn run(self: Arc<Self>) -> Result<()> {
        tracing::info!(
            "BatchScheduler started: max_running={}, max_batch_tokens={}, pages_total={}",
            self.cfg.max_running_requests,
            self.cfg.max_batch_tokens,
            self.pages.num_total_pages(),
        );
        while !self.shutdown.load(Ordering::Relaxed) {
            self.admit_waiting();

            let did_work = match self.step() {
                Ok(n) => n > 0,
                Err(e) => {
                    tracing::error!("scheduler step failed: {e:#}");
                    false
                }
            };

            if !did_work {
                std::thread::sleep(self.cfg.idle_sleep);
            }
        }
        tracing::info!("BatchScheduler shutting down");
        Ok(())
    }

    /// Move requests from `waiting` to `running` while we have free pages
    /// and capacity. Consults the prefix cache to skip prefill on shared
    /// prompt prefixes.
    fn admit_waiting(&self) {
        let mut waiting = self.waiting.lock();
        let mut running = self.running.lock();
        let page_size = self.pages.page_size();

        while running.len() < self.cfg.max_running_requests {
            let Some(mut req) = waiting.pop_front() else {
                break;
            };

            // 1) Consult the prefix cache.
            let (cached_pages, cached_tokens, end_hash) = if let Some(cache) = &self.prefix_cache {
                let mut c = cache.lock();
                let hit = c.lookup(&req.prompt_tokens);
                if hit.matched_pages.is_empty() {
                    let mut s = self.stats.lock();
                    s.prefix_cache_misses += 1;
                } else {
                    let mut s = self.stats.lock();
                    s.prefix_cache_hits += 1;
                    s.prefix_tokens_skipped += hit.matched_tokens as u64;
                    tracing::debug!(
                        "{}: prefix cache hit, skipped {} tokens ({} pages)",
                        req.id,
                        hit.matched_tokens,
                        hit.matched_pages.len()
                    );
                }
                (hit.matched_pages, hit.matched_tokens, hit.next_chain_hash)
            } else {
                (Vec::new(), 0, 0u64)
            };

            // 2) Worst-case page need for the WHOLE sequence (cached + suffix +
            //    generation). The cached portion is already covered.
            let total_pages_needed = req.pages_required(page_size, req.max_new_tokens);
            let pages_to_alloc = total_pages_needed.saturating_sub(cached_pages.len());

            let new_blocks = if pages_to_alloc == 0 {
                Some(Vec::new())
            } else {
                self.pages.alloc(pages_to_alloc)
            };

            match new_blocks {
                Some(extra) => {
                    let mut block_table = cached_pages.clone();
                    block_table.extend(extra);

                    req.block_table = block_table;
                    req.cached_prefix_pages = cached_pages.len();
                    req.prefix_chain_hash = end_hash;
                    req.seq_len = cached_tokens;
                    req.status = RequestStatus::Running;
                    req.admitted_at = Some(Instant::now());
                    tracing::debug!(
                        "{}: admitted (prompt_len={}, pages={}, cached_pages={})",
                        req.id,
                        req.prompt_tokens.len(),
                        total_pages_needed,
                        cached_pages.len()
                    );
                    running.insert(req.id, req);
                }
                None => {
                    // Not enough pages; release any cache refs we just took
                    // and try this request again next round.
                    if !cached_pages.is_empty() {
                        self.pages.free_pages(cached_pages);
                    }
                    waiting.push_front(req);
                    break;
                }
            }
        }
    }

    /// One forward step. Returns the number of NEW tokens processed in
    /// this step (so the caller can decide to sleep or loop).
    fn step(&self) -> Result<usize> {
        let step_start = Instant::now();

        // Snapshot the running set into a stable order. We hold the lock for
        // the whole step because Candle's sync forward call needs exclusive
        // model access and we want the per-request state to stay coherent.
        let mut running = self.running.lock();
        if running.is_empty() {
            return Ok(0);
        }

        // Sort by id for determinism (and for easier debugging).
        let mut ids: Vec<RequestId> = running.keys().copied().collect();
        ids.sort_by_key(|id| id.0);

        // Build the packed batch.
        let mut input_token_buf: Vec<u32> = Vec::new();
        let mut seq_specs: Vec<SeqSpec> = Vec::with_capacity(ids.len());
        let mut total_new = 0usize;

        for id in &ids {
            let req = running.get(id).expect("id from snapshot");
            // How many new tokens does this request contribute this step?
            let new_tokens = if req.needs_prefill() {
                req.prompt_tokens.len() - req.seq_len
            } else {
                1
            };
            if total_new + new_tokens > self.cfg.max_batch_tokens && total_new > 0 {
                // This request would push the batch over budget; defer.
                continue;
            }
            // Slice the input ids: prefill grabs the next chunk of prompt,
            // decode grabs the last generated token (or last prompt token
            // if generation hasn't started yet).
            let input_slice: Vec<u32> = if req.needs_prefill() {
                req.prompt_tokens[req.seq_len..req.seq_len + new_tokens].to_vec()
            } else {
                let last_tok = *req
                    .generated_tokens
                    .last()
                    .or_else(|| req.prompt_tokens.last())
                    .ok_or_else(|| anyhow!("no token to feed for {}", id))?;
                vec![last_tok]
            };
            input_token_buf.extend_from_slice(&input_slice);
            seq_specs.push(SeqSpec {
                id: *id,
                new_tokens,
                kv_offset: req.seq_len,
                block_table: req.block_table.clone(),
                is_prefill: req.needs_prefill(),
            });
            total_new += new_tokens;
        }

        if total_new == 0 {
            return Ok(0);
        }

        // Build the batch step.
        let device = &self.model.device;
        let input_tensor = Tensor::from_vec(input_token_buf, (total_new,), device)
            .context("input tensor")?;
        let seq_slices: Vec<SeqSlice<'_>> = seq_specs
            .iter()
            .map(|s| SeqSlice {
                new_tokens: s.new_tokens,
                kv_offset: s.kv_offset,
                block_table: &s.block_table,
            })
            .collect();
        let step = BatchStep {
            input_ids: &input_tensor,
            seqs: &seq_slices,
        };
        let logits = self.model.forward(&step, &self.pages)?;
        // logits shape: [num_reqs, vocab]. One row per request, at the
        // position of its LAST new token.

        // Sample one token per row in a SINGLE GPU→CPU sync when possible.
        //
        // The naive path (used until commit `feat(perf): GPU-side argmax`)
        // pulled the entire `[vocab]` logits row to the CPU N times per
        // step (once per running request), then ran argmax/top-p in pure
        // Rust. With Qwen3 vocab = 151K and 16 concurrent users that is
        // ~10 MB of GPU→CPU transfer + 16 sync points per generated
        // token — purely wasted bandwidth when we just want one u32 per
        // row.
        //
        // `sample_batch` instead detects greedy batches and dispatches a
        // single Metal `fast_argmax_*` reduce, then transfers exactly
        // `4 * num_reqs` bytes back to host. Mixed-mode batches (some
        // requests greedy, some stochastic) still benefit because the
        // greedy rows are batched and the stochastic ones keep the old
        // per-row sampling path (they need the full logits anyway).
        let next_tokens = sample_batch(&logits, &seq_specs, &running)?;

        // Per-request: advance state, stream.
        let mut to_remove: Vec<RequestId> = Vec::new();
        for (row, spec) in seq_specs.iter().enumerate() {
            let req = running
                .get_mut(&spec.id)
                .expect("request still in running map");

            // Bookkeeping: extend seq_len by the new tokens we just pushed.
            req.seq_len += spec.new_tokens;

            // If we were prefilling, the model has now seen the FULL prefill
            // chunk we sent. The LAST token of that chunk is the one whose
            // logits we just sampled, which gives us the FIRST generated
            // token. Otherwise we're in pure decode mode.
            let next = next_tokens[row];

            // If we still need to prefill more (the prompt was longer than
            // a single chunk allowed), the sampled token is just lookahead;
            // we discard it. The next step will continue prefill.
            if req.needs_prefill() {
                continue;
            }

            // Otherwise: this is a real generated token. Push it.
            req.generated_tokens.push(next);

            // Check EOS BEFORE pushing the token to KV (we don't actually
            // need to — pages already hold this token's position via
            // seq_len bookkeeping. The next step would re-feed `next`).
            let is_eos = req.eos_token_ids.contains(&next);

            // Decode ONLY the generated tokens (we never want to emit the
            // prompt back to the client). Full re-decode each step keeps
            // BPE merges and multi-byte UTF-8 correct.
            let full_text = self
                .tokenizer
                .decode(&req.generated_tokens, true)
                .map_err(|e| anyhow!("decode: {e}"))?;

            // Compute delta vs what we last emitted.
            let prev_decoded = std::mem::take(&mut req.decoded_text);
            match safe_text_delta(&prev_decoded, &full_text) {
                Some(delta) => {
                    req.decoded_text = full_text.clone();
                    if !delta.is_empty()
                        && !req.send_chunk(StreamChunk::Token {
                            token_id: next,
                            text_delta: delta,
                        })
                    {
                        // Client cancelled.
                        let _ = req.send_chunk(StreamChunk::Done {
                            finish_reason: "cancelled".into(),
                            prompt_tokens: req.prompt_tokens.len() as u32,
                            completion_tokens: req.generated_tokens.len() as u32,
                        });
                        to_remove.push(spec.id);
                        continue;
                    }
                }
                None => {
                    // Partial UTF-8: keep the previous decoded buffer and
                    // wait for the next token to complete the codepoint.
                    req.decoded_text = prev_decoded;
                }
            }

            // Stop checks.
            let hit_stop = !req.stop_sequences.is_empty()
                && req
                    .stop_sequences
                    .iter()
                    .any(|s| !s.is_empty() && req.decoded_text.contains(s));
            let hit_max = req.generated_tokens.len() >= req.max_new_tokens;

            if is_eos || hit_stop || hit_max {
                let finish = if hit_max && !is_eos && !hit_stop {
                    "length"
                } else {
                    "stop"
                };
                let _ = req.send_chunk(StreamChunk::Done {
                    finish_reason: finish.into(),
                    prompt_tokens: req.prompt_tokens.len() as u32,
                    completion_tokens: req.generated_tokens.len() as u32,
                });
                to_remove.push(spec.id);
            }
        }

        // Reap finished requests.
        for id in &to_remove {
            if let Some(req) = running.remove(id) {
                // Insert the freshly-prefilled prompt pages into the prefix
                // cache so subsequent requests with the same prefix can
                // reuse them. Only cache the page-aligned portion of the
                // PROMPT (not generation), and only pages that came from
                // *this* request's allocation (skipping the ones we got
                // from the cache to begin with).
                if let Some(cache) = &self.prefix_cache {
                    let prompt_full_pages =
                        req.prompt_tokens.len() / self.pages.page_size();
                    if prompt_full_pages > req.cached_prefix_pages {
                        let new_prompt_pages = &req.block_table
                            [req.cached_prefix_pages..prompt_full_pages];
                        let start_token =
                            req.cached_prefix_pages * self.pages.page_size();
                        tracing::debug!(
                            "{}: inserting {} prompt pages into prefix cache",
                            req.id,
                            new_prompt_pages.len(),
                        );
                        cache.lock().insert(
                            &req.prompt_tokens,
                            start_token,
                            req.prefix_chain_hash,
                            new_prompt_pages,
                        );
                    }
                }
                self.pages.free_pages(req.block_table);
            }
        }

        // Stats.
        let mut stats = self.stats.lock();
        stats.steps += 1;
        stats.tokens_generated += seq_specs
            .iter()
            .filter(|s| !s.is_prefill)
            .map(|s| s.new_tokens as u64)
            .sum::<u64>();
        stats.requests_completed += to_remove.len() as u64;
        stats.last_batch_size = seq_specs.len();
        stats.last_step_us = step_start.elapsed().as_micros() as u64;

        Ok(total_new)
    }
}

struct SeqSpec {
    id: RequestId,
    new_tokens: usize,
    kv_offset: usize,
    block_table: Vec<u32>,
    is_prefill: bool,
}

fn make_sampler(temperature: f64, top_p: f64) -> LogitsProcessor {
    if temperature <= 1e-7 {
        LogitsProcessor::from_sampling(rand::random(), Sampling::ArgMax)
    } else if top_p <= 0.0 || top_p >= 1.0 {
        LogitsProcessor::from_sampling(rand::random(), Sampling::All { temperature })
    } else {
        LogitsProcessor::from_sampling(rand::random(), Sampling::TopP { p: top_p, temperature })
    }
}

/// Whether a request samples greedily (temperature ≈ 0).
#[inline]
fn is_greedy(temperature: f64) -> bool {
    temperature <= 1e-7
}

/// Sample one token per row of `logits` in as few GPU→CPU syncs as possible.
///
/// `logits` has shape `[num_reqs, vocab]`. We return a `Vec<u32>` of length
/// `num_reqs`, one sampled token id per row, in the same order as
/// `seq_specs`.
///
/// Strategy:
/// * If **every** row is greedy, we run a single `argmax` reduction on the
///   whole `[num_reqs, vocab]` tensor (one Metal dispatch) and pull
///   `num_reqs` u32s back. This is the hot path during benchmarks and any
///   `temperature=0` workload (code, structured output, eval suites).
/// * Otherwise we fall back to per-row sampling via Candle's
///   `LogitsProcessor`. This is correct but slower because each row pulls
///   the full `[vocab]` of f32s to the CPU.
///
/// We deliberately do NOT try to do partial GPU argmax + per-row CPU
/// sampling here. The mixed case is rare in practice (most batches are
/// homogeneous), and the bookkeeping/branching cost of doing it is not
/// worth the marginal speedup we could squeeze out.
fn sample_batch(
    logits: &Tensor,
    seq_specs: &[SeqSpec],
    running: &HashMap<RequestId, Request>,
) -> Result<Vec<u32>> {
    debug_assert_eq!(
        logits.dims().first().copied(),
        Some(seq_specs.len()),
        "logits batch dim must match seq_specs len"
    );

    let all_greedy = seq_specs
        .iter()
        .all(|s| running.get(&s.id).map(|r| is_greedy(r.temperature)).unwrap_or(true));

    if all_greedy {
        // FAST PATH: one Metal `fast_argmax_*` over the whole batch, one
        // sync of `4 * num_reqs` bytes.
        let ids = logits
            .argmax(candle_core::D::Minus1)
            .map_err(|e| anyhow!("batched argmax: {e}"))?;
        let ids = match ids.dtype() {
            candle_core::DType::U32 => ids,
            _ => ids
                .to_dtype(candle_core::DType::U32)
                .map_err(|e| anyhow!("argmax cast: {e}"))?,
        };
        let v = ids
            .to_vec1::<u32>()
            .map_err(|e| anyhow!("argmax sync: {e}"))?;
        return Ok(v);
    }

    // SLOW PATH: per-row sampling for non-greedy batches.
    let mut out = Vec::with_capacity(seq_specs.len());
    for (row, spec) in seq_specs.iter().enumerate() {
        let row_logits = logits.i(row)?.contiguous()
            .map_err(|e| anyhow!("logits row {row}: {e}"))?;
        let req = running.get(&spec.id).expect("request still in running map");
        let mut sampler = make_sampler(req.temperature, req.top_p);
        let next = sampler
            .sample(&row_logits)
            .map_err(|e| anyhow!("sampler row {row}: {e}"))?;
        out.push(next);
    }
    Ok(out)
}

/// Render a Hugging Face Jinja2 chat template with minijinja + pycompat.
fn render_chat_template(template: &str, messages: &[ChatMessage]) -> Result<String> {
    use minijinja::{context, Environment};
    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", template)
        .map_err(|e| anyhow!("template parse: {e}"))?;
    let tmpl = env.get_template("chat").map_err(|e| anyhow!("get template: {e}"))?;
    let msgs: Vec<minijinja::value::Value> = messages
        .iter()
        .map(|m| {
            minijinja::value::Value::from_serialize(&serde_json::json!({
                "role": m.role,
                "content": m.content,
            }))
        })
        .collect();
    tmpl.render(context! {
        messages => msgs,
        add_generation_prompt => true,
        bos_token => "",
        eos_token => "",
    })
    .map_err(|e| anyhow!("template render: {e}"))
}

/// Re-export `IndexOp::i()` for `Tensor` so the scheduler can grab a row.
trait TensorRow {
    fn i(&self, idx: usize) -> Result<Tensor>;
}
impl TensorRow for Tensor {
    fn i(&self, idx: usize) -> Result<Tensor> {
        candle_core::IndexOp::i(self, idx).map_err(Into::into)
    }
}
