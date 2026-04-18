// Candle backend — pure-Rust LLM inference, Metal-accelerated on Apple Silicon.
//
// Architecture
//   - CandleBackend holds a DashMap<id, Arc<LoadedCandleModel>>.
//   - ModelHandle.id is the key into that map.
//   - LoadedCandleModel is an enum, one variant per supported architecture.
//   - generate() runs inside spawn_blocking (Candle is sync, GPU/CPU bound).
//   - Chat templates are evaluated with minijinja (HF Jinja2 dialect).
//
// This is the default Phase-1 backend. mlx-rs is gated behind --features mlx
// and currently broken upstream; revisit when the RoPE / KV-cache bugs in
// `mlx-lm` are resolved.

use super::traits::*;
use crate::paged::pages::{PagedKvCache, PagedKvConfig};
use crate::paged::qwen3::{PagedQwen3, Qwen3Config as PagedQwen3Config};
use crate::paged::scheduler::{BatchScheduler, SchedulerConfig, SubmitRequest};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gemma3::{Config as Gemma3Config, Model as Gemma3Model};
use candle_transformers::models::mistral::{Config as MistralConfig, Model as MistralModel};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3ForCausalLM};
use dashmap::DashMap;
use futures::stream::BoxStream;
use minijinja::{context, Environment};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// All supported Candle architectures share the same forward signature
/// (`forward(&mut self, tokens, seqlen_offset)` + `clear_kv_cache`). Wrap them
/// in this enum to keep the per-token loop architecture-agnostic.
enum CandleArch {
    Qwen3(Qwen3ForCausalLM),
    Mistral(MistralModel),
    Phi3(Phi3Model),
    Gemma3(Gemma3Model),
}

impl CandleArch {
    fn forward(&mut self, tokens: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            CandleArch::Qwen3(m) => m.forward(tokens, offset),
            CandleArch::Mistral(m) => m.forward(tokens, offset),
            CandleArch::Phi3(m) => m.forward(tokens, offset),
            CandleArch::Gemma3(m) => m.forward(tokens, offset),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            CandleArch::Qwen3(m) => m.clear_kv_cache(),
            CandleArch::Mistral(m) => m.clear_kv_cache(),
            CandleArch::Phi3(m) => m.clear_kv_cache(),
            CandleArch::Gemma3(m) => m.clear_kv_cache(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CandleArch::Qwen3(_) => "qwen3",
            CandleArch::Mistral(_) => "mistral",
            CandleArch::Phi3(_) => "phi3",
            CandleArch::Gemma3(_) => "gemma3",
        }
    }
}

/// One loaded model with everything the inference loop needs.
struct LoadedCandleModel {
    model: Mutex<CandleArch>,
    tokenizer: Tokenizer,
    chat_template: String,
    eos_token_ids: Vec<u32>,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

/// A Qwen3 model loaded with the paged backend + a running BatchScheduler.
struct PagedHandle {
    scheduler: Arc<BatchScheduler>,
    /// Background scheduler thread; we never join it during normal operation
    /// (it runs forever until shutdown), but we keep the handle so `Drop`
    /// can wait if needed.
    _thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for PagedHandle {
    fn drop(&mut self) {
        self.scheduler.shutdown();
    }
}

pub struct CandleBackend {
    loaded: Arc<DashMap<String, Arc<LoadedCandleModel>>>,
    paged: Arc<DashMap<String, Arc<PagedHandle>>>,
    device: Device,
}

impl CandleBackend {
    pub fn new() -> Self {
        let device = pick_device();
        tracing::info!("Candle backend initialised on device: {:?}", device);
        Self {
            loaded: Arc::new(DashMap::new()),
            paged: Arc::new(DashMap::new()),
            device,
        }
    }

    /// Candle is available on every platform we currently target.
    pub fn available() -> bool {
        true
    }

    /// Build a paged scheduler for a Qwen3 model loaded on disk. Returns the
    /// scheduler handle ready to accept submissions.
    fn build_paged_qwen3(
        path: &Path,
        device: &Device,
    ) -> anyhow::Result<PagedHandle> {
        tracing::info!("Building paged Qwen3 from {:?}", path);
        let cfg: PagedQwen3Config = load_arch_config(path)?;
        let dtype = pick_dtype(path, device);
        let shards = discover_safetensors(path)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, dtype, device)
                .map_err(|e| anyhow::anyhow!("VarBuilder failed: {e}"))?
        };
        let model = Arc::new(
            PagedQwen3::load(&cfg, vb).map_err(|e| anyhow::anyhow!("PagedQwen3::load: {e}"))?,
        );

        // KV pool sizing. Conservative: 256 pages × 16 tokens = 4K total
        // tokens of cache. For Qwen3-0.6B BF16, that's ~3.5 GB on Apple
        // Silicon — reasonable for 16+ GB unified memory machines.
        let pages_cfg = PagedKvConfig {
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            page_size: 16,
            num_pages: 256,
            dtype,
        };
        let pages = Arc::new(PagedKvCache::new(pages_cfg, device)?);

        let tokenizer = Arc::new(load_tokenizer(path)?);
        let chat_template = load_chat_template(path)?;
        let eos_token_ids = read_eos_tokens(path, &tokenizer);

        let scheduler = Arc::new(BatchScheduler::new(
            model,
            pages,
            tokenizer,
            chat_template,
            eos_token_ids,
            SchedulerConfig::default(),
        ));

        let sched_run = scheduler.clone();
        let thread = std::thread::Builder::new()
            .name("velox-batch-scheduler".to_string())
            .spawn(move || {
                if let Err(e) = sched_run.run() {
                    tracing::error!("BatchScheduler thread crashed: {e:#}");
                }
            })
            .map_err(|e| anyhow::anyhow!("spawn scheduler thread: {e}"))?;

        Ok(PagedHandle {
            scheduler,
            _thread: Some(thread),
        })
    }
}

/// Pick the best available compute device for Candle.
///   Apple Silicon → Metal
///   x86 / Linux   → CPU (no CUDA support compiled in)
fn pick_device() -> Device {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "candle-metal"))]
    {
        match Device::new_metal(0) {
            Ok(d) => return d,
            Err(e) => tracing::warn!("Failed to open Metal device, falling back to CPU: {e}"),
        }
    }
    Device::Cpu
}

#[async_trait]
impl InferenceBackend for CandleBackend {
    async fn load_model(&self, path: &Path) -> anyhow::Result<ModelHandle> {
        let path_buf = path.to_path_buf();
        let loaded_map = self.loaded.clone();
        let paged_map = self.paged.clone();
        let device = self.device.clone();

        let handle = tokio::task::spawn_blocking(move || -> anyhow::Result<ModelHandle> {
            let arch = detect_architecture(&path_buf)?;
            tracing::info!("Loading Candle model from {:?} (arch={})", path_buf, arch);

            // Qwen3 takes the new paged-attention path with continuous batching.
            // Other architectures fall back to the sequential per-request path.
            if arch == "qwen3" {
                let paged = Self::build_paged_qwen3(&path_buf, &device)?;
                let id = uuid::Uuid::new_v4().to_string();
                let handle = ModelHandle {
                    id: id.clone(),
                    path: path_buf.to_string_lossy().to_string(),
                    model_type: ModelType::Llm,
                    params_total: 0,
                    params_active: 0,
                };
                paged_map.insert(id, Arc::new(paged));
                return Ok(handle);
            }

            let tokenizer = load_tokenizer(&path_buf)?;
            let chat_template = load_chat_template(&path_buf)?;
            let eos_token_ids = read_eos_tokens(&path_buf, &tokenizer);
            let dtype = pick_dtype(&path_buf, &device);

            let shards = discover_safetensors(&path_buf)?;
            tracing::info!(
                "Loading {} safetensors shard(s), dtype={:?}",
                shards.len(),
                dtype
            );
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)
                    .map_err(|e| anyhow::anyhow!("VarBuilder failed: {e}"))?
            };

            let candle_arch = match arch.as_str() {
                "qwen3" => {
                    let cfg = load_arch_config::<Qwen3Config>(&path_buf)?;
                    CandleArch::Qwen3(
                        Qwen3ForCausalLM::new(&cfg, vb)
                            .map_err(|e| anyhow::anyhow!("Qwen3ForCausalLM::new failed: {e}"))?,
                    )
                }
                "mistral" => {
                    let cfg = load_arch_config::<MistralConfig>(&path_buf)?;
                    CandleArch::Mistral(
                        MistralModel::new(&cfg, vb)
                            .map_err(|e| anyhow::anyhow!("MistralModel::new failed: {e}"))?,
                    )
                }
                "phi3" | "phi-3" => {
                    let cfg = load_arch_config::<Phi3Config>(&path_buf)?;
                    CandleArch::Phi3(
                        Phi3Model::new(&cfg, vb)
                            .map_err(|e| anyhow::anyhow!("Phi3Model::new failed: {e}"))?,
                    )
                }
                "gemma3" | "gemma3_text" => {
                    let cfg = load_arch_config::<Gemma3Config>(&path_buf)?;
                    CandleArch::Gemma3(
                        Gemma3Model::new(false, &cfg, vb)
                            .map_err(|e| anyhow::anyhow!("Gemma3Model::new failed: {e}"))?,
                    )
                }
                other => {
                    anyhow::bail!(
                        "Architecture '{}' not yet supported by Candle backend. \
                         Supported: qwen3, mistral, phi3, gemma3",
                        other
                    );
                }
            };

            let loaded = LoadedCandleModel {
                model: Mutex::new(candle_arch),
                tokenizer,
                chat_template,
                eos_token_ids,
                device: device.clone(),
                dtype,
            };

            let id = uuid::Uuid::new_v4().to_string();
            let handle = ModelHandle {
                id: id.clone(),
                path: path_buf.to_string_lossy().to_string(),
                model_type: ModelType::Llm,
                params_total: 0,
                params_active: 0,
            };
            loaded_map.insert(id, Arc::new(loaded));
            Ok(handle)
        })
        .await??;

        Ok(handle)
    }

    async fn unload_model(&self, handle: &ModelHandle) -> anyhow::Result<()> {
        self.loaded.remove(&handle.id);
        self.paged.remove(&handle.id);
        Ok(())
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        request: &GenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        // Paged path (Qwen3): submit to scheduler, accumulate stream.
        if let Some(paged) = self.paged.get(&handle.id).map(|r| r.clone()) {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamChunk>(64);
            let id = paged.scheduler.submit(SubmitRequest {
                messages: request.messages.clone(),
                prompt_tokens: request.prompt_tokens.clone(),
                max_tokens: request.max_tokens.max(1),
                temperature: request.temperature.max(0.0) as f64,
                top_p: request.top_p.max(0.0).min(1.0) as f64,
                stop_sequences: request.stop_sequences.clone(),
                tx,
            })?;
            let mut text = String::new();
            let mut tokens: Vec<u32> = Vec::new();
            let mut prompt_tokens = 0u32;
            let mut completion_tokens = 0u32;
            let mut finish_reason = String::from("stop");
            while let Some(chunk) = rx.recv().await {
                match chunk {
                    StreamChunk::Token { token_id, text_delta } => {
                        tokens.push(token_id);
                        text.push_str(&text_delta);
                    }
                    StreamChunk::Done {
                        finish_reason: r,
                        prompt_tokens: pt,
                        completion_tokens: ct,
                    } => {
                        finish_reason = r;
                        prompt_tokens = pt;
                        completion_tokens = ct;
                    }
                    StreamChunk::Error(e) => return Err(anyhow::anyhow!(e)),
                }
            }
            let _ = id;
            return Ok(GenerateResult {
                tokens,
                text,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            });
        }

        let model_arc = self
            .loaded
            .get(&handle.id)
            .map(|r| r.clone())
            .ok_or_else(|| anyhow::anyhow!("Model {} not loaded in Candle backend", handle.id))?;

        let messages = request.messages.clone();
        let prompt_tokens_pre = request.prompt_tokens.clone();
        let max_tokens = request.max_tokens.max(1) as usize;
        let temperature = request.temperature.max(0.0) as f64;
        let top_p = request.top_p.max(0.0).min(1.0) as f64;
        let stop_sequences = request.stop_sequences.clone();

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<GenerateResult> {
            let mut full_text = String::new();
            let mut completion_tokens: u32 = 0;
            let mut prompt_tokens: u32 = 0;
            let mut finish_reason = String::from("length");
            let mut all_tokens: Vec<u32> = Vec::new();

            run_qwen3_inference(
                &model_arc,
                &messages,
                &prompt_tokens_pre,
                max_tokens,
                temperature,
                top_p,
                &stop_sequences,
                |chunk| -> bool {
                    match chunk {
                        StreamChunk::Token { token_id, text_delta } => {
                            all_tokens.push(token_id);
                            full_text.push_str(&text_delta);
                            true
                        }
                        StreamChunk::Done {
                            finish_reason: r,
                            prompt_tokens: pt,
                            completion_tokens: ct,
                        } => {
                            finish_reason = r;
                            prompt_tokens = pt;
                            completion_tokens = ct;
                            true
                        }
                        StreamChunk::Error(_) => false,
                    }
                },
            )?;

            Ok(GenerateResult {
                tokens: all_tokens,
                text: full_text,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            })
        })
        .await??;

        Ok(result)
    }

    async fn generate_stream(
        &self,
        handle: &ModelHandle,
        request: &GenerateRequest,
    ) -> anyhow::Result<BoxStream<'static, StreamChunk>> {
        // Paged path: route directly through scheduler. Streaming is native.
        if let Some(paged) = self.paged.get(&handle.id).map(|r| r.clone()) {
            let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(64);
            paged.scheduler.submit(SubmitRequest {
                messages: request.messages.clone(),
                prompt_tokens: request.prompt_tokens.clone(),
                max_tokens: request.max_tokens.max(1),
                temperature: request.temperature.max(0.0) as f64,
                top_p: request.top_p.max(0.0).min(1.0) as f64,
                stop_sequences: request.stop_sequences.clone(),
                tx,
            })?;
            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            return Ok(Box::pin(stream));
        }

        let model_arc = self
            .loaded
            .get(&handle.id)
            .map(|r| r.clone())
            .ok_or_else(|| anyhow::anyhow!("Model {} not loaded in Candle backend", handle.id))?;

        let messages = request.messages.clone();
        let prompt_tokens_pre = request.prompt_tokens.clone();
        let max_tokens = request.max_tokens.max(1) as usize;
        let temperature = request.temperature.max(0.0) as f64;
        let top_p = request.top_p.max(0.0).min(1.0) as f64;
        let stop_sequences = request.stop_sequences.clone();

        // Bounded channel: backpressure if the SSE consumer is slow, but never
        // OOMs the inference task.
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(64);

        tokio::task::spawn_blocking(move || {
            let send_tx = tx.clone();
            let result = run_qwen3_inference(
                &model_arc,
                &messages,
                &prompt_tokens_pre,
                max_tokens,
                temperature,
                top_p,
                &stop_sequences,
                |chunk| -> bool {
                    // `blocking_send` returns Err if the receiver is dropped
                    // (client disconnected); use that as a cancel signal.
                    send_tx.blocking_send(chunk).is_ok()
                },
            );
            if let Err(e) = result {
                let _ = tx.blocking_send(StreamChunk::Error(format!("{e:#}")));
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn prefill(
        &self,
        _handle: &ModelHandle,
        tokens: &[u32],
    ) -> anyhow::Result<PrefillResult> {
        // Phase 2: hook into paged KV cache. For now, no-op accounting.
        Ok(PrefillResult {
            tokens_processed: tokens.len() as u32,
            cache_blocks: vec![],
            time_ms: 0.0,
        })
    }

    async fn embed(&self, _handle: &ModelHandle, _text: &str) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("embed() not implemented for Qwen3 LLM models in Phase 1")
    }

    fn backend_name(&self) -> &str {
        "candle"
    }

    fn available() -> bool {
        true
    }
}

/// Detect architecture from `config.json`.
fn detect_architecture(model_dir: &Path) -> anyhow::Result<String> {
    let cfg_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("Cannot read {:?}: {e}", cfg_path))?;
    let cfg: serde_json::Value = serde_json::from_str(&text)?;
    if let Some(model_type) = cfg.get("model_type").and_then(|v| v.as_str()) {
        return Ok(model_type.to_string());
    }
    if let Some(archs) = cfg.get("architectures").and_then(|v| v.as_array()) {
        if let Some(first) = archs.first().and_then(|v| v.as_str()) {
            return Ok(first.to_lowercase());
        }
    }
    anyhow::bail!("Could not detect model_type from config.json at {:?}", model_dir)
}

fn load_tokenizer(model_dir: &Path) -> anyhow::Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("Tokenizer::from_file({:?}) failed: {e}", path))
}

/// Load the Jinja chat template, preferring the dedicated `chat_template.jinja`
/// file over `tokenizer_config.json` (HF's newer convention).
fn load_chat_template(model_dir: &Path) -> anyhow::Result<String> {
    let dedicated = model_dir.join("chat_template.jinja");
    if dedicated.exists() {
        return std::fs::read_to_string(&dedicated)
            .map_err(|e| anyhow::anyhow!("Cannot read {:?}: {e}", dedicated));
    }
    let cfg_path = model_dir.join("tokenizer_config.json");
    let text = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("Cannot read {:?}: {e}", cfg_path))?;
    let cfg: serde_json::Value = serde_json::from_str(&text)?;
    cfg.get("chat_template")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("No chat_template in tokenizer_config.json"))
}

/// EOS ids: union of `generation_config.json`/`tokenizer_config.json`/`config.json`
/// numeric ids, plus tokenizer-resolved IDs for any EOS token strings.
fn read_eos_tokens(model_dir: &Path, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    for filename in [
        "generation_config.json",
        "tokenizer_config.json",
        "config.json",
    ] {
        let path = model_dir.join(filename);
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else { continue };

        if let Some(v) = json.get("eos_token_id") {
            collect_id_or_array(v, &mut ids);
        }
        // String-form `eos_token` in tokenizer_config.json
        if let Some(s) = json.get("eos_token").and_then(|v| v.as_str()) {
            if let Some(id) = tokenizer.token_to_id(s) {
                ids.push(id);
            }
        }
    }
    // Always-on safety: Qwen3 uses <|im_end|> as turn delimiter
    if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
        ids.push(id);
    }
    ids.sort();
    ids.dedup();
    ids
}

fn collect_id_or_array(v: &serde_json::Value, out: &mut Vec<u32>) {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(id) = n.as_u64() {
                out.push(id as u32);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(id) = item.as_u64() {
                    out.push(id as u32);
                }
            }
        }
        _ => {}
    }
}

/// Pick the best dtype for the device. Metal supports BF16 and F16 well, CPU
/// is fastest with F32 but uses more memory; we prefer BF16 if the model is
/// stored in BF16 on disk.
fn pick_dtype(model_dir: &Path, device: &Device) -> DType {
    if device.is_metal() {
        // Metal handles BF16 natively as of Candle 0.10
        DType::BF16
    } else {
        let _ = model_dir;
        DType::F32
    }
}

/// Generic per-architecture config loader. All Candle arch configs implement
/// `serde::Deserialize` from the model's `config.json`.
fn load_arch_config<C: serde::de::DeserializeOwned>(model_dir: &Path) -> anyhow::Result<C> {
    let cfg_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("Cannot read {:?}: {e}", cfg_path))?;
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("Invalid arch config: {e}"))
}

/// Discover safetensors shards: `model.safetensors.index.json` for sharded
/// models, fall back to a single `model.safetensors`.
fn discover_safetensors(model_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let text = std::fs::read_to_string(&index)?;
        let json: serde_json::Value = serde_json::from_str(&text)?;
        let mut set = std::collections::BTreeSet::<String>::new();
        if let Some(map) = json.get("weight_map").and_then(|v| v.as_object()) {
            for v in map.values() {
                if let Some(s) = v.as_str() {
                    set.insert(s.to_string());
                }
            }
        }
        if set.is_empty() {
            anyhow::bail!("safetensors index has no weight_map");
        }
        Ok(set.into_iter().map(|f| model_dir.join(f)).collect())
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!("No safetensors found in {:?}", model_dir);
        }
        Ok(vec![single])
    }
}

/// Render an HF chat template via minijinja. We enable the pycompat extension
/// so Python-isms in templates (`.split`, `.lstrip`, `[-1]`, `[::-1]`, etc.)
/// work out of the box.
fn render_chat_template(template: &str, messages: &[ChatMessage]) -> anyhow::Result<String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", template)
        .map_err(|e| anyhow::anyhow!("Invalid chat template: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow::anyhow!("Cannot fetch chat template: {e}"))?;

    // Convert to plain serde-friendly maps so the template can `message.role`
    // / `message.content` / `message.tool_calls` etc.
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content,
            })
        })
        .collect();

    tmpl.render(context! {
        messages => msgs,
        add_generation_prompt => true,
        // Qwen3 templates branch on this; default OFF for Phase 1 so the model
        // actually emits real content rather than a long thinking trace.
        enable_thinking => false,
    })
    .map_err(|e| anyhow::anyhow!("Chat template render failed: {e}"))
}

/// Core inference loop, shared by `generate` and `generate_stream` and by all
/// supported architectures (Qwen3 / Mistral / Phi-3 / Gemma3 — they all expose
/// the same `forward(tokens, offset)` API).
///
/// `on_chunk` is invoked for every emitted [`StreamChunk`]. Returning `false`
/// from the callback cancels generation immediately (used by SSE when the
/// client disconnects).
fn run_qwen3_inference(
    model_arc: &Arc<LoadedCandleModel>,
    messages: &[ChatMessage],
    prompt_tokens_pre: &[u32],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    stop_sequences: &[String],
    mut on_chunk: impl FnMut(StreamChunk) -> bool,
) -> anyhow::Result<()> {
    let LoadedCandleModel {
        model,
        tokenizer,
        chat_template,
        eos_token_ids,
        device,
        ..
    } = &**model_arc;

    let prompt_ids: Vec<u32> = if !messages.is_empty() {
        let prompt_text = render_chat_template(chat_template, messages)?;
        tokenizer
            .encode(prompt_text, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?
            .get_ids()
            .to_vec()
    } else {
        prompt_tokens_pre.to_vec()
    };

    if prompt_ids.is_empty() {
        anyhow::bail!("Empty prompt: provide either messages or prompt_tokens");
    }
    let prompt_token_count = prompt_ids.len() as u32;

    let mut model_guard = model.lock();
    model_guard.clear_kv_cache();

    let mut sampler = if temperature <= 1e-7 {
        LogitsProcessor::from_sampling(rand::random(), Sampling::ArgMax)
    } else if top_p <= 0.0 || top_p >= 1.0 {
        LogitsProcessor::from_sampling(rand::random(), Sampling::All { temperature })
    } else {
        LogitsProcessor::from_sampling(rand::random(), Sampling::TopP { p: top_p, temperature })
    };

    // Prefill in one shot at offset = 0.
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| anyhow::anyhow!("Prompt tensor build failed: {e}"))?;
    let logits = model_guard
        .forward(&prompt_tensor, 0)
        .map_err(|e| anyhow::anyhow!("Prefill forward failed: {e}"))?;
    let logits = squeeze_to_1d(&logits)?;
    let mut next = sampler
        .sample(&logits)
        .map_err(|e| anyhow::anyhow!("Sampler failed: {e}"))?;

    let mut offset = prompt_ids.len();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut tail = String::new(); // running concatenation of emitted text (for stop-seq scanning)
    let mut decode_stream = tokenizer.decode_stream(true);
    let mut finish_reason = String::from("length");

    loop {
        if eos_token_ids.contains(&next) {
            finish_reason = "stop".into();
            break;
        }
        generated.push(next);

        // Stateful streaming decode: handles byte-fallback / BPE merges /
        // multi-byte UTF-8 cleanly. May return None when waiting for follow-up
        // bytes; that's expected — we just don't emit a token chunk this step.
        let text_delta = decode_stream
            .step(next)
            .map_err(|e| anyhow::anyhow!("decode_stream.step failed: {e}"))?
            .unwrap_or_default();

        if !text_delta.is_empty() {
            tail.push_str(&text_delta);
            let keep_going = on_chunk(StreamChunk::Token {
                token_id: next,
                text_delta,
            });
            if !keep_going {
                finish_reason = "cancelled".into();
                break;
            }
            if !stop_sequences.is_empty()
                && stop_sequences.iter().any(|s| !s.is_empty() && tail.contains(s))
            {
                finish_reason = "stop".into();
                break;
            }
        }

        if generated.len() >= max_tokens {
            finish_reason = "length".into();
            break;
        }

        // Decode step.
        let input = Tensor::new(&[next], device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| anyhow::anyhow!("Decode tensor build failed: {e}"))?;
        let logits = model_guard
            .forward(&input, offset)
            .map_err(|e| anyhow::anyhow!("Decode forward failed: {e}"))?;
        let logits = squeeze_to_1d(&logits)?;
        next = sampler
            .sample(&logits)
            .map_err(|e| anyhow::anyhow!("Sampler failed: {e}"))?;
        offset += 1;
    }

    on_chunk(StreamChunk::Done {
        finish_reason,
        prompt_tokens: prompt_token_count,
        completion_tokens: generated.len() as u32,
    });

    Ok(())
}

/// `Qwen3ForCausalLM::forward` returns `[B=1, T=1, V]` after the internal
/// `narrow(1, l-1, 1)`. Squeeze to a 1-D vector for the sampler.
fn squeeze_to_1d(logits: &Tensor) -> anyhow::Result<Tensor> {
    let dims = logits.dims().len();
    let mut t = logits.clone();
    for _ in 0..(dims - 1) {
        t = t
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("Logits squeeze failed: {e}"))?;
    }
    Ok(t)
}
