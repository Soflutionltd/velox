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
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3ForCausalLM};
use dashmap::DashMap;
use minijinja::{context, Environment};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// One loaded model. Variants per supported architecture.
enum LoadedCandleModel {
    Qwen3 {
        model: Mutex<Qwen3ForCausalLM>,
        tokenizer: Tokenizer,
        chat_template: String,
        eos_token_ids: Vec<u32>,
        device: Device,
        dtype: DType,
    },
}

pub struct CandleBackend {
    loaded: Arc<DashMap<String, Arc<LoadedCandleModel>>>,
    device: Device,
}

impl CandleBackend {
    pub fn new() -> Self {
        let device = pick_device();
        tracing::info!("Candle backend initialised on device: {:?}", device);
        Self {
            loaded: Arc::new(DashMap::new()),
            device,
        }
    }

    /// Candle is available on every platform we currently target.
    pub fn available() -> bool {
        true
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
        let device = self.device.clone();

        let handle = tokio::task::spawn_blocking(move || -> anyhow::Result<ModelHandle> {
            let arch = detect_architecture(&path_buf)?;
            tracing::info!("Loading Candle model from {:?} (arch={})", path_buf, arch);

            let tokenizer = load_tokenizer(&path_buf)?;
            let chat_template = load_chat_template(&path_buf)?;
            let eos_token_ids = read_eos_tokens(&path_buf, &tokenizer);
            let dtype = pick_dtype(&path_buf, &device);

            let loaded = match arch.as_str() {
                "qwen3" => {
                    let cfg = load_qwen3_config(&path_buf)?;
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
                    let model = Qwen3ForCausalLM::new(&cfg, vb)
                        .map_err(|e| anyhow::anyhow!("Qwen3ForCausalLM::new failed: {e}"))?;
                    LoadedCandleModel::Qwen3 {
                        model: Mutex::new(model),
                        tokenizer,
                        chat_template,
                        eos_token_ids,
                        device: device.clone(),
                        dtype,
                    }
                }
                other => {
                    anyhow::bail!(
                        "Architecture '{}' not yet supported by Candle backend. Supported: qwen3",
                        other
                    );
                }
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
        Ok(())
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        request: &GenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
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

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<GenerateResult> {
            match &*model_arc {
                LoadedCandleModel::Qwen3 {
                    model,
                    tokenizer,
                    chat_template,
                    eos_token_ids,
                    device,
                    ..
                } => {
                    let prompt_ids: Vec<u32> = if !messages.is_empty() {
                        let prompt_text = render_chat_template(chat_template, &messages)?;
                        tokenizer
                            .encode(prompt_text, true)
                            .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?
                            .get_ids()
                            .to_vec()
                    } else {
                        prompt_tokens_pre
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
                        LogitsProcessor::from_sampling(
                            rand::random(),
                            Sampling::All { temperature },
                        )
                    } else {
                        LogitsProcessor::from_sampling(
                            rand::random(),
                            Sampling::TopP { p: top_p, temperature },
                        )
                    };

                    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
                    let mut finish_reason = String::from("length");

                    // Prefill: feed the entire prompt in one shot, offset = 0.
                    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), device)
                        .map_err(|e| anyhow::anyhow!("Prompt tensor build failed: {e}"))?
                        .unsqueeze(0)
                        .map_err(|e| anyhow::anyhow!("Prompt tensor unsqueeze failed: {e}"))?;
                    let logits = model_guard
                        .forward(&prompt_tensor, 0)
                        .map_err(|e| anyhow::anyhow!("Prefill forward failed: {e}"))?;
                    let logits = squeeze_to_1d(&logits)?;
                    let next = sampler
                        .sample(&logits)
                        .map_err(|e| anyhow::anyhow!("Sampler failed: {e}"))?;

                    let mut offset = prompt_ids.len();
                    if eos_token_ids.contains(&next) {
                        finish_reason = "stop".into();
                    } else {
                        generated.push(next);
                    }

                    // Decode loop: one token at a time.
                    while generated.len() < max_tokens && finish_reason == "length" {
                        let last = *generated.last().unwrap();
                        let input = Tensor::new(&[last], device)
                            .map_err(|e| anyhow::anyhow!("Decode tensor build failed: {e}"))?
                            .unsqueeze(0)
                            .map_err(|e| anyhow::anyhow!("Decode unsqueeze failed: {e}"))?;
                        let logits = model_guard
                            .forward(&input, offset)
                            .map_err(|e| anyhow::anyhow!("Decode forward failed: {e}"))?;
                        let logits = squeeze_to_1d(&logits)?;
                        let next = sampler
                            .sample(&logits)
                            .map_err(|e| anyhow::anyhow!("Sampler failed: {e}"))?;
                        offset += 1;

                        if eos_token_ids.contains(&next) {
                            finish_reason = "stop".into();
                            break;
                        }
                        generated.push(next);
                    }

                    let text = tokenizer
                        .decode(&generated, true)
                        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;

                    Ok(GenerateResult {
                        completion_tokens: generated.len() as u32,
                        tokens: generated,
                        text,
                        finish_reason,
                        prompt_tokens: prompt_token_count,
                    })
                }
            }
        })
        .await??;

        Ok(result)
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

fn load_qwen3_config(model_dir: &Path) -> anyhow::Result<Qwen3Config> {
    let cfg_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("Cannot read {:?}: {e}", cfg_path))?;
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("Invalid Qwen3 config: {e}"))
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
