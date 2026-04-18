// MLX backend for Apple Silicon — uses mlx-rs + mlx-lm
//
// Architecture:
//   - MlxBackend holds a DashMap<id, Arc<LoadedMlxModel>>
//   - ModelHandle.id is the key into that map (handle stays cheap to clone)
//   - LoadedMlxModel is an enum, one variant per model architecture
//   - generate() runs inside spawn_blocking (mlx-rs is sync and CPU/GPU bound)
//
// Reference: examples/lm/src/main.rs in oxideai/mlx-rs

use super::traits::*;
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use mlx_lm::cache::ConcatKeyValueCache;
use mlx_lm::models::qwen3::{
    get_qwen3_model_args, Generate as Qwen3Generate, Model as Qwen3Model,
    WeightMap as Qwen3WeightMap,
};
use mlx_rs::quantization::MaybeQuantized;
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Tokenizer,
};
use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
use mlx_rs::nn;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use serde::Serialize;

/// Role that serializes to lowercase strings, compatible with HF chat templates
/// (Qwen3 expects "system", "user", "assistant", "tool").
/// We use this instead of `mlx_lm_utils::tokenizer::Role` which only supports
/// User and Assistant.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One loaded model. Variants per supported architecture.
enum LoadedMlxModel {
    Qwen3 {
        model: Mutex<Qwen3Model>,
        tokenizer: Mutex<Tokenizer>,
        chat_template: String,
        model_id: String,
        eos_token_ids: Vec<u32>,
    },
}

/// Detect the architecture from the model directory's config.json.
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

/// Read EOS token ids from generation_config.json or tokenizer_config.json.
fn read_eos_tokens(model_dir: &Path) -> Vec<u32> {
    let mut ids = Vec::new();
    for filename in ["generation_config.json", "tokenizer_config.json", "config.json"] {
        let path = model_dir.join(filename);
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(v) = json.get("eos_token_id") {
                    match v {
                        serde_json::Value::Number(n) => {
                            if let Some(id) = n.as_u64() {
                                ids.push(id as u32);
                            }
                        }
                        serde_json::Value::Array(arr) => {
                            for item in arr {
                                if let Some(id) = item.as_u64() {
                                    ids.push(id as u32);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if !ids.is_empty() {
                    break;
                }
            }
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

pub struct MlxBackend {
    loaded: Arc<DashMap<String, Arc<LoadedMlxModel>>>,
}

impl MlxBackend {
    pub fn new() -> Self {
        Self {
            loaded: Arc::new(DashMap::new()),
        }
    }

    pub fn available() -> bool {
        cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
    }
}

#[async_trait]
impl InferenceBackend for MlxBackend {
    async fn load_model(&self, path: &Path) -> anyhow::Result<ModelHandle> {
        let path = path.to_path_buf();
        let loaded_map = self.loaded.clone();

        let handle = tokio::task::spawn_blocking(move || -> anyhow::Result<ModelHandle> {
            let arch = detect_architecture(&path)?;
            tracing::info!("Loading MLX model from {:?} (arch={})", path, arch);

            let tokenizer_file = path.join("tokenizer.json");
            let tokenizer_config_file = path.join("tokenizer_config.json");

            let tokenizer = Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
            let chat_template = load_model_chat_template_from_file(&tokenizer_config_file)?
                .ok_or_else(|| anyhow::anyhow!("No chat_template in tokenizer_config.json"))?;
            let eos_token_ids = read_eos_tokens(&path);

            let model_id = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            let loaded = match arch.as_str() {
                "qwen3" => {
                    // If config.json has a "quantization" block, run our custom quantized
                    // loader (which has to work around upstream bugs). Otherwise use
                    // mlx-lm's stock loader for plain bf16/fp16 weights.
                    let model = if read_quantization(&path).is_some() {
                        load_qwen3_quantized(&path)
                            .map_err(|e| anyhow::anyhow!("load_qwen3 (quantized) failed: {e}"))?
                    } else {
                        tracing::info!("Loading non-quantized Qwen3 via stock mlx-lm loader");
                        mlx_lm::models::qwen3::load_qwen3_model(&path)
                            .map_err(|e| anyhow::anyhow!("load_qwen3_model failed: {e:?}"))?
                    };
                    LoadedMlxModel::Qwen3 {
                        model: Mutex::new(model),
                        tokenizer: Mutex::new(tokenizer),
                        chat_template,
                        model_id: model_id.clone(),
                        eos_token_ids,
                    }
                }
                other => {
                    anyhow::bail!(
                        "Architecture '{}' not yet supported. Supported: qwen3",
                        other
                    );
                }
            };

            let id = uuid::Uuid::new_v4().to_string();
            let handle = ModelHandle {
                id: id.clone(),
                path: path.to_string_lossy().to_string(),
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
            .ok_or_else(|| anyhow::anyhow!("Model {} not loaded in MLX backend", handle.id))?;

        let messages = request.messages.clone();
        let prompt_tokens_pre = request.prompt_tokens.clone();
        let max_tokens = request.max_tokens.max(1);
        let temperature = request.temperature.max(0.0);

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<GenerateResult> {
            match &*model_arc {
                LoadedMlxModel::Qwen3 {
                    model,
                    tokenizer,
                    chat_template,
                    model_id,
                    eos_token_ids,
                } => {
                    let mut tokenizer_guard = tokenizer.lock();

                    let prompt_ids: Vec<u32> = if !messages.is_empty() {
                        let convs: Vec<Conversation<ChatRole, String>> = messages
                            .iter()
                            .map(|m| Conversation {
                                role: parse_role(&m.role),
                                content: m.content.clone(),
                            })
                            .collect();

                        let args = ApplyChatTemplateArgs {
                            conversations: vec![convs.into()],
                            documents: None,
                            model_id: model_id.as_str(),
                            chat_template_id: None,
                            add_generation_prompt: Some(true),
                            continue_final_message: None,
                        };
                        let encodings = tokenizer_guard
                            .apply_chat_template_and_encode(chat_template.clone(), args)
                            .map_err(|e| anyhow::anyhow!("apply_chat_template failed: {:?}", e))?;
                        encodings
                            .iter()
                            .flat_map(|enc| enc.get_ids())
                            .copied()
                            .collect()
                    } else {
                        prompt_tokens_pre
                    };

                    if prompt_ids.is_empty() {
                        anyhow::bail!("Empty prompt: provide either messages or prompt_tokens");
                    }

                    let prompt_token_count = prompt_ids.len() as u32;
                    let prompt_array = Array::from(&prompt_ids[..]).index(NewAxis);

                    let mut model_guard = model.lock();
                    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();

                    let generate = Qwen3Generate::<ConcatKeyValueCache>::new(
                        &mut *model_guard,
                        &mut cache,
                        temperature,
                        &prompt_array,
                    );

                    let mut tokens: Vec<Array> = Vec::new();
                    let mut completion_ids: Vec<u32> = Vec::new();
                    let mut finish_reason = String::from("length");

                    for (token, ntoks) in generate.zip(0..max_tokens as usize) {
                        let token = token.map_err(|e| anyhow::anyhow!("Generate step failed: {e}"))?;
                        tokens.push(token.clone());

                        if ntoks == 0 {
                            eval(&tokens).ok();
                        }

                        if tokens.len() >= 16 {
                            eval(&tokens).ok();
                            let drained: Vec<u32> =
                                tokens.drain(..).map(|t| t.item::<u32>()).collect();
                            completion_ids.extend_from_slice(&drained);

                            if drained.iter().any(|id| eos_token_ids.contains(id)) {
                                finish_reason = "stop".into();
                                break;
                            }
                        }
                    }

                    if !tokens.is_empty() {
                        eval(&tokens).ok();
                        let drained: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
                        completion_ids.extend_from_slice(&drained);
                    }

                    if eos_token_ids
                        .iter()
                        .any(|eos| completion_ids.contains(eos))
                    {
                        finish_reason = "stop".into();
                    }

                    let visible: Vec<u32> = completion_ids
                        .iter()
                        .copied()
                        .filter(|id| !eos_token_ids.contains(id))
                        .collect();

                    let text = tokenizer_guard
                        .decode(&visible, true)
                        .map_err(|e| anyhow::anyhow!("Decode failed: {:?}", e))?;

                    Ok(GenerateResult {
                        tokens: completion_ids.clone(),
                        text,
                        finish_reason,
                        prompt_tokens: prompt_token_count,
                        completion_tokens: completion_ids.len() as u32,
                    })
                }
            }
        })
        .await??;

        Ok(result)
    }

    async fn prefill(&self, _handle: &ModelHandle, tokens: &[u32]) -> anyhow::Result<PrefillResult> {
        // Phase 2: integrate with paged KV cache. For now we simply report.
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
        "mlx"
    }

    fn available() -> bool {
        cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
    }
}

fn parse_role(s: &str) -> ChatRole {
    match s.to_lowercase().as_str() {
        "system" => ChatRole::System,
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

/// Quantization parameters from a model's config.json.
#[derive(Debug, serde::Deserialize)]
struct QuantizationConfig {
    bits: i32,
    group_size: i32,
}

/// Read quantization parameters from config.json (if present).
/// MLX models from `mlx-community` store these under the `quantization` field.
fn read_quantization(model_dir: &Path) -> Option<QuantizationConfig> {
    let cfg_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&cfg_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let q = json.get("quantization").or_else(|| json.get("quantization_config"))?;
    serde_json::from_value(q.clone()).ok()
}

/// Load a Qwen3 model, applying quantization if the config specifies it.
///
/// Why we need this: upstream `mlx_lm::models::qwen3::load_qwen3_model` builds a
/// fresh non-quantized `Model` and then loads weights — but the safetensors from
/// `mlx-community/Qwen3-*-4bit` contain pre-quantized tensors with mismatched
/// shapes vs. plain `nn::Linear` weights. We must quantize the model FIRST,
/// then load the weights into the now-correctly-shaped `QuantizedLinear`s.
fn load_qwen3_quantized(model_dir: &Path) -> anyhow::Result<Qwen3Model> {
    let model_args = get_qwen3_model_args(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to read qwen3 model args: {e}"))?;

    let mut model = Qwen3Model::new(model_args)
        .map_err(|e| anyhow::anyhow!("Qwen3Model::new failed: {e}"))?;

    if let Some(q) = read_quantization(model_dir) {
        tracing::info!("Quantizing Qwen3 model: {}-bit, group_size={}", q.bits, q.group_size);
        model = nn::quantize(model, Some(q.group_size), Some(q.bits))
            .map_err(|e| anyhow::anyhow!("nn::quantize failed: {e}"))?;
    } else {
        tracing::info!("No quantization config found, loading as full precision");
    }

    // Discover all safetensors shards from the index file (or fall back to the
    // single-file `model.safetensors`).
    let shard_files: Vec<std::path::PathBuf> = {
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let json = std::fs::read_to_string(&index_path)
                .map_err(|e| anyhow::anyhow!("Cannot read {:?}: {e}", index_path))?;
            let weight_map: Qwen3WeightMap = serde_json::from_str(&json)?;
            let unique: std::collections::HashSet<&String> =
                weight_map.weight_map.values().collect();
            unique.into_iter().map(|f| model_dir.join(f)).collect()
        } else {
            vec![model_dir.join("model.safetensors")]
        }
    };

    for shard in shard_files {
        tracing::debug!("Loading safetensors shard {:?}", shard);
        load_safetensors_remapped(&mut model, &shard)
            .map_err(|e| anyhow::anyhow!("load_safetensors({:?}) failed: {e}", shard))?;
        patch_quantized_embedding(&mut model, &shard)
            .map_err(|e| anyhow::anyhow!("patch_quantized_embedding({:?}) failed: {e}", shard))?;
    }

    Ok(model)
}

/// Workaround for an mlx-rs 0.25.3 bug: `QuantizedEmbedding` doesn't have
/// `#[param]` attributes on its `scales`, `biases`, and `inner` fields, so
/// its parameters never appear in `parameters_mut().flatten()` and cannot
/// be loaded via the normal safetensors flow. We assign them by hand.
fn patch_quantized_embedding(
    model: &mut Qwen3Model,
    shard: &Path,
) -> anyhow::Result<()> {
    let MaybeQuantized::Quantized(qe) = &mut model.model.embed_tokens else {
        // Embedding wasn't quantized → nothing to patch.
        return Ok(());
    };

    let tensors: HashMap<String, Array> = Array::load_safetensors(shard)
        .map_err(|e| anyhow::anyhow!("Array::load_safetensors failed: {e}"))?
        .into_iter()
        .collect();

    let mut patched = 0;
    if let Some(w) = tensors.get("model.embed_tokens.weight") {
        qe.inner.weight.value = w.clone();
        patched += 1;
    }
    if let Some(s) = tensors.get("model.embed_tokens.scales") {
        qe.scales.value = s.clone();
        patched += 1;
    }
    if let Some(b) = tensors.get("model.embed_tokens.biases") {
        qe.biases.value = b.clone();
        patched += 1;
    }
    if patched > 0 {
        tracing::info!("Patched {} quantized-embedding tensors", patched);
    }
    Ok(())
}

/// Load safetensors weights into a model, handling the parameter-key
/// differences between safetensors files and `mlx-rs`'s module structure.
///
/// `mlx-rs`'s `QuantizedLinear` wraps a `Linear` in an `inner` field, so its
/// parameter keys look like `…q_proj.inner.weight` while the safetensors
/// store `…q_proj.weight`. `QuantizedEmbedding` has no such wrapping.
///
/// Strategy: probe the model's actual parameter keys. For each tensor in the
/// safetensors file, try the key as-is; if that misses but `<prefix>.inner.<suffix>`
/// matches a model param, use the wrapped form instead. This is robust to
/// whichever module type a given prefix corresponds to.
fn load_safetensors_remapped(
    model: &mut Qwen3Model,
    path: &Path,
) -> anyhow::Result<()> {
    let loaded: Vec<(String, Array)> = Array::load_safetensors(path)
        .map_err(|e| anyhow::anyhow!("Array::load_safetensors failed: {e}"))?
        .into_iter()
        .collect();

    let mut params = model.parameters_mut().flatten();
    let total_params = params.len();
    let mut assigned = 0usize;
    let mut missing: Vec<String> = Vec::new();

    for (key, value) in loaded {
        let target_key = pick_param_key(&key, &params);
        match target_key.as_deref().and_then(|k| params.get_mut(k)) {
            Some(param) => {
                **param = value;
                assigned += 1;
            }
            None => missing.push(key),
        }
    }

    if !missing.is_empty() {
        tracing::warn!(
            "Safetensors had {} keys with no matching model param. First 5: {:?}",
            missing.len(),
            missing.iter().take(5).collect::<Vec<_>>()
        );
    }
    if assigned < total_params {
        // List which model params went unassigned so we can identify random-init weights.
        let assigned_set: std::collections::HashSet<String> = params.keys().map(|k| k.to_string()).collect();
        let _ = assigned_set;
        tracing::warn!(
            "Only {}/{} model params were assigned from safetensors",
            assigned,
            total_params
        );
    }
    tracing::info!("Loaded {} tensors into model from {:?} (model has {} params)", assigned, path, total_params);

    model
        .eval()
        .map_err(|e| anyhow::anyhow!("model.eval() failed: {e}"))?;
    Ok(())
}

/// Choose the right model-parameter key for a safetensors tensor name.
/// Tries the key as-is first, then `<prefix>.inner.<suffix>` if applicable.
fn pick_param_key<'a>(
    safetensor_key: &'a str,
    params: &mlx_rs::module::FlattenedModuleParamMut<'_>,
) -> Option<String> {
    if params.contains_key(safetensor_key) {
        return Some(safetensor_key.to_string());
    }
    for suffix in [".weight", ".bias"] {
        if let Some(prefix) = safetensor_key.strip_suffix(suffix) {
            let wrapped = format!("{prefix}.inner{suffix}");
            if params.contains_key(wrapped.as_str()) {
                return Some(wrapped);
            }
        }
    }
    None
}
