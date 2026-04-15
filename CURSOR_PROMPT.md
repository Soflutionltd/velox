# AURA Inference Server - Réécriture oMLX en Rust

## Objectif

Réécrire le serveur d'inférence oMLX (Python, 59 000 lignes) en Rust pur.
Le code Python source est dans /tmp/omlx-install/omlx/
Le nouveau projet Rust est dans /Users/antoinepinelli/Cursor/App/aura-inference/

## Pourquoi Rust

- Intégration directe dans l'app AURA (Tauri, déjà en Rust)
- Un seul binaire, pas besoin d'installer Python
- Mémoire déterministe, pas de GC, parfait pour un serveur 24/7
- Cross-compilation possible (macOS + Linux + Windows)
- Les appels MLX passent par mlx-rs (bindings Rust pour MLX d'Apple)
- Sur Windows/Linux, on utilise llama-cpp-rs (bindings Rust pour llama.cpp)

## Architecture cible

```
aura-inference/
├── Cargo.toml
├── src/
│   ├── main.rs                    # Point d'entrée, CLI
│   ├── lib.rs                     # Library pour intégration Tauri
│   ├── server/
│   │   ├── mod.rs
│   │   ├── routes.rs              # Routes API (OpenAI + Anthropic compatible)
│   │   ├── sse.rs                 # Server-Sent Events streaming
│   │   └── middleware.rs          # Auth, CORS, logging
│   ├── engine/
│   │   ├── mod.rs
│   │   ├── pool.rs                # Multi-model engine pool (LRU eviction, TTL)
│   │   ├── batched.rs             # Continuous batching engine (LLM)
│   │   ├── vlm.rs                 # Vision-Language Model engine
│   │   ├── embedding.rs           # Embedding engine
│   │   ├── reranker.rs            # Reranker engine
│   │   └── scheduler.rs           # FCFS scheduler, configurable concurrency
│   ├── cache/
│   │   ├── mod.rs
│   │   ├── paged.rs               # PagedCacheManager (GPU blocks, CoW, prefix sharing)
│   │   ├── hot.rs                 # In-memory hot cache (write-back)
│   │   ├── ssd.rs                 # SSD cold tier (safetensors format)
│   │   └── prefix.rs              # Prefix sharing for repeated system prompts
│   ├── backend/
│   │   ├── mod.rs
│   │   ├── mlx.rs                 # Apple Silicon: mlx-rs bindings
│   │   ├── llamacpp.rs            # Windows/Linux: llama-cpp-rs bindings
│   │   └── traits.rs              # InferenceBackend trait (common interface)
│   ├── api/
│   │   ├── mod.rs
│   │   ├── openai.rs              # OpenAI API compatibility layer
│   │   ├── anthropic.rs           # Anthropic API compatibility layer
│   │   └── tool_calling.rs        # Tool/function calling support
│   ├── model/
│   │   ├── mod.rs
│   │   ├── discovery.rs           # Auto-detect MLX/GGUF models from directory
│   │   ├── registry.rs            # Model registry + settings
│   │   └── download.rs            # HuggingFace model downloader
│   ├── memory/
│   │   ├── mod.rs
│   │   └── enforcer.rs            # Process-level memory limits, TTL checks
│   ├── config.rs                  # Settings (persisted to ~/.aura/settings.json)
│   ├── metrics.rs                 # Server metrics (tokens/s, cache hit rate)
│   └── error.rs                   # Error types
├── tests/
│   ├── test_server.rs
│   ├── test_cache.rs
│   ├── test_engine.rs
│   └── test_api.rs
└── benches/
    └── inference_bench.rs
```

## Correspondance Python -> Rust

Voici la correspondance fichier par fichier entre le code oMLX Python et ce qu'il faut réécrire en Rust :

| Python (oMLX) | Lignes | Rust (aura-inference) | Priorité |
|---|---|---|---|
| server.py | 4359 | server/routes.rs + server/sse.rs | P0 |
| scheduler.py | 4289 | engine/scheduler.rs | P0 |
| cache/paged_ssd_cache.py | 1984 | cache/ssd.rs | P0 |
| cache/prefix_cache.py | 2259 | cache/prefix.rs | P0 |
| cache/paged_cache.py | 1732 | cache/paged.rs | P0 |
| engine_pool.py | 921 | engine/pool.rs | P0 |
| settings.py | 1233 | config.rs | P1 |
| api/tool_calling.py | 1270 | api/tool_calling.rs | P1 |
| api/anthropic_utils.py | 947 | api/anthropic.rs | P1 |
| engine/vlm.py | 1513 | engine/vlm.rs | P2 |
| oq.py (quantization) | 1926 | (non réécrit, utilise JANG externe) | P3 |
| model_discovery.py | 813 | model/discovery.rs | P1 |
| admin/routes.py | 4290 | (admin UI séparé, pas P0) | P3 |
| admin/hf_downloader.py | 849 | model/download.rs | P2 |
| cache/type_handlers.py | 944 | cache/mod.rs | P1 |
| api/utils.py | 839 | api/mod.rs | P1 |
| process_memory_enforcer.py | ~200 | memory/enforcer.rs | P1 |
| memory_monitor.py | ~150 | memory/mod.rs | P1 |
| engine_core.py | ~400 | engine/mod.rs | P0 |

## Dépendances Rust (Cargo.toml)

```toml
[package]
name = "aura-inference"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[dependencies]
# Web server
axum = { version = "0.8", features = ["macros", "ws"] }
tokio = { version = "1", features = ["full"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# MLX bindings (macOS only)
mlx-rs = { version = "0.20", optional = true }

# llama.cpp bindings (cross-platform fallback)
llama-cpp-rs = { version = "0.4", optional = true }

# Cache
safetensors = "0.4"
memmap2 = "0.9"

# Async
async-trait = "0.1"
futures = "0.3"

# Utils
tracing = "0.1"
tracing-subscriber = "0.3"
clap = { version = "4", features = ["derive"] }
uuid = { version = "1", features = ["v4"] }
dashmap = "6"
parking_lot = "0.12"
bytes = "1"

[features]
default = ["mlx"]
mlx = ["mlx-rs"]
llamacpp = ["llama-cpp-rs"]
```

## Instructions de réécriture

### Phase 1 : Serveur HTTP + API (P0)

Commence par réécrire server.py en Rust avec Axum :

1. Lis /tmp/omlx-install/omlx/server.py (4359 lignes)
2. Crée src/server/routes.rs avec toutes les routes :
   - POST /v1/chat/completions (OpenAI compatible)
   - POST /v1/completions
   - POST /v1/embeddings
   - GET /v1/models
   - POST /v1/messages (Anthropic compatible)
3. Crée src/server/sse.rs pour le streaming SSE
4. Implémente le streaming token par token

### Phase 2 : Cache KV deux niveaux (P0)

C'est LE feature clé d'oMLX. Le cache doit être fidèlement réécrit :

1. Lis /tmp/omlx-install/omlx/cache/paged_cache.py (1732 lignes)
   - Block-based KV cache avec Copy-on-Write
   - Pages de taille fixe, allocation/désallocation
   - Prefix sharing (les requêtes avec le même system prompt partagent les blocs)

2. Lis /tmp/omlx-install/omlx/cache/paged_ssd_cache.py (1984 lignes)
   - Cold tier : les blocs inactifs sont sérialisés en safetensors sur SSD
   - Restore : quand un prefix revient, restaurer depuis SSD au lieu de recalculer
   - Persiste entre les redémarrages du serveur

3. Lis /tmp/omlx-install/omlx/cache/prefix_cache.py (2259 lignes)
   - Trie des prefixes de tokens
   - Matching des prefixes communs entre requêtes
   - Eviction LRU quand le cache est plein

Implémente en Rust avec :
- memmap2 pour le mapping mémoire SSD
- safetensors crate pour la sérialisation
- dashmap pour le concurrent access
- parking_lot pour les locks rapides

### Phase 3 : Scheduler + Batching (P0)

1. Lis /tmp/omlx-install/omlx/scheduler.py (4289 lignes)
   - Continuous batching : nouvelles requêtes ajoutées au batch en cours
   - FCFS scheduling avec concurrence configurable
   - Prefill et completion batch sizes séparés
   - Gestion des timeouts et annulations

2. Lis /tmp/omlx-install/omlx/engine_pool.py (921 lignes)
   - Multi-model serving avec LRU eviction
   - TTL par modèle (décharge après inactivité)
   - Load/unload/pin models
   - Memory limits

### Phase 4 : Backend d'inférence (P0)

Crée un trait InferenceBackend que les deux backends implémentent :

```rust
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn load_model(&self, path: &Path, config: &ModelConfig) -> Result<ModelHandle>;
    async fn generate(&self, handle: &ModelHandle, request: &GenerateRequest) -> Result<TokenStream>;
    async fn prefill(&self, handle: &ModelHandle, tokens: &[u32]) -> Result<PrefillResult>;
    async fn embed(&self, handle: &ModelHandle, text: &str) -> Result<Vec<f32>>;
    fn backend_name(&self) -> &str;
    fn available(&self) -> bool;
}
```

Pour macOS : implémenter avec mlx-rs (crate mlx-rs sur crates.io)
- Regarde https://github.com/oxideai/mlx-rs pour l'API
- L'inférence passe par mlx_nn::Module et mlx_optimizers
- Le prefill et la génération utilisent mlx_rs::ops

Pour Windows/Linux : implémenter avec llama-cpp-rs
- Regarde https://github.com/utilityai/llama-cpp-rs
- Supporte CUDA, ROCm, Metal, CPU
- API similaire : load model, tokenize, generate

### Phase 5 : API compatibility layers (P1)

1. Lis /tmp/omlx-install/omlx/api/anthropic_utils.py (947 lignes)
   - Convertir messages Anthropic -> format interne
   - Supporter le thinking/reasoning
   - Supporter les vision inputs (images base64)

2. Lis /tmp/omlx-install/omlx/api/tool_calling.py (1270 lignes)
   - Supporter tous les formats de tool calling de mlx-lm
   - JSON schema validation
   - MCP tool integration

### Phase 6 : Model discovery + Download (P1)

1. Lis /tmp/omlx-install/omlx/model_discovery.py (813 lignes)
   - Scanner un répertoire pour détecter les modèles MLX et GGUF
   - Auto-détecter le type (LLM, VLM, embedding, reranker)
   - Supporter les dossiers à deux niveaux (org/model-name)

2. Lis /tmp/omlx-install/omlx/admin/hf_downloader.py (849 lignes)
   - Télécharger des modèles depuis HuggingFace
   - Progress bar, resume, verification

### Phase 7 : Intégration Tauri (finale)

Une fois que le serveur compile et fonctionne, l'intégrer dans Tauri :

1. Exposer le serveur comme library via lib.rs
2. Dans l'app Tauri, démarrer le serveur en background au lancement
3. L'app communique avec le serveur via localhost
4. Pas besoin d'installer Python, oMLX, ou Ollama séparément

## Contraintes

- Tout en Rust, pas de Python, pas de Node
- Doit compiler avec `cargo build --release`
- macOS : feature "mlx" activée par défaut
- Windows/Linux : feature "llamacpp" à la place
- Le cache SSD utilise le format safetensors (compatible avec oMLX Python)
- L'API est 100% compatible OpenAI et Anthropic
- Performance : le goulot est les kernels MLX/Metal, pas le serveur Rust
- Priorité : faire marcher les P0 d'abord, les P1 ensuite, les P2/P3 plus tard

## Code source Python de référence

Le code Python d'oMLX est dans /tmp/omlx-install/omlx/
Lis chaque fichier Python AVANT de réécrire en Rust.
Ne traduis pas ligne par ligne. Comprends la logique, puis réécris idiomatiquement en Rust.

Fichiers critiques à lire en priorité (dans cet ordre) :
1. /tmp/omlx-install/omlx/server.py (le serveur FastAPI, 4359 lignes)
2. /tmp/omlx-install/omlx/scheduler.py (le scheduler, 4289 lignes)
3. /tmp/omlx-install/omlx/cache/paged_cache.py (cache GPU, 1732 lignes)
4. /tmp/omlx-install/omlx/cache/paged_ssd_cache.py (cache SSD, 1984 lignes)
5. /tmp/omlx-install/omlx/cache/prefix_cache.py (prefix trie, 2259 lignes)
6. /tmp/omlx-install/omlx/engine_pool.py (multi-model pool, 921 lignes)
7. /tmp/omlx-install/omlx/engine_core.py (engine core, ~400 lignes)
8. /tmp/omlx-install/omlx/api/tool_calling.py (tool calling, 1270 lignes)
9. /tmp/omlx-install/omlx/api/anthropic_utils.py (Anthropic compat, 947 lignes)
10. /tmp/omlx-install/omlx/settings.py (config, 1233 lignes)

## Commande de démarrage

Le serveur doit se lancer avec :
```bash
cargo run -- serve --model-dir ~/models --port 8000
# ou
aura-inference serve --model-dir ~/models --port 8000 --hot-cache-max-size 20%
```

## Go !

Commence par Phase 1 (serveur HTTP + routes API).
Crée les fichiers, compile avec `cargo build`, corrige les erreurs.
Puis passe à Phase 2 (cache), Phase 3 (scheduler), etc.
Un commit par phase avec un message clair.
