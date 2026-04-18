//! gRPC surface for Velox.
//!
//! Mirrors the OpenAI/Anthropic HTTP API but with a typed schema and
//! HTTP/2 + Protobuf framing. Useful when:
//!
//!  * the client is a typed language (Rust, Go, Swift, …) that
//!    benefits from generated stubs
//!  * the workload is many concurrent streams (HTTP/2 multiplexing
//!    avoids head-of-line blocking that plagues HTTP/1.1+SSE)
//!  * the caller wants bidirectional cancellation (drop the stream,
//!    server-side dropping detects it next chunk)
//!
//! All RPCs delegate to the same `EnginePool` + `InferenceBackend`
//! that backs the HTTP routes — no duplicate codegen path.

use crate::backend::traits::{ChatMessage, GenerateRequest, StreamChunk};
use crate::server::AppState;
use async_stream::try_stream;
use futures::StreamExt;
use std::pin::Pin;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

pub mod proto {
    tonic::include_proto!("velox.v1");
}

use proto::velox_server::{Velox, VeloxServer};
use proto::{ChatChunk, GenerateRequest as PbGenerateRequest, ListModelsRequest, ListModelsResponse};

/// Concrete service impl. Holds the shared application state by Arc
/// so it can be cloned cheaply for each in-flight RPC.
pub struct VeloxService {
    state: AppState,
}

impl VeloxService {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    pub fn into_server(self) -> VeloxServer<Self> {
        VeloxServer::new(self)
    }
}

#[tonic::async_trait]
impl Velox for VeloxService {
    async fn list_models(
        &self,
        _: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let ids = self.state.pool.list_models();
        Ok(Response::new(ListModelsResponse { ids }))
    }

    type GenerateStream =
        Pin<Box<dyn Stream<Item = Result<ChatChunk, Status>> + Send + 'static>>;

    async fn generate(
        &self,
        request: Request<PbGenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();

        let handle = self
            .state
            .pool
            .get_model(&req.model)
            .await
            .map_err(|e| Status::not_found(format!("model_not_found: {e}")))?;

        let messages: Vec<ChatMessage> = req
            .messages
            .into_iter()
            .map(|m| ChatMessage { role: m.role, content: m.content })
            .collect();

        let gen_req = GenerateRequest {
            prompt_tokens: vec![],
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop_sequences: req.stop,
        };

        let backend = self.state.pool.backend_arc();
        let mut stream = backend
            .generate_stream(&handle, &gen_req)
            .await
            .map_err(|e| Status::internal(format!("generate_stream: {e}")))?;

        // Re-package the backend's StreamChunk into proto ChatChunk.
        // We translate Done/Error into a final chunk with a non-empty
        // finish_reason instead of an out-of-band terminator — this
        // keeps the protocol uniform (one stream message type).
        let s = try_stream! {
            while let Some(c) = stream.next().await {
                match c {
                    StreamChunk::Token { token_id, text_delta } => {
                        yield ChatChunk {
                            text_delta,
                            token_id,
                            finish_reason: String::new(),
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            error: String::new(),
                        };
                    }
                    StreamChunk::Done { finish_reason, prompt_tokens, completion_tokens } => {
                        yield ChatChunk {
                            text_delta: String::new(),
                            token_id: 0,
                            finish_reason,
                            prompt_tokens,
                            completion_tokens,
                            error: String::new(),
                        };
                        break;
                    }
                    StreamChunk::Error(msg) => {
                        yield ChatChunk {
                            text_delta: String::new(),
                            token_id: 0,
                            finish_reason: "error".into(),
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            error: msg,
                        };
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(s)))
    }
}
