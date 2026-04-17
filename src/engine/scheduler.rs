// Scheduler: continuous batching, FCFS, concurrency control
// Reference: /tmp/omlx-install/omlx/scheduler.py (4289 lines)

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot, Semaphore};
use crate::backend::traits::*;

/// A pending inference request in the queue
pub struct PendingRequest {
    pub id: String,
    pub model: String,
    pub request: GenerateRequest,
    pub response_tx: oneshot::Sender<Result<GenerateResult, String>>,
    pub created_at: std::time::Instant,
}

/// Scheduler manages the request queue and dispatches to engines
pub struct Scheduler {
    queue: Arc<Mutex<VecDeque<PendingRequest>>>,
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
}

impl Scheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Submit a request and get a receiver for the result
    pub fn submit(&self, model: String, request: GenerateRequest) -> oneshot::Receiver<Result<GenerateResult, String>> {
        let (tx, rx) = oneshot::channel();
        let pending = PendingRequest {
            id: uuid::Uuid::new_v4().to_string(),
            model,
            request,
            response_tx: tx,
            created_at: std::time::Instant::now(),
        };
        self.queue.lock().push_back(pending);
        rx
    }

    /// Start the scheduler loop (runs in background)
    pub fn start(self: Arc<Self>, backend: Arc<dyn InferenceBackend>) -> mpsc::Sender<()> {
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        let queue = self.queue.clone();
        let semaphore = self.semaphore.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        tracing::info!("Scheduler shutting down");
                        break;
                    }
                    permit = semaphore.clone().acquire_owned() => {
                        let permit = match permit {
                            Ok(p) => p,
                            Err(_) => break,
                        };
                        // Try to dequeue a request
                        let pending = { queue.lock().pop_front() };
                        match pending {
                            Some(req) => {
                                let backend = backend.clone();
                                tokio::spawn(async move {
                                    let _permit = permit; // Hold permit until done
                                    let result = backend.generate(
                                        &ModelHandle {
                                            id: req.model.clone(),
                                            path: String::new(),
                                            model_type: ModelType::Llm,
                                            params_total: 0,
                                            params_active: 0,
                                        },
                                        &req.request,
                                    ).await;
                                    let _ = req.response_tx.send(
                                        result.map_err(|e| e.to_string())
                                    );
                                });
                            }
                            None => {
                                drop(permit);
                                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                            }
                        }
                    }
                }
            }
        });
        shutdown_tx
    }

    pub fn queue_len(&self) -> usize {
        self.queue.lock().len()
    }
}
