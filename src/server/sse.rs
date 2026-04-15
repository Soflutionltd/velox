// SSE streaming for token-by-token output
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use std::convert::Infallible;

/// Create an SSE stream from a token iterator
pub fn token_stream(
    _model: &str,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // TODO: Implement real token streaming from engine
    let stream = futures::stream::once(async {
        Ok(Event::default().data("[DONE]"))
    });
    Sse::new(stream)
}
