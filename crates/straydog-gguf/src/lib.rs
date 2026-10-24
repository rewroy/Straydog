//! Model preflight for the Straydog inference engine.
//!
//! This crate does not yet execute inference. Header validation establishes
//! structural bounds, not model identity, authenticity, or GPU compatibility.
//! Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

#![forbid(unsafe_code)]

mod gguf;
mod snapshot;

pub use gguf::{inspect, Metadata, ModelHeader, TensorInfo, SIDECAR_ARCHITECTURES};
pub use snapshot::inspect_header;
