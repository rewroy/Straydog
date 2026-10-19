//! Containment of the GPU worker: bubblewrap on Linux, an AppContainer of its own on
//! Windows. There is no fallback to an unrestricted child.
//!
//! Interface fixed for phase R5; the bodies are to be written.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// What the contained worker may read.
#[derive(Debug, Clone, Default)]
pub struct Exposure {
    /// The straydog executable the worker runs (the same binary as the server).
    pub executable: PathBuf,
    /// Pinned files the worker reads: model shards and resources, the drafter.
    pub files: Vec<PathBuf>,
    /// Directories the worker opens files in (the model and drafter directories).
    pub directories: Vec<PathBuf>,
    /// The worker's working directory (the model directory).
    pub working_directory: PathBuf,
    /// Never exposed, nor any directory holding one (the API key file).
    pub secrets: Vec<PathBuf>,
}

/// Why this host cannot contain the worker, or None when it can.
pub fn unavailable_reason() -> Option<String> {
    unimplemented!()
}

/// Refuse to expose a secret: no exposed path may be one, hold one or lie inside one.
pub fn check_exposable(paths: &[&Path], secrets: &[PathBuf]) -> Result<(), String> {
    let _ = (paths, secrets);
    unimplemented!()
}

/// The program and arguments that run `executable worker_args...` contained, to be spawned
/// by the server with piped standard handles.
pub fn worker_command(worker_args: &[OsString], exposure: &Exposure) -> Result<(PathBuf, Vec<OsString>), String> {
    let _ = (worker_args, exposure);
    unimplemented!()
}

/// Windows: the `straydog contain` entry point that grants access, starts the worker in the
/// AppContainer with this process's standard handles, and returns its exit status.
#[cfg(windows)]
pub fn launcher_main(args: &[OsString]) -> i32 {
    let _ = args;
    unimplemented!()
}
