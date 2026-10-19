//! Fetching and authenticating the pinned model bundles, and building the MTP drafter from
//! the official tensors. Never used by the inference worker.
//!
//! Interface fixed for phase R5; the bodies are to be written.

use std::path::{Path, PathBuf};

/// Progress of a stage: (stage, bytes done, bytes in the stage).
pub type Progress<'a> = &'a (dyn Fn(&str, u64, u64) + Sync);

/// Download (resuming) and authenticate every pinned file of `quantization` into `root`;
/// with `offline`, only authenticate what is there.
pub fn fetch(root: &Path, quantization: &str, offline: bool, progress: Progress) -> Result<(), String> {
    let _ = (root, quantization, offline, progress);
    unimplemented!()
}

/// Authenticate every pinned file of `quantization` in `root`.
pub fn verify(root: &Path, quantization: &str, progress: Progress) -> Result<(), String> {
    let _ = (root, quantization, progress);
    unimplemented!()
}

/// Build the pinned MTP drafter GGUF in `root` from the authenticated official tensors; its path.
pub fn fetch_drafter(root: &Path, keep_source: bool, progress: Progress) -> Result<PathBuf, String> {
    let _ = (root, keep_source, progress);
    unimplemented!()
}
