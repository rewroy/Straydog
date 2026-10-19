//! Fetching and authenticating the pinned model bundles, and building the MTP drafter from
//! the official tensors. Never used by the inference worker.
//!
//! Every long step reports through [`Progress`] under the label it is shown with
//! ("downloading <file>", "authenticating 2 present file(s)", ...), first with nothing or
//! the resumed part done and last with all of it done. Then the step's outcome, if it has
//! one, is printed to standard output as a line of its own: "verified <file>", "present
//! <file>" or "reused <path> as <file>".

mod assets;
mod http;
mod mtp;

#[cfg(test)]
mod mock;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use straydog_engine::bundle;

/// Progress of a stage: (stage, done, in the stage). Counts are bytes, except while the
/// drafter is built ("building the drafter (Q2_0 experts)"), when they are the routed-expert
/// values quantized. Calls may come from several threads at once (the drafter's download
/// connections, files hashed two at a time).
pub type Progress<'a> = &'a (dyn Fn(&str, u64, u64) + Sync);

/// Download (resuming) and authenticate every pinned file of `quantization` into `root`;
/// with `offline`, only authenticate what is there.
pub fn fetch(root: &Path, quantization: &str, offline: bool, progress: Progress) -> Result<(), String> {
    fetch_assets(root, quantization, offline, true, progress)
}

/// Authenticate every pinned file of `quantization` in `root`.
pub fn verify(root: &Path, quantization: &str, progress: Progress) -> Result<(), String> {
    fetch_assets(root, quantization, true, true, progress)
}

/// [`fetch`], where files already present are authenticated only when `authenticate_existing`:
/// sizes are always checked, but a caller that starts the server right after may leave the
/// hashing to the worker, which authenticates every file before using it. Downloads are
/// always authenticated.
pub fn fetch_assets(root: &Path, quantization: &str, offline: bool, authenticate_existing: bool,
                    progress: Progress) -> Result<(), String> {
    let entries = assets::entries(quantization)?;
    let agent = http::agent(assets::TIMEOUT);
    assets::fetch_with(&agent, root, &entries, offline, authenticate_existing, progress).map_err(|error| error.to_string())
}

/// Build the pinned MTP drafter GGUF in `root` from the authenticated official tensors; its path.
///
/// A drafter already there is authenticated and kept (or removed when it does not match its
/// pin). The official BF16 tensors are fetched into `root` unless an earlier build kept them;
/// `keep_source` keeps them for the next build.
pub fn fetch_drafter(root: &Path, keep_source: bool, progress: Progress) -> Result<PathBuf, String> {
    build_drafter(root, keep_source, progress).map_err(|error| error.to_string())
}

fn build_drafter(root: &Path, keep_source: bool, progress: Progress) -> io::Result<PathBuf> {
    let drafter = bundle::drafter_source().map_err(invalid)?;
    fs::create_dir_all(root).map_err(at(root))?;
    let target = root.join(&drafter.asset.name);
    let label = format!("authenticating {}", drafter.asset.name);
    if target.is_file() {
        match assets::verify_file(&target, &drafter.asset, &label, progress) {
            Ok(()) => return Ok(target),
            Err(error) if error.kind() == io::ErrorKind::InvalidData => fs::remove_file(&target).map_err(at(&target))?,
            Err(error) => return Err(error),
        }
    }
    let source = root.join(mtp::SOURCE_FILE);
    if !source.is_file() {
        let stage = "downloading the official MTP tensors";
        let total = drafter.tensors.iter().map(|tensor| tensor.bytes).sum();
        let done = AtomicU64::new(0);
        progress(stage, 0, total);
        let count = |bytes: u64| progress(stage, done.fetch_add(bytes, Ordering::Relaxed) + bytes, total);
        let base = format!("https://huggingface.co/{}/resolve/{}/", drafter.repository, drafter.revision);
        let agent = http::agent(mtp::TIMEOUT);
        mtp::fetch_source(&agent, &base, &drafter.tensors, &source, mtp::PIECE, &count)?;
    }
    let stage = "building the drafter (Q2_0 experts)";
    mtp::convert(&source, &target, &drafter.revision, mtp::EXPERT_GROUP, &|done, total| progress(stage, done, total))?;
    assets::verify_file(&target, &drafter.asset, &label, progress)?;
    if !keep_source {
        fs::remove_file(&source).map_err(at(&source))?;
    }
    Ok(target)
}

/// A file or an answer that is not what it must be (the reference's ValueError), as opposed
/// to an I/O error: callers that try the next candidate tell them apart by this kind.
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// An I/O error naming the path it concerns.
fn at(path: &Path) -> impl FnOnce(io::Error) -> io::Error + '_ {
    move |error| io::Error::new(error.kind(), format!("{}: {error}", path.display()))
}

/// Where a file is written until it is complete, then renamed to `target`.
fn partial_path(target: &Path) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    target.with_file_name(name)
}
