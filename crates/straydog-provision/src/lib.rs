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
mod vision;

/// A local HTTP server for the download tests (also for a crate's tests of what it builds on fetching: feature `mock`).
#[cfg(any(test, feature = "mock"))]
pub mod mock;

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

/// The optional files of a bundle read on demand that a fetch adds to its shards: DeepSeek-V4.1's
/// DSpark drafter and its vision encoder (a model without either pinned has nothing to add).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extras {
    pub dspark: bool,
    pub vision: bool,
}

impl Extras {
    pub const ALL: Extras = Extras { dspark: true, vision: true };
}

/// Download (resuming) and authenticate `files` (their pins and plain-HTTP URLs) into `root`, as the
/// pinned bundles are fetched, from a test server: the pinned downloads are HTTPS only.
#[cfg(any(test, feature = "mock"))]
pub fn fetch_files_plain(root: &Path, files: &[(bundle::Asset, String)], progress: Progress) -> Result<(), String> {
    assets::fetch_with(&http::plain(assets::TIMEOUT), root, files, false, true, progress).map_err(|error| error.to_string())
}

/// Download (resuming) and authenticate every pinned file of `quantization` into `root`;
/// with `offline`, only authenticate what is there. The shards' verity trees are then built into
/// `data_dir` (by default `.straydog` in `root`): all of a bundle read on demand's, which they
/// authenticate, and the others' that are missing.
pub fn fetch(root: &Path, quantization: &str, offline: bool, data_dir: Option<&Path>, progress: Progress) -> Result<(), String> {
    fetch_extras(root, quantization, offline, data_dir, Extras::ALL, progress)
}

/// [`fetch`], adding of a bundle read on demand's optional files only those `extras` names (those
/// already there are authenticated all the same).
pub fn fetch_extras(root: &Path, quantization: &str, offline: bool, data_dir: Option<&Path>, extras: Extras,
                    progress: Progress) -> Result<(), String> {
    let data_dir = data_dir.map_or_else(|| bundle::default_data_dir(root), Path::to_path_buf);
    if bundle::is_on_demand(quantization) {
        let entries = assets::entries(quantization)?;
        let agent = http::agent(assets::TIMEOUT);
        // The trees authenticate the files present: they are not hashed twice.
        assets::fetch_with(&agent, root, &entries, offline, false, progress).map_err(|error| error.to_string())?;
        build_vision(root, quantization, offline || !extras.vision, progress).map_err(|error| error.to_string())?;
        fetch_dspark(root, quantization, offline || !extras.dspark, progress).map_err(|error| error.to_string())?;
        return print_trees(root, quantization, &data_dir, false, progress);
    }
    fetch_assets(root, quantization, offline, true, progress)?;
    print_trees(root, quantization, &data_dir, true, progress)
}

/// Authenticate every pinned file of `quantization` in `root` (a bundle read on demand: by
/// building its verity trees into `data_dir`, by default `.straydog` in `root`; another: by
/// hashing, then building the trees missing there).
pub fn verify(root: &Path, quantization: &str, data_dir: Option<&Path>, progress: Progress) -> Result<(), String> {
    let data_dir = data_dir.map_or_else(|| bundle::default_data_dir(root), Path::to_path_buf);
    if bundle::is_on_demand(quantization) {
        return print_trees(root, quantization, &data_dir, false, progress);
    }
    fetch_assets(root, quantization, true, true, progress)?;
    print_trees(root, quantization, &data_dir, true, progress)
}

/// Build the verity trees of `quantization`'s shards that `data_dir` lacks (a server's first
/// start): each kept only when its digest is the pinned one. A tree there already is checked
/// against its pin whenever it is opened, and each block it covers whenever it is used.
pub fn ensure_trees(root: &Path, quantization: &str, data_dir: &Path, progress: Progress) -> Result<Vec<String>, String> {
    build_trees(root, quantization, data_dir, true, progress).map_err(|error| error.to_string())
}

/// A directory trees are about to be written into: made if missing, and found writable, or an
/// error naming the fix.
fn writable(directory: &Path) -> io::Result<()> {
    let probe = directory.join(".straydog-write-test");
    let tried = fs::create_dir_all(directory).and_then(|()| fs::write(&probe, b"")).and_then(|()| fs::remove_file(&probe));
    tried.map_err(|error| {
        io::Error::new(error.kind(), format!("cannot write verity trees into {} ({error}): give --data-dir DIR, a directory \
                                              Straydog may write, to keep them there", directory.display()))
    })
}

/// The verity trees of a bundle's shards (straydog_engine::verity), built from the files into
/// `data_dir/verity` (all of them, or with `missing_only` those not there yet) and kept only
/// when their digests are the pinned ones. That authenticates the files: every block hashes
/// into the pinned root. Nothing but `data_dir` is written (by default in the model directory).
fn build_trees(root: &Path, quantization: &str, data_dir: &Path, missing_only: bool, progress: Progress) -> io::Result<Vec<String>> {
    let mut pins = bundle::verity_files_in(root, quantization).map_err(invalid)?;
    if missing_only {
        pins.retain(|(_, asset, _)| !bundle::tree_path(data_dir, asset).is_file());
    }
    let mut names = Vec::new();
    if pins.is_empty() {
        return Ok(names);
    }
    writable(&data_dir.join("verity"))?;
    for (path, asset, digest) in &pins {
        if fs::metadata(path).map_err(at(path))?.len() != asset.bytes {
            return Err(invalid(format!("size mismatch: {}", asset.name)));
        }
        let file = fs::File::open(path).map_err(at(path))?;
        let tree = bundle::tree_path(data_dir, asset);
        let partial = partial_path(&tree);
        let label = format!("building the verity tree of {}", asset.name);
        let done = AtomicU64::new(0);
        progress(&label, 0, asset.bytes);
        let count = |bytes: u64| progress(&label, done.fetch_add(bytes, Ordering::Relaxed) + bytes, asset.bytes);
        let built = straydog_engine::verity::build(&file, &partial, &count).map_err(at(path))?;
        let built: String = built.iter().map(|byte| format!("{byte:02x}")).collect();
        if built != *digest {
            fs::remove_file(&partial).map_err(at(&partial))?;
            return Err(invalid(format!("verity digest mismatch: {}", asset.name)));
        }
        fs::rename(&partial, &tree).map_err(at(&tree))?;
        names.push(asset.name.clone());
    }
    Ok(names)
}

/// [`build_trees`] for a provisioning command: each file authenticated by its new tree is
/// printed as verified.
fn print_trees(root: &Path, quantization: &str, data_dir: &Path, missing_only: bool, progress: Progress) -> Result<(), String> {
    for name in build_trees(root, quantization, data_dir, missing_only, progress).map_err(|error| error.to_string())? {
        println!("verified {name}");
    }
    Ok(())
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

/// Download Qwen3.8's vision encoder and projector into `root` (a model's directory or the one above it: it serves every
/// quantization), authenticated before it is published; one there already is authenticated. With `offline`, only that;
/// none there is left missing (the model then reads no images). Its path when there.
pub fn fetch_qwen_vision(root: &Path, offline: bool, progress: Progress) -> Result<Option<PathBuf>, String> {
    let source = bundle::qwen_vision_source()?;
    let target = root.join(&source.asset.name);
    if offline && !target.is_file() {
        return Ok(None);
    }
    let url = format!("https://huggingface.co/{}/resolve/{}/{}", source.repository, source.revision, source.path);
    assets::fetch_with(&http::agent(assets::TIMEOUT), root, &[(source.asset, url)], offline, true, progress).map_err(|error| error.to_string())?;
    Ok(Some(target))
}

/// Build DeepSeek-V4.1's vision encoder file in `root` from the authenticated official tensors,
/// when the quantization's model has one pinned and it is not there yet; with `offline`, leave it
/// missing (the model then reads no images). Its verity tree then authenticates it.
fn build_vision(root: &Path, quantization: &str, offline: bool, progress: Progress) -> io::Result<()> {
    let Some(source_pins) = bundle::vision_source(quantization).map_err(invalid)? else { return Ok(()) };
    let target = root.join(&source_pins.asset.name);
    if offline || target.is_file() {
        return Ok(());
    }
    let source = root.join(vision::SOURCE_FILE);
    if !source.is_file() {
        let stage = "downloading the official vision tensors";
        let total = source_pins.tensors.iter().map(|tensor| tensor.bytes).sum();
        let done = AtomicU64::new(0);
        progress(stage, 0, total);
        let count = |bytes: u64| progress(stage, done.fetch_add(bytes, Ordering::Relaxed) + bytes, total);
        let base = format!("https://huggingface.co/{}/resolve/{}/", source_pins.repository, source_pins.revision);
        let agent = http::agent(mtp::TIMEOUT);
        mtp::fetch_source(&agent, &base, &source_pins.tensors, &source, mtp::PIECE, &count)?;
    }
    vision::convert(&source, &target, &source_pins.revision)?;
    let label = format!("authenticating {}", source_pins.asset.name);
    if let Err(error) = assets::verify_file(&target, &source_pins.asset, &label, progress) {
        fs::remove_file(&target).map_err(at(&target))?;
        return Err(error);
    }
    fs::remove_file(&source).map_err(at(&source))
}

/// Where DSpark's drafter is downloaded from, with its pin.
fn dspark_entry(source: &bundle::DsparkSource) -> (bundle::Asset, String) {
    (source.asset.clone(), format!("https://huggingface.co/{}/resolve/{}/{}", source.repository, source.revision, source.path))
}

/// Download DSpark's drafter into `root/dspark` when the quantization's model has one pinned and it
/// is in neither of its directories ([`bundle::dspark_in`]) yet; with `offline`, leave it missing
/// (the model then drafts from its own text only). Authenticated by its SHA-256 before it is
/// published; its verity tree then authenticates it for good.
fn fetch_dspark(root: &Path, quantization: &str, offline: bool, progress: Progress) -> io::Result<()> {
    let Some(source) = bundle::dspark_source(quantization).map_err(invalid)? else { return Ok(()) };
    if offline || bundle::dspark_in(root, quantization).map_err(invalid)?.is_some() {
        return Ok(());
    }
    let agent = http::agent(assets::TIMEOUT);
    assets::fetch_with(&agent, &root.join(bundle::DSPARK_DIRECTORY), &[dspark_entry(&source)], false, false, progress)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::Scratch;

    fn quiet(_stage: &str, _done: u64, _total: u64) {}

    /// DSpark's drafter is fetched from Unsloth's repository at its pinned revision into the model's
    /// `dspark` directory, and only when the model has one pinned, it is not there already (in either
    /// directory) and the fetch is not offline.
    #[test]
    fn the_dspark_drafter_is_fetched_from_its_pinned_revision_and_only_when_missing() {
        let source = bundle::dspark_source("GSQ-RCO-IQ2_XS").unwrap().expect("a pinned DSpark");
        let (asset, url) = dspark_entry(&source);
        assert_eq!(asset, source.asset);
        assert_eq!(url, "https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/60ad3651641b5c97679c807e064ab1d7a3ca1455/\
                         dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf");
        let scratch = Scratch::new("dspark");
        let root = scratch.path().join("GSQ-RCO-IQ2_XS");
        fs::create_dir_all(&root).unwrap();
        // (no request is made in any of these)
        fetch_dspark(&root, "GSQ-RCO-IQ2_XS", true, &quiet).unwrap();
        fetch_dspark(&root, "UD-IQ2_XXS", false, &quiet).unwrap();
        assert!(!root.join("dspark").exists() && !scratch.path().join("dspark").exists());
        fs::create_dir_all(scratch.path().join("dspark")).unwrap();
        fs::write(scratch.path().join("dspark").join(&asset.name), b"").unwrap();
        fetch_dspark(&root, "GSQ-RCO-IQ2_XS", false, &quiet).unwrap();
        assert!(!root.join("dspark").exists());
    }
}
