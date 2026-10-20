//! The pinned files themselves (design phase R5): a real bundle authenticates, and the
//! drafter built from the official tensors equals its pin byte for byte.
//!
//! cargo test -p straydog-provision --features hip --test pinned -- --nocapture, with
//! - STRAYDOG_MODEL: an IQ3_S bundle directory, only read (about 30 s with the files cached);
//! - STRAYDOG_DRAFTER_BUILD: a directory to build the drafter in. It downloads the official
//!   tensors (5.2 GB) and leaves the drafter there; the tensors are deleted.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use straydog_engine::{bundle, digest};

/// Progress that prints each stage once, when it starts, and records them all.
struct Stages(Mutex<Vec<(String, u64, u64)>>);

impl Stages {
    fn new() -> Stages {
        Stages(Mutex::new(Vec::new()))
    }

    fn report(&self, stage: &str, done: u64, total: u64) {
        let mut stages = self.0.lock().unwrap();
        match stages.iter_mut().find(|(name, _, _)| name == stage) {
            Some(last) => *last = (stage.to_string(), last.1.max(done), total),
            None => {
                eprintln!("{stage} ({total})");
                stages.push((stage.to_string(), done, total));
            }
        }
    }

    fn names(&self) -> Vec<String> {
        self.0.lock().unwrap().iter().map(|(name, _, _)| name.clone()).collect()
    }

    /// Whether every stage ended with all of it done.
    fn completed(&self) -> bool {
        self.0.lock().unwrap().iter().all(|(_, done, total)| done == total)
    }
}

fn sha256(path: &Path) -> String {
    digest::file_sha256(&File::open(path).unwrap(), &|_| {}).unwrap()
}

#[test]
fn a_bundle_authenticates() {
    let Some(root) = std::env::var_os("STRAYDOG_MODEL").map(PathBuf::from) else {
        eprintln!("skipped: set STRAYDOG_MODEL to an IQ3_S bundle directory");
        return;
    };
    let stages = Stages::new();
    straydog_provision::verify(&root, "IQ3_S", None, &|stage, done, total| stages.report(stage, done, total)).unwrap();
    let pins = bundle::bundle("IQ3_S").unwrap();
    let total: u64 = pins.shards.iter().chain(&pins.resources).map(|asset| asset.bytes).sum();
    assert_eq!(*stages.0.lock().unwrap(), [("authenticating 5 present file(s)".to_string(), total, total)]);
    // The Q2_0 bundle's first shard is not there.
    let message = straydog_provision::verify(&root, "Q2_0", None, &|_, _, _| {}).unwrap_err();
    assert!(message.contains("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"), "{message}");
}

#[test]
fn the_drafter_equals_its_pin() {
    let Some(root) = std::env::var_os("STRAYDOG_DRAFTER_BUILD").map(PathBuf::from) else {
        eprintln!("skipped: set STRAYDOG_DRAFTER_BUILD to a directory to build the drafter in (downloads 5.2 GB)");
        return;
    };
    let pin = bundle::drafter_source().unwrap().asset;
    let stages = Stages::new();
    let built = straydog_provision::fetch_drafter(&root, true, &|stage, done, total| stages.report(stage, done, total))
        .unwrap();
    let digest = sha256(&built);
    eprintln!("built {} bytes, SHA-256 {digest} (pinned {})", fs::metadata(&built).unwrap().len(), pin.sha256);
    assert_eq!(digest, pin.sha256);
    assert!(stages.completed());
    if !stages.names().iter().any(|name| name == "building the drafter (Q2_0 experts)") {
        eprintln!("the drafter was already there: built nothing");
        return;
    }
    // Again from the kept tensors, without downloading them; the tensors then go.
    fs::remove_file(&built).unwrap();
    let stages = Stages::new();
    let rebuilt = straydog_provision::fetch_drafter(&root, false, &|stage, done, total| stages.report(stage, done, total))
        .unwrap();
    assert_eq!(stages.names(), ["building the drafter (Q2_0 experts)", "authenticating Qwen3.8-Flash-Next-MTP-Q2_0.gguf"]);
    assert_eq!(sha256(&rebuilt), pin.sha256);
    assert!(!root.join("mtp-bf16.safetensors").exists());
    // A drafter already there is authenticated, not built.
    let stages = Stages::new();
    straydog_provision::fetch_drafter(&root, false, &|stage, done, total| stages.report(stage, done, total)).unwrap();
    assert_eq!(stages.names(), ["authenticating Qwen3.8-Flash-Next-MTP-Q2_0.gguf"]);
}
