//! DeepSeek-V4.1's vision encoder file: the official checkpoint's vision tower, aligner, image
//! span embeddings and image routing biases (range-fetched, each tensor's SHA-256 pinned:
//! [`crate::mtp::fetch_source`]) in one GGUF beside the bundle. Its tensors keep their official
//! names and bytes (BF16; the routing biases F32), their dimensions listed innermost first as
//! GGUF lists them.

use std::borrow::Cow;
use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;
use straydog_engine::bundle::VISION_ARCHITECTURE;

use crate::at;
use crate::mtp::{self, Entry};

/// The official tensors, kept between the download and the conversion.
pub const SOURCE_FILE: &str = "vision-source.safetensors";

const F32: u32 = 0;
const BF16: u32 = 30;

/// Write the vision encoder's GGUF from the authenticated source tensors.
pub fn convert(source: &Path, target: &Path, revision: &str) -> io::Result<()> {
    let file = File::open(source).map_err(at(source))?;
    // SAFETY: mapped read-only; the source is this process's own download, which nothing else
    // writes while the file is built.
    let map = unsafe { Mmap::map(&file) }.map_err(at(source))?;
    let tensors = mtp::source_tensors(&map, true)?;
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let entries: Vec<Entry> = names.into_iter().map(|name| {
        let tensor = &tensors[name];
        Entry { name, dims: tensor.shape.iter().rev().copied().collect(), kind: if tensor.float { F32 } else { BF16 },
                data: Cow::Borrowed(tensor.data) }
    }).collect();
    mtp::write_gguf(target, &[("general.architecture", VISION_ARCHITECTURE), ("straydog.vision.source_revision", revision)], &entries)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use sha2::{Digest, Sha256};
    use straydog_engine::bundle;

    use super::*;
    use crate::http;

    /// Build the file into a model directory and print the pins its manifest takes (bytes, SHA-256,
    /// fs-verity digest, with its tree written where the server reads it):
    /// STRAYDOG_VISION_DIR=<model dir> cargo test --release -p straydog-provision build_the_vision_file -- --ignored --nocapture
    #[test]
    #[ignore]
    fn build_the_vision_file() {
        let directory = PathBuf::from(std::env::var("STRAYDOG_VISION_DIR").expect("STRAYDOG_VISION_DIR"));
        let pins = bundle::vision_source(bundle::Id::find(Some(bundle::DEEPSEEK), None, "IQ2_XS").unwrap()).unwrap().expect("a pinned vision encoder");
        let source = directory.join(SOURCE_FILE);
        if !source.is_file() {
            let base = format!("https://huggingface.co/{}/resolve/{}/", pins.repository, pins.revision);
            mtp::fetch_source(&http::agent(mtp::TIMEOUT), &base, &pins.tensors, &source, mtp::PIECE, &|_| {}).unwrap();
        }
        let target = directory.join(&pins.asset.name);
        convert(&source, &target, &pins.revision).unwrap();
        let bytes = std::fs::read(&target).unwrap();
        let tree = bundle::tree_path(&bundle::default_data_dir(&directory), &pins.asset);
        std::fs::create_dir_all(tree.parent().unwrap()).unwrap();
        let digest = straydog_engine::verity::build(&File::open(&target).unwrap(), &tree, &|_| {}).unwrap();
        let verity: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        println!("\"bytes\": {}, \"sha256\": \"{:x}\", \"verity\": \"{verity}\"", bytes.len(), Sha256::digest(&bytes));
    }
}
