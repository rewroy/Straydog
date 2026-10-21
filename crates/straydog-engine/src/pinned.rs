//! The shards of a bundle read on demand ([`crate::bundle::OnDemand`]), authenticated block by
//! block against their pinned fs-verity digests ([`crate::verity`]) and Merkle trees built into
//! Straydog's data directory. Nothing is used before its 4 KiB blocks verified: each header when
//! the shards open, a tensor the engine takes whole (dense weights, the token embeddings) when it
//! is first taken, a gathered table's rows as they are read (through a row cache,
//! [`crate::rows`], or the mapping: [`Tensors::rows`]), and routed experts by the expert store
//! as it reads them ([`Tensors::verifier`]). The files are mapped read-only and
//! checked unchanged (size and modification time) before every generation.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use memmap2::Mmap;
use straydog_gguf::{ModelHeader, TensorInfo};

use crate::arch::{Tensor, Tensors};
use crate::bundle::{self, OnDemand};
use crate::rows::{ReadPool, RowTable};
use crate::verity::{Verifier, BLOCK};

/// A shard's header lies in its first bytes: verified and parsed from at most this many.
const HEADER_BYTES: u64 = 16 << 20;
/// Checked level-0 tree blocks kept per shard.
const TREE_CACHE_BYTES: usize = 32 << 20;
/// A whole tensor is verified by this many threads when it is large.
const THREADS: usize = 8;

type Signature = (u64, Option<SystemTime>);

fn signature(file: &File) -> Result<Signature, String> {
    let metadata = file.metadata().map_err(|exc| exc.to_string())?;
    Ok((metadata.len(), metadata.modified().ok()))
}

/// Whether the engine gathers rows of a table rather than taking it whole.
fn gathered(name: &str) -> bool {
    name == "per_layer_token_embd.weight"
}

/// Whether the engine takes a tensor whole (checked when taken) rather than in pieces read later:
/// routed experts (the expert store's direct reads) and gathered tables (rows).
fn taken_whole(name: &str) -> bool {
    !name.ends_with("_exps.weight") && !gathered(name)
}

pub struct Pinned {
    /// The first shard's header: the model's metadata.
    pub header: ModelHeader,
    pub quantization: String,
    files: Vec<File>,
    maps: Vec<Mmap>,
    signatures: Vec<Signature>,
    verifiers: Vec<Arc<Verifier>>,
    tensors: HashMap<String, (usize, TensorInfo)>,
    /// Gathered tables read through a row cache ([`Pinned::cache_tables`]).
    rows: HashMap<String, RowTable>,
    /// Tensors verified whole.
    checked: Mutex<HashSet<String>>,
    /// Bytes of whole tensors verified so far.
    verified: AtomicU64,
}

impl Pinned {
    /// Open the pinned shards in `root` with their trees from `data_dir`: sizes checked, each tree's
    /// upper levels checked against the pinned digest, and every header verified, then parsed.
    pub fn open(root: &Path, data_dir: &Path, pins: &OnDemand) -> Result<Pinned, String> {
        let (mut files, mut maps, mut signatures, mut verifiers) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut tensors, mut first) = (HashMap::new(), None);
        let count = pins.shards.len();
        for (index, (asset, digest)) in pins.shards.iter().zip(&pins.verity).enumerate() {
            let file = File::open(root.join(&asset.name)).map_err(|exc| format!("{}: {exc}", asset.name))?;
            let before = signature(&file)?;
            if before.0 != asset.bytes {
                return Err(format!("Straydog asset size mismatch: {}", asset.name));
            }
            let tree = bundle::tree_path(data_dir, asset);
            if !tree.is_file() {
                return Err(format!(
                    "no verity tree for {} in {}: build the trees with `straydog verify -m {} --quant {} --data-dir {}`",
                    asset.name, data_dir.display(), root.display(), pins.quantization, data_dir.display()));
            }
            let verifier = Verifier::open(&tree, asset.bytes, &crate::verity::parse_digest(digest)?, TREE_CACHE_BYTES)
                .map_err(|exc| format!("Straydog verity tree of {}: {exc}", asset.name))?;
            // SAFETY: a read-only mapping; every byte used is verified first and the file is
            // checked unchanged before each generation.
            let map = unsafe { Mmap::map(&file) }.map_err(|exc| exc.to_string())?;
            let length = asset.bytes.min(HEADER_BYTES) as usize;
            let mut head = vec![0u8; length.next_multiple_of(BLOCK)];
            head[..length].copy_from_slice(&map[..length]);
            verifier.verify(0, &head).map_err(|exc| format!("Straydog asset {}: {exc}", asset.name))?;
            let header = straydog_gguf::inspect_header(&head[..length], asset.bytes)
                .map_err(|exc| format!("Straydog GGUF preflight of {}: {exc}", asset.name))?;
            if header.shard_index != index as u64 || header.shard_count != count as u64 {
                return Err(format!("Straydog shard identity mismatch: {}", asset.name));
            }
            if index == 0 && header.architecture != pins.architecture {
                return Err(format!("Straydog architecture mismatch: {} is not {}", header.architecture, pins.architecture));
            }
            for tensor in &header.tensors {
                if tensor.offset.checked_add(tensor.bytes).is_none_or(|end| end > asset.bytes) {
                    return Err(format!("Straydog tensor {} lies beyond its file", tensor.name));
                }
                if tensors.insert(tensor.name.clone(), (index, tensor.clone())).is_some() {
                    return Err(format!("duplicate tensor across Straydog shards: {}", tensor.name));
                }
            }
            if signature(&file)? != before {
                return Err("Straydog asset changed during authentication".into());
            }
            if index == 0 {
                first = Some(header);
            }
            files.push(file);
            maps.push(map);
            signatures.push(before);
            verifiers.push(Arc::new(verifier));
        }
        Ok(Pinned {
            header: first.ok_or("an on-demand bundle without shards")?,
            quantization: pins.quantization.clone(),
            files, maps, signatures, verifiers, tensors, rows: HashMap::new(),
            checked: Mutex::new(HashSet::new()),
            verified: AtomicU64::new(0),
        })
    }

    /// Read the gathered tables' rows through row caches sharing `budget` bytes instead of the
    /// mapping (0 keeps the mapping), each read's blocks verified between the read and the use.
    pub fn cache_tables(&mut self, budget: u64) -> Result<(), String> {
        let names: Vec<String> = self.tensors.keys().filter(|name| gathered(name)).cloned().collect();
        if budget == 0 || names.is_empty() {
            return Ok(());
        }
        let pool = Arc::new(ReadPool::new(crate::rows::THREADS));
        for name in &names {
            let (file, info) = &self.tensors[name];
            let rows = table_rows(name, info)?;
            let verifier = self.verifiers[*file].clone();
            let check: crate::rows::Verifier = Arc::new(move |offset: u64, data: &[u8]| {
                verifier.verify(offset, data).map_err(|error| format!("Straydog model bytes do not verify: {error}"))
            });
            let table = RowTable::new(&self.files[*file], info.offset, (info.bytes / rows) as usize, rows,
                                      budget / names.len() as u64, pool.clone(), Some(check))
                .map_err(|exc| format!("{name}: {exc}"))?;
            self.rows.insert(name.clone(), table);
        }
        Ok(())
    }

    /// Bytes of the tensors taken whole and verified so far (loading progress).
    pub fn verified_bytes(&self) -> u64 {
        self.verified.load(Ordering::Relaxed)
    }

    /// Bytes of the tensors the engine takes whole (as `filter` selects), for progress.
    pub fn whole_bytes(&self, filter: impl Fn(&str) -> bool) -> u64 {
        self.tensors.iter().filter(|(name, _)| taken_whole(name) && filter(name)).map(|(_, (_, info))| info.bytes).sum()
    }

    /// Tensor names and sizes, for planning.
    pub fn tensor_bytes(&self) -> impl Iterator<Item = (&str, u64)> {
        self.tensors.iter().map(|(name, (_, info))| (name.as_str(), info.bytes))
    }

    /// Refuse a new generation if a file has changed since it was opened.
    pub fn check_unchanged(&self) -> Result<(), String> {
        for (file, expected) in self.files.iter().zip(&self.signatures) {
            if signature(file)? != *expected {
                return Err("Straydog model changed after authentication".into());
            }
        }
        Ok(())
    }

    /// Verify bytes `offset .. offset + length` of shard `file`, as the whole blocks around them.
    fn check(&self, file: usize, offset: u64, length: u64) -> Result<(), String> {
        crate::verity::verify_mapped(&self.maps[file], &self.verifiers[file], offset, length, THREADS)
            .map_err(|error| format!("Straydog model bytes do not verify: {error}"))
    }

}

/// Rows of a table tensor (its second dimension).
fn table_rows(name: &str, info: &TensorInfo) -> Result<u64, String> {
    info.dimensions.get(1).copied().filter(|&rows| rows > 0 && info.bytes.is_multiple_of(rows))
        .ok_or_else(|| format!("{name} is not a table of rows"))
}

impl Tensors for Pinned {
    fn tensor(&self, name: &str) -> Result<Tensor<'_>, String> {
        let (file, info) = self.tensors.get(name).ok_or_else(|| format!("tensor {name} is missing"))?;
        if taken_whole(name) && !self.checked.lock().unwrap().contains(name) {
            self.check(*file, info.offset, info.bytes)?;
            self.checked.lock().unwrap().insert(name.to_string());
            self.verified.fetch_add(info.bytes, Ordering::Relaxed);
        }
        let start = info.offset as usize;
        Ok(Tensor {
            kind: info.quant_type,
            dims: info.dimensions.iter().map(|&d| d as usize).collect(),
            data: &self.maps[*file][start..start + info.bytes as usize],
        })
    }

    fn prefetch(&self, ranges: &[&[u8]]) {
        crate::storage::read_ahead(ranges.iter().copied());
    }

    fn files(&self) -> Result<Vec<File>, String> {
        self.files.iter().map(|file| file.try_clone().map_err(|exc| exc.to_string())).collect()
    }

    fn location(&self, name: &str) -> Option<(usize, u64)> {
        self.tensors.get(name).map(|(file, info)| (*file, info.offset))
    }

    fn verifier(&self) -> Option<crate::experts::Verifier> {
        let verifiers = self.verifiers.clone();
        Some(Arc::new(move |file: usize, offset: u64, data: &[u8]| {
            verifiers.get(file).ok_or_else(|| "a read outside the pinned files".to_string())?.verify(offset, data)
        }))
    }

    /// From the table's row cache, or else from the mapping, each row's blocks checked.
    fn rows(&self, name: &str, rows: &[u32]) -> Result<Vec<u8>, String> {
        if let Some(table) = self.rows.get(name) {
            return table.read(rows);
        }
        let (file, info) = self.tensors.get(name).ok_or_else(|| format!("tensor {name} is missing"))?;
        let (count, width) = (table_rows(name, info)?, info.bytes / table_rows(name, info)?);
        let mut out = Vec::with_capacity(rows.len() * width as usize);
        for &row in rows {
            if row as u64 >= count {
                return Err(format!("{name} has no row {row}"));
            }
            let at = info.offset + row as u64 * width;
            let data = crate::verity::read_verified(&self.maps[*file], &self.verifiers[*file], at, width)
                .map_err(|error| format!("Straydog model bytes do not verify: {error}"))?;
            out.extend_from_slice(&data);
        }
        Ok(out)
    }

    fn prefetch_rows(&self, name: &str, rows: &[u32]) -> Result<(), String> {
        if let Some(table) = self.rows.get(name) {
            return table.prefetch(rows);
        }
        let (_, info) = self.tensors.get(name).ok_or_else(|| format!("tensor {name} is missing"))?;
        let width = (info.bytes / table_rows(name, info)?) as usize;
        let tensor = self.tensor(name)?;
        self.prefetch(&rows.iter().filter_map(|&row| tensor.data.get(row as usize * width..(row as usize + 1) * width)).collect::<Vec<_>>());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::Asset;

    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    /// A GGUF shard: architecture and split metadata, F32 tensors of these dimensions laid out in
    /// order, their values `seed + index`.
    fn shard(index: u64, count: u64, tensors: &[(&str, &[u64])], seed: f32) -> Vec<u8> {
        let mut out = Vec::from(*b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // As the model's: only the first shard names the architecture.
        out.extend_from_slice(&(if index == 0 { 3u64 } else { 2 }).to_le_bytes());
        if index == 0 {
            string(&mut out, "general.architecture");
            out.extend_from_slice(&8u32.to_le_bytes());
            string(&mut out, "deepseek41");
        }
        for (key, value) in [("split.no", index), ("split.count", count)] {
            string(&mut out, key);
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        let mut offset = 0u64;
        for (name, dims) in tensors {
            let floats: u64 = dims.iter().product();
            string(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|dim| out.extend_from_slice(&dim.to_le_bytes()));
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            offset += (floats * 4).next_multiple_of(32);
        }
        out.resize(out.len().next_multiple_of(32), 0);
        let mut value = seed;
        for (_, dims) in tensors {
            for _ in 0..dims.iter().product::<u64>() {
                out.extend_from_slice(&value.to_le_bytes());
                value += 1.0;
            }
            out.resize(out.len().next_multiple_of(32), 0);
        }
        out
    }

    struct Bundle {
        root: std::path::PathBuf,
        pins: OnDemand,
    }

    impl Drop for Bundle {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// Two shards with their trees in `root/data/verity` and their pins.
    fn bundle(name: &str) -> Bundle {
        let root = std::env::temp_dir().join(format!("straydog-pinned-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("data/verity")).unwrap();
        // The second shard's table (rows of 176 bytes) runs past the header's 16 MiB, so later
        // bytes are read only when used.
        let shards = [
            shard(0, 2, &[("token_embd.weight", &[3000])], 0.0),
            shard(1, 2, &[("per_layer_token_embd.weight", &[44, TABLE_ROWS]), ("blk.0.attn_norm.weight", &[5000]),
                          ("blk.0.ffn_gate_exps.weight", &[4096])], 0.5),
        ];
        let (mut assets, mut verity) = (Vec::new(), Vec::new());
        for (index, bytes) in shards.iter().enumerate() {
            let asset = Asset { name: format!("shard-{index}.gguf"), bytes: bytes.len() as u64, sha256: String::new() };
            std::fs::write(root.join(&asset.name), bytes).unwrap();
            let file = File::open(root.join(&asset.name)).unwrap();
            let digest = crate::verity::build(&file, &bundle::tree_path(&root.join("data"), &asset), &|_| {}).unwrap();
            verity.push(digest.iter().map(|byte| format!("{byte:02x}")).collect());
            assets.push(asset);
        }
        let pins = OnDemand { model: "test".into(), architecture: "deepseek41".into(), quantization: "TEST".into(), shards: assets, verity };
        Bundle { root, pins }
    }

    const TABLE_ROWS: u64 = 120_000;
    const ROW: u64 = 176;

    fn open(bundle: &Bundle) -> Result<Pinned, String> {
        Pinned::open(&bundle.root, &bundle.root.join("data"), &bundle.pins)
    }

    /// Overwrite one byte of a shard (after its tree was built).
    fn tamper(bundle: &Bundle, shard: usize, offset: u64) {
        let path = bundle.root.join(&bundle.pins.shards[shard].name);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[offset as usize] ^= 1;
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn intact_bytes_verify_where_they_are_used() {
        let bundle = bundle("intact");
        let files = open(&bundle).unwrap();
        assert_eq!(files.header.architecture, "deepseek41");
        let norm = files.tensor("blk.0.attn_norm.weight").unwrap();
        assert_eq!((norm.dims.as_slice(), &norm.data[..4]), (&[5000usize][..], &(0.5 + (44 * TABLE_ROWS) as f32).to_le_bytes()[..]));
        assert_eq!(files.verified_bytes(), 20000);
        // A table's rows verify as they are read; experts through the store's check of whole blocks.
        let table = files.tensor("per_layer_token_embd.weight").unwrap();
        let wanted = [3u32, 100_000, 0, 3];
        let expected: Vec<u8> = wanted.iter().flat_map(|&row| &table.data[(row as u64 * ROW) as usize..((row as u64 + 1) * ROW) as usize]).copied().collect();
        assert_eq!(files.rows("per_layer_token_embd.weight", &wanted).unwrap(), expected);
        assert!(files.rows("per_layer_token_embd.weight", &[TABLE_ROWS as u32]).is_err());
        let (file, offset) = files.location("blk.0.ffn_gate_exps.weight").unwrap();
        let start = offset / BLOCK as u64 * BLOCK as u64;
        let mut blocks = vec![0u8; 2 * BLOCK];
        crate::storage::read_at(&files.files[file], start, &mut blocks).unwrap();
        (files.verifier().unwrap())(file, start, &blocks).unwrap();
        files.check_unchanged().unwrap();
        let mut cached = open(&bundle).unwrap();
        cached.cache_tables(1 << 20).unwrap();
        cached.prefetch_rows("per_layer_token_embd.weight", &wanted).unwrap();
        assert_eq!(cached.rows("per_layer_token_embd.weight", &wanted).unwrap(), expected);
    }

    #[test]
    fn changed_bytes_are_refused_before_use() {
        let bundle = bundle("tampered");
        let ((file, offset), row) = {
            let files = open(&bundle).unwrap();
            let norm = &files.tensors["blk.0.attn_norm.weight"];
            ((norm.0, norm.1.offset), files.tensors["per_layer_token_embd.weight"].1.offset + 110_000 * ROW)
        };
        tamper(&bundle, file, offset + 7777);
        tamper(&bundle, 1, row);
        let files = open(&bundle).unwrap();
        assert!(files.tensor("blk.0.attn_norm.weight").err().unwrap().contains("do not verify"));
        // The tensor taken whole before the change is still fine; a changed row is not.
        assert!(files.tensor("token_embd.weight").is_ok());
        let mut cached = open(&bundle).unwrap();
        cached.cache_tables(1 << 20).unwrap();
        for files in [&files, &cached] {
            assert!(files.rows("per_layer_token_embd.weight", &[0, 7]).is_ok());
            assert!(files.rows("per_layer_token_embd.weight", &[0, 110_000]).unwrap_err().contains("do not verify"));
            // Refused again, not cached.
            assert!(files.rows("per_layer_token_embd.weight", &[110_000]).unwrap_err().contains("do not verify"));
        }
        // A changed header (the first block) fails the opening itself.
        tamper(&bundle, 0, 10);
        assert!(open(&bundle).err().unwrap().contains("does not verify"));
    }

    #[test]
    fn missing_trees_and_wrong_digests_stop_the_opening() {
        let mut bundle = bundle("pins");
        bundle.pins.verity[1] = "00".repeat(32);
        assert!(open(&bundle).err().unwrap().contains("does not verify"));
        std::fs::remove_file(bundle::tree_path(&bundle.root.join("data"), &bundle.pins.shards[0])).unwrap();
        assert!(open(&bundle).err().unwrap().contains("straydog verify"));
    }
}
