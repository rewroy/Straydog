//! Authenticated GGUF weights.
//!
//! Opening reads and authenticates each file's first chunk and parses the header from it;
//! placement then runs on the header alone, and `crate::load` reads every chunk once,
//! verifying it before its bytes reach the GPUs or the host copies kept here (token
//! embeddings, vectors, the operands passes upload at first use). Only what the engine
//! reads through the mapping at run time, the PLE table and partial residency's expert
//! banks, is used from the page cache after its chunk was verified. With the shards' verity
//! trees ([`Weights::open_trees`]) the PLE table's rows are checked block by block as they are
//! read and the expert banks page-locked in host memory once locked; the banks left to the page
//! cache are not (docs/DESIGN.md §3).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use memmap2::Mmap;

use crate::bundle::{self, Asset};
use crate::digest::{self, CHUNK_BYTES};
use crate::formats;
use crate::rows::{ReadPool, RowTable};
use crate::verity;

pub const CONTEXT_LIMIT: usize = 262144;
const TENSOR_COUNT: usize = 1224;
const SNAPSHOT_BYTES: u64 = 16 << 20;
const RESOURCE_LIMIT: u64 = 32 << 20;
/// Rows gathered by several threads: uncached rows of the 28.8 GB PLE table are random reads.
const PARALLEL_ROWS: usize = 256;
const ROW_THREADS: usize = 16;
/// Decoded token embeddings kept: 4096 rows of 2560 floats (40 MiB), for batches up to eight.
const CACHED_BATCH: usize = 8;
const CACHED_ROWS: usize = 4096;
/// Checked level-0 tree blocks kept per file.
const TREE_CACHE_BYTES: usize = 32 << 20;
/// The drafter's tensors (the pinned MTP layer).
pub const DRAFTER_TENSORS: [&str; 33] = [
    "mtp.embed_norm.weight", "mtp.fc_embedding.weight", "mtp.hidden_norm.weight", "mtp.fc_hidden.weight",
    "mtp.hc_out_norm.weight", "mtp.hc_out_down.weight", "mtp.hc_out_up.weight", "blk.48.attn_q.weight",
    "blk.48.attn_k.weight", "blk.48.attn_v.weight", "blk.48.attn_output.weight", "blk.48.attn_q_norm.weight",
    "blk.48.attn_k_norm.weight", "blk.48.indexer.q_proj.weight", "blk.48.indexer.k_proj.weight",
    "blk.48.indexer.q_norm.weight", "blk.48.indexer.k_norm.weight", "blk.48.ffn_gate_inp.weight",
    "blk.48.ffn_gate_inp_shexp.weight", "blk.48.ffn_gate_shexp.weight", "blk.48.ffn_up_shexp.weight",
    "blk.48.ffn_down_shexp.weight", "blk.48.ffn_gate_exps.weight", "blk.48.ffn_up_exps.weight",
    "blk.48.ffn_down_exps.weight", "blk.48.hc_attn_norm.weight", "blk.48.hc_attn_down.weight",
    "blk.48.hc_attn_up.weight", "blk.48.hc_attn_inject.weight", "blk.48.hc_ffn_norm.weight",
    "blk.48.hc_ffn_down.weight", "blk.48.hc_ffn_up.weight", "blk.48.hc_ffn_inject.weight",
];

/// Authentication progress: (stage, bytes done, bytes in the stage).
pub type Progress<'a> = &'a (dyn Fn(&str, u64, u64) + Sync);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub file: usize,
    /// GGUF order: the input dimension first.
    pub dims: Vec<u64>,
    pub kind: u32,
    pub offset: u64,
    pub bytes: u64,
}

impl Tensor {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// GPU bytes when pinned: vectors and convolution weights are uploaded as F32.
    fn device_bytes(&self, name: &str) -> u64 {
        if self.dims.len() == 1 || name.ends_with("ssm_conv1d.weight") {
            self.bytes.max(self.elements() * 4)
        } else if (hc_quantized(name) || drafter_quantized(name) && matches!(self.kind, 0 | 1 | 30)) && self.elements().is_multiple_of(32) {
            // (a HyperConnection's down and up projections and the drafter's dense ones go up as Q8_0: see Model::q8_projection)
            self.elements() / 32 * 34
        } else {
            self.bytes
        }
    }
}

/// A decoded vector (norm, bias, convolution): row-major with the GGUF dimensions reversed.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub values: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Vector {
    pub fn bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }
}

/// Decoded token-embedding rows and their insertion order (the oldest is evicted first).
type EmbeddingCache = (HashMap<i64, Arc<Vec<f32>>>, VecDeque<i64>);

pub struct Weights {
    pub root: PathBuf,
    pub bundle: bundle::Id,
    /// The engram table read in the last shard's place, when it is one of the model's engram files ([`bundle::table_in`]).
    pub table: Option<bundle::Table>,
    pub quantization: String,
    /// Measured outside isolation by the runtime (cgroup limits included).
    pub host_memory_bytes: u64,
    pub integer_arrays: BTreeMap<String, Vec<i64>>,
    pub resources: HashMap<String, Vec<u8>>,
    /// Whether the pinned MTP drafter is authenticated and in use.
    pub drafter: bool,
    drafter_names: HashSet<String>,
    /// Per file: its asset name, handle, mapping, pinned chunk digests and signature.
    assets: Vec<String>,
    files: Vec<File>,
    maps: Vec<Mmap>,
    chunk_digests: Vec<Vec<String>>,
    signatures: Vec<Signature>,
    tensors: HashMap<String, Tensor>,
    order: Vec<String>,
    /// Host copies of tensors made from their verified chunks (see `crate::load`).
    kept: HashMap<String, Vec<u8>>,
    /// Whether every chunk is verified: before that only the header may be used.
    authenticated: bool,
    /// Tables read through a row cache instead of the mapping (see `crate::rows`).
    row_tables: HashMap<String, RowTable>,
    /// Per file: its verity tree, when opened ([`Weights::open_trees`]).
    verifiers: Vec<Option<Arc<verity::Verifier>>>,
    /// Per file: its verity tree is younger than it. A tree is built from the whole file and checked against the pin (at download
    /// or at the first start), so a file not changed since needs no reading through at a start.
    tree_current: Vec<bool>,
    vectors: Mutex<HashMap<String, Arc<Vector>>>,
    /// The F32 routers whose every value is a BF16 value, which go up as BF16 ([`Weights::narrowed`]).
    narrow: std::sync::OnceLock<HashSet<String>>,
    embeddings: Mutex<EmbeddingCache>,
}

/// A file's size and modification time, compared before each generation.
type Signature = (u64, Option<SystemTime>);

fn signature(file: &File) -> std::io::Result<Signature> {
    let metadata = file.metadata()?;
    Ok((metadata.len(), metadata.modified().ok()))
}

fn error(message: impl Into<String>) -> String {
    message.into()
}

/// A model or drafter directory as an absolute path, without asking the file system anything.
///
/// The server has already resolved it, as the user (links followed, and on Windows the `\\?\` form), before it starts
/// the worker, so resolving it again changes nothing. It also asks the contained worker for more than it needs:
/// `canonicalize` opens the directory itself and, on Windows, asks for its final name (`GetFinalPathNameByHandleW`,
/// the drive-letter lookup), and in the worker's AppContainer that open can be refused (os error 5) where reading the
/// pinned files inside the directory is not. An absolute path, a `\\?\` one too, comes back as it was given. A relative
/// one (an example or a test run by hand) is made absolute against the current directory.
fn absolute_directory(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::path::absolute(path).map_err(|exc| format!("{}: {exc}", path.display()))
}

/// An opened file, its signature then, and the chunk digests reading will verify.
type Opened = ((File, Signature), Vec<String>);

/// Open the pinned files, checking their sizes. A file without pinned chunk digests is hashed
/// whole now, which also yields them.
fn open_pinned(root: &Path, assets: &[Asset], stage: &str, progress: Progress) -> Result<Vec<Opened>, String> {
    let mut seen = HashSet::new();
    let mut opened = Vec::new();
    for asset in assets {
        if !seen.insert(&asset.name) {
            return Err(error("invalid Straydog asset name"));
        }
        let file = File::open(root.join(&asset.name)).map_err(|exc| format!("{}: {exc}", asset.name))?;
        if file.metadata().map_err(|exc| exc.to_string())?.len() != asset.bytes {
            return Err(format!("Straydog asset size mismatch: {}", asset.name));
        }
        let pieces = match bundle::chunk_digests(&asset.sha256)? {
            Some(pieces) => pieces,
            None => {
                let total = asset.bytes;
                let done = AtomicU64::new(0);
                let count = |bytes: u64| progress(stage, done.fetch_add(bytes, Ordering::Relaxed) + bytes, total);
                let (whole, pieces) = digest::file_digests(&file, &count).map_err(|exc| exc.to_string())?;
                if whole != asset.sha256 {
                    return Err(format!("Straydog asset digest mismatch: {}", asset.name));
                }
                pieces
            }
        };
        if pieces.len() as u64 != asset.bytes.div_ceil(CHUNK_BYTES) {
            return Err(format!("Straydog asset digest mismatch: {}", asset.name));
        }
        let before = signature(&file).map_err(|exc| exc.to_string())?;
        opened.push(((file, before), pieces));
    }
    Ok(opened)
}

/// A resource read whole and checked against its pin.
fn read_resource(root: &Path, asset: &Asset) -> Result<Vec<u8>, String> {
    if asset.bytes > RESOURCE_LIMIT {
        return Err(error("Straydog resource exceeds size limit"));
    }
    let mut file = File::open(root.join(&asset.name)).map_err(|exc| format!("{}: {exc}", asset.name))?;
    let mut data = Vec::new();
    (&mut file).take(RESOURCE_LIMIT + 1).read_to_end(&mut data).map_err(|exc| exc.to_string())?;
    let digest = format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&data));
    if data.len() as u64 != asset.bytes || digest != asset.sha256 {
        return Err(format!("Straydog asset digest mismatch: {}", asset.name));
    }
    Ok(data)
}

/// The header of a GGUF file, parsed from its first chunk once that matched its digest.
fn header(file: &File, name: &str, first: &str) -> Result<straydog_gguf::ModelHeader, String> {
    let size = file.metadata().map_err(|exc| exc.to_string())?.len();
    let mut chunk = vec![0u8; size.min(CHUNK_BYTES) as usize];
    digest::will_need(file, &[(0, chunk.len() as u64)]);
    crate::storage::read_at(file, 0, &mut chunk).map_err(|exc| format!("{name}: {exc}"))?;
    if format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&chunk)) != first {
        return Err(format!("Straydog asset digest mismatch: {name}"));
    }
    let snapshot = &chunk[..chunk.len().min(SNAPSHOT_BYTES as usize)];
    straydog_gguf::inspect_header(snapshot, size).map_err(|exc| format!("Straydog GGUF preflight: {exc}"))
}

/// A file no younger than its verity tree (and its size unchanged is the tree's own check, at `Verifier::open`). This is what lets a start
/// read the file unhashed, so on Unix it reads the change time, which no tool can set back as it can a modification time. It catches accidents
/// and careless edits, not someone who can write the data directory too: they can make a tree look newer.
fn tree_is_current(file: &File, tree: &Path) -> bool {
    let tree = std::fs::metadata(tree).and_then(|metadata| metadata.modified()).ok();
    #[cfg(unix)]
    let file = file.metadata().ok().map(|metadata| {
        use std::os::unix::fs::MetadataExt;
        std::time::UNIX_EPOCH + std::time::Duration::new(metadata.ctime().max(0) as u64, metadata.ctime_nsec().max(0) as u32)
    });
    #[cfg(not(unix))]
    let file = file.metadata().and_then(|metadata| metadata.modified()).ok();
    matches!((file, tree), (Some(file), Some(tree)) if file <= tree)
}

impl Weights {
    /// Open the pinned bundle: resources and headers authenticated, tensors not yet read
    /// (placement may run; [`crate::load::load`] or [`Weights::authenticate`] reads them).
    pub fn open(root: &Path, id: bundle::Id, drafter: Option<&Path>, host_memory_bytes: u64,
                progress: Progress) -> Result<Weights, String> {
        let pins = bundle::bundle(id)?;
        // (GSQ-RCO's two shards, Unsloth's three: its first holds the metadata alone)
        let shards = pins.shards.len();
        if shards == 0 {
            return Err(error("Straydog bundle without shards"));
        }
        let names: HashSet<&str> = pins.resources.iter().map(|asset| asset.name.as_str()).collect();
        if names != HashSet::from(["tokenizer.json", "chat_template.jinja", "config.json"]) {
            return Err(error("Straydog requires authenticated tokenizer, template, and config"));
        }
        let root = absolute_directory(root)?;
        // (Unsloth's Qwen3.8: a more exact engram table there is read in place of the last shard, which holds the bundled one)
        let table = bundle::table_in(&root, id)?.filter(|(table, _)| !table.bundled);
        let mut weights = Weights {
            root: root.clone(),
            bundle: id,
            table: table.as_ref().map(|(table, _)| table.clone()),
            quantization: id.quant.to_string(),
            host_memory_bytes: if host_memory_bytes > 0 { host_memory_bytes } else { crate::host::host_memory_bytes() },
            integer_arrays: BTreeMap::new(),
            resources: HashMap::new(),
            drafter: false,
            drafter_names: HashSet::new(),
            assets: Vec::new(),
            files: Vec::new(),
            maps: Vec::new(),
            chunk_digests: Vec::new(),
            signatures: Vec::new(),
            tensors: HashMap::new(),
            order: Vec::new(),
            kept: HashMap::new(),
            authenticated: false,
            row_tables: HashMap::new(),
            verifiers: Vec::new(),
            tree_current: Vec::new(),
            vectors: Mutex::new(HashMap::new()),
            narrow: std::sync::OnceLock::new(),
            embeddings: Mutex::new((HashMap::new(), VecDeque::new())),
        };
        for asset in &pins.resources {
            weights.resources.insert(asset.name.clone(), read_resource(&root, asset)?);
        }
        let mut files: Vec<Asset> = pins.shards.clone();
        let mut opened = match &table {
            Some((table, path)) => {
                files.pop();
                let mut opened = open_pinned(&root, &files, "open", progress)?;
                let directory = path.parent().ok_or("Straydog engram table without a directory")?;
                opened.extend(open_pinned(directory, std::slice::from_ref(&table.asset), "open", progress)?);
                files.push(table.asset.clone());
                opened
            }
            None => open_pinned(&root, &pins.shards, "open", progress)?,
        };
        if let Some(directory) = drafter {
            let directory = absolute_directory(directory)?;
            opened.extend(open_pinned(&directory, std::slice::from_ref(&pins.drafter), "drafter", progress)?);
        }
        // Each header's chunk is read and hashed on a thread of its own.
        let assets: Vec<&Asset> = files.iter().chain([&pins.drafter]).collect();
        let headers: Vec<Result<straydog_gguf::ModelHeader, String>> = std::thread::scope(|scope| {
            let threads: Vec<_> = opened.iter().zip(&assets)
                .map(|(((file, _), pieces), asset)| scope.spawn(move || header(file, &asset.name, &pieces[0])))
                .collect();
            threads.into_iter().map(|thread| thread.join().unwrap()).collect()
        });
        for (index, (((file, pieces), header), asset)) in opened.into_iter().zip(headers).zip(assets).enumerate() {
            let header = header?;
            if index == shards {
                weights.add_drafter(asset, file, pieces, header)?;
                continue;
            }
            // (an engram file is a file of one: its table alone, where the last shard's was)
            let engram = index == shards - 1 && table.is_some();
            let identity = if engram { (1, 0) } else { (shards as u64, index as u64) };
            if (header.shard_count, header.shard_index) != identity {
                return Err(error("Straydog shard identity mismatch"));
            }
            if engram && !(header.tensors.len() == 1 && header.tensors[0].name == "per_layer_token_embd.weight") {
                return Err(error("Straydog engram table holds other tensors"));
            }
            let architecture_ok = header.architecture == "qwen4exp" || (index > 0 && header.architecture.is_empty());
            if !architecture_ok {
                return Err(error("Straydog architecture mismatch"));
            }
            if index == 0 && header.declared_context != Some(CONTEXT_LIMIT as u64) {
                return Err(error("Straydog model context declaration mismatch"));
            }
            weights.add_tensors(index, &header.tensors)?;
            weights.integer_arrays.extend(header.integer_arrays.clone());
            weights.map(&asset.name, file, pieces)?;
            if index == shards - 1 && weights.tensors.len() != TENSOR_COUNT {
                return Err(error("Straydog model tensor count mismatch"));
            }
        }
        Ok(weights)
    }

    /// Open the pinned shards' verity trees in `data_dir` (`bundle::tree_path`, checked against
    /// their pinned digests): from then on the PLE table's rows are checked block by block as they
    /// are read, and partial residency's page-locked expert banks once locked.
    pub fn open_trees(&mut self, data_dir: &Path) -> Result<(), String> {
        let mut verifiers = vec![None; self.files.len()];
        let mut current = vec![false; self.files.len()];
        let table = self.table.as_ref().map(|table| -> Result<_, String> {
            let digest = bundle::verity_digest(&table.asset.sha256)?.ok_or_else(|| format!("{} has no pinned verity digest", table.asset.name))?;
            Ok((table.asset.clone(), digest))
        }).transpose()?;
        for (asset, digest) in bundle::verity_pins(self.bundle)?.into_iter().chain(table) {
            let Some(index) = self.assets.iter().position(|name| *name == asset.name) else { continue };
            let tree = bundle::tree_path(data_dir, &asset);
            current[index] = tree_is_current(&self.files[index], &tree);
            let verifier = verity::Verifier::open(&tree, asset.bytes, &verity::parse_digest(&digest)?, TREE_CACHE_BYTES)
                .map_err(|error| format!("Straydog verity tree of {}: {error}", asset.name))?;
            verifiers[index] = Some(Arc::new(verifier));
        }
        self.verifiers = verifiers;
        self.tree_current = current;
        Ok(())
    }

    /// Whether a file was read through and checked against its pin, and not changed since.
    pub fn file_checked(&self, file: usize) -> bool {
        self.tree_current.get(file).copied().unwrap_or(false)
    }

    pub(crate) fn verifier_of(&self, file: usize) -> Option<&Arc<verity::Verifier>> {
        self.verifiers.get(file).and_then(Option::as_ref)
    }

    /// Whether a tensor's reads after loading are checked against its file's verity tree.
    pub fn verified(&self, name: &str) -> bool {
        self.tensor(name).is_ok_and(|tensor| self.verifier_of(tensor.file).is_some())
    }

    /// Check a tensor's bytes in the mapping, block by block on every core, against its file's
    /// tree: for bytes that cannot change under the check (page-locked). Ok(false) without a tree.
    pub fn verify_mapped(&self, name: &str) -> Result<bool, String> {
        let tensor = self.tensor(name)?;
        let Some(verifier) = self.verifier_of(tensor.file) else { return Ok(false) };
        let threads = std::thread::available_parallelism().map_or(8, |count| count.get());
        verity::verify_mapped(&self.maps[tensor.file], verifier, tensor.offset, tensor.bytes, threads)
            .map_err(|error| format!("Straydog model bytes do not verify ({name}): {error}"))?;
        Ok(true)
    }

    /// Read and verify every chunk, keeping host copies of the token embeddings (for callers
    /// that use the weights on the host only; a model loads them with its placement).
    pub fn authenticate(&mut self, progress: Progress) -> Result<(), String> {
        crate::load::authenticate(self, &["token_embd.weight".to_string()], progress)
    }

    fn add_tensors(&mut self, index: usize, tensors: &[straydog_gguf::TensorInfo]) -> Result<(), String> {
        for tensor in tensors {
            if self.tensors.contains_key(&tensor.name) {
                return Err(error("duplicate tensor across Straydog shards"));
            }
            self.order.push(tensor.name.clone());
            self.tensors.insert(tensor.name.clone(), Tensor {
                file: index,
                dims: tensor.dimensions.clone(),
                kind: tensor.quant_type,
                offset: tensor.offset,
                bytes: tensor.bytes,
            });
        }
        Ok(())
    }

    /// Map an opened file, refusing one that changed since it was opened.
    fn map(&mut self, name: &str, (file, before): (File, Signature), pieces: Vec<String>) -> Result<(), String> {
        // SAFETY: the mapping is read-only and the file is checked unchanged before every
        // generation; the files are pinned assets nothing is expected to write.
        let map = unsafe { Mmap::map(&file) }.map_err(|exc| exc.to_string())?;
        if signature(&file).map_err(|exc| exc.to_string())? != before {
            return Err(error("Straydog asset changed during authentication"));
        }
        self.assets.push(name.to_string());
        self.signatures.push(before);
        self.files.push(file);
        self.maps.push(map);
        self.chunk_digests.push(pieces);
        Ok(())
    }

    fn add_drafter(&mut self, pin: &Asset, file: (File, Signature), pieces: Vec<String>,
                   header: straydog_gguf::ModelHeader) -> Result<(), String> {
        let names: HashSet<String> = header.tensors.iter().map(|tensor| tensor.name.clone()).collect();
        let expected: HashSet<String> = DRAFTER_TENSORS.iter().map(|name| name.to_string()).collect();
        if header.shard_count != 1 || header.architecture != "qwen4exp" || names != expected {
            return Err(error("Straydog drafter layout mismatch"));
        }
        if names.iter().any(|name| self.tensors.contains_key(name)) {
            return Err(error("duplicate tensor across Straydog drafter and model"));
        }
        self.add_tensors(self.files.len(), &header.tensors)?;
        self.map(&pin.name, file, pieces)?;
        self.drafter_names = names;
        self.drafter = true;
        Ok(())
    }

    /// Stop using the drafter (it does not fit this placement); it stays mapped.
    pub fn drop_drafter(&mut self) {
        self.drafter = false;
    }

    /// Decoder layers to place: the 48 model layers, and the MTP layer when drafting.
    pub fn layers(&self) -> std::ops::Range<usize> {
        0..if self.drafter { 49 } else { 48 }
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors.get(name).ok_or_else(|| format!("Straydog tensor missing: {name}"))
    }

    /// The tensor's GGUF dimensions (input dimension first).
    pub fn shape(&self, name: &str) -> Result<&[u64], String> {
        Ok(&self.tensor(name)?.dims)
    }

    /// Tensor names in file order.
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// The bytes of a tensor slice: its host copy if kept, else the mapping. Before the weights
    /// are authenticated the mapped bytes may only be placed, never read (see `crate::load`).
    pub fn raw(&self, name: &str, start: u64, size: Option<u64>) -> Result<(&[u8], u32), String> {
        let tensor = self.tensor(name)?;
        let size = size.unwrap_or(tensor.bytes);
        if start.checked_add(size).is_none_or(|end| end > tensor.bytes) {
            return Err(error("Straydog tensor slice is out of range"));
        }
        if let Some(kept) = self.kept.get(name) {
            return Ok((&kept[start as usize..(start + size) as usize], tensor.kind));
        }
        let begin = (tensor.offset + start) as usize;
        let map = &self.maps[tensor.file];
        let end = begin.checked_add(size as usize).filter(|end| *end <= map.len()).ok_or("tensor beyond its file")?;
        Ok((&map[begin..end], tensor.kind))
    }

    fn require_authenticated(&self) -> Result<(), String> {
        if self.authenticated { Ok(()) } else { Err(error("Straydog weights are not authenticated yet")) }
    }

    // ---- reading (crate::load) ----

    pub(crate) fn file_count(&self) -> usize {
        self.files.len()
    }

    /// The files to read, in order: the shards, and the drafter's while it is in use.
    pub(crate) fn files_to_read(&self) -> Vec<(&str, &File, &[u8], &[String])> {
        let count = if self.drafter || self.drafter_names.is_empty() { self.files.len() } else { self.files.len() - 1 };
        (0..count)
            .map(|index| (self.assets[index].as_str(), &self.files[index], &self.maps[index][..],
                          self.chunk_digests[index].as_slice()))
            .collect()
    }

    /// Every mapping with its file.
    #[cfg(windows)]
    pub(crate) fn mapped_files(&self) -> impl Iterator<Item = (&[u8], &File)> {
        self.maps.iter().map(|map| &map[..]).zip(&self.files)
    }

    /// The file and offset of mapped bytes (an operand's address), if they are mapped.
    pub(crate) fn locate(&self, address: *const u8, bytes: usize) -> Option<(usize, u64)> {
        let address = address as usize;
        self.maps.iter().enumerate().find_map(|(index, map)| {
            let start = map.as_ptr() as usize;
            (address >= start && address + bytes <= start + map.len()).then(|| (index, (address - start) as u64))
        })
    }

    /// The tensor holding byte `offset` of a file.
    pub(crate) fn tensor_at(&self, file: usize, offset: u64) -> Result<&str, String> {
        self.tensors
            .iter()
            .find(|(_, tensor)| tensor.file == file && (tensor.offset..tensor.offset + tensor.bytes).contains(&offset))
            .map(|(name, _)| name.as_str())
            .ok_or_else(|| error("Straydog upload is not a tensor"))
    }

    /// Every tensor's byte range (offset, length) in a file.
    pub(crate) fn tensor_ranges(&self, file: usize) -> Vec<(u64, u64)> {
        self.tensors.values().filter(|tensor| tensor.file == file).map(|tensor| (tensor.offset, tensor.bytes)).collect()
    }

    /// Keep a host copy of a tensor's verified bytes.
    pub(crate) fn keep(&mut self, name: String, bytes: Vec<u8>) {
        self.kept.insert(name, bytes);
    }

    pub(crate) fn set_authenticated(&mut self) {
        self.authenticated = true;
    }

    /// The file a tensor lives in (for read-ahead hints).
    pub fn file_of(&self, name: &str) -> Result<&File, String> {
        Ok(&self.files[self.tensor(name)?.file])
    }

    /// A vector's shape (rows, columns): a norm, bias or convolution tensor, small enough to
    /// be uploaded decoded to F32.
    pub fn vector_shape(&self, name: &str) -> Result<(usize, usize), String> {
        let tensor = self.tensor(name)?;
        if tensor.elements() > 65536 {
            return Err(error("Straydog vector request is too large"));
        }
        match tensor.dims.as_slice() {
            [cols] => Ok((1, *cols as usize)),
            [inner, outer] => Ok((*outer as usize, *inner as usize)),
            _ => Err(format!("{name} is not a vector")),
        }
    }

    /// A small norm, bias or convolution tensor, decoded once.
    pub fn vector(&self, name: &str) -> Result<Arc<Vector>, String> {
        self.require_authenticated()?;
        if let Some(vector) = self.vectors.lock().unwrap().get(name) {
            return Ok(vector.clone());
        }
        let (rows, cols) = self.vector_shape(name)?;
        let (raw, kind) = self.raw(name, 0, None)?;
        let values = crate::gpu::decode_vector(raw, kind, rows * cols, name)?;
        let vector = Arc::new(Vector { values, rows, cols });
        self.vectors.lock().unwrap().insert(name.to_string(), vector.clone());
        Ok(vector)
    }

    /// Embedding rows, decoded as one batch; row `i` of the result is `indices[i]`.
    pub fn rows(&self, name: &str, indices: &[i64]) -> Result<Vec<f32>, String> {
        self.require_authenticated()?;
        if name == "token_embd.weight" && (1..=CACHED_BATCH).contains(&indices.len()) {
            return self.token_rows(indices);
        }
        self.decoded_rows(name, indices)
    }

    /// Read a table's rows through a cache of `budget` bytes (`crate::rows`) instead of the
    /// mapping: from the authenticated file, as aligned blocks, past the page cache, each read
    /// checked against the file's verity tree when it is open.
    pub fn cache_rows(&mut self, name: &str, budget: u64) -> Result<(), String> {
        let tensor = self.tensor(name)?;
        let [_, count] = tensor.dims[..] else { return Err(format!("{name} is not a table of rows")) };
        let pool = Arc::new(ReadPool::new(crate::rows::THREADS));
        let verifier = self.verifier_of(tensor.file).cloned().map(|verifier| -> crate::rows::Verifier {
            Arc::new(move |offset: u64, data: &[u8]| {
                verifier.verify(offset, data).map_err(|error| format!("Straydog model bytes do not verify: {error}"))
            })
        });
        let table = RowTable::new(&self.files[tensor.file], tensor.offset, (tensor.bytes / count) as usize, count, budget, pool,
                                  verifier)
            .map_err(|exc| format!("{name}: {exc}"))?;
        self.row_tables.insert(name.to_string(), table);
        Ok(())
    }

    /// Start reading these rows of a table in the background, so that a later `rows` waits
    /// only for those still on their way: through its row cache, or the mapping's read-ahead.
    pub fn prefetch_rows(&self, name: &str, indices: &[i64]) -> Result<(), String> {
        let tensor = self.tensor(name)?;
        let [_, count] = tensor.dims[..] else { return Err(format!("{name} is not a table of rows")) };
        if indices.iter().any(|index| *index < 0 || *index as u64 >= count) {
            return Err(error("invalid Straydog embedding row batch"));
        }
        if let Some(rows) = self.row_tables.get(name) {
            return rows.prefetch(&indices.iter().map(|&index| index as u32).collect::<Vec<_>>());
        }
        if !self.kept.contains_key(name) {
            let (table, _) = self.raw(name, 0, None)?;
            let bytes = (tensor.bytes / count) as usize;
            crate::storage::read_ahead(indices.iter().map(|&index| &table[index as usize * bytes..(index as usize + 1) * bytes]));
        }
        Ok(())
    }

    /// A row-cached table's (rows asked for, rows read, blocks read), see [`Weights::cache_rows`].
    pub fn row_counts(&self, name: &str) -> Option<(u64, u64, u64)> {
        self.row_tables.get(name).map(RowTable::counts)
    }

    fn token_rows(&self, indices: &[i64]) -> Result<Vec<f32>, String> {
        {
            let cache = self.embeddings.lock().unwrap();
            if indices.iter().all(|index| cache.0.contains_key(index)) {
                return Ok(indices.iter().flat_map(|index| cache.0[index].iter().copied()).collect());
            }
        }
        let decoded = self.decoded_rows("token_embd.weight", indices)?;
        let width = decoded.len() / indices.len();
        let mut cache = self.embeddings.lock().unwrap();
        for (index, row) in indices.iter().zip(decoded.chunks_exact(width)) {
            if cache.0.contains_key(index) {
                continue;
            }
            if cache.0.len() >= CACHED_ROWS {
                if let Some(oldest) = cache.1.pop_front() {
                    cache.0.remove(&oldest);
                }
            }
            cache.0.insert(*index, Arc::new(row.to_vec()));
            cache.1.push_back(*index);
        }
        Ok(decoded)
    }

    fn decoded_rows(&self, name: &str, indices: &[i64]) -> Result<Vec<f32>, String> {
        let tensor = self.tensor(name)?;
        let [width, count] = tensor.dims[..] else {
            return Err(error("invalid Straydog embedding row batch"));
        };
        if !(1..=65536).contains(&indices.len())
            || indices.len() as u64 * width > 16 << 20
            || indices.iter().any(|index| *index < 0 || *index as u64 >= count)
        {
            return Err(error("invalid Straydog embedding row batch"));
        }
        let (table, kind) = self.raw(name, 0, None)?;
        let row_bytes = (tensor.bytes / count) as usize;
        let width = width as usize;
        // The rows' bytes: through the table's row cache when it has one (see `crate::rows`),
        // else copied from the mapping and checked when the file's tree is open, else from the
        // mapping, every row's read started at once so that decoding waits on the slowest rather
        // than on each in turn.
        let verifier = self.verifier_of(tensor.file).filter(|_| !self.kept.contains_key(name));
        let fetched = match (self.row_tables.get(name), verifier) {
            (Some(rows), _) => Some(rows.read(&indices.iter().map(|&index| index as u32).collect::<Vec<_>>())?),
            (None, Some(verifier)) => {
                let mut bytes = Vec::with_capacity(indices.len() * row_bytes);
                for &index in indices {
                    let at = tensor.offset + index as u64 * row_bytes as u64;
                    bytes.extend(verity::read_verified(&self.maps[tensor.file], verifier, at, row_bytes as u64)
                        .map_err(|error| format!("Straydog model bytes do not verify: {error}"))?);
                }
                Some(bytes)
            }
            (None, None) => {
                if !self.kept.contains_key(name) {
                    crate::storage::read_ahead(indices.iter().map(|&index| &table[index as usize * row_bytes..(index as usize + 1) * row_bytes]));
                }
                None
            }
        };
        let row = |position: usize| match &fetched {
            Some(bytes) => &bytes[position * row_bytes..(position + 1) * row_bytes],
            None => &table[indices[position] as usize * row_bytes..(indices[position] as usize + 1) * row_bytes],
        };
        let decode = |first: usize, out: &mut [f32]| -> Result<(), String> {
            let mut values = Vec::with_capacity(width);
            for (position, target) in (first..).zip(out.chunks_exact_mut(width)) {
                values.clear();
                formats::decode_into(kind, row(position), &mut values)?;
                target.copy_from_slice(&values);
            }
            Ok(())
        };
        let mut output = vec![0f32; indices.len() * width];
        if indices.len() >= PARALLEL_ROWS {
            // Several threads decode (and, through the mapping, keep the device queue full).
            let per = indices.len().div_ceil(ROW_THREADS);
            std::thread::scope(|scope| {
                let workers: Vec<_> = output
                    .chunks_mut(per * width)
                    .enumerate()
                    .map(|(part, out)| scope.spawn(move || decode(part * per, out)))
                    .collect();
                workers.into_iter().try_for_each(|worker| worker.join().unwrap())
            })?;
        } else {
            decode(0, &mut output)?;
        }
        if !output.iter().all(|value| value.is_finite()) {
            return Err(error("non-finite Straydog embedding row batch"));
        }
        Ok(output)
    }

    /// Refuse a new generation if an authenticated file has changed.
    pub fn check_unchanged(&self) -> Result<(), String> {
        for (file, expected) in self.files.iter().zip(&self.signatures) {
            if signature(file).map_err(|exc| exc.to_string())? != *expected {
                return Err(error("Straydog model changed after authentication"));
            }
        }
        Ok(())
    }

    fn device_tensors(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.order
            .iter()
            .filter(|name| !matches!(name.as_str(), "token_embd.weight" | "per_layer_token_embd.weight"))
            .map(|name| (name, &self.tensors[name]))
    }

    /// Whether a MoE router is F32 in the bundle but every value a BF16 value (Unsloth's Qwen3.8: the model's own BF16 weights written
    /// as F32), so it goes up as BF16: the same values, half the bytes. Decided from the mapping; the upload converts the verified
    /// copy and refuses any value that is not BF16 (gpu::narrow_bf16), so no unverified byte reaches a GPU.
    pub fn narrowed(&self, name: &str) -> bool {
        self.narrow.get_or_init(|| {
            self.order.iter().filter(|name| name.ends_with("ffn_gate_inp.weight"))
                .filter(|name| self.tensors[*name].kind == 0 && self.tensors[*name].dims.len() == 2)
                .filter(|name| self.raw(name, 0, None).is_ok_and(|(bytes, _)| crate::gpu::bf16_exact(bytes)))
                .cloned().collect()
        }).contains(name)
    }

    /// GPU bytes of a tensor when pinned (Tensor::device_bytes), a narrowed router's as BF16.
    fn device_bytes(&self, name: &str, tensor: &Tensor) -> u64 {
        if self.narrowed(name) { tensor.elements() * 2 } else { tensor.device_bytes(name) }
    }

    /// GPU bytes of each decoder layer and of the rest placed with the last layer (head,
    /// MTP). Under expert parallelism the routed experts are split wherever the layer
    /// boundary falls, so only the rest is balanced.
    pub fn placement_bytes(&self, expert_parallel: bool) -> (Vec<u64>, u64) {
        let (mut layers, mut tail) = (vec![0u64; 48], 0u64);
        for (name, tensor) in self.device_tensors() {
            let layer = layer_of(name);
            if expert_parallel && layer < 48 && name.ends_with("_exps.weight") {
                continue;
            }
            match layers.get_mut(layer) {
                Some(slot) => *slot += self.device_bytes(name, tensor),
                None => tail += self.device_bytes(name, tensor),
            }
        }
        (layers, tail)
    }

    /// GPU bytes of every weight full residency pins, with or without the drafter's.
    pub fn resident_requirements(&self, drafter: bool) -> Vec<(String, u64)> {
        self.device_tensors()
            .filter(|(name, _)| drafter || !self.drafter_names.contains(*name))
            .map(|(name, tensor)| (name.clone(), self.device_bytes(name, tensor)))
            .collect()
    }

    /// GPU bytes of every dense operand (partial residency pins these, not routed experts).
    pub fn dense_requirements(&self) -> Vec<(String, u64)> {
        self.device_tensors()
            .filter(|(name, _)| !name.ends_with("_exps.weight") && !self.drafter_names.contains(*name))
            .map(|(name, tensor)| (name.clone(), self.device_bytes(name, tensor)))
            .collect()
    }

    /// What the MTP drafter pins when kept beside a partially resident target.
    pub fn drafter_requirements(&self) -> Vec<(String, u64)> {
        self.device_tensors()
            .filter(|(name, _)| self.drafter_names.contains(*name))
            .map(|(name, tensor)| (name.clone(), self.device_bytes(name, tensor)))
            .collect()
    }

    /// GGML kinds of a layer's routed gate, up and down banks.
    pub fn expert_kinds(&self, layer: usize) -> Result<[u32; 3], String> {
        let kind = |role: &str| self.tensor(&format!("blk.{layer}.ffn_{role}_exps.weight")).map(|tensor| tensor.kind);
        Ok([kind("gate")?, kind("up")?, kind("down")?])
    }
}

/// Whether a tensor is a HyperConnection's down or up projection or the PLE layer's key or value projection (BF16 matrices a
/// pass reads whole), uploaded quantized to Q8_0 (`STRAYDOG_HC_Q8=0`: as it is).
pub fn hc_quantized(name: &str) -> bool {
    (name.ends_with("hc_attn_down.weight") || name.ends_with("hc_attn_up.weight") || name.ends_with("hc_ffn_down.weight")
     || name.ends_with("hc_ffn_up.weight") || name.ends_with("output_hc_down.weight") || name.ends_with("output_hc_up.weight")
     || name.ends_with("hc_out_down.weight") || name.ends_with("hc_out_up.weight") || name.ends_with(".ple_key.weight")
     || name.ends_with(".ple_value.weight"))
        && std::env::var("STRAYDOG_HC_Q8").map_or(true, |value| value != "0")
}

/// Whether a tensor is one of the MTP drafter's dense projections but its router, uploaded quantized to Q8_0 when it is
/// a float matrix: a draft step reads them all for one row (BF16, they were a fifth of a step), and only the drafts
/// depend on them.
pub fn drafter_quantized(name: &str) -> bool {
    matches!(name, "mtp.fc_embedding.weight" | "mtp.fc_hidden.weight")
        || name.strip_prefix("blk.48.").is_some_and(|rest| {
            matches!(rest, "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_output.weight" | "indexer.q_proj.weight"
                     | "indexer.k_proj.weight" | "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" | "ffn_down_shexp.weight")
        })
}

/// The decoder layer a tensor belongs to; 48 for the rest (head, embeddings, MTP).
pub fn layer_of(name: &str) -> usize {
    name.strip_prefix("blk.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|layer| layer.parse().ok())
        .unwrap_or(48)
}

/// The PLE n-gram rows of `tokens` after `history`: 16 heads of bigram (8) and trigram (8)
/// hashes. An end token restarts the n-grams. Exact integer arithmetic (as Python's).
pub fn ngram_rows(history: &[i64], tokens: &[i64], arrays: &BTreeMap<String, Vec<i64>>)
                  -> Result<Vec<i64>, String> {
    const EOS: i128 = 248044;
    let get = |key: &str| arrays.get(key).ok_or_else(|| "invalid Straydog PLE constants".to_string());
    let (multipliers, offsets, sizes) =
        (get("qwen4exp.ple.layer_multipliers")?, get("qwen4exp.ple.head_offsets")?, get("qwen4exp.ple.head_vocab_sizes")?);
    if multipliers.len() != 3 || offsets.len() != 16 || sizes.len() != 16 || sizes.iter().any(|size| *size <= 0) {
        return Err("invalid Straydog PLE constants".into());
    }
    if tokens.is_empty() || tokens.len() > 4096 {
        return Err("invalid Straydog PLE token batch".into());
    }
    let mut previous: Vec<i128> = history[history.len().saturating_sub(2)..].iter().map(|&value| value.into()).collect();
    let mut rows = Vec::with_capacity(tokens.len() * 16);
    let [first, second, third] = [multipliers[0], multipliers[1], multipliers[2]].map(i128::from);
    for &token in tokens {
        let token = i128::from(token);
        let last = previous.last().copied().unwrap_or(EOS);
        let earlier = if previous.len() > 1 && last != EOS { previous[previous.len() - 2] } else { EOS };
        let bigram = (token * first) ^ (last * second);
        let trigram = bigram ^ (earlier * third);
        for head in 0..16 {
            let hash = if head < 8 { bigram } else { trigram };
            rows.push((hash.rem_euclid(i128::from(sizes[head])) + i128::from(offsets[head])) as i64);
        }
        previous.push(token);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An absolute path to a directory that is not there.
    fn missing() -> PathBuf {
        let path = std::env::temp_dir().join(format!("straydog-no-such-directory-{}", std::process::id())).join("model");
        assert!(path.is_absolute() && !path.exists());
        path
    }

    #[test]
    fn an_absolute_path_comes_back_as_given_without_the_file_system() {
        // `canonicalize` fails on a directory that is not there: this must not ask.
        let path = missing();
        assert_eq!(absolute_directory(&path).unwrap(), path);
        // Nothing resolves a `..` either.
        let dotted = path.join("..").join("other");
        assert_eq!(absolute_directory(&dotted).unwrap(), dotted);
    }

    #[cfg(windows)]
    #[test]
    fn a_verbatim_path_comes_back_as_given() {
        for text in [r"\\?\C:\straydog-no-such-directory\IQ2_XS", r"\\?\UNC\straydog-no-such-server\share\IQ2_XS"] {
            let path = Path::new(text);
            assert!(path.is_absolute());
            assert_eq!(absolute_directory(path).unwrap(), path);
        }
    }

    #[test]
    fn a_relative_path_is_made_absolute_against_the_current_directory() {
        let relative = Path::new("straydog-no-such-directory").join("model");
        let made = absolute_directory(&relative).unwrap();
        assert!(made.is_absolute());
        assert_eq!(made, std::env::current_dir().unwrap().join(&relative));
    }

    #[test]
    fn an_empty_path_is_refused() {
        assert!(absolute_directory(Path::new("")).is_err());
    }

    #[test]
    fn opening_a_directory_does_not_resolve_it_first() {
        // With the directory resolved, the error would be about the directory; it is about the first file read in it.
        let path = missing();
        let Err(message) = Weights::open(&path, bundle::Id::find(Some(bundle::MODEL), None, "IQ2_XS").unwrap(), None, 0, &|_, _, _| {}) else { panic!("a missing directory opened") };
        assert!(!message.contains(&path.display().to_string()), "{message}");
        assert!(message.contains("os error"), "{message}");
    }
}
