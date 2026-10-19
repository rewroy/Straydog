//! Authenticated, demand-paged GGUF weights.
//!
//! Every file is hashed from the handle that is then mapped, and its header is checked
//! from a snapshot of that same handle. Weights reach the GPU as the mapped bytes; the
//! host decodes only embedding rows and small vectors.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use memmap2::Mmap;

use crate::bundle::{self, Asset};
use crate::digest;
use crate::formats;

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
    pub quantization: String,
    /// Measured outside isolation by the runtime (cgroup limits included).
    pub host_memory_bytes: u64,
    pub integer_arrays: BTreeMap<String, Vec<i64>>,
    pub resources: HashMap<String, Vec<u8>>,
    /// Whether the pinned MTP drafter is authenticated and in use.
    pub drafter: bool,
    drafter_names: HashSet<String>,
    files: Vec<File>,
    maps: Vec<Mmap>,
    signatures: Vec<(u64, Option<SystemTime>)>,
    tensors: HashMap<String, Tensor>,
    order: Vec<String>,
    vectors: Mutex<HashMap<String, Arc<Vector>>>,
    embeddings: Mutex<EmbeddingCache>,
}

fn signature(file: &File) -> std::io::Result<(u64, Option<SystemTime>)> {
    let metadata = file.metadata()?;
    Ok((metadata.len(), metadata.modified().ok()))
}

fn error(message: impl Into<String>) -> String {
    message.into()
}

/// Open the pinned files and authenticate them by size and SHA-256 from these handles.
fn open_authenticated(root: &Path, assets: &[Asset], stage: &str, progress: Progress) -> Result<Vec<File>, String> {
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for asset in assets {
        if !seen.insert(&asset.name) {
            return Err(error("invalid Straydog asset name"));
        }
        let file = File::open(root.join(&asset.name)).map_err(|exc| format!("{}: {exc}", asset.name))?;
        if file.metadata().map_err(|exc| exc.to_string())?.len() != asset.bytes {
            return Err(format!("Straydog asset size mismatch: {}", asset.name));
        }
        files.push(file);
    }
    let total: u64 = assets.iter().map(|asset| asset.bytes).sum();
    let done = AtomicU64::new(0);
    progress(stage, 0, total);
    let count = |bytes: u64| progress(stage, done.fetch_add(bytes, Ordering::Relaxed) + bytes, total);
    // Files with pinned chunk digests are hashed piece by piece on every core; the rest whole.
    let pieces = assets.iter().map(|asset| bundle::chunk_digests(&asset.sha256)).collect::<Result<Vec<_>, _>>()?;
    let chunked: Vec<(&File, &[String])> = files.iter().zip(&pieces)
        .filter_map(|(file, pieces)| pieces.as_deref().map(|pieces| (file, pieces)))
        .collect();
    let whole: Vec<&File> = files.iter().zip(&pieces).filter(|(_, pieces)| pieces.is_none()).map(|(file, _)| file).collect();
    let mut chunked_results = digest::verify_chunks(&chunked, &count).map_err(|exc| exc.to_string())?.into_iter();
    let mut whole_digests = digest::files_sha256(&whole, &count).map_err(|exc| exc.to_string())?.into_iter();
    for (asset, pieces) in assets.iter().zip(&pieces) {
        let authentic = match pieces {
            Some(_) => chunked_results.next() == Some(true),
            None => whole_digests.next().as_deref() == Some(asset.sha256.as_str()),
        };
        if !authentic {
            return Err(format!("Straydog asset digest mismatch: {}", asset.name));
        }
    }
    Ok(files)
}

/// The header of a GGUF file, from a snapshot of the authenticated handle.
fn header(file: &mut File) -> Result<straydog_gguf::ModelHeader, String> {
    let size = file.metadata().map_err(|exc| exc.to_string())?.len();
    let mut snapshot = vec![0u8; size.min(SNAPSHOT_BYTES) as usize];
    file.seek(SeekFrom::Start(0)).and_then(|_| file.read_exact(&mut snapshot)).map_err(|exc| exc.to_string())?;
    straydog_gguf::inspect_header(&snapshot, size).map_err(|exc| format!("Straydog GGUF preflight: {exc}"))
}

impl Weights {
    pub fn open(root: &Path, quantization: &str, drafter: Option<&Path>, host_memory_bytes: u64,
                progress: Progress) -> Result<Weights, String> {
        let pins = bundle::bundle(quantization)?;
        if pins.shards.len() != 2 {
            return Err(error("Straydog requires both fixed model shards"));
        }
        let names: HashSet<&str> = pins.resources.iter().map(|asset| asset.name.as_str()).collect();
        if names != HashSet::from(["tokenizer.json", "chat_template.jinja", "config.json"]) {
            return Err(error("Straydog requires authenticated tokenizer, template, and config"));
        }
        let root = root.canonicalize().map_err(|exc| format!("{}: {exc}", root.display()))?;
        let assets: Vec<Asset> = pins.shards.iter().chain(&pins.resources).cloned().collect();
        let opened = open_authenticated(&root, &assets, "authenticate", progress)?;
        let mut weights = Weights {
            root: root.clone(),
            quantization: quantization.to_string(),
            host_memory_bytes: if host_memory_bytes > 0 { host_memory_bytes } else { crate::host::host_memory_bytes() },
            integer_arrays: BTreeMap::new(),
            resources: HashMap::new(),
            drafter: false,
            drafter_names: HashSet::new(),
            files: Vec::new(),
            maps: Vec::new(),
            signatures: Vec::new(),
            tensors: HashMap::new(),
            order: Vec::new(),
            vectors: Mutex::new(HashMap::new()),
            embeddings: Mutex::new((HashMap::new(), VecDeque::new())),
        };
        for (asset, mut file) in assets.iter().zip(opened) {
            if pins.resources.contains(asset) {
                if asset.bytes > RESOURCE_LIMIT {
                    return Err(error("Straydog resource exceeds size limit"));
                }
                let mut data = Vec::new();
                file.seek(SeekFrom::Start(0)).and_then(|_| file.read_to_end(&mut data)).map_err(|exc| exc.to_string())?;
                if format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&data)) != asset.sha256 {
                    return Err(error("Straydog resource changed during authentication"));
                }
                weights.resources.insert(asset.name.clone(), data);
                continue;
            }
            let before = signature(&file).map_err(|exc| exc.to_string())?;
            let header = header(&mut file)?;
            let index = weights.files.len();
            if header.shard_count != 2 || header.shard_index != index as u64 {
                return Err(error("Straydog shard identity mismatch"));
            }
            let architecture_ok = header.architecture == "qwen4exp" || (index == 1 && header.architecture.is_empty());
            if !architecture_ok {
                return Err(error("Straydog architecture mismatch"));
            }
            if index == 0 && header.declared_context != Some(CONTEXT_LIMIT as u64) {
                return Err(error("Straydog model context declaration mismatch"));
            }
            weights.add_tensors(index, &header.tensors)?;
            weights.integer_arrays.extend(header.integer_arrays.clone());
            weights.map(file, before)?;
        }
        if weights.tensors.len() != TENSOR_COUNT {
            return Err(error("Straydog model tensor count mismatch"));
        }
        if let Some(directory) = drafter {
            weights.load_drafter(directory, &pins.drafter, progress)?;
        }
        Ok(weights)
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

    /// Map the authenticated handle, refusing a file that changed since it was hashed.
    fn map(&mut self, file: File, before: (u64, Option<SystemTime>)) -> Result<(), String> {
        // SAFETY: the mapping is read-only and the file is checked unchanged before every
        // generation; the files are pinned assets nothing is expected to write.
        let map = unsafe { Mmap::map(&file) }.map_err(|exc| exc.to_string())?;
        if signature(&file).map_err(|exc| exc.to_string())? != before {
            return Err(error("Straydog asset changed during authentication"));
        }
        self.signatures.push(before);
        self.files.push(file);
        self.maps.push(map);
        Ok(())
    }

    fn load_drafter(&mut self, directory: &Path, pin: &Asset, progress: Progress) -> Result<(), String> {
        let directory = directory.canonicalize().map_err(|exc| format!("{}: {exc}", directory.display()))?;
        let mut file = open_authenticated(&directory, std::slice::from_ref(pin), "drafter", progress)?.remove(0);
        let before = signature(&file).map_err(|exc| exc.to_string())?;
        let header = header(&mut file)?;
        let names: HashSet<String> = header.tensors.iter().map(|tensor| tensor.name.clone()).collect();
        let expected: HashSet<String> = DRAFTER_TENSORS.iter().map(|name| name.to_string()).collect();
        if header.shard_count != 1 || header.architecture != "qwen4exp" || names != expected {
            return Err(error("Straydog drafter layout mismatch"));
        }
        if names.iter().any(|name| self.tensors.contains_key(name)) {
            return Err(error("duplicate tensor across Straydog drafter and model"));
        }
        self.add_tensors(self.files.len(), &header.tensors)?;
        self.map(file, before)?;
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

    /// The mapped bytes of a tensor slice.
    pub fn raw(&self, name: &str, start: u64, size: Option<u64>) -> Result<(&[u8], u32), String> {
        let tensor = self.tensor(name)?;
        let size = size.unwrap_or(tensor.bytes);
        if start.checked_add(size).is_none_or(|end| end > tensor.bytes) {
            return Err(error("Straydog tensor slice is out of range"));
        }
        let begin = (tensor.offset + start) as usize;
        let map = &self.maps[tensor.file];
        let end = begin.checked_add(size as usize).filter(|end| *end <= map.len()).ok_or("tensor beyond its file")?;
        Ok((&map[begin..end], tensor.kind))
    }

    /// The file a tensor lives in (for read-ahead hints).
    pub fn file_of(&self, name: &str) -> Result<&File, String> {
        Ok(&self.files[self.tensor(name)?.file])
    }

    /// A small norm, bias or convolution tensor, decoded once.
    pub fn vector(&self, name: &str) -> Result<Arc<Vector>, String> {
        if let Some(vector) = self.vectors.lock().unwrap().get(name) {
            return Ok(vector.clone());
        }
        let tensor = self.tensor(name)?;
        if tensor.elements() > 65536 {
            return Err(error("Straydog vector request is too large"));
        }
        let (raw, kind) = self.raw(name, 0, None)?;
        let mut values = Vec::new();
        formats::decode_into(kind, raw, &mut values).map_err(|exc| format!("{name}: {exc}"))?;
        if values.len() as u64 != tensor.elements() || !values.iter().all(|value| value.is_finite()) {
            return Err(error("non-finite Straydog model parameters"));
        }
        let (rows, cols) = match tensor.dims.as_slice() {
            [cols] => (1, *cols as usize),
            [inner, outer] => (*outer as usize, *inner as usize),
            _ => return Err(format!("{name} is not a vector")),
        };
        let vector = Arc::new(Vector { values, rows, cols });
        self.vectors.lock().unwrap().insert(name.to_string(), vector.clone());
        Ok(vector)
    }

    /// Embedding rows, decoded as one batch; row `i` of the result is `indices[i]`.
    pub fn rows(&self, name: &str, indices: &[i64]) -> Result<Vec<f32>, String> {
        if name == "token_embd.weight" && (1..=CACHED_BATCH).contains(&indices.len()) {
            return self.token_rows(indices);
        }
        self.decoded_rows(name, indices)
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
        let decode = |part: &[i64], out: &mut [f32]| -> Result<(), String> {
            let mut values = Vec::with_capacity(width);
            for (index, target) in part.iter().zip(out.chunks_exact_mut(width)) {
                values.clear();
                let start = *index as usize * row_bytes;
                formats::decode_into(kind, &table[start..start + row_bytes], &mut values)?;
                target.copy_from_slice(&values);
            }
            Ok(())
        };
        let mut output = vec![0f32; indices.len() * width];
        if indices.len() >= PARALLEL_ROWS {
            // Several threads keep the device queue full while uncached rows fault in.
            let per = indices.len().div_ceil(ROW_THREADS);
            std::thread::scope(|scope| {
                let workers: Vec<_> = indices
                    .chunks(per)
                    .zip(output.chunks_mut(per * width))
                    .map(|(part, out)| scope.spawn(move || decode(part, out)))
                    .collect();
                workers.into_iter().try_for_each(|worker| worker.join().unwrap())
            })?;
        } else {
            decode(indices, &mut output)?;
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
                Some(slot) => *slot += tensor.device_bytes(name),
                None => tail += tensor.device_bytes(name),
            }
        }
        (layers, tail)
    }

    /// GPU bytes of every weight full residency pins, with or without the drafter's.
    pub fn resident_requirements(&self, drafter: bool) -> Vec<(String, u64)> {
        self.device_tensors()
            .filter(|(name, _)| drafter || !self.drafter_names.contains(*name))
            .map(|(name, tensor)| (name.clone(), tensor.device_bytes(name)))
            .collect()
    }

    /// GPU bytes of every dense operand (partial residency pins these, not routed experts).
    pub fn dense_requirements(&self) -> Vec<(String, u64)> {
        self.device_tensors()
            .filter(|(name, _)| !name.ends_with("_exps.weight") && !self.drafter_names.contains(*name))
            .map(|(name, tensor)| (name.clone(), tensor.device_bytes(name)))
            .collect()
    }

    /// GGML kinds of a layer's routed gate, up and down banks.
    pub fn expert_kinds(&self, layer: usize) -> Result<[u32; 3], String> {
        let kind = |role: &str| self.tensor(&format!("blk.{layer}.ffn_{role}_exps.weight")).map(|tensor| tensor.kind);
        Ok([kind("gate")?, kind("up")?, kind("down")?])
    }
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
