//! Straydog's MTP drafter: the base model's own multi-token-prediction layer.
//!
//! The GSQ-RCO GGUF bundles omit the MTP layer, so its tensors are range-fetched from the
//! pinned official checkpoint (each tensor's SHA-256 is pinned in the manifest) and
//! converted into one GGUF named like the decoder layers (`blk.48.*` plus `mtp.*`
//! prologue and mixer operands). Gemma norm weights gain their +1 offset like the main
//! bundle; routed experts are encoded as Q2_0 by the GPU library's deterministic CPU
//! quantizer, everything else stays BF16 or F32. Drafts are always verified by the main
//! model, so the drafter's precision affects only the acceptance rate, never the
//! generated tokens.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use memmap2::Mmap;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use straydog_engine::bundle::SourceTensor;
use straydog_gpu::ffi;
use ureq::Agent;

use crate::assets::read_body;
use crate::{at, invalid, partial_path};

/// The official tensors, kept between the download and the conversion.
pub const SOURCE_FILE: &str = "mtp-bf16.safetensors";
/// A tensor download fails after this long without progress (connecting, each read).
pub const TIMEOUT: Duration = Duration::from_secs(600);
/// Download pieces and parallel connections (one connection reaches ~42 MB/s from the Hub,
/// four ~57), and experts per quantization call (progress steps; the Q2_0 bytes do not
/// depend on it).
pub const PIECE: u64 = 64 << 20;
const CONNECTIONS: usize = 4;
pub const EXPERT_GROUP: usize = 32;
const BLOCK: usize = 1 << 20;
const HASH_BLOCK: usize = 8 << 20;
const ALIGNMENT: usize = 32;
const F32: u32 = 0;
const BF16: u32 = 30;
const Q2_0: u32 = 42;

#[derive(Debug, Clone, Copy)]
enum Transform {
    /// F32, with the Gemma norm's +1 offset.
    Norm,
    /// F32.
    Vector,
    /// BF16 as it is.
    Bf16,
    /// BF16 rows `first..last` of the source.
    Rows(usize, usize),
    /// Q2_0 of rows `first..last` of each expert.
    Experts(usize, usize),
}

macro_rules! layer {
    ($name:literal) => {
        concat!("mtp.layers.0.", $name)
    };
}

/// The drafter's tensors in file order: GGUF name, source tensor, transform.
const PLAN: [(&str, &str, Transform); 33] = [
    ("mtp.embed_norm.weight", "mtp.pre_fc_norm_embedding.weight", Transform::Norm),
    ("mtp.fc_embedding.weight", "mtp.fc_embedding.weight", Transform::Bf16),
    ("mtp.hidden_norm.weight", "mtp.pre_fc_norm_hidden.weight", Transform::Norm),
    ("mtp.fc_hidden.weight", "mtp.fc_hidden.weight", Transform::Bf16),
    ("mtp.hc_out_norm.weight", "mtp.hyper_connection_mixer.hc_norm.weight", Transform::Norm),
    ("mtp.hc_out_down.weight", "mtp.hyper_connection_mixer.input_mix_weight_down.weight", Transform::Bf16),
    ("mtp.hc_out_up.weight", "mtp.hyper_connection_mixer.input_mix_weight_up.weight", Transform::Bf16),
    ("blk.48.attn_q.weight", layer!("self_attn.q_proj.weight"), Transform::Bf16),
    ("blk.48.attn_k.weight", layer!("self_attn.k_proj.weight"), Transform::Bf16),
    ("blk.48.attn_v.weight", layer!("self_attn.v_proj.weight"), Transform::Bf16),
    ("blk.48.attn_output.weight", layer!("self_attn.o_proj.weight"), Transform::Bf16),
    ("blk.48.attn_q_norm.weight", layer!("self_attn.q_norm.weight"), Transform::Norm),
    ("blk.48.attn_k_norm.weight", layer!("self_attn.k_norm.weight"), Transform::Norm),
    ("blk.48.indexer.q_proj.weight", layer!("self_attn.indexer.index_qk_proj.weight"), Transform::Rows(0, 512)),
    ("blk.48.indexer.k_proj.weight", layer!("self_attn.indexer.index_qk_proj.weight"), Transform::Rows(512, 640)),
    ("blk.48.indexer.q_norm.weight", layer!("self_attn.indexer.q_layernorm.weight"), Transform::Norm),
    ("blk.48.indexer.k_norm.weight", layer!("self_attn.indexer.k_layernorm.weight"), Transform::Norm),
    ("blk.48.ffn_gate_inp.weight", layer!("mlp.gate.weight"), Transform::Bf16),
    ("blk.48.ffn_gate_inp_shexp.weight", layer!("mlp.shared_expert_gate.weight"), Transform::Vector),
    ("blk.48.ffn_gate_shexp.weight", layer!("mlp.shared_expert.gate_proj.weight"), Transform::Bf16),
    ("blk.48.ffn_up_shexp.weight", layer!("mlp.shared_expert.up_proj.weight"), Transform::Bf16),
    ("blk.48.ffn_down_shexp.weight", layer!("mlp.shared_expert.down_proj.weight"), Transform::Bf16),
    ("blk.48.ffn_gate_exps.weight", layer!("mlp.experts.gate_up_proj"), Transform::Experts(0, 640)),
    ("blk.48.ffn_up_exps.weight", layer!("mlp.experts.gate_up_proj"), Transform::Experts(640, 1280)),
    ("blk.48.ffn_down_exps.weight", layer!("mlp.experts.down_proj"), Transform::Experts(0, 2560)),
    ("blk.48.hc_attn_norm.weight", layer!("attn_hyper_connection.hc_norm.weight"), Transform::Norm),
    ("blk.48.hc_attn_down.weight", layer!("attn_hyper_connection.input_mix_weight_down.weight"), Transform::Bf16),
    ("blk.48.hc_attn_up.weight", layer!("attn_hyper_connection.input_mix_weight_up.weight"), Transform::Bf16),
    ("blk.48.hc_attn_inject.weight", layer!("attn_hyper_connection.block_inject_weight.weight"), Transform::Bf16),
    ("blk.48.hc_ffn_norm.weight", layer!("mlp_hyper_connection.hc_norm.weight"), Transform::Norm),
    ("blk.48.hc_ffn_down.weight", layer!("mlp_hyper_connection.input_mix_weight_down.weight"), Transform::Bf16),
    ("blk.48.hc_ffn_up.weight", layer!("mlp_hyper_connection.input_mix_weight_up.weight"), Transform::Bf16),
    ("blk.48.hc_ffn_inject.weight", layer!("mlp_hyper_connection.block_inject_weight.weight"), Transform::Bf16),
];

/// Range-fetch the pinned MTP tensors from `base` (the checkpoint's URL, ending in a slash)
/// into one safetensors file at `target`, verifying each SHA-256.
///
/// Several connections fetch pieces of `piece` bytes straight into their places in the file;
/// each tensor is then hashed from the file, so no tensor is held in memory. `progress`
/// (called from those connections' threads) receives the byte counts.
pub fn fetch_source(agent: &Agent, base: &str, pinned: &[SourceTensor], target: &Path, piece: u64,
                    progress: &(dyn Fn(u64) + Sync)) -> io::Result<()> {
    let mut tensors: Vec<&SourceTensor> = pinned.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    // Each shard's header length and table, read once.
    let mut headers: HashMap<&str, (u64, Map<String, Value>)> = HashMap::new();
    let mut meta = Map::new();
    let mut spans = Vec::new();
    let mut offset = 0;
    for tensor in &tensors {
        if !headers.contains_key(tensor.shard.as_str()) {
            headers.insert(&tensor.shard, shard_header(agent, &format!("{base}{}", tensor.shard))?);
        }
        let (size, table) = &headers[tensor.shard.as_str()];
        let changed = || invalid(format!("MTP tensor changed upstream: {}", tensor.name));
        let info = table.get(&tensor.name).ok_or_else(changed)?;
        let (first, last) = data_offsets(info).ok_or_else(changed)?;
        if info["dtype"] != "BF16" || last - first != tensor.bytes {
            return Err(changed());
        }
        // Where the tensor starts in the shard: after the length, the header and its offset
        // there (whose end must be a possible one).
        let start = (8 + size).checked_add(last).map(|_| 8 + size + first).ok_or_else(changed)?;
        let span = [offset, offset + tensor.bytes];
        meta.insert(tensor.name.clone(), json!({"dtype": "BF16", "shape": info["shape"], "data_offsets": span}));
        spans.push((start, offset));
        offset += tensor.bytes;
    }
    let mut blob = serde_json::to_vec(&meta)?;
    blob.resize(blob.len().next_multiple_of(8), b' ');
    let data = 8 + blob.len() as u64;
    // (URL, first byte there, first byte here, length)
    let mut pieces = Vec::new();
    for (tensor, &(source, offset)) in tensors.iter().zip(&spans) {
        let url = format!("{base}{}", tensor.shard);
        for start in (0..tensor.bytes).step_by(piece as usize) {
            pieces.push((url.clone(), source + start, data + offset + start, piece.min(tensor.bytes - start)));
        }
    }
    let partial = partial_path(target);
    let result = (|| {
        let mut output = File::create(&partial).map_err(at(&partial))?;
        output.write_all(&(blob.len() as u64).to_le_bytes())?;
        output.write_all(&blob)?;
        output.set_len(data + offset)?;
        drop(output);
        fetch_pieces(agent, &pieces, &partial, progress)?;
        let mut output = File::open(&partial).map_err(at(&partial))?;
        let mut buffer = vec![0u8; HASH_BLOCK];
        for (tensor, &(_, offset)) in tensors.iter().zip(&spans) {
            output.seek(SeekFrom::Start(data + offset))?;
            let (mut digest, mut remaining) = (Sha256::new(), tensor.bytes);
            while remaining > 0 {
                let count = output.read(&mut buffer[..remaining.min(HASH_BLOCK as u64) as usize])?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
                remaining -= count as u64;
            }
            if remaining > 0 || format!("{:x}", digest.finalize()) != tensor.sha256 {
                return Err(invalid(format!("MTP tensor failed authentication: {}", tensor.name)));
            }
        }
        fs::rename(&partial, target).map_err(at(target))
    })();
    if result.is_err() {
        fs::remove_file(&partial).ok();
    }
    result
}

/// A tensor's `data_offsets` in its safetensors header.
fn data_offsets(info: &Value) -> Option<(u64, u64)> {
    let offsets = info["data_offsets"].as_array()?;
    let (first, last) = (offsets.first()?.as_u64()?, offsets.get(1)?.as_u64()?);
    (offsets.len() == 2 && first <= last).then_some((first, last))
}

/// Bytes `start..=end` of `url`, which must be exactly those.
fn read_range(agent: &Agent, url: &str, start: u64, end: u64) -> io::Result<Vec<u8>> {
    let response = agent.get(url).header("Range", format!("bytes={start}-{end}")).call().map_err(ureq::Error::into_io)?;
    let length = end + 1 - start;
    let mut bytes = Vec::new();
    // One byte more than asked shows a server that ignored the range, without reading all it sends.
    response.into_body().into_reader().take(length + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != length {
        return Err(invalid("invalid safetensors header"));
    }
    Ok(bytes)
}

/// A safetensors shard's header length and table.
fn shard_header(agent: &Agent, url: &str) -> io::Result<(u64, Map<String, Value>)> {
    let size = u64::from_le_bytes(read_range(agent, url, 0, 7)?.try_into().expect("eight bytes"));
    if !(1..64 << 20).contains(&size) {
        return Err(invalid("invalid safetensors header"));
    }
    let header = read_range(agent, url, 8, 7 + size)?;
    Ok((size, serde_json::from_slice(&header).map_err(|_| invalid("invalid safetensors header"))?))
}

/// Fetch every (URL, source offset, destination offset, length) piece into `partial`, on
/// several connections. After a failure no new piece starts, and the first failure is returned.
fn fetch_pieces(agent: &Agent, pieces: &[(String, u64, u64, u64)], partial: &Path, progress: &(dyn Fn(u64) + Sync))
                -> io::Result<()> {
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<io::Error>> = Mutex::new(None);
    std::thread::scope(|scope| -> io::Result<()> {
        for _ in 0..CONNECTIONS.min(pieces.len()) {
            std::thread::Builder::new().name("straydog-mtp".into()).spawn_scoped(scope, || loop {
                if failure.lock().unwrap().is_some() {
                    return;
                }
                let Some((url, start, destination, length)) = pieces.get(next.fetch_add(1, Ordering::Relaxed)) else {
                    return;
                };
                if let Err(error) = fetch_piece(agent, url, *start, *destination, *length, partial, progress) {
                    failure.lock().unwrap().get_or_insert(error);
                    return;
                }
            })?;
        }
        Ok(())
    })?;
    match failure.into_inner().unwrap() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn fetch_piece(agent: &Agent, url: &str, start: u64, destination: u64, length: u64, partial: &Path,
               progress: &(dyn Fn(u64) + Sync)) -> io::Result<()> {
    let range = format!("bytes={start}-{}", start + length - 1);
    let response = agent.get(url).header("Range", range).call().map_err(ureq::Error::into_io)?;
    let mut output = OpenOptions::new().write(true).open(partial).map_err(at(partial))?;
    output.seek(SeekFrom::Start(destination))?;
    let (mut body, mut buffer, mut received) = (response.into_body().into_reader(), vec![0u8; BLOCK], 0);
    loop {
        let count = read_body(&mut body, &mut buffer)?;
        if count == 0 {
            break;
        }
        received += count as u64;
        if received > length {
            return Err(invalid("MTP tensor download exceeded its range"));
        }
        output.write_all(&buffer[..count])?;
        progress(count as u64);
    }
    if received != length {
        return Err(invalid("MTP tensor download ended early"));
    }
    Ok(())
}

/// A tensor of the drafter GGUF.
struct Entry<'a> {
    name: &'a str,
    /// GGUF order: the input dimension first.
    dims: Vec<usize>,
    kind: u32,
    data: Cow<'a, [u8]>,
}

/// A source tensor: its shape and BF16 bytes.
struct Tensor<'a> {
    shape: Vec<usize>,
    data: &'a [u8],
}

/// The tensors of a safetensors file, as views of its bytes.
fn source_tensors(file: &[u8]) -> io::Result<HashMap<String, Tensor<'_>>> {
    let broken = || invalid("invalid MTP source file");
    let size = u64::from_le_bytes(file.get(..8).ok_or_else(broken)?.try_into().expect("eight bytes"));
    let base = usize::try_from(size).ok().and_then(|size| size.checked_add(8)).ok_or_else(broken)?;
    let header: Map<String, Value> = serde_json::from_slice(file.get(8..base).ok_or_else(broken)?).map_err(|_| broken())?;
    // A position in the file from one in its data.
    let at = |offset: u64| usize::try_from(offset).ok()?.checked_add(base);
    let mut tensors = HashMap::new();
    for (name, info) in header.iter().filter(|(name, _)| *name != "__metadata__") {
        let (first, last) = data_offsets(info).ok_or_else(broken)?;
        let data = at(first).zip(at(last)).and_then(|(first, last)| file.get(first..last)).ok_or_else(broken)?;
        let shape: Vec<usize> = info["shape"].as_array().ok_or_else(broken)?.iter()
            .map(|dim| dim.as_u64().and_then(|dim| usize::try_from(dim).ok())).collect::<Option<_>>().ok_or_else(broken)?;
        let bytes = shape.iter().try_fold(2usize, |bytes, &dim| bytes.checked_mul(dim));
        if info["dtype"] != "BF16" || bytes != Some(data.len()) {
            return Err(invalid(format!("invalid MTP source tensor: {name}")));
        }
        tensors.insert(name.clone(), Tensor { shape, data });
    }
    Ok(tensors)
}

fn bf16_values(data: &[u8]) -> impl Iterator<Item = u16> + '_ {
    data.chunks_exact(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
}

/// BF16 widened to F32, plus `offset` when there is one (an exact F32 sum).
fn f32_bytes(data: &[u8], offset: Option<f32>) -> Vec<u8> {
    bf16_values(data)
        .map(|bits| f32::from_bits(u32::from(bits) << 16))
        .map(|value| offset.map_or(value, |offset| value + offset))
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn quantize_q2_0(bits: &[u16]) -> io::Result<Vec<u8>> {
    let mut output = vec![0u8; bits.len() / 64 * 18];
    // SAFETY: the quantizer reads `bits.len()` values and writes 18 bytes per 64 of them; it
    // rejects a count that is not a positive multiple of 64 before touching either buffer.
    if unsafe { ffi::sd_quantize_q2_0(bits.as_ptr(), bits.len() as u64, output.as_mut_ptr()) } != 0 {
        return Err(io::Error::other("Straydog Q2_0 quantization failed"));
    }
    Ok(output)
}

/// Write the drafter GGUF from the authenticated BF16 source tensors, quantizing the routed
/// experts `group` at a time. `progress(done, total)` follows the routed-expert values
/// quantized, nearly all of the work.
pub fn convert(source: &Path, target: &Path, revision: &str, group: usize, progress: &dyn Fn(u64, u64))
               -> io::Result<()> {
    let file = File::open(source).map_err(at(source))?;
    // SAFETY: mapped read-only; the source is this process's own download, which nothing
    // else writes while the drafter is built.
    let map = unsafe { Mmap::map(&file) }.map_err(at(source))?;
    let tensors = source_tensors(&map)?;
    let tensor = |origin: &str| tensors.get(origin).ok_or_else(|| invalid(format!("MTP source tensor missing: {origin}")));
    let mismatch = |origin: &str| invalid(format!("MTP source tensor has an unexpected shape: {origin}"));
    let mut total = 0;
    for (_, origin, transform) in PLAN {
        if let Transform::Experts(first, last) = transform {
            match tensor(origin)?.shape[..] {
                [experts, rows, cols] if last <= rows => total += (experts * (last - first) * cols) as u64,
                _ => return Err(mismatch(origin)),
            }
        }
    }
    progress(0, total);
    let mut done = 0;
    let mut entries = Vec::new();
    for (name, origin, transform) in PLAN {
        let source = tensor(origin)?;
        let count = source.data.len() / 2;
        let reversed = |shape: &[usize]| shape.iter().rev().copied().collect::<Vec<_>>();
        let (dims, kind, data) = match transform {
            Transform::Norm => (vec![count], F32, Cow::Owned(f32_bytes(source.data, Some(1.0)))),
            Transform::Vector => (vec![count], F32, Cow::Owned(f32_bytes(source.data, None))),
            Transform::Bf16 => (reversed(&source.shape), BF16, Cow::Borrowed(source.data)),
            Transform::Rows(first, last) => {
                let Some(&rows) = source.shape.first().filter(|&&rows| last <= rows) else { return Err(mismatch(origin)) };
                let row = source.data.len() / rows;
                let mut shape = source.shape.clone();
                shape[0] = last - first;
                (reversed(&shape), BF16, Cow::Borrowed(&source.data[first * row..last * row]))
            }
            Transform::Experts(first, last) => {
                let (experts, rows, cols) = (source.shape[0], source.shape[1], source.shape[2]);
                // Q2_0 blocks are independent, so a few experts at a time give the same bytes as all at once.
                let mut quantized = Vec::new();
                for expert in (0..experts).step_by(group) {
                    let mut bits = Vec::with_capacity(group.min(experts - expert) * (last - first) * cols);
                    for index in expert..experts.min(expert + group) {
                        let at = (index * rows + first) * cols * 2;
                        bits.extend(bf16_values(&source.data[at..at + (last - first) * cols * 2]));
                    }
                    quantized.extend(quantize_q2_0(&bits)?);
                    done += bits.len() as u64;
                    progress(done, total);
                }
                (vec![cols, last - first, experts], Q2_0, Cow::Owned(quantized))
            }
        };
        entries.push(Entry { name, dims, kind, data });
    }
    write_gguf(target, revision, &entries)
}

fn push_string(header: &mut Vec<u8>, text: &str) {
    header.extend((text.len() as u64).to_le_bytes());
    header.extend(text.as_bytes());
}

/// One GGUF (version 3) of `entries` with the drafter's metadata and 32-byte alignment,
/// written to a partial file and then renamed into place.
fn write_gguf(target: &Path, revision: &str, entries: &[Entry]) -> io::Result<()> {
    let metadata = [("general.architecture", "qwen4exp"), ("straydog.drafter", "mtp"),
                    ("straydog.drafter.source_revision", revision)];
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((entries.len() as u64).to_le_bytes());
    header.extend((metadata.len() as u64 + 1).to_le_bytes());
    for (key, value) in metadata {
        push_string(&mut header, key);
        header.extend(8u32.to_le_bytes());
        push_string(&mut header, value);
    }
    push_string(&mut header, "general.alignment");
    header.extend(4u32.to_le_bytes());
    header.extend((ALIGNMENT as u32).to_le_bytes());
    let mut offset = 0;
    for entry in entries {
        push_string(&mut header, entry.name);
        header.extend((entry.dims.len() as u32).to_le_bytes());
        for &dim in &entry.dims {
            header.extend((dim as u64).to_le_bytes());
        }
        header.extend(entry.kind.to_le_bytes());
        header.extend((offset as u64).to_le_bytes());
        offset += entry.data.len().next_multiple_of(ALIGNMENT);
    }
    header.resize(header.len().next_multiple_of(ALIGNMENT), 0);
    let partial = partial_path(target);
    let mut output = BufWriter::new(File::create(&partial).map_err(at(&partial))?);
    output.write_all(&header)?;
    for entry in entries {
        output.write_all(&entry.data)?;
        output.write_all(&[0; ALIGNMENT][..entry.data.len().next_multiple_of(ALIGNMENT) - entry.data.len()])?;
    }
    output.into_inner().map_err(io::IntoInnerError::into_error)?;
    fs::rename(&partial, target).map_err(at(target))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::atomic::AtomicU64;

    use super::*;
    use crate::http;
    use crate::mock::{self, Scratch, Server};

    fn words(values: impl IntoIterator<Item = u16>) -> Vec<u8> {
        values.into_iter().flat_map(u16::to_le_bytes).collect()
    }

    /// A safetensors file of BF16 tensors in the given order.
    fn safetensors(tensors: &[(&str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let (mut header, mut offset) = (Map::new(), 0);
        for (name, shape, data) in tensors {
            let span = [offset, offset + data.len()];
            header.insert(name.to_string(), json!({"dtype": "BF16", "shape": shape, "data_offsets": span}));
            offset += data.len();
        }
        let blob = serde_json::to_vec(&header).unwrap();
        let data: Vec<u8> = tensors.iter().flat_map(|(_, _, data)| data.clone()).collect();
        [&(blob.len() as u64).to_le_bytes()[..], &blob, &data].concat()
    }

    /// One shard holding `payload` (one-dimensional tensors), served with ranges, and the pins.
    fn upstream(payload: &[(&str, Vec<u8>)]) -> (Server, Vec<SourceTensor>) {
        let tensors: Vec<_> = payload.iter().map(|(name, data)| (*name, vec![data.len() / 2], data.clone())).collect();
        let shard = safetensors(&tensors);
        let server = Server::start(move |request| mock::ranged(&shard, request.range.as_deref()));
        let pins = payload.iter().map(|(name, data)| SourceTensor {
            name: name.to_string(),
            shard: "model.safetensors".into(),
            bytes: data.len() as u64,
            sha256: format!("{:x}", Sha256::digest(data)),
        }).collect();
        (server, pins)
    }

    fn header(file: &[u8]) -> &str {
        let size = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
        std::str::from_utf8(&file[8..8 + size]).unwrap()
    }

    #[test]
    fn drafter_fetch_authenticates_each_tensor() {
        let (server, mut pins) = upstream(&[("mtp.b.weight", words([1; 4])), ("mtp.a.weight", words(0..8))]);
        let (agent, base) = (http::plain(TIMEOUT), format!("{}/", server.url));
        let scratch = Scratch::new("mtp-authenticate");
        let target = scratch.path().join("source.safetensors");
        fetch_source(&agent, &base, &pins, &target, PIECE, &|_| {}).unwrap();
        // The reference's header: compact JSON in name order, padded with spaces to eight bytes.
        let expected = r#"{"mtp.a.weight":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]},"mtp.b.weight":{"dtype":"BF16","shape":[4],"data_offsets":[16,24]}}"#;
        assert_eq!(header(&fs::read(&target).unwrap()), format!("{expected}{}", " ".repeat(2)));
        pins[0].sha256 = "0".repeat(64);
        let tampered = scratch.path().join("tampered.safetensors");
        let message = fetch_source(&agent, &base, &pins, &tampered, PIECE, &|_| {}).unwrap_err().to_string();
        assert_eq!(message, "MTP tensor failed authentication: mtp.b.weight");
        assert!(!tampered.exists() && !scratch.path().join("tampered.safetensors.part").exists());
        pins[0].bytes = 6;
        let message = fetch_source(&agent, &base, &pins, &tampered, PIECE, &|_| {}).unwrap_err().to_string();
        assert_eq!(message, "MTP tensor changed upstream: mtp.b.weight");
    }

    #[test]
    fn drafter_fetch_splits_tensors_into_pieces_on_several_connections() {
        let payload = [("mtp.a.weight", words(0..13)), ("mtp.b.weight", words([1; 4]))];
        let (server, pins) = upstream(&payload);
        let scratch = Scratch::new("mtp-pieces");
        let target = scratch.path().join("source.safetensors");
        let counted = AtomicU64::new(0);
        fetch_source(&http::plain(TIMEOUT), &format!("{}/", server.url), &pins, &target, 6, &|bytes| {
            counted.fetch_add(bytes, Ordering::Relaxed);
        }).unwrap();
        let written = fs::read(&target).unwrap();
        // The same layout a single sequential fetch writes: the header, then the tensors in name order.
        assert_eq!(&written[8 + header(&written).len()..], [payload[0].1.clone(), payload[1].1.clone()].concat());
        assert_eq!(counted.into_inner(), 34);
        let mut lengths: Vec<u64> = server.ranges()[2..].iter().map(|range| {
            let (first, last) = range.as_deref().unwrap().strip_prefix("bytes=").unwrap().split_once('-').unwrap();
            last.parse::<u64>().unwrap() + 1 - first.parse::<u64>().unwrap()
        }).collect();
        lengths.sort();
        assert_eq!(lengths, [2, 2, 6, 6, 6, 6, 6]);
        assert!(!scratch.path().join("source.safetensors.part").exists());
    }

    /// Every source tensor of the plan with arbitrary finite BF16 values: experts [3, 2560, 64],
    /// the indexer projection [640, 64], vectors [64] and matrices [8, 64].
    fn synthetic_source(path: &Path) {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut tensors: Vec<(&str, Vec<usize>, Vec<u8>)> = Vec::new();
        for (_, origin, transform) in PLAN {
            if tensors.iter().any(|(name, _, _)| *name == origin) {
                continue;
            }
            let shape = match transform {
                Transform::Experts(..) => vec![3, 2560, 64],
                Transform::Rows(..) => vec![640, 64],
                Transform::Norm | Transform::Vector => vec![64],
                Transform::Bf16 => vec![8, 64],
            };
            let values = (0..shape.iter().product::<usize>()).map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                (value.to_bits() >> 16) as u16
            });
            tensors.push((origin, shape, words(values)));
        }
        fs::write(path, safetensors(&tensors)).unwrap();
    }

    #[test]
    fn drafter_experts_quantized_in_groups_match_one_call() {
        let scratch = Scratch::new("mtp-convert");
        let source = scratch.path().join("source.safetensors");
        synthetic_source(&source);
        let total = 3 * 64 * (640 + 640 + 2560);
        let mut outputs = Vec::new();
        for (group, calls) in [(1, 9), (512, 3)] {
            let target = scratch.path().join(format!("drafter-{group}.gguf"));
            let steps = RefCell::new(Vec::new());
            convert(&source, &target, "r", group, &|done, total| steps.borrow_mut().push((done, total))).unwrap();
            let steps = steps.into_inner();
            assert_eq!((steps[0], *steps.last().unwrap(), steps.len() - 1), ((0, total), (total, total), calls));
            outputs.push(fs::read(&target).unwrap());
        }
        assert_eq!(outputs[0], outputs[1]);
        let drafter = straydog_gguf::inspect(&mut io::Cursor::new(&outputs[0])).unwrap();
        let names: Vec<&str> = drafter.tensors.iter().map(|tensor| tensor.name.as_str()).collect();
        assert_eq!(names, straydog_engine::weights::DRAFTER_TENSORS);
        let tensor = |name: &str| drafter.tensors.iter().find(|tensor| tensor.name == name).unwrap();
        assert_eq!((tensor("blk.48.ffn_up_exps.weight").dimensions.clone(), tensor("blk.48.ffn_up_exps.weight").quant_type),
                   (vec![64, 640, 3], Q2_0));
        assert_eq!(tensor("blk.48.indexer.k_proj.weight").dimensions, [64, 128]);
        assert_eq!(tensor("blk.48.attn_q.weight").dimensions, [64, 8]);
        // A norm is its BF16 widened, plus one.
        let map = fs::read(&source).unwrap();
        let tensors = source_tensors(&map).unwrap();
        let norm = tensor("mtp.embed_norm.weight");
        assert_eq!(norm.quant_type, F32);
        let start = norm.offset as usize;
        let written: Vec<f32> = outputs[0][start..start + 256].chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
        let expected: Vec<f32> = bf16_values(tensors["mtp.pre_fc_norm_embedding.weight"].data)
            .map(|bits| f32::from_bits(u32::from(bits) << 16) + 1.0).collect();
        assert_eq!(written, expected);
    }
}
