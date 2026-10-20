use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Seek, SeekFrom};

// Bounds are independent of any lengths claimed by the file. Tokenizer arrays
// are large, so bound total bytes AND element count before iterating them.
const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;
const MAX_STRING_BYTES: u64 = 1024 * 1024;
const MAX_METADATA: u64 = 65_536;
const MAX_ARRAY_ITEMS: u64 = 1_048_576;
/// Array elements kept across the whole header: a one-byte element becomes a value of some
/// 32 bytes, so the input bound alone would allow gigabytes (tokenizers use under 1M).
const MAX_KEPT_ITEMS: u64 = 4_194_304;
const MAX_TENSORS: u64 = 16_384;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub quant_type: u32,
    /// Absolute offset within this shard, never a path supplied by metadata.
    pub offset: u64,
    pub bytes: u64,
}

/// A metadata value as the file stores it (arrays hold one element type, never arrays).
#[derive(Debug, Clone, PartialEq)]
pub enum Metadata {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Array(Vec<Metadata>),
}

impl Metadata {
    /// An integer of either sign that fits `i64`.
    pub fn integer(&self) -> Option<i64> {
        match self {
            Metadata::Unsigned(value) => i64::try_from(*value).ok(),
            Metadata::Signed(value) => Some(*value),
            _ => None,
        }
    }

    pub fn float(&self) -> Option<f64> {
        match self {
            Metadata::Float(value) => Some(*value),
            _ => self.integer().map(|value| value as f64),
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Metadata::Text(value) => Some(value),
            _ => None,
        }
    }

    pub fn array(&self) -> Option<&[Metadata]> {
        match self {
            Metadata::Array(items) => Some(items),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelHeader {
    /// `general.architecture`; which ones are supported is the engine's decision.
    pub architecture: String,
    /// GGUF declaration only. This is NOT an observed running context capacity.
    pub declared_context: Option<u64>,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_bytes: u64,
    /// Zero-based shard number, as encoded by GGUF split.no.
    pub shard_index: u64,
    pub shard_count: u64,
    pub tensors: Vec<TensorInfo>,
    /// Small architecture-specific integer arrays, preserving exact PLE hashes.
    pub integer_arrays: BTreeMap<String, Vec<i64>>,
    /// Every metadata key, within the header bounds above.
    pub metadata: BTreeMap<String, Metadata>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

struct Input<'a, R> {
    reader: &'a mut R,
    pos: u64,
    file_bytes: u64,
    kept_items: u64,
}

impl<R: Read + Seek> Input<'_, R> {
    fn claim(&mut self, bytes: u64) -> io::Result<()> {
        let end = self
            .pos
            .checked_add(bytes)
            .ok_or_else(|| invalid("header overflow"))?;
        if end > self.file_bytes || end > MAX_HEADER_BYTES {
            return Err(invalid("truncated or oversized GGUF header"));
        }
        self.pos = end;
        Ok(())
    }

    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        self.claim(N as u64)?;
        let mut out = [0; N];
        self.reader.read_exact(&mut out)?;
        Ok(out)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn string(&mut self, limit: u64) -> io::Result<String> {
        let len = self.u64()?;
        if len > limit {
            return Err(invalid("GGUF string exceeds limit"));
        }
        self.claim(len)?;
        let mut bytes = vec![0; len as usize];
        self.reader.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| invalid("GGUF string is not UTF-8"))
    }

    fn value(&mut self, kind: u32) -> io::Result<Metadata> {
        match kind {
            0 => Ok(Metadata::Unsigned(u8::from_le_bytes(self.bytes()?) as u64)),
            2 => Ok(Metadata::Unsigned(u16::from_le_bytes(self.bytes()?) as u64)),
            4 => Ok(Metadata::Unsigned(self.u32()? as u64)),
            10 => Ok(Metadata::Unsigned(self.u64()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => Ok(Metadata::Bool(false)),
                1 => Ok(Metadata::Bool(true)),
                _ => Err(invalid("invalid GGUF boolean")),
            },
            8 => Ok(Metadata::Text(self.string(MAX_STRING_BYTES)?)),
            9 => {
                let element_kind = self.u32()?;
                let count = self.u64()?;
                self.kept_items = self.kept_items.saturating_add(count);
                if count > MAX_ARRAY_ITEMS || self.kept_items > MAX_KEPT_ITEMS || element_kind > 12 || element_kind == 9 {
                    return Err(invalid("invalid, nested, or oversized GGUF array"));
                }
                // One element at a time: each is read (and so bounded by the header's bytes)
                // before the next is allocated.
                let mut items = Vec::new();
                for _ in 0..count {
                    items.push(self.value(element_kind)?);
                }
                Ok(Metadata::Array(items))
            }
            1 => Ok(Metadata::Signed(i8::from_le_bytes(self.bytes()?) as i64)),
            3 => Ok(Metadata::Signed(i16::from_le_bytes(self.bytes()?) as i64)),
            5 => Ok(Metadata::Signed(i32::from_le_bytes(self.bytes()?) as i64)),
            6 => Ok(Metadata::Float(f32::from_le_bytes(self.bytes()?) as f64)),
            11 => Ok(Metadata::Signed(i64::from_le_bytes(self.bytes()?))),
            12 => Ok(Metadata::Float(f64::from_le_bytes(self.bytes()?))),
            _ => Err(invalid("unknown GGUF metadata type")),
        }
    }
}

fn unsigned(metadata: &BTreeMap<String, Metadata>, key: &str) -> io::Result<Option<u64>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(Metadata::Unsigned(value)) => Ok(Some(*value)),
        _ => Err(invalid(format!("{key} must be an unsigned integer"))),
    }
}

// GGML block geometry is required to check byte ranges, including mixed dense
// tensors in low-bit models. This does not implement any dequantization kernel.
// Q2_0 (type 42) is the GSQ-RCO extension: 64 elements in 18 bytes.
fn geometry(kind: u32) -> io::Result<(u64, u64)> {
    match kind {
        0 => Ok((1, 4)),
        1 | 30 => Ok((1, 2)),
        2 | 20 => Ok((32, 18)),
        3 => Ok((32, 20)),
        6 => Ok((32, 22)),
        7 => Ok((32, 24)),
        8 => Ok((32, 34)),
        10 => Ok((256, 84)),
        11 => Ok((256, 110)),
        12 => Ok((256, 144)),
        13 => Ok((256, 176)),
        14 => Ok((256, 210)),
        16 => Ok((256, 66)),
        17 => Ok((256, 74)),
        18 => Ok((256, 98)),
        19 => Ok((256, 50)),
        21 => Ok((256, 110)),
        22 => Ok((256, 82)),
        23 => Ok((256, 136)),
        29 => Ok((256, 56)),
        39 => Ok((32, 17)),
        42 => Ok((64, 18)),
        _ => Err(invalid(format!("unsupported tensor type {kind}"))),
    }
}

/// Inspect one little-endian GGUF v3 shard without mapping or loading weights.
///
/// Rejects unbounded counts, duplicate names/keys, unsupported formats, invalid
/// alignment, arithmetic overflow, overlapping tensors, and out-of-file ranges.
/// Any well-formed architecture name is reported; whether it runs is the engine's
/// decision. Identity and cross-shard consistency still need a pinned manifest
/// before any execution is permitted.
pub fn inspect<R: Read + Seek>(reader: &mut R) -> io::Result<ModelHeader> {
    let file_bytes = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    let mut input = Input {
        reader,
        pos: 0,
        file_bytes,
        kept_items: 0,
    };
    if input.bytes::<4>()? != *b"GGUF" || input.u32()? != 3 {
        return Err(invalid("expected little-endian GGUF version 3"));
    }
    let tensor_count = input.u64()?;
    let metadata_count = input.u64()?;
    // A split model's first shard may hold only the metadata.
    if tensor_count > MAX_TENSORS || metadata_count > MAX_METADATA {
        return Err(invalid("invalid GGUF tensor or metadata count"));
    }
    let mut metadata = BTreeMap::new();
    let mut integer_arrays = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = input.string(65_535)?;
        if key.is_empty() || key.contains('\0') {
            return Err(invalid("invalid GGUF metadata key"));
        }
        let kind = input.u32()?;
        let value = if kind == 9 && key.starts_with("qwen4exp.ple.") {
            let element_kind = input.u32()?;
            let count = input.u64()?;
            if count > 4096 || !matches!(element_kind, 0..=5 | 10 | 11) {
                return Err(invalid("invalid PLE integer array"));
            }
            let mut numbers = Vec::new();
            for _ in 0..count {
                numbers.push(match input.value(element_kind)? {
                    Metadata::Unsigned(value) => {
                        i64::try_from(value).map_err(|_| invalid("PLE integer overflow"))?
                    }
                    Metadata::Signed(value) => value,
                    _ => return Err(invalid("invalid PLE integer")),
                });
            }
            integer_arrays.insert(key.clone(), numbers.clone());
            Metadata::Array(numbers.into_iter().map(Metadata::Signed).collect())
        } else {
            input.value(kind)?
        };
        if metadata.insert(key, value).is_some() {
            return Err(invalid("duplicate GGUF metadata key"));
        }
    }
    let architecture = match metadata.get("general.architecture") {
        Some(Metadata::Text(value))
            if !value.is_empty()
                && value.len() <= 64
                && value.bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_') =>
        {
            value.clone()
        }
        None => String::new(),
        _ => return Err(invalid("invalid GGUF architecture name")),
    };
    let declared_context = if architecture.is_empty() {
        None
    } else {
        unsigned(&metadata, &format!("{architecture}.context_length"))?
    };
    if declared_context == Some(0) {
        return Err(invalid("declared context must be positive"));
    }
    let alignment = unsigned(&metadata, "general.alignment")?.unwrap_or(32);
    if alignment == 0 || alignment > 4096 || !alignment.is_power_of_two() {
        return Err(invalid("invalid GGUF alignment"));
    }
    let shard_count = unsigned(&metadata, "split.count")?.unwrap_or(1);
    let shard_index = unsigned(&metadata, "split.no")?.unwrap_or(0);
    if shard_count == 0 || shard_count > 1024 || shard_index >= shard_count {
        return Err(invalid("invalid GGUF shard numbering"));
    }
    if shard_count > 1 && !metadata.contains_key("split.no") {
        return Err(invalid("split GGUF is missing its shard number"));
    }
    if architecture.is_empty() && shard_index == 0 {
        return Err(invalid("the first GGUF shard must declare its architecture"));
    }
    let mut names = BTreeSet::new();
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = input.string(256)?;
        if name.is_empty() || name.contains('\0') || !names.insert(name.clone()) {
            return Err(invalid("invalid or duplicate tensor name"));
        }
        let rank = input.u32()?;
        if !(1..=4).contains(&rank) {
            return Err(invalid("tensor rank must be between 1 and 4"));
        }
        let mut dimensions = Vec::new();
        let mut elements = 1u64;
        for _ in 0..rank {
            let dim = input.u64()?;
            if dim == 0 {
                return Err(invalid("zero tensor dimension"));
            }
            elements = elements
                .checked_mul(dim)
                .ok_or_else(|| invalid("tensor shape overflow"))?;
            dimensions.push(dim);
        }
        let quant_type = input.u32()?;
        let (block_elements, block_bytes) = geometry(quant_type)?;
        if dimensions[0] % block_elements != 0 {
            return Err(invalid(
                "tensor row does not contain whole quantization blocks",
            ));
        }
        let bytes = (elements / block_elements)
            .checked_mul(block_bytes)
            .ok_or_else(|| invalid("tensor byte size overflow"))?;
        let offset = input.u64()?;
        if offset % alignment != 0 {
            return Err(invalid("unaligned tensor offset"));
        }
        tensors.push(TensorInfo {
            name,
            dimensions,
            quant_type,
            offset,
            bytes,
        });
    }
    let data_offset = input
        .pos
        .checked_add(alignment - 1)
        .ok_or_else(|| invalid("data offset overflow"))?
        & !(alignment - 1);
    let mut ranges = Vec::new();
    for tensor in &mut tensors {
        tensor.offset = data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        let end = tensor
            .offset
            .checked_add(tensor.bytes)
            .ok_or_else(|| invalid("tensor end overflow"))?;
        if end > file_bytes {
            return Err(invalid("tensor extends past end of shard"));
        }
        ranges.push((tensor.offset, end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(invalid("overlapping tensor data"));
    }
    Ok(ModelHeader {
        architecture,
        declared_context,
        alignment,
        data_offset,
        file_bytes,
        shard_index,
        shard_count,
        tensors,
        integer_arrays,
        metadata,
    })
}
