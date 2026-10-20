//! Model architectures (docs/DESIGN.md §10): each reads its hyperparameters and layer plan
//! from the GGUF metadata, so nothing about a model's shape is a constant of the engine.

pub mod deepseek41;

use straydog_gguf::{Metadata, ModelHeader};

/// A tensor's GGUF type, extents (innermost first) and bytes.
pub struct Tensor<'a> {
    pub kind: u32,
    pub dims: Vec<usize>,
    pub data: &'a [u8],
}

/// Where an engine reads its weights: the authenticated files, or a development reader.
pub trait Tensors {
    fn tensor(&self, name: &str) -> Result<Tensor<'_>, String>;

    /// Start reading these byte ranges of tensors (slices of their `data`) in the background,
    /// so a later access finds them in memory; the default reads nothing ahead.
    fn prefetch(&self, _ranges: &[&[u8]]) {}

    /// The files the tensors lie in, for reads that go around their mapping (routed experts,
    /// read directly into the expert store's tiers); none by default.
    fn files(&self) -> Result<Vec<std::fs::File>, String> {
        Ok(Vec::new())
    }

    /// Which of [`files`](Self::files) holds a tensor's bytes, and from which offset.
    fn location(&self, _name: &str) -> Option<(usize, u64)> {
        None
    }

    /// The check of every block read directly from [`files`](Self::files), before its use. No
    /// default: a source says explicitly that it reads unchecked (a development reader).
    fn verifier(&self) -> Option<crate::experts::Verifier>;

    /// Rows of a table (a 2-D tensor), `rows[i]`'s bytes at `i` times the row size; the
    /// default copies them from the tensor's data (a reader may serve them from a row cache,
    /// see `crate::rows`).
    fn rows(&self, name: &str, rows: &[u32]) -> Result<Vec<u8>, String> {
        let tensor = self.tensor(name)?;
        let width = row_size(&tensor, name)?;
        let mut out = Vec::with_capacity(rows.len() * width);
        for &row in rows {
            let start = row as usize * width;
            out.extend_from_slice(tensor.data.get(start..start + width).ok_or_else(|| format!("{name} has no row {row}"))?);
        }
        Ok(out)
    }

    /// Start reading rows of a table that [`Tensors::rows`] will be asked for; the default
    /// prefetches their bytes.
    fn prefetch_rows(&self, name: &str, rows: &[u32]) -> Result<(), String> {
        let tensor = self.tensor(name)?;
        let width = row_size(&tensor, name)?;
        let ranges: Vec<&[u8]> = rows.iter().filter_map(|&row| tensor.data.get(row as usize * width..(row as usize + 1) * width)).collect();
        self.prefetch(&ranges);
        Ok(())
    }
}

/// Bytes per row of a table tensor.
fn row_size(tensor: &Tensor, name: &str) -> Result<usize, String> {
    match tensor.dims[..] {
        [_, rows] if rows > 0 => Ok(tensor.data.len() / rows),
        _ => Err(format!("{name} is not a table of rows")),
    }
}

/// An architecture's metadata keys, `{architecture}.{key}`, with typed access.
pub struct Keys<'a> {
    header: &'a ModelHeader,
}

impl<'a> Keys<'a> {
    pub fn new(header: &'a ModelHeader) -> Keys<'a> {
        Keys { header }
    }

    pub fn raw(&self, key: &str) -> Option<&'a Metadata> {
        self.header.metadata.get(key)
    }

    fn get(&self, key: &str) -> Result<&'a Metadata, String> {
        let name = format!("{}.{key}", self.header.architecture);
        self.header.metadata.get(&name).ok_or_else(|| format!("GGUF metadata {name} is missing"))
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_ok()
    }

    pub fn integer(&self, key: &str) -> Result<i64, String> {
        self.get(key)?.integer().ok_or_else(|| format!("GGUF metadata {key} is not an integer"))
    }

    pub fn count(&self, key: &str) -> Result<usize, String> {
        usize::try_from(self.integer(key)?).map_err(|_| format!("GGUF metadata {key} is negative"))
    }

    pub fn float(&self, key: &str) -> Result<f64, String> {
        self.get(key)?.float().ok_or_else(|| format!("GGUF metadata {key} is not a number"))
    }

    pub fn flag(&self, key: &str) -> Result<bool, String> {
        match self.get(key)? {
            Metadata::Bool(value) => Ok(*value),
            _ => Err(format!("GGUF metadata {key} is not a boolean")),
        }
    }

    /// An integer array (as `i64`, which every count, layer index and hash parameter fits).
    pub fn integers(&self, key: &str) -> Result<Vec<i64>, String> {
        let items = self.get(key)?.array().ok_or_else(|| format!("GGUF metadata {key} is not an array"))?;
        items.iter().map(|item| item.integer().ok_or_else(|| format!("GGUF metadata {key} holds a non-integer"))).collect()
    }

    /// Unsigned 64-bit values kept exactly (hash multipliers can exceed `i64`).
    pub fn words(&self, key: &str) -> Result<Vec<u64>, String> {
        let items = self.get(key)?.array().ok_or_else(|| format!("GGUF metadata {key} is not an array"))?;
        items
            .iter()
            .map(|item| match item {
                Metadata::Unsigned(value) => Ok(*value),
                Metadata::Signed(value) if *value >= 0 => Ok(*value as u64),
                _ => Err(format!("GGUF metadata {key} holds a negative or non-integer value")),
            })
            .collect()
    }

    pub fn floats(&self, key: &str) -> Result<Vec<f64>, String> {
        let items = self.get(key)?.array().ok_or_else(|| format!("GGUF metadata {key} is not an array"))?;
        items.iter().map(|item| item.float().ok_or_else(|| format!("GGUF metadata {key} holds a non-number"))).collect()
    }

    /// A per-layer value given either once or as one entry per layer.
    pub fn per_layer(&self, key: &str, layers: usize) -> Result<Vec<f64>, String> {
        match self.get(key)? {
            Metadata::Array(_) => {
                let values = self.floats(key)?;
                if values.len() < layers {
                    return Err(format!("GGUF metadata {key} has fewer entries than layers"));
                }
                Ok(values)
            }
            _ => Ok(vec![self.float(key)?; layers]),
        }
    }
}
