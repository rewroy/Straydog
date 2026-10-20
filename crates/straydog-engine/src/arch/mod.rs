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

    /// The check of every block read directly from [`files`](Self::files), before its use.
    fn verifier(&self) -> Option<crate::experts::Verifier> {
        None
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
