//! The development weight reader shared by the DeepSeek examples: every GGUF shard of a
//! directory mapped and its tensors found by name, without authentication.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use straydog_engine::arch::{Tensor, Tensors};
use straydog_engine::rows::{ReadPool, RowTable, Verifier};
use straydog_gguf::{ModelHeader, TensorInfo};

pub struct Files {
    handles: Vec<File>,
    maps: Vec<Mmap>,
    tensors: HashMap<String, (usize, TensorInfo)>,
    rows: HashMap<String, RowTable>,
    pub header: ModelHeader,
}

impl Files {
    pub fn open(dir: &Path) -> Result<Files, String> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|exc| exc.to_string())?
            .flatten().map(|entry| entry.path()).filter(|path| path.extension().is_some_and(|ext| ext == "gguf")).collect();
        shards.sort();
        let (mut handles, mut maps, mut tensors, mut first) = (Vec::new(), Vec::new(), HashMap::new(), None);
        for (index, path) in shards.iter().enumerate() {
            let file = File::open(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
            // SAFETY: a read-only mapping of a file this process never writes.
            let map = unsafe { Mmap::map(&file) }.map_err(|exc| exc.to_string())?;
            let header = straydog_gguf::inspect(&mut std::io::Cursor::new(&map[..])).map_err(|exc| exc.to_string())?;
            for tensor in &header.tensors {
                tensors.insert(tensor.name.clone(), (index, tensor.clone()));
            }
            if header.shard_index == 0 {
                first = Some(header);
            }
            maps.push(map);
            handles.push(file);
        }
        Ok(Files { handles, maps, tensors, rows: HashMap::new(), header: first.ok_or("no first shard")? })
    }

    /// Read a table's rows through a row cache of `budget` bytes instead of the mapping, each
    /// read checked as the source's direct reads are ([`Tensors::verifier`]).
    #[allow(dead_code)]
    pub fn cache_rows(&mut self, name: &str, budget: u64) -> Result<(), String> {
        let (shard, info) = self.tensors.get(name).ok_or_else(|| format!("tensor {name} is missing"))?;
        let rows = *info.dimensions.get(1).ok_or_else(|| format!("{name} is not a table of rows"))?;
        let pool = Arc::new(ReadPool::new(straydog_engine::rows::THREADS));
        let shard = *shard;
        let verifier = self.verifier().map(|verify| -> Verifier { Arc::new(move |offset, data| verify(shard, offset, data)) });
        let table = RowTable::new(&self.handles[shard], info.offset, (info.bytes / rows) as usize, rows, budget, pool, verifier)
            .map_err(|exc| format!("{name}: {exc}"))?;
        self.rows.insert(name.to_string(), table);
        Ok(())
    }

    /// A row-cached table's (rows asked for, rows read, blocks read).
    #[allow(dead_code)]
    pub fn row_counts(&self, name: &str) -> Option<(u64, u64, u64)> {
        self.rows.get(name).map(RowTable::counts)
    }
}

impl Tensors for Files {
    fn prefetch(&self, ranges: &[&[u8]]) {
        if std::env::var_os("DS41_NO_PREFETCH").is_none() {
            straydog_engine::storage::read_ahead(ranges.iter().copied());
        }
    }

    /// Development only: direct reads go unchecked, as the mapping does.
    fn verifier(&self) -> Option<straydog_engine::experts::Verifier> {
        None
    }

    fn files(&self) -> Result<Vec<File>, String> {
        self.handles.iter().map(|file| file.try_clone().map_err(|exc| exc.to_string())).collect()
    }

    fn location(&self, name: &str) -> Option<(usize, u64)> {
        self.tensors.get(name).map(|(shard, info)| (*shard, info.offset))
    }

    fn rows(&self, name: &str, rows: &[u32]) -> Result<Vec<u8>, String> {
        match self.rows.get(name) {
            Some(table) => table.read(rows),
            None => {
                let tensor = self.tensor(name)?;
                let width = tensor.data.len() / tensor.dims[1];
                Ok(rows.iter().flat_map(|&row| &tensor.data[row as usize * width..(row as usize + 1) * width]).copied().collect())
            }
        }
    }

    fn prefetch_rows(&self, name: &str, rows: &[u32]) -> Result<(), String> {
        match self.rows.get(name) {
            Some(table) => table.prefetch(rows),
            None => {
                let tensor = self.tensor(name)?;
                let width = tensor.data.len() / tensor.dims[1];
                self.prefetch(&rows.iter().map(|&row| &tensor.data[row as usize * width..(row as usize + 1) * width]).collect::<Vec<_>>());
                Ok(())
            }
        }
    }

    fn tensor(&self, name: &str) -> Result<Tensor<'_>, String> {
        let (shard, info) = self.tensors.get(name).ok_or_else(|| format!("tensor {name} is missing"))?;
        let start = info.offset as usize;
        Ok(Tensor {
            kind: info.quant_type,
            dims: info.dimensions.iter().map(|&d| d as usize).collect(),
            data: &self.maps[*shard][start..start + info.bytes as usize],
        })
    }
}
