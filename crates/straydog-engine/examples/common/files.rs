//! The development weight reader shared by the DeepSeek examples: every GGUF shard of a
//! directory mapped and its tensors found by name, without authentication.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use straydog_engine::arch::{Tensor, Tensors};
use straydog_gguf::{ModelHeader, TensorInfo};

pub struct Files {
    maps: Vec<Mmap>,
    tensors: HashMap<String, (usize, TensorInfo)>,
    pub header: ModelHeader,
}

impl Files {
    pub fn open(dir: &Path) -> Result<Files, String> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|exc| exc.to_string())?
            .flatten().map(|entry| entry.path()).filter(|path| path.extension().is_some_and(|ext| ext == "gguf")).collect();
        shards.sort();
        let (mut maps, mut tensors, mut first) = (Vec::new(), HashMap::new(), None);
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
        }
        Ok(Files { maps, tensors, header: first.ok_or("no first shard")? })
    }
}

impl Tensors for Files {
    fn prefetch(&self, ranges: &[&[u8]]) {
        if std::env::var_os("DS41_NO_PREFETCH").is_some() {
            return;
        }
        const PAGE: usize = 4096;
        for range in ranges {
            let start = range.as_ptr() as usize / PAGE * PAGE;
            let end = (range.as_ptr() as usize + range.len()).div_ceil(PAGE) * PAGE;
            // SAFETY: an advice on pages of a live read-only mapping; it reads ahead, never writes.
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_WILLNEED) };
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
