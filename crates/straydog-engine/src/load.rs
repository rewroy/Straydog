//! Reading the weights once: every chunk of the pinned files is read into page-locked host
//! memory, hashed there, and only after its pinned digest matched copied to the GPU
//! allocations placement made for it and to the host copies the engine keeps.
//!
//! The bytes hashed are the bytes used, so nothing depends on the page cache between the
//! two, and reading, hashing and the copies of different chunks overlap: each core reads and
//! hashes chunks of its own while the GPUs take the verified ones. A start is as fast as the
//! slowest of the SSD (cold), SHA-256 on every core (warm) and the copies. What the engine
//! reads through the mapping at run time (the PLE table, partial residency's expert banks)
//! is read through the page cache, so that it stays there; the rest is read from the cache
//! where it is cached, and around it only when the files could not stay there (see
//! [`bypass_cache`]).

use std::collections::BTreeSet;
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::digest::CHUNK_BYTES;
use crate::gpu::{self, Filler, PageLock, Pending};
use crate::group::Group;
use crate::storage::{self, Buffer, CACHED_PIECE, PIECE};
use crate::weights::{Progress, Weights};

/// A chunk at least this cached is read through the page cache: memory is faster than the SSD.
const CACHED: f64 = 0.95;

/// Whether chunks not in the page cache are read past it (direct reads, Linux): only when the
/// files could not stay cached, as a model larger than three quarters of the host memory.
/// Both read the SSD as fast (4.7 GB/s for the IQ3_S bundle); direct reads cost no kernel copy
/// and evict nothing, but leave the next start to read the SSD again (15 s instead of 11).
fn bypass_cache(total: u64, memory: u64) -> bool {
    memory > 0 && total > memory / 4 * 3
}

/// Where a verified range of a chunk goes.
#[derive(Clone, Copy)]
enum Target {
    /// A device's pending upload, at an offset into it.
    Device { device: usize, upload: Pending, at: u64 },
    /// A host copy, at an offset into it.
    Host { copy: usize, at: usize },
}

/// A file range (offset, length) and its target.
type Range = (u64, u64, Target);

/// One file to read: its handles, pinned chunk digests and each chunk's pieces (offset into
/// the chunk, length, target).
struct Source<'a> {
    name: &'a str,
    file: &'a File,
    direct: Option<File>,
    map: &'a [u8],
    digests: &'a [String],
    /// Per chunk: whether the engine reads some of it through the mapping later.
    mapped: Vec<bool>,
    pieces: Vec<Vec<(usize, usize, Target)>>,
}

// The targets' uploads hold device addresses the fillers copy to; nothing dereferences them here.
unsafe impl Sync for Source<'_> {}

/// Authenticate every chunk of the weights' files, filling the pending uploads (one list per
/// device of `group`) and keeping host copies of the tensors in `keep` that no upload holds
/// whole; the weights are usable after.
pub fn load(weights: &mut Weights, group: &mut Group, pending: Vec<Vec<Pending>>, keep: &[String],
            progress: Progress) -> Result<(), String> {
    let mut ranges: Vec<Vec<Range>> = vec![Vec::new(); weights.file_count()];
    let uploaded = |name: &String| {
        let tensor = weights.tensor(name).ok();
        // (whole: a part's pieces, Data::Gather, are not a copy of the tensor)
        pending.iter().flatten().filter(|upload| upload.decode.is_none() && upload.offset == 0 && upload.columns.is_none()).any(|upload| {
            tensor.is_some_and(|tensor| {
                weights.locate(upload.source, upload.bytes) == Some((tensor.file, tensor.offset))
                    && upload.bytes as u64 == tensor.bytes
            })
        })
    };
    let mut kept: BTreeSet<String> = keep.iter().filter(|name| !uploaded(name)).cloned().collect();
    // Vectors are uploaded decoded to F32: from host copies, once those are verified.
    let mut decodes = Vec::new();
    for (device, uploads) in pending.iter().enumerate() {
        for upload in uploads {
            let (file, offset) = weights.locate(upload.source, upload.bytes).ok_or("Straydog upload is not mapped")?;
            match upload.decode {
                Some(kind) => {
                    let name = weights.tensor_at(file, offset)?.to_string();
                    let within = offset - weights.tensor(&name)?.offset;
                    kept.insert(name.clone());
                    decodes.push((device, *upload, kind, name, within));
                }
                None => ranges[file].push((offset, upload.bytes as u64, Target::Device { device, upload: *upload, at: 0 })),
            }
        }
    }
    let copies = read(weights, ranges, &kept, progress, |workers| {
        let workers = (0..workers)
            .map(|_| {
                let fillers = group.backends.iter().map(|backend| backend.filler()).collect::<Result<Vec<_>, _>>()?;
                Ok(Devices { fillers, lock: None })
            })
            .collect();
        // Host K/V page-locks beside the reading, once the readers' fillers exist: its thread
        // holds up the devices' other calls meanwhile.
        group.prefetch_host_states()?;
        workers
    })?;
    keep_copies(weights, copies);
    for (device, upload, kind, name, within) in decodes {
        // (the upload's bytes of the tensor, a part's rows or, with columns, a run of each row: whole blocks)
        let (bytes, _) = weights.raw(&name, 0, None)?;
        let Some(source) = bytes.get(within as usize..within as usize + upload.bytes) else {
            return Err(format!("Straydog upload is not within its tensor: {name}"));
        };
        let gathered;
        let source = match upload.columns {
            None => source,
            Some(columns) if columns.pitch > 0 && (source.len() as u64).is_multiple_of(columns.pitch)
                              && columns.first + columns.width <= columns.pitch => {
                let mut packed = vec![0u8; (source.len() as u64 / columns.pitch * columns.width) as usize];
                gpu::Columns { to_pitch: columns.width, ..columns }.gather(source, &mut packed);
                gathered = packed;
                &gathered[..]
            }
            Some(_) => return Err(format!("Straydog upload columns do not fit {name}")),
        };
        let tensor = weights.tensor(&name)?;
        let elements = source.len() as u64 * tensor.elements() / tensor.bytes;
        if elements * tensor.bytes != source.len() as u64 * tensor.elements() {
            return Err(format!("Straydog upload is not whole blocks of {name}"));
        }
        let values = gpu::decode_vector(source, kind, elements as usize, &name)?;
        let converted: Vec<u8> = if upload.quantize { crate::formats::quantize_q8_0(&values) }
                                 else { values.iter().flat_map(|value| value.to_le_bytes()).collect() };
        group.backends[device].write(&upload, &converted)?;
    }
    Ok(())
}

/// [`load`] without GPUs: authenticate every chunk, keeping host copies of `keep`.
pub fn authenticate(weights: &mut Weights, keep: &[String], progress: Progress) -> Result<(), String> {
    let ranges = vec![Vec::new(); weights.file_count()];
    let kept: BTreeSet<String> = keep.iter().cloned().collect();
    let copies = read(weights, ranges, &kept, progress, |workers| Ok((0..workers).map(|_| Host).collect()))?;
    keep_copies(weights, copies);
    Ok(())
}

type Copies = Vec<(String, Mutex<Vec<u8>>)>;

/// Read every chunk into `ranges` and fresh host copies of `kept`, a worker per core.
fn read<W: Worker>(weights: &Weights, mut ranges: Vec<Vec<Range>>, kept: &BTreeSet<String>, progress: Progress,
                   workers: impl FnOnce(usize) -> Result<Vec<W>, String>) -> Result<Copies, String> {
    let mut copies = Vec::new();
    for name in kept {
        let tensor = weights.tensor(name)?;
        ranges[tensor.file].push((tensor.offset, tensor.bytes, Target::Host { copy: copies.len(), at: 0 }));
        copies.push((name.clone(), Mutex::new(vec![0u8; tensor.bytes as usize])));
    }
    let sources = sources(weights, &ranges)?;
    let total: u64 = sources.iter().map(|source| source.map.len() as u64).sum();
    let chunks: usize = sources.iter().map(|source| source.digests.len()).sum();
    let workers = workers(std::thread::available_parallelism().map_or(4, |count| count.get()).min(chunks).max(1))?;
    progress("load", 0, total);
    stream(&sources, workers, &copies, total, progress)?;
    Ok(copies)
}

fn keep_copies(weights: &mut Weights, copies: Copies) {
    for (name, copy) in copies {
        weights.keep(name, copy.into_inner().unwrap());
    }
    weights.set_authenticated();
}

/// What a reader does with its slot and with the verified pieces bound for a device.
trait Worker: Send {
    fn start(&mut self, slot: &Buffer) -> Result<(), String>;
    fn fill(&self, device: usize, upload: &Pending, offset: u64, bytes: &[u8]) -> Result<(), String>;
}

/// A reader filling device allocations: a filler per device, DMA from its page-locked slot.
struct Devices<'a> {
    fillers: Vec<Filler<'a>>,
    lock: Option<PageLock>,
}

impl Worker for Devices<'_> {
    fn start(&mut self, slot: &Buffer) -> Result<(), String> {
        self.lock = Some(PageLock::new(slot.as_slice())?);
        Ok(())
    }

    fn fill(&self, device: usize, upload: &Pending, offset: u64, bytes: &[u8]) -> Result<(), String> {
        self.fillers[device].fill(upload, offset, bytes)
    }
}

/// A reader without devices (authentication only).
struct Host;

impl Worker for Host {
    fn start(&mut self, _slot: &Buffer) -> Result<(), String> {
        Ok(())
    }

    fn fill(&self, _device: usize, _upload: &Pending, _offset: u64, _bytes: &[u8]) -> Result<(), String> {
        Err("Straydog has no device to fill".into())
    }
}

/// Check a whole file against chunk digests, reading it as loading does (past the page cache
/// where it is not cached), on every core.
pub fn verify_file(name: &str, file: &File, digests: &[String]) -> Result<(), String> {
    // SAFETY: a read-only mapping, used here only to probe the page cache.
    let map = unsafe { memmap2::Mmap::map(file) }.map_err(|exc| exc.to_string())?;
    let chunks = (map.len() as u64).div_ceil(CHUNK_BYTES) as usize;
    if digests.len() != chunks {
        return Err(format!("Straydog chunk digests do not match {name}"));
    }
    let direct = storage::direct(file);
    let source = Source { name, file, direct, map: &map, digests, mapped: vec![false; chunks],
                          pieces: vec![Vec::new(); chunks] };
    let workers = std::thread::available_parallelism().map_or(4, |count| count.get()).min(chunks).max(1);
    stream(std::slice::from_ref(&source), (0..workers).map(|_| Host).collect(), &[], map.len() as u64, &|_, _, _| {})
}

/// Each file's chunks with the targets of their ranges.
fn sources<'a>(weights: &'a Weights, ranges: &[Vec<Range>]) -> Result<Vec<Source<'a>>, String> {
    let files = weights.files_to_read();
    if ranges[files.len()..].iter().any(|ranges| !ranges.is_empty()) {
        return Err("Straydog places weights of the drafter it dropped".into());
    }
    let total = files.iter().map(|(_, _, map, _)| map.len() as u64).sum();
    let bypass = bypass_cache(total, weights.host_memory_bytes);
    let mut sources = Vec::new();
    for (index, (name, file, map, digests)) in files.into_iter().enumerate() {
        let chunks = (map.len() as u64).div_ceil(CHUNK_BYTES) as usize;
        if digests.len() != chunks {
            return Err(format!("Straydog chunk digests do not match {name}"));
        }
        let mut pieces = vec![Vec::new(); chunks];
        for &(offset, length, target) in &ranges[index] {
            let mut done = 0;
            while done < length {
                let at = offset + done;
                let (chunk, within) = ((at / CHUNK_BYTES) as usize, at % CHUNK_BYTES);
                let part = (CHUNK_BYTES - within).min(length - done);
                let target = match target {
                    Target::Device { device, upload, .. } => Target::Device { device, upload, at: done },
                    Target::Host { copy, .. } => Target::Host { copy, at: done as usize },
                };
                pieces[chunk].push((within as usize, part as usize, target));
                done += part;
            }
        }
        // Tensors no range reads stay mapped: the chunks they touch go through the page cache.
        let mut mapped = vec![false; chunks];
        for (offset, length) in weights.tensor_ranges(index) {
            let read = ranges[index].iter().any(|&(start, bytes, _)| start < offset + length && offset < start + bytes);
            if !read && length > 0 {
                mapped[(offset / CHUNK_BYTES) as usize..=((offset + length - 1) / CHUNK_BYTES) as usize].fill(true);
            }
        }
        let direct = if bypass { storage::direct(file) } else { None };
        sources.push(Source { name, file, direct, map, digests, mapped, pieces });
    }
    Ok(sources)
}

/// Read, verify and deliver every chunk of `sources`, each worker taking the next chunk: first
/// those with nothing for the GPUs (the host K/V page-locking meanwhile holds up the devices'
/// copies, see `Backend::prefetch_host_states`), then the rest, each in file order.
fn stream<W: Worker>(sources: &[Source], workers: Vec<W>, copies: &[(String, Mutex<Vec<u8>>)], total: u64,
                     progress: Progress) -> Result<(), String> {
    let mut chunks: Vec<(usize, usize)> = sources
        .iter()
        .enumerate()
        .flat_map(|(source, file)| (0..file.digests.len()).map(move |chunk| (source, chunk)))
        .collect();
    chunks.sort_by_key(|&(source, chunk)| {
        sources[source].pieces[chunk].iter().any(|piece| matches!(piece.2, Target::Device { .. }))
    });
    let (next, done, failed) = (AtomicUsize::new(0), AtomicU64::new(0), AtomicBool::new(false));
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let count = |bytes: u64| progress("load", done.fetch_add(bytes, Ordering::Relaxed) + bytes, total);
    std::thread::scope(|scope| {
        for mut worker in workers {
            let (chunks, next, failed, failure, count) = (&chunks, &next, &failed, &failure, &count);
            scope.spawn(move || {
                let mut slot = Buffer::new(CHUNK_BYTES as usize);
                let result = worker.start(&slot).and_then(|()| {
                    while let Some(&(source, chunk)) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if failed.load(Ordering::Relaxed) {
                            break;
                        }
                        let source = &sources[source];
                        let length = read_chunk(source, chunk, &mut slot)?;
                        count(length as u64);
                        let bytes = &slot.as_slice()[..length];
                        for &(at, part, target) in &source.pieces[chunk] {
                            match target {
                                Target::Device { device, upload, at: offset } => {
                                    worker.fill(device, &upload, offset, &bytes[at..at + part])?
                                }
                                Target::Host { copy, at: offset } => {
                                    copies[copy].1.lock().unwrap()[offset..offset + part]
                                        .copy_from_slice(&bytes[at..at + part])
                                }
                            }
                        }
                    }
                    Ok(())
                });
                // The worker (its page lock) goes before the slot.
                drop(worker);
                if let Err(message) = result {
                    failed.store(true, Ordering::Relaxed);
                    failure.lock().unwrap().get_or_insert(message);
                }
            });
        }
    });
    failure.into_inner().unwrap().map_or(Ok(()), Err)
}

/// Read one chunk into `slot`, a piece at a time, hashing each piece as it arrives; check it
/// against the pinned digest. The chunk's length.
fn read_chunk(source: &Source, chunk: usize, slot: &mut Buffer) -> Result<usize, String> {
    let offset = chunk as u64 * CHUNK_BYTES;
    let length = CHUNK_BYTES.min(source.map.len() as u64 - offset) as usize;
    let cached = storage::resident(&source.map[offset as usize..offset as usize + length]) >= CACHED;
    let direct = source.direct.as_ref().filter(|_| !source.mapped[chunk] && !cached);
    if direct.is_none() && !cached {
        // Read ahead the whole chunk: one read waits on a 128 KiB read-ahead window at a time.
        crate::digest::will_need(source.file, &[(offset, length as u64)]);
    }
    let mut digest = Sha256::new();
    let mut done = 0;
    while done < length {
        let part = if direct.is_some() { PIECE } else { CACHED_PIECE }.min(length - done);
        storage::read_into(direct.unwrap_or(source.file), offset + done as u64, slot, done, part, direct.is_some())
            .map_err(|exc| format!("{}: {exc}", source.name))?;
        digest.update(&slot.as_slice()[done..done + part]);
        done += part;
    }
    if format!("{:x}", digest.finalize()) != source.digests[chunk] {
        return Err(format!("Straydog asset digest mismatch: {}", source.name));
    }
    Ok(length)
}
