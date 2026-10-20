//! Rows of tables too large for memory (embedding tables of tens to hundreds of GB: the PLE
//! table, Engram tables), read on demand into a bounded cache of our own.
//!
//! A pass needs a few rows per token (16 of 90 B, 48 of 176 B), scattered over the table, and
//! many recur: in agent sessions and chats 70% of row uses repeat a row seen before. A 4 KiB
//! page holds dozens of rows of which one is wanted, so caching rows rather than pages makes
//! memory go 30-45 times further (64 MB of rows hit as often as 2 GB of page cache). Misses
//! are deduplicated and read together, as soon as their ids are known, on a pool of threads
//! with many requests in flight: a token's 16 cold rows take 0.4 ms instead of 4.4 ms of page
//! faults in turn, and the reads of a prompt's next batch hide behind the batch before it.
//! Reads bypass the page cache where the platform allows (O_DIRECT), so rows are not cached
//! twice; blocks the page cache already holds (a table read at load, say) are read through it
//! instead, without touching the drive.
//!
//! Every read is of whole, aligned 4 KiB blocks of the file: the unit a Merkle tree over the
//! file (fs-verity's format, [`crate::verity`]) hashes, so a verifier checks each block between
//! its read and the rows' use.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use memmap2::Mmap;

use crate::storage::{self, Buffer, ALIGN};

/// The default cache budget across a model's row tables.
pub const DEFAULT_CACHE_BYTES: u64 = 256 << 20;
/// Reads in flight: the device needs dozens to reach its random-read rate.
pub const THREADS: usize = 32;
/// The most a read covers when rows lie in neighbouring blocks.
const EXTENT: u64 = 64 << 10;
/// Index and recency bytes per cached row, counted against the budget.
const OVERHEAD: u64 = 24;
const EMPTY: u32 = u32::MAX;

/// Checks a read of the table's file at `offset` (whole aligned blocks, zero past the file's
/// end) before its rows are used.
pub type Verifier = Arc<dyn Fn(u64, &[u8]) -> Result<(), String> + Send + Sync>;

type Job = Box<dyn FnOnce(&mut Buffer) + Send>;
/// Jobs waiting, whether the pool is closing, and the signal of either.
type Queue = Arc<(Mutex<(VecDeque<Job>, bool)>, Condvar)>;

/// Threads that run reads, each with an aligned buffer of its own.
pub struct ReadPool {
    queue: Queue,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl ReadPool {
    pub fn new(threads: usize) -> ReadPool {
        let queue: Queue = Arc::default();
        let workers = (0..threads.max(1))
            .map(|_| {
                let queue = queue.clone();
                std::thread::spawn(move || {
                    let mut buffer = Buffer::new(EXTENT as usize);
                    loop {
                        let job = {
                            let mut state = queue.0.lock().unwrap();
                            loop {
                                if let Some(job) = state.0.pop_front() {
                                    break job;
                                }
                                if state.1 {
                                    return;
                                }
                                state = queue.1.wait(state).unwrap();
                            }
                        };
                        job(&mut buffer);
                    }
                })
            })
            .collect();
        ReadPool { queue, workers }
    }

    fn submit(&self, jobs: Vec<Job>) {
        if jobs.is_empty() {
            return;
        }
        self.queue.0.lock().unwrap().0.extend(jobs);
        self.queue.1.notify_all();
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        self.queue.0.lock().unwrap().1 = true;
        self.queue.1.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// The cached rows: fixed slots, an index, and CLOCK recency (a slot's bit is set by use and
/// cleared as the hand passes; the hand evicts the first slot found clear).
struct Cache {
    slots: Vec<u8>,
    owner: Vec<u32>,
    used: Vec<bool>,
    index: HashMap<u32, u32>,
    hand: usize,
    /// Rows being read, by the batch reading them; each batch's reads still running (a reader
    /// waits for its rows' batches, woken as each batch completes); a read's failure not yet
    /// reported.
    pending: HashMap<u32, u64>,
    batches: HashMap<u64, usize>,
    next_batch: u64,
    failure: Option<String>,
}

impl Cache {
    fn insert(&mut self, row: u32, bytes: &[u8]) {
        let width = bytes.len();
        if self.owner.is_empty() || self.index.contains_key(&row) {
            return;
        }
        let slot = loop {
            let slot = self.hand;
            self.hand = (self.hand + 1) % self.owner.len();
            if self.used[slot] {
                self.used[slot] = false;
            } else {
                break slot;
            }
        };
        if self.owner[slot] != EMPTY {
            self.index.remove(&self.owner[slot]);
        }
        self.owner[slot] = row;
        self.used[slot] = true;
        self.index.insert(row, slot as u32);
        self.slots[slot * width..(slot + 1) * width].copy_from_slice(bytes);
    }

    fn get(&mut self, row: u32, width: usize) -> Option<&[u8]> {
        let slot = *self.index.get(&row)? as usize;
        self.used[slot] = true;
        Some(&self.slots[slot * width..(slot + 1) * width])
    }
}

/// Rows asked for, rows read from the file, and the blocks read from the drive for them (not
/// those the page cache held).
#[derive(Default)]
struct Counts {
    requested: AtomicU64,
    read: AtomicU64,
    blocks: AtomicU64,
}

struct Table {
    file: File,
    /// A handle reading past the page cache, and a mapping that tells which blocks the page
    /// cache holds (those are read through it, without touching the drive).
    direct: Option<(File, Mmap)>,
    /// File offset of row 0, bytes per row, rows.
    base: u64,
    width: usize,
    rows: u64,
    size: u64,
    cache: Mutex<Cache>,
    arrived: Condvar,
    counts: Counts,
    verifier: Option<Verifier>,
}

impl Table {
    /// The aligned block range covering a row.
    fn blocks(&self, row: u32) -> (u64, u64) {
        let start = self.base + row as u64 * self.width as u64;
        (start / ALIGN as u64 * ALIGN as u64, (start + self.width as u64).next_multiple_of(ALIGN as u64))
    }

    /// Read `[start, end)` (aligned blocks, the last possibly cut by the end of the file) into
    /// `buffer`, grown if a row spans more, and verify it.
    fn read(&self, start: u64, end: u64, buffer: &mut Buffer) -> Result<(), String> {
        let span = (end - start) as usize;
        if buffer.len() < span {
            *buffer = Buffer::new(span);
        }
        let length = (end.min(self.size) - start) as usize;
        let read = match &self.direct {
            Some((_, map)) if storage::resident(&map[start as usize..start as usize + length]) >= 1.0 => {
                storage::read_into(&self.file, start, buffer, 0, length, false)
            }
            direct => {
                self.counts.blocks.fetch_add(length.div_ceil(ALIGN) as u64, Ordering::Relaxed);
                let file = direct.as_ref().map_or(&self.file, |(file, _)| file);
                storage::read_into(file, start, buffer, 0, length, direct.is_some())
            }
        };
        read.map_err(|exc| exc.to_string())?;
        if let Some(verify) = &self.verifier {
            // The last block past the file's end is zero, as its tree hashed it.
            let blocks = &mut buffer.as_mut_slice()[..span];
            blocks[length..].fill(0);
            verify(start, blocks)?;
        }
        Ok(())
    }

    /// One read job of `batch`: the extent, then every row it covers into the cache.
    fn finish(&self, batch: u64, start: u64, end: u64, rows: &[u32], buffer: &mut Buffer) {
        // A read that panics (in its verifier, say) fails like one that errs: no reader waits on
        // its batch forever, and the thread lives on.
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.read(start, end, buffer)))
            .unwrap_or_else(|_| Err("a row read panicked".into()));
        let mut cache = self.cache.lock().unwrap();
        match read {
            Ok(()) => {
                for &row in rows {
                    let at = (self.base + row as u64 * self.width as u64 - start) as usize;
                    cache.insert(row, &buffer.as_slice()[at..at + self.width]);
                }
            }
            Err(exc) => {
                cache.failure.get_or_insert(exc);
            }
        }
        for row in rows {
            cache.pending.remove(row);
        }
        self.counts.read.fetch_add(rows.len() as u64, Ordering::Relaxed);
        let remaining = cache.batches.get_mut(&batch).map(|left| {
            *left -= 1;
            *left
        });
        if remaining == Some(0) {
            cache.batches.remove(&batch);
            drop(cache);
            self.arrived.notify_all();
        }
    }
}

/// A table of fixed-size rows in a file, read through a bounded row cache.
pub struct RowTable {
    table: Arc<Table>,
    pool: Arc<ReadPool>,
}

impl RowTable {
    /// Rows of `width` bytes from `base` in `file` (the authenticated handle; direct reads use
    /// a second handle on the same file), cached within `budget` bytes, read on `pool`, each
    /// read checked by `verifier` if given.
    pub fn new(file: &File, base: u64, width: usize, rows: u64, budget: u64, pool: Arc<ReadPool>,
               verifier: Option<Verifier>) -> std::io::Result<RowTable> {
        let size = file.metadata()?.len();
        if width == 0 || rows > EMPTY as u64 || base + rows * width as u64 > size {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid row table"));
        }
        let capacity = (budget / (width as u64 + OVERHEAD)).min(rows) as usize;
        let cache = Cache {
            slots: vec![0; capacity * width],
            owner: vec![EMPTY; capacity],
            used: vec![false; capacity],
            index: HashMap::with_capacity(capacity),
            hand: 0,
            pending: HashMap::new(),
            batches: HashMap::new(),
            next_batch: 0,
            failure: None,
        };
        let direct = match storage::direct(file) {
            Some(handle) => {
                // SAFETY: a read-only mapping, used only to probe the page cache.
                Some((handle, unsafe { Mmap::map(file) }?))
            }
            None => None,
        };
        let table = Table { file: file.try_clone()?, direct, base, width, rows, size,
                            cache: Mutex::new(cache), arrived: Condvar::new(), counts: Counts::default(), verifier };
        Ok(RowTable { table: Arc::new(table), pool })
    }

    /// Start reading the rows not cached or already on their way; returns at once.
    pub fn prefetch(&self, rows: &[u32]) -> Result<(), String> {
        let table = &self.table;
        if let Some(&row) = rows.iter().find(|&&row| row as u64 >= table.rows) {
            return Err(format!("row {row} is outside the table"));
        }
        let mut wanted: Vec<u32> = rows.to_vec();
        wanted.sort_unstable();
        wanted.dedup();
        let mut cache = table.cache.lock().unwrap();
        if cache.owner.is_empty() {
            // No room to keep what a prefetch reads: `read` reads the rows itself.
            return Ok(());
        }
        wanted.retain(|row| match cache.index.get(row) {
            Some(&slot) => {
                // Kept from eviction until used.
                cache.used[slot as usize] = true;
                false
            }
            None => !cache.pending.contains_key(row),
        });
        if wanted.is_empty() {
            return Ok(());
        }
        // Rows in neighbouring blocks share a read, up to an extent.
        let batch = cache.next_batch;
        cache.next_batch += 1;
        let mut extents: Vec<(u64, u64, Vec<u32>)> = Vec::new();
        for row in wanted {
            cache.pending.insert(row, batch);
            let (start, end) = table.blocks(row);
            match extents.last_mut() {
                Some((first, last, members)) if start <= *last && end - *first <= EXTENT => {
                    *last = (*last).max(end);
                    members.push(row);
                }
                _ => extents.push((start, end, vec![row])),
            }
        }
        cache.batches.insert(batch, extents.len());
        drop(cache);
        let jobs = extents
            .into_iter()
            .map(|(start, end, rows)| {
                let table = table.clone();
                Box::new(move |buffer: &mut Buffer| table.finish(batch, start, end, &rows, buffer)) as Job
            })
            .collect();
        self.pool.submit(jobs);
        Ok(())
    }

    /// The rows' bytes in order (`rows[i]`'s at `i * width`): cached ones at once, the rest
    /// read (or awaited, if a prefetch is reading them).
    pub fn read(&self, rows: &[u32]) -> Result<Vec<u8>, String> {
        let table = &self.table;
        let width = table.width;
        self.prefetch(rows)?;
        let mut out = vec![0u8; rows.len() * width];
        let mut missing = Vec::new();
        {
            let mut cache = table.cache.lock().unwrap();
            let mut batches: Vec<u64> = rows.iter().filter_map(|row| cache.pending.get(row).copied()).collect();
            batches.sort_unstable();
            batches.dedup();
            while batches.iter().any(|batch| cache.batches.contains_key(batch)) {
                cache = table.arrived.wait(cache).unwrap();
            }
            if let Some(failure) = cache.failure.take() {
                return Err(format!("row table read failed: {failure}"));
            }
            for (index, &row) in rows.iter().enumerate() {
                match cache.get(row, width) {
                    Some(bytes) => out[index * width..(index + 1) * width].copy_from_slice(bytes),
                    None => missing.push(index),
                }
            }
        }
        table.counts.requested.fetch_add(rows.len() as u64, Ordering::Relaxed);
        table.counts.read.fetch_add(missing.len() as u64, Ordering::Relaxed);
        // Evicted before use (more distinct rows than the cache holds): read them here.
        let mut buffer = Buffer::new(width + 2 * ALIGN);
        for index in missing {
            let (start, end) = table.blocks(rows[index]);
            table.read(start, end, &mut buffer).map_err(|exc| format!("row table read failed: {exc}"))?;
            let at = (table.base + rows[index] as u64 * width as u64 - start) as usize;
            out[index * width..(index + 1) * width].copy_from_slice(&buffer.as_slice()[at..at + width]);
        }
        Ok(out)
    }

    /// (rows asked for, rows read from the file, 4 KiB blocks of them not in the page cache)
    /// since the table opened.
    pub fn counts(&self) -> (u64, u64, u64) {
        let counts = &self.table.counts;
        (counts.requested.load(Ordering::Relaxed), counts.read.load(Ordering::Relaxed),
         counts.blocks.load(Ordering::Relaxed))
    }
}
