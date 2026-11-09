//! Integrity of model bytes read on demand, in fs-verity's format (docs/DESIGN.md §10). A model
//! whose tables and experts are read long after load (hundreds of GB, never all in memory)
//! cannot be hashed whole at each start. Its files are hashed once instead into Merkle trees:
//! SHA-256 over 4 KiB blocks, 128 hashes per tree block, up to one root, and a file's digest is
//! the SHA-256 of a 256-byte descriptor naming its size and root (the kernel's `fsverity digest`,
//! so a kernel that has fs-verity can enforce the same digest). The tree is 0.79% of the file
//! and lives beside it; its upper levels (0.006%) are checked against the pinned digest when
//! opened and kept, and each level-0 block is read and checked when a data block under it is
//! first verified, then kept in a bounded cache. Reads verify every 4 KiB block before use.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use sha2::{Digest, Sha256};

use crate::host::Priority;

pub const BLOCK: usize = 4096;
const FAN: u64 = (BLOCK / 32) as u64;
/// Data blocks hashed per work item while building: whole level-0 blocks (128 data blocks).
const BUILD_STEP: u64 = 16 * FAN;

pub type Hash = [u8; 32];

/// A digest written as 64 hex digits (as the pins are).
pub fn parse_digest(text: &str) -> Result<Hash, String> {
    let mut digest = [0u8; 32];
    if text.len() != 64 || !text.is_ascii() {
        return Err("invalid Straydog verity digest".into());
    }
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * index..2 * index + 2], 16).map_err(|_| "invalid Straydog verity digest")?;
    }
    Ok(digest)
}

/// A file's tree: tree blocks per level (level 0 hashes the data), and where each level starts
/// in the tree file, which stores the top level first (as the kernel does).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub size: u64,
    pub levels: Vec<u64>,
    starts: Vec<u64>,
}

impl Geometry {
    pub fn new(size: u64) -> Geometry {
        // As the kernel: levels while more than one block remains (one data block is its own root).
        let mut levels = Vec::new();
        let mut blocks = size.div_ceil(BLOCK as u64);
        while blocks > 1 {
            blocks = blocks.div_ceil(FAN);
            levels.push(blocks);
        }
        let mut starts = vec![0; levels.len()];
        let mut at = 0;
        for level in (0..levels.len()).rev() {
            starts[level] = at;
            at += levels[level] * BLOCK as u64;
        }
        Geometry { size, levels, starts }
    }

    pub fn tree_bytes(&self) -> u64 {
        self.levels.iter().sum::<u64>() * BLOCK as u64
    }
}

/// The file digest: SHA-256 of the descriptor (version 1, SHA-256, 4 KiB blocks, no salt).
pub fn descriptor_digest(size: u64, root: &Hash) -> Hash {
    let mut descriptor = [0u8; 256];
    descriptor[..4].copy_from_slice(&[1, 1, 12, 0]);
    descriptor[8..16].copy_from_slice(&size.to_le_bytes());
    descriptor[16..48].copy_from_slice(root);
    Sha256::digest(descriptor).into()
}

fn hash_block(block: &[u8]) -> Hash {
    if block.len() == BLOCK {
        return Sha256::digest(block).into();
    }
    // The file's last block, zero padded.
    let mut padded = [0u8; BLOCK];
    padded[..block.len()].copy_from_slice(block);
    Sha256::digest(padded).into()
}

/// Hash `file` into its tree at `tree` (all cores, one pass) and return the file digest.
/// `progress` receives each work item's data bytes.
pub fn build(file: &File, tree: &Path, progress: &(dyn Fn(u64) + Sync)) -> std::io::Result<Hash> {
    let size = file.metadata()?.len();
    let geometry = Geometry::new(size);
    let mut out = File::create(tree)?;
    if geometry.levels.is_empty() {
        let mut block = vec![0u8; size as usize];
        read_exact_at(file, &mut block, 0)?;
        let root = if size == 0 { [0; 32] } else { hash_block(&block) };
        return Ok(descriptor_digest(size, &root));
    }
    // Level 0 goes to the tree file as it is made; the hashes of its blocks stay in memory
    // (1/128 of level 0) to build the levels above.
    let blocks = size.div_ceil(BLOCK as u64);
    // (the threads below read the file at once: through a handle that lets them)
    let concurrent = crate::storage::concurrent(file);
    let file = concurrent.as_ref().unwrap_or(file);
    let level0_blocks = geometry.levels[0];
    let mut parents = vec![[0u8; 32]; level0_blocks as usize];
    let next = AtomicU64::new(0);
    let failure = Mutex::new(None);
    let parents_at = parents.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            scope.spawn(|| {
                let mut data = vec![0u8; (BUILD_STEP * BLOCK as u64) as usize];
                let mut hashes = vec![0u8; (BUILD_STEP / FAN) as usize * BLOCK];
                loop {
                    let first = next.fetch_add(BUILD_STEP, Ordering::Relaxed);
                    if first >= blocks || failure.lock().unwrap().is_some() {
                        return;
                    }
                    let count = BUILD_STEP.min(blocks - first);
                    let bytes = (count * BLOCK as u64).min(size - first * BLOCK as u64) as usize;
                    let result = read_exact_at(file, &mut data[..bytes], first * BLOCK as u64).and_then(|()| {
                        data[bytes..(count as usize) * BLOCK].fill(0);
                        hashes.fill(0);
                        for index in 0..count as usize {
                            hashes[index * 32..(index + 1) * 32].copy_from_slice(&hash_block(&data[index * BLOCK..(index + 1) * BLOCK]));
                        }
                        let tree_blocks = count.div_ceil(FAN) as usize;
                        let first_tree_block = first / FAN;
                        for index in 0..tree_blocks {
                            let hash = hash_block(&hashes[index * BLOCK..(index + 1) * BLOCK]);
                            // SAFETY: work items cover disjoint level-0 blocks, so disjoint entries.
                            unsafe { *(parents_at as *mut Hash).add(first_tree_block as usize + index) = hash };
                        }
                        write_at(&out, &hashes[..tree_blocks * BLOCK], geometry.starts[0] + first_tree_block * BLOCK as u64)
                    });
                    match result {
                        Ok(()) => progress(bytes as u64),
                        Err(error) => {
                            failure.lock().unwrap().get_or_insert(error);
                            return;
                        }
                    }
                }
            });
        }
    });
    if let Some(error) = failure.into_inner().unwrap() {
        return Err(error);
    }
    // Levels above: each packs the hashes of the blocks below.
    let mut below = parents;
    for level in 1..geometry.levels.len() {
        let mut packed = vec![0u8; (geometry.levels[level] * BLOCK as u64) as usize];
        for (index, hash) in below.iter().enumerate() {
            packed[index * 32..(index + 1) * 32].copy_from_slice(hash);
        }
        write_at(&out, &packed, geometry.starts[level])?;
        below = packed.chunks(BLOCK).map(hash_block).collect();
    }
    out.flush()?;
    out.sync_all()?;
    Ok(descriptor_digest(size, &below[0]))
}

fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut done = 0;
    while done < buffer.len() {
        let count = crate::storage::read_partial(file, &mut buffer[done..], offset + done as u64)?;
        if count == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file shorter than its tree"));
        }
        done += count;
    }
    Ok(())
}

fn write_at(file: &File, buffer: &[u8], offset: u64) -> std::io::Result<()> {
    let mut done = 0;
    while done < buffer.len() {
        #[cfg(unix)]
        let count = std::os::unix::fs::FileExt::write_at(file, &buffer[done..], offset + done as u64)?;
        #[cfg(windows)]
        let count = std::os::windows::fs::FileExt::seek_write(file, &buffer[done..], offset + done as u64)?;
        done += count;
    }
    Ok(())
}

/// Checks blocks of one file against its tree, from any thread.
/// Bytes `offset .. offset + length` of a mapped file, copied out of the whole blocks around them
/// once that copy verified: what is checked is what is used, whatever the page cache does meanwhile.
pub fn read_verified(map: &[u8], verifier: &Verifier, offset: u64, length: u64) -> Result<Vec<u8>, String> {
    let size = map.len() as u64;
    if offset.checked_add(length).is_none_or(|end| end > size) {
        return Err("a verified read past the file's end".into());
    }
    let start = offset / BLOCK as u64 * BLOCK as u64;
    let end = (offset + length).next_multiple_of(BLOCK as u64);
    let mut blocks = vec![0u8; (end - start) as usize];
    let present = (end.min(size) - start) as usize;
    blocks[..present].copy_from_slice(&map[start as usize..start as usize + present]);
    verifier.verify(start, &blocks)?;
    let from = (offset - start) as usize;
    Ok(blocks[from..from + length as usize].to_vec())
}

/// Check bytes `offset .. offset + length` of a mapped file in place, as the whole blocks around
/// them, in pieces on up to `threads` threads: for memory that cannot change under the check
/// (page-locked, or used only after a copy). A last partial block is checked zero-padded.
pub fn verify_mapped(map: &[u8], verifier: &Verifier, offset: u64, length: u64, threads: usize) -> Result<(), String> {
    verify_mapped_at(map, verifier, offset, length, threads, Priority::Normal)
}

/// [`verify_mapped`] with the priority the threads it starts work at (the calling thread keeps its own, which it uses when `threads` is 1 or the
/// range is one piece): the check of the banks behind "ready" gives way to a request's work, the load's does not.
pub fn verify_mapped_at(map: &[u8], verifier: &Verifier, offset: u64, length: u64, threads: usize, priority: Priority) -> Result<(), String> {
    const PIECE: u64 = 16 << 20;
    let size = map.len() as u64;
    if offset.checked_add(length).is_none_or(|end| end > size) {
        return Err("a verified range past the file's end".into());
    }
    let start = offset / BLOCK as u64 * BLOCK as u64;
    let end = (offset + length).next_multiple_of(BLOCK as u64);
    // Whole blocks within the file go straight from the mapping; a last partial one is padded.
    let whole_end = end.min(size / BLOCK as u64 * BLOCK as u64).max(start);
    let pieces: Vec<(u64, u64)> = (start..whole_end).step_by(PIECE as usize).map(|at| (at, (at + PIECE).min(whole_end))).collect();
    let failure = Mutex::new(None);
    let check = |(from, to): (u64, u64)| {
        let data = &map[from as usize..to as usize];
        // Issue the cold piece's reads before hashing its 4 KiB blocks. Sequential demand
        // faults otherwise serialize I/O with hashing. Stay inside this worker's 16 MiB
        // piece (at most 128 MiB across the batch), and skip advice on resident pieces.
        // Advice is not authentication: every block below still checks its Merkle proof.
        if crate::storage::resident(data) < 1.0 {
            crate::storage::read_ahead(data.chunks(2 << 20));
        }
        if let Err(error) = verifier.verify(from, data) {
            failure.lock().unwrap().get_or_insert(error);
        }
    };
    run_pieces(&pieces, threads, priority, check);
    if end > whole_end && whole_end < size {
        let mut last = vec![0u8; BLOCK];
        last[..(size - whole_end) as usize].copy_from_slice(&map[whole_end as usize..]);
        if let Err(error) = verifier.verify(whole_end, &last) {
            failure.lock().unwrap().get_or_insert(error);
        }
    }
    match failure.into_inner().unwrap() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Run `check` on every piece, on the calling thread when there is one piece or one thread, else on `threads` threads (as many as there are pieces
/// at most) started here, each at `priority` for as long as it works.
fn run_pieces(pieces: &[(u64, u64)], threads: usize, priority: Priority, check: impl Fn((u64, u64)) + Sync) {
    if pieces.len() > 1 && threads > 1 {
        let next = AtomicU64::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads.min(pieces.len()) {
                scope.spawn(|| {
                    let _lowered = crate::host::Lowered::begin(priority);
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed) as usize;
                        let Some(piece) = pieces.get(index) else { return };
                        check(*piece);
                    }
                });
            }
        });
    } else {
        pieces.iter().copied().for_each(check);
    }
}

/// A whole tensor to check during a scoped batch. The mapping is borrowed, never copied into
/// the work queue; the same immutability assumptions as `verify_mapped` apply.
pub struct MappedRange<'a> {
    pub map: &'a [u8],
    pub verifier: &'a Verifier,
    pub offset: u64,
    pub length: u64,
}

struct CheckState {
    remaining: usize,
    // Keep the earliest failing piece, independently of worker completion order.
    failure: Option<(u64, String)>,
}

/// Completion of each input range. A successful wait means ALL blocks of that range verified;
/// a later range's failure cannot replace the result of an earlier consumer access.
pub struct MappedChecks {
    states: Vec<(Mutex<CheckState>, Condvar)>,
}

impl MappedChecks {
    pub fn wait(&self, index: usize) -> Result<(), String> {
        let (state, ready) = &self.states[index];
        let state = ready.wait_while(state.lock().unwrap(), |state| state.remaining != 0).unwrap();
        match &state.failure {
            Some((_, error)) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn complete(&self, index: usize, offset: u64, result: Result<(), String>) {
        let (state, ready) = &self.states[index];
        let mut state = state.lock().unwrap();
        if let Err(error) = result {
            if state.failure.as_ref().is_none_or(|(first, _)| offset < *first) {
                state.failure = Some((offset, error));
            }
        }
        state.remaining -= 1;
        if state.remaining == 0 {
            ready.notify_all();
        }
    }
}

/// One persistent group of bounded workers for pieces from ALL input tensors. Work descriptors
/// contain only indices/ranges: at most `min(available_parallelism, 8)` pieces fault/hash at once,
/// with no model-sized payload allocation or nested per-tensor thread groups.
///
/// `run` waits for tensors in its original order and can overlap their uploads with later checks.
/// It must not return success after ignoring a failed wait; callers check planned results in
/// input order after `run` succeeds. Dropping the cancellation guard also handles unwinding:
/// scoped threads finish their current piece and join before any borrowed mapping can disappear.
pub fn with_mapped_checks<R>(ranges: &[MappedRange<'_>], run: impl FnOnce(&MappedChecks) -> R) -> Result<R, String> {
    const PIECE: u64 = 16 << 20;
    let mut jobs = Vec::new();
    let mut states = Vec::with_capacity(ranges.len());
    for (index, range) in ranges.iter().enumerate() {
        let size = range.map.len() as u64;
        if range.offset.checked_add(range.length).is_none_or(|end| end > size) {
            return Err("a verified range past the file's end".into());
        }
        let first = range.offset / BLOCK as u64 * BLOCK as u64;
        let end = (range.offset + range.length).next_multiple_of(BLOCK as u64).min(size);
        let before = jobs.len();
        for offset in (first..end).step_by(PIECE as usize) {
            jobs.push((index, offset, (end - offset).min(PIECE)));
        }
        states.push((Mutex::new(CheckState { remaining: jobs.len() - before, failure: None }), Condvar::new()));
    }
    let checks = MappedChecks { states };
    let next = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    struct Cancel<'a>(&'a AtomicBool);
    impl Drop for Cancel<'_> {
        fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); }
    }
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get()).min(8).min(jobs.len());
    eprintln!("Straydog dense authentication: {} tensors, {} workers, {} MiB maximum read-ahead",
              ranges.len(), threads, threads * (PIECE as usize >> 20));
    Ok(std::thread::scope(|scope| {
        let _cancel = Cancel(&cancelled);
        for _ in 0..threads {
            scope.spawn(|| {
                while !cancelled.load(Ordering::Relaxed) {
                    let Some(&(index, offset, length)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) else { return };
                    let range = &ranges[index];
                    // A worker panic must complete this item with an error, rather than leave
                    // a consumer asleep forever waiting for a piece that can never finish.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        verify_mapped(range.map, range.verifier, offset, length, 1)
                    })).unwrap_or_else(|_| Err("authentication worker panicked".into()));
                    checks.complete(index, offset, result);
                }
            });
        }
        run(&checks)
    }))
}

pub struct Verifier {
    geometry: Geometry,
    tree: File,
    digest: Hash,
    /// The hash of each level-0 block, from the checked level 1 (empty when level 0 is one
    /// block, which the digest checks).
    level0: Vec<Hash>,
    /// A checked, owned copy of every level-0 block, when the row-file budget holds it.
    resident: Option<Box<[u8]>>,
    /// Checked level-0 blocks, bounded, CLOCK-evicted.
    cache: Mutex<Level0>,
    pub tree_reads: AtomicU64,
}

struct Level0 {
    blocks: HashMap<u64, (Box<[u8; BLOCK]>, bool)>,
    order: Vec<u64>,
    hand: usize,
    capacity: usize,
}

impl Verifier {
    /// Open the tree of a file of `size` bytes whose pinned digest is `digest`; its upper levels
    /// are read and checked now. `cache_bytes` bounds the level-0 blocks kept.
    pub fn open(tree_path: &Path, size: u64, digest: &Hash, cache_bytes: usize) -> Result<Verifier, String> {
        let geometry = Geometry::new(size);
        let name = tree_path.display();
        let tree = File::open(tree_path).map_err(|exc| format!("{name}: {exc}"))?;
        let length = tree.metadata().map_err(|exc| exc.to_string())?.len();
        if length != geometry.tree_bytes() {
            return Err(format!("{name}: a tree of {length} bytes for a file that needs {}", geometry.tree_bytes()));
        }
        if size == 0 && descriptor_digest(0, &[0; 32]) != *digest {
            return Err(format!("{name}: an empty file's digest does not match"));
        }
        // Walk down from the top block, checking every block of each level above level 0 against
        // the level above it (the top against the digest). Level 1's entries are then the hashes
        // of level 0's blocks, which are read and checked as data under them is verified. A
        // single level-0 block is the top, checked here; a file of one block has no tree, and
        // its digest is checked by each verification.
        let top = geometry.levels.len();
        let mut expected: Vec<Hash> = Vec::new();
        for level in (0..top).rev() {
            if level == 0 && top > 1 {
                break;
            }
            let mut bytes = vec![0u8; (geometry.levels[level] * BLOCK as u64) as usize];
            read_exact_at(&tree, &mut bytes, geometry.starts[level]).map_err(|exc| format!("{name}: {exc}"))?;
            for (index, block) in bytes.chunks(BLOCK).enumerate() {
                let hash = hash_block(block);
                let good = if level + 1 == top { descriptor_digest(size, &hash) == *digest } else { hash == expected[index] };
                if !good {
                    return Err(format!("{name}: tree block ({level}, {index}) does not verify"));
                }
            }
            expected = if level == 0 {
                Vec::new()
            } else {
                bytes.chunks(32).take(geometry.levels[level - 1] as usize).map(|h| h.try_into().unwrap()).collect()
            };
        }
        let level0 = expected;
        let capacity = (cache_bytes / BLOCK).max(1);
        // (verifications on many threads read its blocks at once)
        let tree = crate::storage::concurrent(&tree).unwrap_or(tree);
        Ok(Verifier {
            geometry, tree, digest: *digest, level0, resident: None,
            cache: Mutex::new(Level0 { blocks: HashMap::new(), order: Vec::new(), hand: 0, capacity }),
            tree_reads: AtomicU64::new(0),
        })
    }

    /// Keep the whole leaf level as an immutable, checked copy, only when it fits `budget`.
    /// Called before readers share the verifier. The old bounded cache remains the fallback;
    /// no partially checked copy is published on a read, allocation or authentication failure.
    pub fn preload_level0(&mut self, budget: u64) -> Result<u64, String> {
        let bytes = self.geometry.levels.first().copied().unwrap_or(0) * BLOCK as u64;
        if bytes == 0 || bytes > budget {
            return Ok(0);
        }
        if self.resident.is_some() {
            return Ok(bytes);
        }
        let Ok(length) = usize::try_from(bytes) else { return Ok(0) };
        let mut blocks = Vec::new();
        if blocks.try_reserve_exact(length).is_err() {
            return Ok(0);
        }
        blocks.resize(length, 0);
        read_exact_at(&self.tree, &mut blocks, self.geometry.starts[0]).map_err(|error| format!("tree read: {error}"))?;
        self.tree_reads.fetch_add(bytes / BLOCK as u64, Ordering::Relaxed);
        for (index, block) in blocks.chunks_exact(BLOCK).enumerate() {
            if !self.level0_matches(index as u64, block) {
                return Err(format!("tree block (0, {index}) does not verify"));
            }
        }
        self.resident = Some(blocks.into_boxed_slice());
        let cache = self.cache.get_mut().unwrap();
        cache.blocks.clear();
        cache.blocks.shrink_to_fit();
        cache.order.clear();
        cache.order.shrink_to_fit();
        cache.hand = 0;
        Ok(bytes)
    }

    fn level0_matches(&self, number: u64, block: &[u8]) -> bool {
        let hash = hash_block(block);
        match self.level0.get(number as usize) {
            Some(expected) => hash == *expected,
            None => self.level0.is_empty() && descriptor_digest(self.geometry.size, &hash) == self.digest,
        }
    }

    /// Check `data`, whole blocks of the file from `offset` (a block boundary); bytes past the
    /// file's end must be zero.
    pub fn verify(&self, offset: u64, data: &[u8]) -> Result<(), String> {
        if !offset.is_multiple_of(BLOCK as u64) || !data.len().is_multiple_of(BLOCK) {
            return Err("a verified read must be whole blocks".into());
        }
        let size = self.geometry.size;
        // The level-0 tree block of the blocks being checked: taken once per FAN data blocks.
        let mut current: Option<(u64, Box<[u8; BLOCK]>)> = None;
        for (index, block) in data.chunks(BLOCK).enumerate() {
            let at = offset + (index * BLOCK) as u64;
            if at >= size {
                if block.iter().any(|&byte| byte != 0) {
                    return Err(format!("bytes past the file's end at {at} are not zero"));
                }
                continue;
            }
            let hash = hash_block(block);
            let number = at / BLOCK as u64;
            if self.geometry.levels.is_empty() {
                if descriptor_digest(size, &hash) != self.digest {
                    return Err(format!("block {number} does not verify"));
                }
                continue;
            }
            let (entries, entry) = match self.resident.as_deref() {
                Some(entries) => (entries, (number * 32) as usize),
                None => {
                    let tree_block = number / FAN;
                    if current.as_ref().is_none_or(|(held, _)| *held != tree_block) {
                        current = Some((tree_block, self.level0_block(tree_block)?));
                    }
                    (&current.as_ref().unwrap().1[..], ((number % FAN) * 32) as usize)
                }
            };
            if entries[entry..entry + 32] != hash {
                return Err(format!("block {number} does not verify"));
            }
        }
        Ok(())
    }

    /// A (checked) level-0 tree block: the hashes of FAN data blocks.
    fn level0_block(&self, tree_block: u64) -> Result<Box<[u8; BLOCK]>, String> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some((block, used)) = cache.blocks.get_mut(&tree_block) {
                *used = true;
                return Ok(block.clone());
            }
        }
        let mut block = Box::new([0u8; BLOCK]);
        read_exact_at(&self.tree, &mut block[..], self.geometry.starts[0] + tree_block * BLOCK as u64)
            .map_err(|exc| format!("tree read: {exc}"))?;
        self.tree_reads.fetch_add(1, Ordering::Relaxed);
        if !self.level0_matches(tree_block, &block[..]) {
            return Err(format!("tree block (0, {tree_block}) does not verify"));
        }
        let mut cache = self.cache.lock().unwrap();
        // Readers verify in parallel: another may have cached this block since the check above.
        if let Some(entry) = cache.blocks.get_mut(&tree_block) {
            entry.1 = true;
            return Ok(block);
        }
        if cache.blocks.len() >= cache.capacity {
            // CLOCK: pass over used blocks, clearing their bit; evict the first clear one.
            loop {
                let hand = cache.hand % cache.order.len();
                let key = cache.order[hand];
                cache.hand = hand + 1;
                let entry = cache.blocks.get_mut(&key).expect("the CLOCK order names only cached blocks");
                if entry.1 {
                    entry.1 = false;
                } else {
                    cache.blocks.remove(&key);
                    cache.order[hand] = tree_block;
                    break;
                }
            }
        } else {
            cache.order.push(tree_block);
        }
        cache.blocks.insert(tree_block, (block.clone(), true));
        Ok(block)
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn pooled_mapped_ranges_match_serial_checks_and_join_on_early_exit() {
        let size = (33 << 20) + 137;
        let mut data: Vec<u8> = (0..size).map(|at| (at * 37 % 251) as u8).collect();
        let file = std::env::temp_dir().join(format!("straydog-auth-pool-{}.data", std::process::id()));
        let tree = file.with_extension("tree");
        std::fs::write(&file, &data).unwrap();
        let digest = build(&File::open(&file).unwrap(), &tree, &|_| {}).unwrap();
        let verifier = Verifier::open(&tree, size as u64, &digest, 4 * BLOCK).unwrap();
        // Small tensors, several pieces of one tensor, unaligned/shared blocks, partial EOF,
        // and zero-length ranges. All use the identical block checker and pinned root.
        let spans = [(0, 1 << 20), (100_000, 30 << 20), (size as u64 - 100, 100),
                     (0, size as u64), (4095, 2), (size as u64, 0)];
        let pooled = |data: &[u8]| {
            let ranges: Vec<_> = spans.iter().map(|&(offset, length)| MappedRange { map: data, verifier: &verifier, offset, length }).collect();
            with_mapped_checks(&ranges, |checks| {
                spans.iter().enumerate().map(|(index, &(offset, length))| {
                    let expected = verify_mapped(data, &verifier, offset, length, 1);
                    assert_eq!(checks.wait(index), expected, "range {index}");
                    expected
                }).collect::<Vec<_>>()
            }).unwrap()
        };
        assert!(pooled(&data).iter().all(Result::is_ok));
        for at in [17 << 20, size - 3] {
            data[at] ^= 1;
            let results = pooled(&data);
            assert!(results[0].is_ok());
            assert!(results[3].is_err());
            data[at] ^= 1;
        }
        let range = MappedRange { map: &data, verifier: &verifier, offset: 0, length: size as u64 };
        let early: Result<(), &str> = with_mapped_checks(&[range], |_| Err("upload failed")).unwrap();
        assert_eq!(early, Err("upload failed"));
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let range = MappedRange { map: &data, verifier: &verifier, offset: 0, length: size as u64 };
            with_mapped_checks(&[range], |_| panic!("consumer failed"))
        }));
        assert!(unwound.is_err());
        assert!(pooled(&data).iter().all(Result::is_ok));
        assert!(with_mapped_checks(&[], |_| true).unwrap());
        let invalid = MappedRange { map: &data, verifier: &verifier, offset: size as u64, length: 1 };
        assert!(with_mapped_checks(&[invalid], |_| ()).is_err());
        drop(verifier);
        std::fs::remove_file(file).unwrap();
        std::fs::remove_file(tree).unwrap();
    }
}

#[cfg(test)]
mod piece_tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Condvar;
    use std::thread::ThreadId;
    use std::time::Duration;

    /// The threads `run_pieces` ran 24 pieces on, with the pieces it ran, for `threads` allowed.
    fn ran_on(threads: usize, priority: Priority) -> (HashSet<ThreadId>, Vec<(u64, u64)>) {
        let pieces: Vec<(u64, u64)> = (0..24).map(|index| (index * 100, index * 100 + 100)).collect();
        let workers = if threads > 1 { threads.min(pieces.len()) } else { 1 };
        let seen = (Mutex::new((HashSet::new(), Vec::new())), Condvar::new());
        run_pieces(&pieces, threads, priority, |piece| {
            let (state, wake) = &seen;
            let mut state = state.lock().unwrap();
            state.0.insert(std::thread::current().id());
            state.1.push(piece);
            wake.notify_all();
            // Hold each worker on its first piece until all workers have taken one. A sleep here
            // lets a late-scheduled worker lose its piece to one that already finished sleeping.
            let (state, _) = wake.wait_timeout_while(state, Duration::from_secs(30), |(ids, _)| ids.len() < workers).unwrap();
            assert_eq!(state.0.len(), workers, "all requested workers must take a piece");
        });
        let (ids, mut done) = seen.0.into_inner().unwrap();
        done.sort_unstable();
        assert_eq!(done, pieces, "every piece ran once");
        (ids, done)
    }

    /// Decides: the threads a check starts are as many as it was given (and not more than the pieces), none when there is one thread or one
    /// piece (the calling thread does the work), at either priority.
    #[test]
    fn a_check_starts_as_many_threads_as_it_was_given() {
        let here = std::thread::current().id();
        for priority in [Priority::Normal, Priority::Background] {
            for threads in [2, 3, 4] {
                let (ids, _) = ran_on(threads, priority);
                assert_eq!(ids.len(), threads, "{threads} threads asked for at {priority:?}");
                assert!(!ids.contains(&here));
            }
            assert_eq!(ran_on(1, priority).0, HashSet::from([here]), "one thread is the caller's");
            assert_eq!(ran_on(100, priority).0.len(), 24, "no more threads than pieces");
        }
        let one_piece = Mutex::new(HashSet::new());
        run_pieces(&[(0, 10)], 8, Priority::Background, |_| {
            one_piece.lock().unwrap().insert(std::thread::current().id());
        });
        assert_eq!(one_piece.into_inner().unwrap(), HashSet::from([here]));
    }

    /// Decides: a check at the lower priority still finds a changed byte and passes bytes that are as they were, on the threads it starts.
    #[test]
    fn a_check_at_the_lower_priority_checks_the_same() {
        let size = (40 << 20) + 321;
        let mut data: Vec<u8> = (0..size).map(|at| (at * 31 % 253) as u8).collect();
        let file = std::env::temp_dir().join(format!("straydog-auth-priority-{}.data", std::process::id()));
        let tree = file.with_extension("tree");
        std::fs::write(&file, &data).unwrap();
        let digest = build(&File::open(&file).unwrap(), &tree, &|_| {}).unwrap();
        let verifier = Verifier::open(&tree, size as u64, &digest, 4 * BLOCK).unwrap();
        for priority in [Priority::Normal, Priority::Background] {
            verify_mapped_at(&data, &verifier, 100_000, 38 << 20, 3, priority).unwrap();
            for at in [20 << 20, size - 2] {
                data[at] ^= 1;
                assert!(verify_mapped_at(&data, &verifier, 0, size as u64, 3, priority).is_err(), "a changed byte at {at} under {priority:?}");
                data[at] ^= 1;
            }
        }
        drop(verifier);
        std::fs::remove_file(file).unwrap();
        std::fs::remove_file(tree).unwrap();
    }
}


#[cfg(test)]
mod resident_tests {
    use super::*;

    struct Fixture {
        file: std::path::PathBuf,
        tree: std::path::PathBuf,
        size: usize,
        data: Vec<u8>,
        digest: Hash,
    }

    impl Fixture {
        fn new(label: &str, size: usize) -> Self {
            let file = std::env::temp_dir().join(format!("straydog-proof-{label}-{}", std::process::id()));
            let tree = file.with_extension("tree");
            let mut data: Vec<u8> = (0..size).map(|at| (at * 31 % 251) as u8).collect();
            std::fs::write(&file, &data).unwrap();
            let digest = build(&File::open(&file).unwrap(), &tree, &|_| {}).unwrap();
            data.resize(size.next_multiple_of(BLOCK), 0);
            Self { file, tree, size, data, digest }
        }

        fn verifier(&self) -> Verifier {
            Verifier::open(&self.tree, self.size as u64, &self.digest, BLOCK).unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.file);
            let _ = std::fs::remove_file(&self.tree);
        }
    }

    #[test]
    fn resident_proofs_match_bounded_checks_without_later_tree_reads() {
        let fixture = Fixture::new("resident", (2 * FAN as usize + 1) * BLOCK + 137);
        let bounded = fixture.verifier();
        let mut resident = fixture.verifier();
        let required = resident.geometry.levels[0] * BLOCK as u64;
        assert_eq!(resident.preload_level0(required - 1).unwrap(), 0);
        bounded.verify(0, &fixture.data).unwrap();
        assert_eq!(resident.preload_level0(required).unwrap(), required);
        let reads = resident.tree_reads.load(Ordering::Relaxed);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let resident = &resident;
                let data = &fixture.data;
                scope.spawn(move || resident.verify(0, data).unwrap());
            }
        });
        assert_eq!(resident.tree_reads.load(Ordering::Relaxed), reads);
        let mut changed = fixture.data.clone();
        for at in [0, FAN as usize * BLOCK + 9, fixture.size - 1, changed.len() - 1] {
            changed[at] ^= 1;
            assert!(resident.verify(0, &changed).is_err(), "changed byte {at}");
            assert!(bounded.verify(0, &changed).is_err(), "changed byte {at}");
            changed[at] ^= 1;
        }
        // A later change to the source tree cannot alter the authenticated copy being used.
        let tree = std::fs::OpenOptions::new().write(true).open(&fixture.tree).unwrap();
        let at = resident.geometry.starts[0] + BLOCK as u64 + 7;
        let mut byte = [0];
        read_exact_at(&File::open(&fixture.tree).unwrap(), &mut byte, at).unwrap();
        byte[0] ^= 1;
        write_at(&tree, &byte, at).unwrap();
        resident.verify(0, &fixture.data).unwrap();
        assert_eq!(resident.tree_reads.load(Ordering::Relaxed), reads);
    }

    #[test]
    fn a_bad_late_leaf_never_publishes_a_partly_checked_copy() {
        let fixture = Fixture::new("late-leaf", (2 * FAN as usize + 1) * BLOCK);
        let mut verifier = fixture.verifier();
        let tree = std::fs::OpenOptions::new().write(true).open(&fixture.tree).unwrap();
        let bad = verifier.geometry.starts[0] + 2 * BLOCK as u64;
        let mut byte = [0];
        read_exact_at(&File::open(&fixture.tree).unwrap(), &mut byte, bad).unwrap();
        byte[0] ^= 1;
        write_at(&tree, &byte, bad).unwrap();
        assert!(verifier.preload_level0(u64::MAX).unwrap_err().contains("(0, 2)"));
        assert!(verifier.resident.is_none());
        verifier.verify(0, &fixture.data[..BLOCK]).unwrap();
        let at = 2 * FAN as usize * BLOCK;
        assert!(verifier.verify(at as u64, &fixture.data[at..at + BLOCK]).is_err());
    }

    #[test]
    fn empty_single_block_and_single_leaf_roots_still_authenticate() {
        for size in [0, BLOCK, BLOCK + 1, 2 * BLOCK] {
            let fixture = Fixture::new(&format!("small-{size}"), size);
            let mut verifier = fixture.verifier();
            assert_eq!(verifier.preload_level0(0).unwrap(), 0);
            let expected = if size > BLOCK { BLOCK as u64 } else { 0 };
            assert_eq!(verifier.preload_level0(BLOCK as u64).unwrap(), expected);
            verifier.verify(0, &fixture.data).unwrap();
            if size > 0 {
                let mut changed = fixture.data.clone();
                changed[0] ^= 1;
                assert!(verifier.verify(0, &changed).is_err());
            }
        }
    }
}
