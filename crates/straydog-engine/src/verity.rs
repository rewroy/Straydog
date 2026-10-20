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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

pub const BLOCK: usize = 4096;
const FAN: u64 = (BLOCK / 32) as u64;
/// Data blocks hashed per work item while building: whole level-0 blocks (128 data blocks).
const BUILD_STEP: u64 = 16 * FAN;

pub type Hash = [u8; 32];

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
        #[cfg(unix)]
        let count = std::os::unix::fs::FileExt::read_at(file, &mut buffer[done..], offset + done as u64)?;
        #[cfg(windows)]
        let count = std::os::windows::fs::FileExt::seek_read(file, &mut buffer[done..], offset + done as u64)?;
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
pub struct Verifier {
    geometry: Geometry,
    tree: File,
    digest: Hash,
    /// The hash of each level-0 block, from the checked level 1 (empty when level 0 is one
    /// block, which the digest checks).
    level0: Vec<Hash>,
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
        Ok(Verifier {
            geometry, tree, digest: *digest, level0,
            cache: Mutex::new(Level0 { blocks: HashMap::new(), order: Vec::new(), hand: 0, capacity }),
            tree_reads: AtomicU64::new(0),
        })
    }

    /// Check `data`, whole blocks of the file from `offset` (a block boundary); bytes past the
    /// file's end must be zero.
    pub fn verify(&self, offset: u64, data: &[u8]) -> Result<(), String> {
        if !offset.is_multiple_of(BLOCK as u64) || !data.len().is_multiple_of(BLOCK) {
            return Err("a verified read must be whole blocks".into());
        }
        let size = self.geometry.size;
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
            if self.entry(number)? != hash {
                return Err(format!("block {number} does not verify"));
            }
        }
        Ok(())
    }

    /// Data block `number`'s hash from its (checked) level-0 tree block.
    fn entry(&self, number: u64) -> Result<Hash, String> {
        let tree_block = number / FAN;
        let at = ((number % FAN) * 32) as usize;
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some((block, used)) = cache.blocks.get_mut(&tree_block) {
                *used = true;
                return Ok(block[at..at + 32].try_into().unwrap());
            }
        }
        let mut block = Box::new([0u8; BLOCK]);
        read_exact_at(&self.tree, &mut block[..], self.geometry.starts[0] + tree_block * BLOCK as u64)
            .map_err(|exc| format!("tree read: {exc}"))?;
        self.tree_reads.fetch_add(1, Ordering::Relaxed);
        let good = match self.level0.get(tree_block as usize) {
            Some(expected) => hash_block(&block[..]) == *expected,
            None => self.level0.is_empty() && descriptor_digest(self.geometry.size, &hash_block(&block[..])) == self.digest,
        };
        if !good {
            return Err(format!("tree block (0, {tree_block}) does not verify"));
        }
        let hash = block[at..at + 32].try_into().unwrap();
        let mut cache = self.cache.lock().unwrap();
        if cache.blocks.len() >= cache.capacity {
            // CLOCK: pass over used blocks, clearing their bit; evict the first clear one.
            loop {
                let hand = cache.hand % cache.order.len();
                let key = cache.order[hand];
                let entry = cache.blocks.get_mut(&key).unwrap();
                if entry.1 {
                    entry.1 = false;
                    cache.hand = hand + 1;
                } else {
                    cache.blocks.remove(&key);
                    cache.order[hand] = tree_block;
                    cache.hand = hand + 1;
                    break;
                }
            }
        } else {
            cache.order.push(tree_block);
        }
        cache.blocks.insert(tree_block, (block, true));
        Ok(hash)
    }
}
