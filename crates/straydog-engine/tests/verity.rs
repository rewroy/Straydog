//! Merkle trees in fs-verity's format (src/verity.rs): file digests against the fs-verity crate's
//! (0.2.0, computed once into the constants below) at the sizes where the tree's shape changes,
//! every block verified through a small level-0 cache, and tampered data, tree blocks and pins
//! refused.

use std::path::PathBuf;

use straydog_engine::verity::{self, Geometry, Verifier, BLOCK};

/// The test files' byte at `i`.
fn byte(i: u64) -> u8 {
    ((i * 131 + (i / 4096) * 7) & 255) as u8
}

/// (size, fs-verity digest) from the fs-verity crate.
const VECTORS: [(u64, &str); 12] = [
    (0, "3d248ca542a24fc62d1c43b916eae5016878e2533c88238480b26128a1f1af95"),
    (1, "b803429503d95915829b29fdbc8bbad142f3abfd11b1cadf5526582e685c0551"),
    (4095, "de15d5663f795e809c7e0c4c468790d71f2aa25eab530244231e05b19c673f2d"),
    (4096, "0d5fa6409d4b558cae4fb7fd1e06719f23b827c810519a104556c55134466f70"),
    (4097, "ab71e9c03fb5d95e0d005f06788e0d342a4803f4d7f14c807bf90b809d1eaa83"),
    (8192, "65889f7acb7619847a098f205592b32fb7cd0e9e36280add1a4254c80c58e71f"),
    (524288, "042744fe4d531f05e90da8ca33033019661f079daaf8696a8bd471ff99bbe1a5"),
    (524289, "bf241f51902864fb979f1bd9f4ab067beb68816cd1f804eaa5ae9df90d5ba5cd"),
    (528384, "5b3bc3af4e6fd756836a712c5ce55a42cb3120ca44f6736ce82ec86ab7aee555"),
    (3145745, "7879dfb52672843e2929bbd2506903888d735d305527a55e5439288949d3bd12"),
    (67108864, "1d84bdf54c029d5e3fc0d8c8ba0c61b25bc6dcc396e257c443eb65b82718d20a"),
    (67121157, "13cf9922fc2b7efc592a8df0de04de97e104afd0efe9110913ee47dc94e6f8f9"),
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("straydog-verity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// The file's whole blocks from `block`, zero padded past its end.
fn blocks(data: &[u8], block: usize, count: usize) -> Vec<u8> {
    let mut out = vec![0u8; count * BLOCK];
    let start = (block * BLOCK).min(data.len());
    let end = ((block + count) * BLOCK).min(data.len());
    out[..end - start].copy_from_slice(&data[start..end]);
    out
}

#[test]
fn digests_match_fs_verity_and_every_block_verifies() {
    for (size, want) in VECTORS {
        let data: Vec<u8> = (0..size).map(byte).collect();
        let (file_path, tree_path) = (scratch(&format!("f{size}")), scratch(&format!("f{size}.tree")));
        std::fs::write(&file_path, &data).unwrap();
        let digest = verity::build(&std::fs::File::open(&file_path).unwrap(), &tree_path, &|_| {}).unwrap();
        assert_eq!(hex(&digest), want, "size {size}");
        assert_eq!(std::fs::metadata(&tree_path).unwrap().len(), Geometry::new(size).tree_bytes(), "size {size}");
        // Two level-0 blocks of cache: verification walks the file, refilling it.
        let verifier = Verifier::open(&tree_path, size, &digest, 2 * BLOCK).unwrap();
        let count = size.div_ceil(BLOCK as u64) as usize;
        for first in (0..count).step_by(37) {
            let n = 37.min(count - first);
            verifier.verify((first * BLOCK) as u64, &blocks(&data, first, n)).unwrap();
        }
        // A block past the end must be zeros.
        let past = count.max(1);
        verifier.verify((past * BLOCK) as u64, &[0u8; BLOCK]).unwrap();
        assert!(verifier.verify((past * BLOCK) as u64, &[1u8; BLOCK]).is_err());
        if size > 0 {
            // One flipped bit in any data block fails.
            for block in [0, count / 2, count - 1] {
                let mut tampered = blocks(&data, block, 1);
                tampered[BLOCK / 3] ^= 4;
                assert!(verifier.verify((block * BLOCK) as u64, &tampered).is_err(), "size {size} block {block}");
            }
        }
        // A wrong pin refuses the tree (a file of one block has none: then its block).
        let mut wrong = digest;
        wrong[0] ^= 1;
        match Verifier::open(&tree_path, size, &wrong, BLOCK) {
            Err(_) => {}
            Ok(verifier) => {
                assert!(size <= BLOCK as u64, "size {size}");
                assert!(verifier.verify(0, &blocks(&data, 0, 1)).is_err() || size == 0, "size {size}");
            }
        }
        std::fs::remove_file(&file_path).unwrap();
        std::fs::remove_file(&tree_path).unwrap();
    }
}

#[test]
fn tampered_tree_blocks_are_refused() {
    let size = 67121157u64;
    let data: Vec<u8> = (0..size).map(byte).collect();
    let (file_path, tree_path) = (scratch("tree-tamper"), scratch("tree-tamper.tree"));
    std::fs::write(&file_path, &data).unwrap();
    let digest = verity::build(&std::fs::File::open(&file_path).unwrap(), &tree_path, &|_| {}).unwrap();
    let pristine = std::fs::read(&tree_path).unwrap();
    let geometry = Geometry::new(size);
    assert_eq!(geometry.levels.len(), 3);
    // Every level: an upper-level block fails at open, a level-0 block when a read needs it.
    let level0_start = (geometry.levels[1] + geometry.levels[2]) as usize * BLOCK;
    for (at, at_open) in [(100, true), (BLOCK + 64, true), (level0_start + 5 * BLOCK + 32, false)] {
        let mut tree = pristine.clone();
        tree[at] ^= 0x80;
        std::fs::write(&tree_path, &tree).unwrap();
        match Verifier::open(&tree_path, size, &digest, 4 * BLOCK) {
            Err(_) => assert!(at_open, "byte {at} refused at open"),
            Ok(verifier) => {
                assert!(!at_open, "byte {at} not refused at open");
                // Data blocks 5 * 128 .. under the tampered level-0 block, honest data.
                let first = 5 * 128;
                assert!(verifier.verify((first * BLOCK) as u64, &blocks(&data, first, 1)).is_err());
                verifier.verify(0, &blocks(&data, 0, 1)).unwrap();
            }
        }
    }
    std::fs::remove_file(&file_path).unwrap();
    std::fs::remove_file(&tree_path).unwrap();
}

/// Many threads verifying overlapping blocks through a small level-0 cache (as the expert and
/// row readers do): every block verifies, the cache never holds a block twice, and nothing
/// panics.
#[test]
fn parallel_verification_through_a_small_cache() {
    let size = 67121157u64;
    let data: Vec<u8> = (0..size).map(byte).collect();
    let (file_path, tree_path) = (scratch("parallel"), scratch("parallel.tree"));
    std::fs::write(&file_path, &data).unwrap();
    let digest = verity::build(&std::fs::File::open(&file_path).unwrap(), &tree_path, &|_| {}).unwrap();
    let verifier = Verifier::open(&tree_path, size, &digest, 3 * BLOCK).unwrap();
    let count = size.div_ceil(BLOCK as u64) as usize;
    std::thread::scope(|scope| {
        for thread in 0..16usize {
            let (verifier, data) = (&verifier, &data);
            scope.spawn(move || {
                let mut seed = 0x9e37_79b9u64 + thread as u64;
                for _ in 0..3000 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    // Few distinct level-0 blocks, so threads miss the same one at once.
                    let first = ((seed % 8) * 128 + (seed >> 8) % 120) as usize % count;
                    let n = 1 + (seed >> 20) as usize % 8;
                    let n = n.min(count - first);
                    verifier.verify((first * BLOCK) as u64, &blocks(data, first, n)).unwrap();
                }
            });
        }
    });
    std::fs::remove_file(&file_path).unwrap();
    std::fs::remove_file(&tree_path).unwrap();
}

/// Reads of a mapped file as the weights make them: unaligned bytes copied out of their verified
/// blocks (`read_verified`), and long ranges checked in place on several threads
/// (`verify_mapped`), the last partial block zero-padded; one changed byte refuses either.
#[test]
fn mapped_ranges_verify_in_place_and_by_copy() {
    let size = 67121157u64;
    let mut data: Vec<u8> = (0..size).map(byte).collect();
    let (file_path, tree_path) = (scratch("mapped"), scratch("mapped.tree"));
    std::fs::write(&file_path, &data).unwrap();
    let digest = verity::build(&std::fs::File::open(&file_path).unwrap(), &tree_path, &|_| {}).unwrap();
    let verifier = Verifier::open(&tree_path, size, &digest, 4 * BLOCK).unwrap();
    for (offset, length) in [(0u64, 1u64), (4095, 2), (12345, 176), (size - 100, 100), (size - 1, 1)] {
        let read = verity::read_verified(&data, &verifier, offset, length).unwrap();
        assert_eq!(read, &data[offset as usize..(offset + length) as usize]);
    }
    assert!(verity::read_verified(&data, &verifier, size - 10, 11).is_err());
    for threads in [1, 8] {
        verity::verify_mapped(&data, &verifier, 0, size, threads).unwrap();
        verity::verify_mapped(&data, &verifier, 100_000, 40 << 20, threads).unwrap();
    }
    for at in [17u64 << 20, size - 3] {
        data[at as usize] ^= 1;
        let start = at.saturating_sub(5000);
        assert!(verity::read_verified(&data, &verifier, at, 1).is_err());
        assert!(verity::verify_mapped(&data, &verifier, start, size - start, 8).is_err());
        // Bytes elsewhere still verify.
        verity::verify_mapped(&data, &verifier, 0, 1 << 20, 8).unwrap();
        data[at as usize] ^= 1;
    }
    std::fs::remove_file(&file_path).unwrap();
    std::fs::remove_file(&tree_path).unwrap();
}
