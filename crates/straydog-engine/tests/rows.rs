//! The row cache returns exactly the file's rows: cached, prefetched, evicted before use, and
//! read past the page cache; with a verifier, every read is checked against the file's tree.

use std::io::Write;
use std::sync::Arc;

use straydog_engine::rows::{self, ReadPool, RowTable};
use straydog_engine::verity;

fn table_file(rows: u64, width: usize) -> (std::path::PathBuf, Vec<u8>) {
    let path = std::env::temp_dir().join(format!("straydog-rows-{}-{rows}.bin", std::process::id()));
    let mut seed = 0x2545f4914f6cdd1du64;
    let bytes: Vec<u8> = (0..123 + rows as usize * width).map(|_| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u8
    }).collect();
    std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
    (path, bytes)
}

#[test]
fn rows_equal_the_file_through_a_small_cache() {
    let (rows, width) = (5000u64, 90usize);
    let (path, bytes) = table_file(rows, width);
    let file = std::fs::File::open(&path).unwrap();
    let pool = Arc::new(ReadPool::new(4));
    // About 100 rows fit: batches larger than that are evicted before use and read again.
    for budget in [0u64, 100 * 114, 1 << 20] {
        let table = RowTable::new(&file, 123, width, rows, budget, pool.clone(), None).unwrap();
        let mut seed = 7u64;
        for step in 0..40 {
            let wanted: Vec<u32> = (0..1 + step * 5).map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) % if step % 3 == 0 { 40 } else { rows }) as u32
            }).collect();
            if step % 2 == 0 {
                table.prefetch(&wanted).unwrap();
            }
            let got = table.read(&wanted).unwrap();
            for (index, &row) in wanted.iter().enumerate() {
                let start = 123 + row as usize * width;
                assert_eq!(&got[index * width..(index + 1) * width], &bytes[start..start + width], "row {row}");
            }
        }
        let (asked, read, _) = table.counts();
        if budget == 1 << 20 {
            // The whole table fits: each row is read once.
            assert!(read <= rows && read < asked, "{read} rows read for {asked}");
        }
    }
    assert!(RowTable::new(&file, 123, width, rows + 1, 0, pool.clone(), None).is_err());
    let table = RowTable::new(&file, 123, width, rows, 1 << 20, pool, None).unwrap();
    assert!(table.read(&[rows as u32]).is_err());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn verified_rows_and_a_tampered_block_refused() {
    // Rows wider than a pool read, ending inside the file's last block.
    let (rows, width) = (300u64, 70_000usize);
    let (path, mut bytes) = table_file(rows, width);
    let tree = path.with_extension("tree");
    let digest = verity::build(&std::fs::File::open(&path).unwrap(), &tree, &|_| {}).unwrap();
    let checker = Arc::new(verity::Verifier::open(&tree, bytes.len() as u64, &digest, 8 << 10).unwrap());
    let verifier: rows::Verifier = Arc::new(move |offset, data| checker.verify(offset, data));
    let pool = Arc::new(ReadPool::new(4));
    let row = |bytes: &[u8], row: u32| bytes[123 + row as usize * width..123 + (row as usize + 1) * width].to_vec();
    for budget in [0u64, 4 * 70_024, 64 << 20] {
        let file = std::fs::File::open(&path).unwrap();
        let table = RowTable::new(&file, 123, width, rows, budget, pool.clone(), Some(verifier.clone())).unwrap();
        let wanted: Vec<u32> = (0..rows as u32).rev().step_by(7).chain([rows as u32 - 1, 0]).collect();
        table.prefetch(&wanted).unwrap();
        assert_eq!(table.read(&wanted).unwrap(), wanted.iter().flat_map(|&r| row(&bytes, r)).collect::<Vec<_>>());
    }
    // One bit flipped in row 150: its reads fail, every time; rows elsewhere still read.
    bytes[123 + 150 * width + 33_333] ^= 4;
    let tampered = path.with_extension("tampered");
    std::fs::write(&tampered, &bytes).unwrap();
    let file = std::fs::File::open(&tampered).unwrap();
    let table = RowTable::new(&file, 123, width, rows, 64 << 20, pool.clone(), Some(verifier)).unwrap();
    for _ in 0..2 {
        assert!(table.read(&[150]).unwrap_err().contains("does not verify"));
    }
    assert_eq!(table.read(&[3, 299]).unwrap(), [row(&bytes, 3), row(&bytes, 299)].concat());
    // A verifier that panics fails its read, not the pool's threads.
    let panics: rows::Verifier = Arc::new(|offset, _| if offset == 0 { panic!("a verifier bug") } else { Ok(()) });
    let table = RowTable::new(&file, 123, width, rows, 64 << 20, pool.clone(), Some(panics)).unwrap();
    for _ in 0..8 {
        assert!(table.read(&[0]).unwrap_err().contains("panicked"));
    }
    assert_eq!(table.read(&[3, 299]).unwrap(), [row(&bytes, 3), row(&bytes, 299)].concat());
    for file in [path, tree, tampered] {
        std::fs::remove_file(file).unwrap();
    }
}

/// Blocks out of the page cache are read past it; blocks it holds are read through it.
#[cfg(target_os = "linux")]
#[test]
fn cached_blocks_come_from_the_page_cache_and_the_rest_past_it() {
    use std::os::fd::AsRawFd;
    let (rows, width) = (2000u64, 90usize);
    let (path, bytes) = table_file(rows, width);
    let file = std::fs::File::open(&path).unwrap();
    file.sync_all().unwrap();
    // SAFETY: advice on an open file; dropping its clean cached pages changes no contents.
    assert_eq!(unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
    let pool = Arc::new(ReadPool::new(4));
    let wanted: Vec<u32> = (0..rows as u32).step_by(45).collect();
    let expected: Vec<u8> = wanted.iter().flat_map(|&row| bytes[123 + row as usize * width..123 + (row as usize + 1) * width].to_vec()).collect();
    for cached in [false, true] {
        if cached {
            std::fs::read(&path).unwrap();
        }
        let table = RowTable::new(&file, 123, width, rows, 1 << 20, pool.clone(), None).unwrap();
        assert_eq!(table.read(&wanted).unwrap(), expected);
        let (_, read, blocks) = table.counts();
        assert_eq!(read, wanted.len() as u64);
        assert_eq!(blocks == 0, cached, "{blocks} blocks read");
    }
    std::fs::remove_file(&path).unwrap();
}
