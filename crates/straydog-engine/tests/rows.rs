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

/// Threads asking for overlapping sets of rows of one table at once (duplicates inside a set and between threads, rows that straddle the
/// 4 KiB blocks, a pool with fewer threads than askers) all get the file's rows, each checked against the tree, whether the cache holds
/// none of them, a few, or all; with room for all, no row is read twice.
#[test]
fn threads_asking_for_overlapping_rows_all_get_the_files() {
    let (rows, width) = (3000u64, 3001usize);
    let (path, bytes) = table_file(rows, width);
    let tree = path.with_extension("tree");
    let digest = verity::build(&std::fs::File::open(&path).unwrap(), &tree, &|_| {}).unwrap();
    let checker = Arc::new(verity::Verifier::open(&tree, bytes.len() as u64, &digest, 16 << 10).unwrap());
    let verifier: rows::Verifier = Arc::new(move |offset, data| checker.verify(offset, data));
    let pool = Arc::new(ReadPool::new(8));
    for budget in [0u64, 20 * (3001 + 24), 64 << 20] {
        let file = std::fs::File::open(&path).unwrap();
        let table = RowTable::new(&file, 123, width, rows, budget, pool.clone(), Some(verifier.clone())).unwrap();
        std::thread::scope(|scope| {
            for thread in 0..12u64 {
                let (table, bytes) = (&table, &bytes);
                scope.spawn(move || {
                    let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (thread + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                    for step in 0..40usize {
                        // (a hot range every thread shares, a row twice in a set now and then, the rest anywhere)
                        let mut wanted: Vec<u32> = (0..1 + step % 60).map(|index| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            ((seed >> 20) % if index % 3 == 0 { 50 } else { rows }) as u32
                        }).collect();
                        wanted.extend_from_within(..wanted.len().min(3));
                        if step % 2 == 0 {
                            table.prefetch(&wanted).unwrap();
                        }
                        let got = table.read(&wanted).unwrap();
                        for (index, &row) in wanted.iter().enumerate() {
                            let start = 123 + row as usize * width;
                            assert_eq!(&got[index * width..(index + 1) * width], &bytes[start..start + width], "row {row}");
                        }
                    }
                });
            }
        });
        let (asked, read, _) = table.counts();
        assert!(asked > 0);
        if budget == 64 << 20 {
            assert!(read <= rows, "{read} rows read, the table has {rows}");
        }
    }
    for file in [path, tree] {
        std::fs::remove_file(file).unwrap();
    }
}

/// A row in a block that does not match the tree fails every read of it, whoever asks, and its bytes are never cached: once the file is
/// whole again the same row reads right. The rows that touch the block fail with it, and no others.
#[test]
fn a_block_that_does_not_verify_fails_its_rows_and_is_never_cached() {
    let (rows, width) = (4000u64, 90usize);
    let (path, bytes) = table_file(rows, width);
    let tree = path.with_extension("tree");
    let digest = verity::build(&std::fs::File::open(&path).unwrap(), &tree, &|_| {}).unwrap();
    let checker = Arc::new(verity::Verifier::open(&tree, bytes.len() as u64, &digest, 16 << 10).unwrap());
    let verifier: rows::Verifier = Arc::new(move |offset, data| checker.verify(offset, data));
    // One bit of row 1000 flipped. It is at 123 + 90000 = 90123, in block 22 (bytes 90112 to 94207), which rows 999 to 1045 touch.
    let mut tampered = bytes.clone();
    tampered[123 + 1000 * 90 + 5] ^= 1;
    let copy = path.with_extension("copy");
    std::fs::write(&copy, &tampered).unwrap();
    let file = std::fs::File::open(&copy).unwrap();
    let table = RowTable::new(&file, 123, width, rows, 64 << 20, Arc::new(ReadPool::new(8)), Some(verifier)).unwrap();
    let row = |row: usize| bytes[123 + row * width..123 + (row + 1) * width].to_vec();
    std::thread::scope(|scope| {
        for thread in 0..6usize {
            let table = &table;
            scope.spawn(move || {
                for step in 0..20usize {
                    let wanted = if thread % 2 == 0 { vec![1000u32] } else { vec![(999 + (step * 7) % 47) as u32, 1000] };
                    if step % 2 == 0 {
                        table.prefetch(&wanted).unwrap();
                    }
                    assert!(table.read(&wanted).unwrap_err().contains("does not verify"));
                }
            });
        }
    });
    // (the rows around the block, in the blocks before and after it, read)
    assert_eq!(table.read(&[998, 1046, 5]).unwrap(), [row(998), row(1046), row(5)].concat());
    assert!(table.read(&[999]).is_err() && table.read(&[1045]).is_err());
    // The file whole again: nothing of the bad block was kept.
    std::fs::write(&copy, &bytes).unwrap();
    assert_eq!(table.read(&[1000, 999, 1045]).unwrap(), [row(1000), row(999), row(1045)].concat());
    for file in [path, tree, copy] {
        std::fs::remove_file(file).unwrap();
    }
}

/// A table over a file that is mapped in the process, and read through the page cache, reads the same bytes (on Windows the engine lets go
/// of such a file before its table reads it, `Weights::cache_rows`; a table that is given one still works).
#[test]
fn a_mapped_file_reads_right() {
    let (rows, width) = (5001u64, 90usize);
    let (path, bytes) = table_file(rows, width);
    let file = std::fs::File::open(&path).unwrap();
    // SAFETY: a read-only mapping of a file nothing writes.
    let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    assert_eq!(map[1000], bytes[1000]);
    let table = RowTable::new(&file, 123, width, rows, 1 << 20, Arc::new(ReadPool::new(8)), None).unwrap();
    let wanted: Vec<u32> = (0..rows as u32).step_by(37).collect();
    table.prefetch(&wanted).unwrap();
    let got = table.read(&wanted).unwrap();
    for (index, &row) in wanted.iter().enumerate() {
        let start = 123 + row as usize * width;
        assert_eq!(&got[index * width..(index + 1) * width], &bytes[start..start + width], "row {row}");
    }
    std::fs::remove_file(&path).unwrap();
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

/// What Qwen3.8's PLE table gets when `--row-cache` is not given: all of it while half of the host
/// memory beside the sequence state holds it, else that half, never less than the default budget.
#[test]
fn the_automatic_budget_holds_the_table_while_half_the_free_memory_does() {
    const GIB: u64 = 1 << 30;
    let (count, width) = (320_001_536u64, 90u64);
    let all = rows::full_budget(count, width);
    assert_eq!(all, count * 114);
    // Two R9700s and 125 GiB with 14 GiB of state, half of 111 GiB holds the 34 GiB table.
    assert_eq!(rows::auto_budget(count, width, 125 * GIB, 14 * GIB), all);
    // 64 GiB: half of what the state leaves, not the table.
    assert_eq!(rows::auto_budget(count, width, 64 * GIB, 14 * GIB), 25 * GIB);
    assert_eq!(rows::auto_budget(count, width, 16 * GIB, 14 * GIB), GIB);
    // Memory short, or unknown: today's default.
    assert_eq!(rows::auto_budget(count, width, 14 * GIB + GIB / 4, 14 * GIB), rows::DEFAULT_CACHE_BYTES);
    assert_eq!(rows::auto_budget(count, width, 0, 0), rows::DEFAULT_CACHE_BYTES);
    assert_eq!(rows::auto_budget(count, width, 8 * GIB, 14 * GIB), rows::DEFAULT_CACHE_BYTES);
    // A table smaller than the default has no use for more than itself.
    assert_eq!(rows::auto_budget(1000, width, 125 * GIB, 0), rows::full_budget(1000, width));
}

/// A cache that holds a whole table of tens of millions of rows takes no memory until rows are met:
/// its slots, owners and recency flags are zeroed memory and its index grows as it is used.
#[cfg(target_os = "linux")]
#[test]
fn a_cache_for_a_whole_table_takes_memory_only_for_the_rows_met() {
    fn resident_bytes() -> u64 {
        let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
        statm.split_whitespace().nth(1).unwrap().parse::<u64>().unwrap() * 4096
    }
    let (rows, width) = (40_000_000u64, 90usize);
    let path = std::env::temp_dir().join(format!("straydog-rows-whole-{}.bin", std::process::id()));
    let file = std::fs::File::create(&path).unwrap();
    // (sparse: zeros but for a few rows near the end)
    file.set_len(rows * width as u64).unwrap();
    std::os::unix::fs::FileExt::write_all_at(&file, &[7u8; 90], (rows - 1) * 90).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let before = resident_bytes();
    let table = RowTable::new(&file, 0, width, rows, rows::full_budget(rows, width as u64), Arc::new(ReadPool::new(4)), None).unwrap();
    let after = resident_bytes();
    assert!(after - before < 100 << 20, "{} MiB resident for an empty cache of {} MiB", (after - before) >> 20,
            rows::full_budget(rows, width as u64) >> 20);
    let wanted = [0u32, 12_345_678, (rows - 1) as u32];
    let got = table.read(&wanted).unwrap();
    assert_eq!(&got[..90], &[0u8; 90]);
    assert_eq!(&got[180..], &[7u8; 90]);
    // Cached now: asked again, nothing is read.
    let (_, read, _) = table.counts();
    assert_eq!(table.read(&wanted).unwrap(), got);
    assert_eq!(table.counts().1, read);
    std::fs::remove_file(&path).unwrap();
}
