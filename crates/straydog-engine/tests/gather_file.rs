//! A gather's reads of a mapped file (Windows): what the file gives where it can, and the mapping's bytes where a read falls short.
//!
//! The "mapping" here is a buffer that differs from the file, so a test can tell which of the two a byte came from. No model or GPU work.
#![cfg(all(windows, feature = "cuda"))]

use std::io::Write;

use straydog_engine::gpu::CpuExperts;
use straydog_gpu::ffi;

const FILE_BYTES: usize = 300 << 10;

fn gather(engine: &CpuExperts, source: &[u8]) -> Vec<u8> {
    let mut target = vec![0u8; source.len()];
    let (from, into, bytes) = (source.as_ptr(), target.as_mut_ptr(), source.len() as u64);
    // SAFETY: a live engine, one range in live buffers of that many bytes.
    assert_eq!(unsafe { ffi::sd_cpu_gather(engine.handle(), &from, &into, &bytes, 1) }, 0);
    target
}

#[test]
fn a_short_read_leaves_the_mappings_bytes() {
    let path = std::env::temp_dir().join(format!("straydog-gather-{}", std::process::id()));
    std::fs::File::create(&path).unwrap().write_all(&vec![0xAAu8; FILE_BYTES]).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    // Longer than the file, as no real mapping is: its reads past the file's end fall short or fail.
    let mapping = vec![0x55u8; 2 * FILE_BYTES];
    let beside = vec![0x33u8; 4096];
    let engine = CpuExperts::new(2).unwrap();
    engine.gather_file(&mapping, &file).unwrap();

    // Whole reads: the file's bytes, a piece at a time.
    assert!(gather(&engine, &mapping[4096..4096 + (200 << 10)]).iter().all(|&byte| byte == 0xAA));
    // A read across the file's end comes back short (it has written what it read): the whole range is the mapping's.
    assert!(gather(&engine, &mapping[FILE_BYTES - 1000..FILE_BYTES + 1000]).iter().all(|&byte| byte == 0x55));
    // Wholly past the end: the read fails.
    assert!(gather(&engine, &mapping[FILE_BYTES + 4096..FILE_BYTES + 8192]).iter().all(|&byte| byte == 0x55));
    // Bytes of no mapped file are copied.
    assert!(gather(&engine, &beside).iter().all(|&byte| byte == 0x33));

    drop(engine);
    drop(file);
    std::fs::remove_file(&path).unwrap();
}
