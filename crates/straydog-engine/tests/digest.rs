use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use straydog_engine::bundle;
use straydog_engine::digest;
use straydog_engine::load;

#[test]
fn every_pinned_shard_and_drafter_has_its_chunk_digests() {
    for quantization in bundle::QUANTIZATIONS {
        let pins = bundle::bundle(quantization).unwrap();
        for asset in pins.shards.iter().chain([&pins.drafter]) {
            let pieces = bundle::chunk_digests(&asset.sha256).unwrap().unwrap_or_else(|| panic!("{} unpinned", asset.name));
            assert_eq!(pieces.len() as u64, asset.bytes.div_ceil(digest::CHUNK_BYTES), "{}", asset.name);
        }
    }
}

#[test]
fn chunk_digests_authenticate_a_file_and_catch_a_changed_byte() {
    let path = std::env::temp_dir().join(format!("straydog-digest-{}.bin", std::process::id()));
    let mut file = File::create(&path).unwrap();
    // Not a whole number of blocks: a direct read stops at the end of the file.
    file.write_all(&(0..100_001u32).flat_map(u32::to_le_bytes).collect::<Vec<u8>>()).unwrap();
    let file = File::open(&path).unwrap();
    let (whole, pieces) = digest::file_digests(&file, &|_| {}).unwrap();
    assert_eq!(whole, digest::file_sha256(&file, &|_| {}).unwrap());
    load::verify_file("test", &file, &pieces).unwrap();
    assert!(load::verify_file("test", &file, &[]).is_err());
    let mut changed = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    changed.seek(SeekFrom::Start(12_345)).unwrap();
    changed.write_all(&[0xff]).unwrap();
    changed.sync_all().unwrap();
    assert!(load::verify_file("test", &file, &pieces).unwrap_err().contains("digest mismatch"));
    std::fs::remove_file(&path).unwrap();
}
