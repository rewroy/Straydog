//! SHA-256 of large files at the speed of a core, not of one read-ahead window. Reads only.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

pub const HASH_CHUNK: usize = 8 << 20;
const HASH_AHEAD: u64 = 256 << 20;
pub const HINT_STEP: u64 = 128 << 10;

/// Hex SHA-256 of an open file from its start. `progress` receives each chunk's byte count.
///
/// A plain sequential read waits on one small kernel read-ahead window at a time (about
/// 1.2 GB/s from an NVMe that reads 4-5 GB/s with more requests in flight), so on Linux
/// a helper thread asks for the next 256 MiB in 128 KiB hints: one WILLNEED reads at most
/// the device's largest request and is submitted at device speed.
pub fn file_sha256(file: &File, progress: &(dyn Fn(u64) + Sync)) -> std::io::Result<String> {
    let mut reader = file;
    reader.seek(SeekFrom::Start(0))?;
    let size = file.metadata()?.len();
    let position = Arc::new(AtomicU64::new(0));
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    std::thread::scope(|scope| {
        let helper = {
            let (position, done) = (position.clone(), done.clone());
            scope.spawn(move || read_ahead(file, size, &position, &done))
        };
        let result = (|| {
            let mut digest = Sha256::new();
            let mut buffer = vec![0u8; HASH_CHUNK];
            loop {
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
                position.fetch_add(count as u64, Ordering::Relaxed);
                progress(count as u64);
            }
            Ok(format!("{:x}", digest.finalize()))
        })();
        done.store(true, Ordering::Relaxed);
        helper.join().ok();
        result
    })
}

/// Each file's hex SHA-256, two files at a time (one digest cannot use two cores).
pub fn files_sha256(files: &[&File], progress: &(dyn Fn(u64) + Sync)) -> std::io::Result<Vec<String>> {
    let next = AtomicU64::new(0);
    let mut results: Vec<Option<std::io::Result<String>>> = (0..files.len()).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<&mut Option<std::io::Result<String>>>> =
        results.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|scope| {
        for _ in 0..2.min(files.len()) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed) as usize;
                let Some(file) = files.get(index) else { return };
                **slots[index].lock().unwrap() = Some(file_sha256(file, progress));
            });
        }
    });
    drop(slots);
    results.into_iter().map(|result| result.expect("every file hashed")).collect()
}

#[cfg(target_os = "linux")]
fn read_ahead(file: &File, size: u64, position: &AtomicU64, done: &std::sync::atomic::AtomicBool) {
    use std::os::fd::AsRawFd;
    let mut hinted = 0u64;
    while hinted < size && !done.load(Ordering::Relaxed) {
        if hinted >= position.load(Ordering::Relaxed) + HASH_AHEAD {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        // SAFETY: a valid descriptor; the hint reads nothing into this process.
        if unsafe { libc::posix_fadvise(file.as_raw_fd(), hinted as i64, HINT_STEP as i64, libc::POSIX_FADV_WILLNEED) }
            != 0
        {
            return;
        }
        hinted += HINT_STEP;
    }
}

#[cfg(not(target_os = "linux"))]
fn read_ahead(_file: &File, _size: u64, _position: &AtomicU64, _done: &std::sync::atomic::AtomicBool) {}

/// Ask the device for these byte ranges of `file` ahead of use (Linux; hints only).
pub fn will_need(file: &File, ranges: &[(u64, u64)]) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        for &(offset, length) in ranges {
            let mut at = offset;
            while at < offset + length {
                // SAFETY: as above.
                if unsafe { libc::posix_fadvise(file.as_raw_fd(), at as i64, HINT_STEP as i64, libc::POSIX_FADV_WILLNEED) }
                    != 0
                {
                    return;
                }
                at += HINT_STEP;
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, ranges);
}
