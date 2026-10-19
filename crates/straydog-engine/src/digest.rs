//! SHA-256 of large files at the speed of a core, not of one read-ahead window. Reads only.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

pub const HASH_CHUNK: usize = 8 << 20;
#[cfg(target_os = "linux")]
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

/// Bytes per piece of a file's chunk digests: the SHA-256 of each consecutive piece (the last
/// one shorter), pinned beside the whole file's (see `bundle::chunk_digests`).
pub const CHUNK_BYTES: u64 = 256 << 20;

/// Whether each file's pieces have these chunk digests. Every core hashes pieces of its own,
/// so a file authenticates at the speed of all of them instead of one: a single SHA-256 is a
/// chain no second core can join. A file whose piece count does not match fails.
pub fn verify_chunks(files: &[(&File, &[String])], progress: &(dyn Fn(u64) + Sync)) -> std::io::Result<Vec<bool>> {
    let mut sizes = Vec::with_capacity(files.len());
    let mut pieces = Vec::new();
    for (index, (file, digests)) in files.iter().enumerate() {
        let size = file.metadata()?.len();
        if digests.len() as u64 == size.div_ceil(CHUNK_BYTES) {
            pieces.extend((0..digests.len()).map(|piece| (index, piece)));
        }
        sizes.push(size);
    }
    let matched: Vec<std::sync::atomic::AtomicBool> = files
        .iter()
        .zip(&sizes)
        .map(|((_, digests), size)| std::sync::atomic::AtomicBool::new(digests.len() as u64 == size.div_ceil(CHUNK_BYTES)))
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failure: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    let threads = std::thread::available_parallelism().map_or(4, |count| count.get()).min(pieces.len()).max(1);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut buffer = vec![0u8; HASH_CHUNK];
                loop {
                    let Some(&(index, piece)) = pieces.get(next.fetch_add(1, Ordering::Relaxed)) else { return };
                    if !matched[index].load(Ordering::Relaxed) || failure.lock().unwrap().is_some() {
                        continue;
                    }
                    let (file, digests) = files[index];
                    let offset = piece as u64 * CHUNK_BYTES;
                    let length = CHUNK_BYTES.min(sizes[index] - offset);
                    match piece_sha256(file, offset, length, &mut buffer, progress) {
                        Ok(digest) if digest == digests[piece] => {}
                        Ok(_) => matched[index].store(false, Ordering::Relaxed),
                        Err(exc) => {
                            failure.lock().unwrap().get_or_insert(exc);
                        }
                    }
                }
            });
        }
    });
    if let Some(exc) = failure.into_inner().unwrap() {
        return Err(exc);
    }
    Ok(matched.iter().map(|matched| matched.load(Ordering::Relaxed)).collect())
}

/// A file's whole SHA-256 and its chunk digests in one pass (for pinning them).
pub fn file_digests(file: &File, progress: &(dyn Fn(u64) + Sync)) -> std::io::Result<(String, Vec<String>)> {
    let size = file.metadata()?.len();
    let mut whole = Sha256::new();
    let mut pieces = Vec::new();
    let mut buffer = vec![0u8; HASH_CHUNK];
    let mut offset = 0;
    while offset < size {
        let length = CHUNK_BYTES.min(size - offset);
        let mut piece = Sha256::new();
        read_range(file, offset, length, &mut buffer, &mut |bytes| {
            whole.update(bytes);
            piece.update(bytes);
            progress(bytes.len() as u64);
        })?;
        pieces.push(format!("{:x}", piece.finalize()));
        offset += length;
    }
    Ok((format!("{:x}", whole.finalize()), pieces))
}

fn piece_sha256(file: &File, offset: u64, length: u64, buffer: &mut [u8], progress: &(dyn Fn(u64) + Sync))
                -> std::io::Result<String> {
    will_need(file, &[(offset, length)]);
    let mut digest = Sha256::new();
    read_range(file, offset, length, buffer, &mut |bytes| {
        digest.update(bytes);
        progress(bytes.len() as u64);
    })?;
    Ok(format!("{:x}", digest.finalize()))
}

/// `length` bytes from `offset`, a buffer at a time, without moving the handle's position.
fn read_range(file: &File, offset: u64, length: u64, buffer: &mut [u8], sink: &mut dyn FnMut(&[u8]))
              -> std::io::Result<()> {
    let mut done = 0;
    while done < length {
        let want = buffer.len().min((length - done) as usize);
        #[cfg(unix)]
        let count = std::os::unix::fs::FileExt::read_at(file, &mut buffer[..want], offset + done)?;
        #[cfg(windows)]
        let count = std::os::windows::fs::FileExt::seek_read(file, &mut buffer[..want], offset + done)?;
        if count == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file shorter than its size"));
        }
        sink(&buffer[..count]);
        done += count as u64;
    }
    Ok(())
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
