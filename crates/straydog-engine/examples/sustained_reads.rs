//! Random 4 KiB unbuffered reads of a large file from N threads for a number of seconds, reads a second, on Windows, with or without a
//! data section on the file.
//!
//! While a file is mapped, or has pages in the system cache, NTFS flushes the range of every unbuffered read before the read, and one such
//! flush runs at a time on the file: 128 threads read at the rate of one (11K reads a second where the drive does 650K). The same file
//! opened for unbuffered reads alone, with nothing else open on it, reads at the drive's speed, and opening it so drops the cached pages
//! it had. A mapping made in the process, or a handle opened for cached reads that stays, brings the slow rate back; so does cached I/O
//! on the file while the process runs. The engine's row tables run into it (`rows`, `Weights::cache_rows`).
//!
//! usage: cargo run --release --example sustained_reads -- FILE threads seconds [--map] [--cached-handle]
//!   --map: map the whole file first and touch a page of it; --cached-handle: open the file for cached reads first and keep that handle
//!   (the reads are then on a handle reopened from it for unbuffered access, as the engine's were)

#[cfg(windows)]
fn main() {
    use std::os::windows::fs::OpenOptionsExt;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use straydog_engine::storage::{self, Buffer, ALIGN};

    const NO_BUFFERING: u32 = 0x2000_0000;
    const OVERLAPPED: u32 = 0x4000_0000;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, threads, seconds, ..] = &args[..] else {
        eprintln!("usage: sustained_reads FILE threads seconds [--map] [--cached-handle]");
        return;
    };
    let (threads, seconds): (usize, f64) = (threads.parse().unwrap(), seconds.parse().unwrap());
    let (map, cached) = (args.iter().any(|arg| arg == "--map"), args.iter().any(|arg| arg == "--cached-handle"));
    let cached_handle = cached.then(|| std::fs::File::open(path).unwrap());
    // SAFETY: a read-only mapping, of which one byte is read.
    let _mapping = map.then(|| {
        let file = std::fs::File::open(path).unwrap();
        let mapping = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        std::hint::black_box(mapping[0]);
        mapping
    });
    let direct = Arc::new(match &cached_handle {
        Some(file) => storage::direct(file).unwrap(),
        None => std::fs::OpenOptions::new().read(true).custom_flags(NO_BUFFERING | OVERLAPPED).open(path).unwrap(),
    });
    let blocks = direct.metadata().unwrap().len() / ALIGN as u64;
    let stop = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicU64::new(0));
    let workers: Vec<_> = (0..threads)
        .map(|index| {
            let (direct, stop, count) = (direct.clone(), stop.clone(), count.clone());
            std::thread::spawn(move || {
                let mut buffer = Buffer::new(ALIGN);
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (index as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                    ^ (std::process::id() as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
                while !stop.load(Ordering::Relaxed) {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    storage::read_partial(&direct, &mut buffer.as_mut_slice()[..ALIGN], seed % blocks * ALIGN as u64).unwrap();
                    count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    let started = Instant::now();
    std::thread::sleep(Duration::from_secs_f64(seconds));
    let done = count.load(Ordering::Relaxed);
    let elapsed = started.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    for worker in workers {
        worker.join().unwrap();
    }
    println!("{threads} threads, mapped {map}, cached handle {cached}: {done} reads in {elapsed:.2} s = {:.0} reads/s", done as f64 / elapsed);
}

#[cfg(not(windows))]
fn main() {
    eprintln!("sustained_reads times Windows handles");
}
