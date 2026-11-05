//! Times one round of N aligned 4 KiB unbuffered reads at random offsets of a large file, the reads of the N threads released together,
//! from the release to the last one done (no thread start in the time): on one synchronous handle read with std's `seek_read` (what the
//! direct reads were before), on one handle opened for overlapped I/O read through `storage` (what they are now), on a handle each, and
//! one after another.
//!
//! On a synchronous handle the I/O manager runs the reads issued on it one at a time, whatever the number of threads.
//!
//! The last lines time the row pool's way: the reads queued on a pool of 128 threads waiting on a condition variable, woken one by one,
//! and the caller waiting on another for all of them (what a token's PLE rows cost with no verification).
//!
//! A fourth argument starts that many threads reading 64 KiB blocks at random from the same file the whole time (the drive busy with
//! other reads, as it is while a model streams its experts), for the timings under that load.
//!
//! A fifth starts that many threads computing the whole time (the cores busy, as they are while the CPU's experts run).
//!
//! usage: cargo run --release --example read_concurrency -- FILE [threads] [rounds] [load threads] [computing threads]

#[cfg(windows)]
fn main() {
    use std::fs::{File, OpenOptions};
    use std::os::windows::fs::{FileExt, OpenOptionsExt};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Condvar, Mutex};
    use std::time::Instant;
    use straydog_engine::storage::{self, Buffer, ALIGN};

    const NO_BUFFERING: u32 = 0x2000_0000;
    const OVERLAPPED: u32 = 0x4000_0000;

    /// How a thread reads a block.
    #[derive(Clone, Copy)]
    enum Way {
        /// std's `seek_read`.
        Std,
        /// `storage::read_into`'s read, with an event of the thread's own.
        Storage,
    }

    fn read(way: Way, file: &File, buffer: &mut Buffer, offset: u64) {
        let target = &mut buffer.as_mut_slice()[..ALIGN];
        match way {
            Way::Std => assert_eq!(file.seek_read(target, offset).unwrap(), ALIGN),
            Way::Storage => assert_eq!(storage::read_partial(file, target, offset).unwrap(), ALIGN),
        }
    }

    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: read_concurrency FILE [threads] [rounds]");
    let threads: usize = args.next().map_or(12, |text| text.parse().unwrap());
    let rounds: usize = args.next().map_or(80, |text| text.parse().unwrap());
    let load: usize = args.next().map_or(0, |text| text.parse().unwrap());
    let computing: usize = args.next().map_or(0, |text| text.parse().unwrap());
    let size = std::fs::metadata(&path).unwrap().len();
    let blocks = size / ALIGN as u64;
    let open = |flags: u32| OpenOptions::new().read(true).custom_flags(flags).open(&path).unwrap();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_offset = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % blocks * ALIGN as u64
    };

    // `threads` threads, each reading through handle `index % handles.len()`, released together every round.
    let run = |way: Way, handles: Vec<Arc<File>>, next_offset: &mut dyn FnMut() -> u64| -> Vec<f64> {
        let start = Arc::new(Barrier::new(threads + 1));
        let end = Arc::new(Barrier::new(threads + 1));
        let offsets: Arc<Vec<AtomicU64>> = Arc::new((0..threads).map(|_| AtomicU64::new(0)).collect());
        let stop = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..threads)
            .map(|index| {
                let (start, end, offsets, stop) = (start.clone(), end.clone(), offsets.clone(), stop.clone());
                let file = handles[index % handles.len()].clone();
                std::thread::spawn(move || {
                    let mut buffer = Buffer::new(ALIGN);
                    loop {
                        start.wait();
                        if stop.load(Ordering::Relaxed) != 0 {
                            return;
                        }
                        read(way, &file, &mut buffer, offsets[index].load(Ordering::Relaxed));
                        end.wait();
                    }
                })
            })
            .collect();
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            for slot in offsets.iter() {
                slot.store(next_offset(), Ordering::Relaxed);
            }
            let began = Instant::now();
            start.wait();
            end.wait();
            times.push(began.elapsed().as_secs_f64());
        }
        stop.store(1, Ordering::Relaxed);
        start.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        times.sort_by(|a, b| a.total_cmp(b));
        times
    };

    // The reads of a round queued on `POOL` threads that wait on a condition variable, the caller waiting on another.
    const POOL: usize = 128;
    type Queue = (Mutex<(Vec<u64>, usize, bool)>, Condvar, Condvar);
    let pooled = |way: Way, file: Arc<File>, next_offset: &mut dyn FnMut() -> u64| -> Vec<f64> {
        let queue: Arc<Queue> = Arc::new((Mutex::new((Vec::new(), 0, false)), Condvar::new(), Condvar::new()));
        let workers: Vec<_> = (0..POOL)
            .map(|_| {
                let (queue, file) = (queue.clone(), file.clone());
                std::thread::spawn(move || {
                    let mut buffer = Buffer::new(ALIGN);
                    loop {
                        let offset = {
                            let mut state = queue.0.lock().unwrap();
                            loop {
                                if let Some(offset) = state.0.pop() {
                                    break offset;
                                }
                                if state.2 {
                                    return;
                                }
                                state = queue.1.wait(state).unwrap();
                            }
                        };
                        read(way, &file, &mut buffer, offset);
                        let mut state = queue.0.lock().unwrap();
                        state.1 += 1;
                        if state.1 == threads {
                            drop(state);
                            queue.2.notify_all();
                        }
                    }
                })
            })
            .collect();
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let offsets: Vec<u64> = (0..threads).map(|_| next_offset()).collect();
            let began = Instant::now();
            {
                let mut state = queue.0.lock().unwrap();
                state.0.extend(offsets);
                state.1 = 0;
            }
            for _ in 0..threads {
                queue.1.notify_one();
            }
            let mut state = queue.0.lock().unwrap();
            while state.1 < threads {
                state = queue.2.wait(state).unwrap();
            }
            drop(state);
            times.push(began.elapsed().as_secs_f64());
        }
        queue.0.lock().unwrap().2 = true;
        queue.1.notify_all();
        for worker in workers {
            worker.join().unwrap();
        }
        times.sort_by(|a, b| a.total_cmp(b));
        times
    };

    let sequential = |way: Way, file: &File, next_offset: &mut dyn FnMut() -> u64| -> Vec<f64> {
        let mut buffer = Buffer::new(ALIGN);
        let mut times: Vec<f64> = (0..rounds)
            .map(|_| {
                let offsets: Vec<u64> = (0..threads).map(|_| next_offset()).collect();
                let began = Instant::now();
                for offset in offsets {
                    read(way, file, &mut buffer, offset);
                }
                began.elapsed().as_secs_f64()
            })
            .collect();
        times.sort_by(|a, b| a.total_cmp(b));
        times
    };

    let show = |name: &str, times: Vec<f64>| {
        println!("{name}: median {:.0} us, p90 {:.0} us", times[times.len() / 2] * 1e6, times[times.len() * 9 / 10] * 1e6);
    };

    // The background load: `load` threads reading 64 KiB blocks at random through a handle of their own until the timings are done.
    let busy = Arc::new(AtomicUsize::new(0));
    let loaders: Vec<_> = (0..load)
        .map(|index| {
            let (file, busy) = (open(NO_BUFFERING | OVERLAPPED), busy.clone());
            std::thread::spawn(move || {
                let mut buffer = Buffer::new(64 << 10);
                let mut seed = 0x1234_5678_9ABC_DEF1u64 + index as u64 * 7919;
                while busy.load(Ordering::Relaxed) == 0 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let offset = seed % (size / (64 << 10)) * (64 << 10);
                    storage::read_partial(&file, buffer.as_mut_slice(), offset).unwrap();
                }
            })
        })
        .collect();
    let computers: Vec<_> = (0..computing)
        .map(|_| {
            let busy = busy.clone();
            std::thread::spawn(move || {
                let mut sum = 1u64;
                while busy.load(Ordering::Relaxed) == 0 {
                    for step in 0..100_000u64 {
                        sum = std::hint::black_box(sum.wrapping_mul(6364136223846793005).wrapping_add(step));
                    }
                }
                sum
            })
        })
        .collect();
    let synchronous = Arc::new(open(NO_BUFFERING));
    let overlapped = Arc::new(open(NO_BUFFERING | OVERLAPPED));
    let each_sync: Vec<Arc<File>> = (0..threads).map(|_| Arc::new(open(NO_BUFFERING))).collect();
    let each_overlapped: Vec<Arc<File>> = (0..threads).map(|_| Arc::new(open(NO_BUFFERING | OVERLAPPED))).collect();
    println!("file {:.1} GiB, {threads} reads of 4 KiB, unbuffered, {rounds} rounds, {load} threads of 64 KiB reads and {computing} computing threads beside", size as f64 / (1u64 << 30) as f64);
    show(&format!("{threads} one after another (synchronous handle, seek_read)"), sequential(Way::Std, &synchronous, &mut next_offset));
    show(&format!("{threads} one after another (overlapped handle, storage)"), sequential(Way::Storage, &overlapped, &mut next_offset));
    show(&format!("{threads} threads, one synchronous handle, seek_read (before)"), run(Way::Std, vec![synchronous.clone()], &mut next_offset));
    show(&format!("{threads} threads, one overlapped handle, storage (after)"), run(Way::Storage, vec![overlapped.clone()], &mut next_offset));
    show(&format!("{threads} threads, a synchronous handle each, seek_read"), run(Way::Std, each_sync, &mut next_offset));
    show(&format!("{threads} threads, an overlapped handle each, storage"), run(Way::Storage, each_overlapped, &mut next_offset));
    show(&format!("{threads} threads, one synchronous handle, storage (a synchronous handle stays serial)"), run(Way::Storage, vec![synchronous.clone()], &mut next_offset));
    show(&format!("{threads} threads, one overlapped handle, storage (again)"), run(Way::Storage, vec![overlapped.clone()], &mut next_offset));
    show(&format!("{threads} reads queued on a {POOL}-thread pool, one overlapped handle, storage"), pooled(Way::Storage, overlapped.clone(), &mut next_offset));
    show(&format!("{threads} reads queued on a {POOL}-thread pool, one synchronous handle, seek_read (before)"), pooled(Way::Std, synchronous.clone(), &mut next_offset));
    // `storage::direct` itself: the handle the engine reads with.
    let reopened = File::open(&path).unwrap();
    if let Some(direct) = storage::direct(&reopened) {
        show(&format!("{threads} threads, storage::direct's handle"), run(Way::Storage, vec![Arc::new(direct)], &mut next_offset));
    }
    busy.store(1, Ordering::Relaxed);
    for loader in loaders {
        loader.join().unwrap();
    }
    for computer in computers {
        computer.join().unwrap();
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("read_concurrency times Windows handles");
}
