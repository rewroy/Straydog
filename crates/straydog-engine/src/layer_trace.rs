//! Explicit diagnostic builds only: wall/CPU time and process I/O around layer-prefill phases.
#[cfg(feature = "prefill-observer")]
mod enabled {
    use std::sync::atomic::{AtomicU64, Ordering};
    static VERIFY_NS: AtomicU64 = AtomicU64::new(0);
    static VERIFY_BYTES: AtomicU64 = AtomicU64::new(0);
    static PROOF_BYTES: AtomicU64 = AtomicU64::new(0);
    fn totals() -> [u64; 3] {
        [
            VERIFY_NS.load(Ordering::Relaxed),
            VERIFY_BYTES.load(Ordering::Relaxed),
            PROOF_BYTES.load(Ordering::Relaxed),
        ]
    }
    pub(crate) struct Verify {
        at: std::time::Instant,
        bytes: u64,
    }
    impl Verify {
        pub(crate) fn new(bytes: u64) -> Self {
            Self {
                at: std::time::Instant::now(),
                bytes,
            }
        }
    }
    impl Drop for Verify {
        fn drop(&mut self) {
            VERIFY_NS.fetch_add(self.at.elapsed().as_nanos() as u64, Ordering::Relaxed);
            VERIFY_BYTES.fetch_add(self.bytes, Ordering::Relaxed);
        }
    }
    pub(crate) fn proof_read(bytes: u64) {
        PROOF_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) struct Span {
        phase: &'static str,
        layer: Option<usize>,
        begin: usize,
        end: usize,
        bytes: u64,
        clock: std::time::Instant,
        at: u64,
        io: Option<u64>,
        cpu: Option<u64>,
        counts: [u64; 3],
    }
    pub(crate) fn tick() -> u64 {
        #[cfg(target_os = "linux")]
        {
            let mut t = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: a writable timespec and a valid monotonic clock.
            if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) } == 0 {
                return t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64;
            }
        }
        0
    }
    fn io() -> Option<u64> {
        if !cfg!(target_os = "linux") {
            return None;
        }
        std::fs::read_to_string("/proc/self/io")
            .ok()?
            .lines()
            .find_map(|line| line.strip_prefix("read_bytes:")?.trim().parse().ok())
    }
    fn cpu() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: rusage is plain storage filled by getrusage for this process.
            let mut r: libc::rusage = unsafe { std::mem::zeroed() };
            if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut r) } == 0 {
                return Some(
                    (r.ru_utime.tv_sec as u64 + r.ru_stime.tv_sec as u64) * 1_000_000_000
                        + (r.ru_utime.tv_usec as u64 + r.ru_stime.tv_usec as u64) * 1000,
                );
            }
        }
        None
    }
    impl Span {
        pub(crate) fn new(
            phase: &'static str,
            layer: Option<usize>,
            begin: usize,
            end: usize,
            bytes: u64,
        ) -> Self {
            Self {
                phase,
                layer,
                begin,
                end,
                bytes,
                clock: std::time::Instant::now(),
                at: tick(),
                io: io(),
                cpu: cpu(),
                counts: totals(),
            }
        }
    }
    impl Drop for Span {
        fn drop(&mut self) {
            let wall_ns = self.clock.elapsed().as_nanos() as u64;
            let counts = totals();
            eprintln!(
                "Straydog layer trace: {}",
                serde_json::json!({
                    "source": "rust", "phase": self.phase, "layer": self.layer, "begin": self.begin, "end": self.end,
                    "logical_bytes": self.bytes, "start_ns": self.at, "end_ns": tick(), "wall_ns": wall_ns,
                    "read_bytes": self.io.zip(io()).map(|(a,b)| b.saturating_sub(a)),
                    "cpu_ns": self.cpu.zip(cpu()).map(|(a,b)| b.saturating_sub(a)),
                    "verification_ns_sum": counts[0].saturating_sub(self.counts[0]),
                    "verification_attempted_bytes": counts[1].saturating_sub(self.counts[1]),
                    "proof_requested_bytes": counts[2].saturating_sub(self.counts[2]),
                    "io_scope": "whole process, including concurrent read-ahead; nested phases overlap"
                })
            );
        }
    }
}
#[cfg(feature = "prefill-observer")]
pub(crate) use enabled::{proof_read, Span, Verify};
#[cfg(not(feature = "prefill-observer"))]
pub(crate) struct Span;
#[cfg(not(feature = "prefill-observer"))]
impl Span {
    #[inline]
    pub(crate) fn new(_: &'static str, _: Option<usize>, _: usize, _: usize, _: u64) -> Self {
        Self
    }
}

#[cfg(not(feature = "prefill-observer"))]
pub(crate) struct Verify;
#[cfg(not(feature = "prefill-observer"))]
impl Verify {
    #[inline]
    pub(crate) fn new(_: u64) -> Self {
        Self
    }
}
#[cfg(not(feature = "prefill-observer"))]
#[inline]
pub(crate) fn proof_read(_: u64) {}
