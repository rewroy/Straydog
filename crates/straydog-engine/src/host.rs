//! Host facts the isolated worker cannot observe itself: the runtime measures them before
//! spawning it (the worker runs in its own cgroup namespace, where its limits are invisible).

/// Physical memory, bounded on Linux by every cgroup memory limit over this process (v2
/// `memory.max`, v1 `memory.limit_in_bytes`); 0 if unknown.
pub fn host_memory_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: sysconf has no preconditions.
        let (pages, page) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGE_SIZE)) };
        if pages <= 0 || page <= 0 {
            return 0;
        }
        linux_memory(pages as u64 * page as u64, std::path::Path::new("/proc/self/cgroup"), std::path::Path::new("/sys/fs/cgroup"))
    }
    #[cfg(windows)]
    {
        windows_memory()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        0
    }
}

/// Windows: this process asks never to have its execution speed throttled (EcoQoS: lower clocks and efficient cores for a process
/// the system judges to be in the background, which a worker without a window is) and keeps the timer resolution it asks for.
/// Returns whether the system took it; elsewhere there is nothing to ask, and it returns true.
pub fn full_speed() -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        };
        // (each bit in ControlMask and clear in StateMask: that throttling is off, whatever the system's own policy)
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
            StateMask: 0,
        };
        // SAFETY: this process's pseudo handle and an initialized structure of the size passed.
        unsafe {
            SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, std::ptr::from_ref(&state).cast(),
                                  std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32) != 0
        }
    }
    #[cfg(not(windows))]
    true
}

/// How a thread of the engine's own work competes with the rest of it for the cores, the memory bandwidth and the drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    /// As the thread was made.
    Normal,
    /// Work that gives way to the request's: the page-locking of the banks behind "ready" and the check of what it locks.
    Background,
}

/// A thread's lower priority ([`Priority::Background`]) for as long as this is held; [`Priority::Normal`] changes nothing. Windows begins the
/// thread's background mode (`THREAD_MODE_BACKGROUND_BEGIN`: lower CPU, I/O and memory priority) and ends it when this is dropped. Linux sets the
/// thread's nice value to 19 and its I/O class to idle, and keeps them: a thread cannot take its own nice value back without a privilege, and the
/// threads that ask for this end with the work. Only the calling thread is touched, never the process or the threads it started before.
#[must_use = "the priority is lowered for as long as this is held"]
pub struct Lowered {
    /// The thread's background mode was begun here, and is ended on drop.
    #[cfg(windows)]
    began: bool,
}

impl Lowered {
    /// Lower the calling thread's priority as `priority` says. A system that refuses leaves the thread as it was.
    pub fn begin(priority: Priority) -> Lowered {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_MODE_BACKGROUND_BEGIN};
            // SAFETY: the calling thread's pseudo handle.
            let began = priority == Priority::Background && unsafe { SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN) } != 0;
            Lowered { began }
        }
        #[cfg(target_os = "linux")]
        {
            if priority == Priority::Background {
                // (the idle class of `ioprio_set` in the class bits of the value, its level 0; `IOPRIO_WHO_PROCESS` with 0 names the caller)
                const IOPRIO_IDLE: libc::c_long = 3 << 13;
                const IOPRIO_WHO_PROCESS: libc::c_long = 1;
                // SAFETY: both calls change only the calling thread's own scheduling; a refusal is not an error here, the thread is as it was.
                unsafe {
                    let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
                    libc::setpriority(libc::PRIO_PROCESS, tid, 19);
                    libc::syscall(libc::SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0 as libc::c_long, IOPRIO_IDLE);
                }
            }
            Lowered {}
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            let _ = priority;
            Lowered {}
        }
    }
}

#[cfg(windows)]
impl Drop for Lowered {
    fn drop(&mut self) {
        if self.began {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_MODE_BACKGROUND_END};
            // SAFETY: the calling thread's pseudo handle; this is held on the stack of the thread that began it.
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END) };
        }
    }
}

/// The physical memory available on the machine now (what other programs leave), 0 where unknown.
pub fn available_bytes() -> u64 {
    #[cfg(windows)]
    {
        windows_available().0
    }
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        meminfo.lines().find_map(|line| line.strip_prefix("MemAvailable:")).and_then(|value| value.split_whitespace().next()?.parse::<u64>().ok())
            .map_or(0, |kib| kib * 1024)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    0
}

/// This process's memory for a request's log line: resident bytes, the bytes available on the machine, and its page faults so far.
pub fn memory_note() -> Option<(u64, u64, u64)> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: the pseudo handle of this process and an initialized structure with its size set.
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) } == 0 {
            return None;
        }
        Some((counters.WorkingSetSize as u64, windows_available().0, u64::from(counters.PageFaultCount)))
    }
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
        // (past the command's closing parenthesis: state is field 3, minflt 10, majflt 12, rss 24 in pages)
        let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
        let number = |field: usize| fields.get(field - 3)?.parse::<u64>().ok();
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let available = meminfo.lines().find_map(|line| line.strip_prefix("MemAvailable:"))?.split_whitespace().next()?.parse::<u64>().ok()? * 1024;
        Some((number(24)? * 4096, available, number(10)? + number(12)?))
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    None
}

/// Ask the system for the pages of these byte ranges (address, length) of a mapped file in as few large requests as the caller makes of them: Windows
/// reads each range in one request of the I/O manager's (`PrefetchVirtualMemory`), Linux in the device's request size (`madvise(MADV_WILLNEED)` reads at
/// most that much of its range, so a range goes in steps of 128 KiB). Only a hint on the mapping: nothing is locked, no commit is taken, and the pages
/// wait in memory (Windows' standby list) for their first touch, a soft fault, without joining the working set. The ranges must lie in a mapping the
/// caller keeps mapped meanwhile; one that is not mapped is refused and nothing more.
pub fn prefetch(ranges: &[(usize, usize)]) -> Result<(), String> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const PAGE: usize = 4096;
        let entries: Vec<WIN32_MEMORY_RANGE_ENTRY> = ranges.iter().filter(|range| range.1 > 0).map(|&(start, length)| {
            let (begin, end) = (start & !(PAGE - 1), (start + length + PAGE - 1) & !(PAGE - 1));
            WIN32_MEMORY_RANGE_ENTRY { VirtualAddress: begin as *mut std::ffi::c_void, NumberOfBytes: end - begin }
        }).collect();
        if entries.is_empty() {
            return Ok(());
        }
        // SAFETY: the pseudo handle of this process and ranges of its address space, which the call only reads the file pages of.
        if unsafe { PrefetchVirtualMemory(GetCurrentProcess(), entries.len(), entries.as_ptr(), 0) } == 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        Ok(())
    }
    #[cfg(target_os = "linux")]
    {
        const PAGE: usize = 4096;
        const STEP: usize = 128 << 10;
        for &(start, length) in ranges {
            let end = start + length;
            let mut at = start & !(PAGE - 1);
            while at < end {
                // SAFETY: the hint reads nothing into this process, and an address that is not mapped is refused.
                if unsafe { libc::madvise(at as *mut libc::c_void, STEP.min(end - at), libc::MADV_WILLNEED) } != 0 {
                    return Err(std::io::Error::last_os_error().to_string());
                }
                at += STEP;
            }
        }
        Ok(())
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = ranges;
        Ok(())
    }
}

/// Physical memory a guaranteed working set and page locks leave to the rest of the system, which
/// they otherwise take without looking at what the other applications and the kernel need.
pub fn left_free(memory: u64) -> u64 {
    (memory / 8).max(4 << 30)
}

/// `wanted` bytes held in memory by a process with `held` there already, at most what `available`
/// physical memory gives while [`left_free`] stays free.
pub fn within_free(wanted: u64, held: u64, available: u64, memory: u64) -> u64 {
    wanted.min(held.saturating_add(available.saturating_sub(left_free(memory))))
}

/// The marks of `memory` that the working set's limit follows: the lower under which it gives back, the upper over which it may take more.
#[cfg(any(windows, test))]
fn working_set_marks(memory: u64) -> (u64, u64) {
    ((memory / 20).max(3 << 30), (memory / 13).max(4 << 30))
}

/// The limit for a working set of `held` with `available` free, of what it wants and at least `least`. Under the lower mark it gives back, over the
/// upper it may take more, and between them it stays. The marks are low because giving back is dear: IQ3_XXS on 62 GiB read a 33K prompt in 58 s with
/// 6M page faults at a working set of 40 GiB (2 GiB available), in 93 s with 37M at 32 GiB (7 GiB available), and in 117 s with 63M at 28 GiB.
#[cfg(any(windows, test))]
fn working_set_limit(wanted: u64, least: u64, held: u64, available: u64, memory: u64) -> u64 {
    let (low, high) = working_set_marks(memory);
    let limit = if available < low {
        held.saturating_sub(low - available)
    } else if available > high {
        held + (available - high)
    } else {
        held
    };
    limit.min(wanted).max(least.min(wanted))
}

/// Nonzero while the machine pages, 0 otherwise: [`keep_resident`] keeps it with every limit it sets, and the CPU engine's read-ahead hint on
/// Windows reads it with one load instead of asking the system (`CpuExperts::watch_paging`). The pages the CPU's experts touch are out of
/// memory only while it does: with RAM to spare the banks stay in the standby list, and a read-ahead of pages already there only costs time
/// (113 us a layer, 5.4 ms a token, 30.4 to 24.9 tok/s on a machine with RAM to spare). It is set while the system reads the drive for page
/// faults ([`drive_busy`]), which covers a cold start with memory to spare as well as a working set held at its limit; where the system does
/// not say, while its free memory is short ([`memory_short`]).
pub static PAGING: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether free memory is short with `available`: from the upper mark of the working set's limit, over which it may take more, until one band
/// above it. A working set held at its limit rests between the marks (RTX 5070 Ti, 62 GiB: 2.5 to 5.3 GiB available while serving, marks 3.1
/// and 4.7), so the lower mark alone would miss most of the time it pages. The band keeps the hint from being switched on and off by a few
/// hundred MiB either way.
#[cfg(any(windows, test))]
fn memory_short(available: u64, memory: u64, was_short: bool) -> bool {
    let (low, high) = working_set_marks(memory);
    available < high || (was_short && available <= high + (high - low))
}

/// Keeps the working set's limit to what is free, four times a second, whatever the engine is doing: a wide batch reads a gigabyte a second, and
/// read at each batch the limit came too late (0.47 GiB available). `pinned` follows the K/V.
pub fn watch_working_set(memory: u64, pinned: std::sync::Arc<std::sync::atomic::AtomicU64>, banks: u64) {
    #[cfg(windows)]
    {
        let _ = std::thread::Builder::new().name("working set".into()).spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(250));
            keep_resident(memory, pinned.load(std::sync::atomic::Ordering::Relaxed), banks);
        });
    }
    #[cfg(not(windows))]
    let _ = (memory, pinned, banks);
}

/// What takes commit after the banks are locked besides the expert slots, which CUDA charges 1 to 1 and
/// which fill what the VRAM budget has left: 0.6 GiB on an RTX 5070 Ti at --vram 6 and 8.
#[cfg(any(windows, test))]
const COMMIT_AFTER_LOCKS: u64 = 1 << 30;

/// Bytes of the expert banks that may be page-locked given the `available` physical memory, the `commit`
/// left and the `vram_room` the slots will take. Windows charges commit for what is locked (1 to 1 over 29 GiB at 62 GiB with an 8 GiB
/// page file, which left 0.5 GiB of commit with physical memory to spare), and a GPU budget has taken its
/// commit by then.
#[cfg(any(windows, test))]
fn lockable_of(available: u64, commit: u64, vram_room: u64, memory: u64) -> u64 {
    within_free(u64::MAX, 0, available.min(commit.saturating_sub(COMMIT_AFTER_LOCKS + vram_room)), memory)
}

/// Bytes of the expert banks that may be page-locked now (no bound off Windows).
pub fn lockable(memory: u64, vram_room: u64) -> u64 {
    #[cfg(windows)]
    {
        let (available, commit) = windows_available();
        return lockable_of(available, commit, vram_room, memory);
    }
    #[cfg(not(windows))]
    {
        let _ = (memory, vram_room);
        u64::MAX
    }
}

/// Ask the system to leave this process what its share of `memory` holds beside `pinned` bytes of
/// page-locked state: pages it trims from the working set come back one fault at a time under the
/// CPU's expert threads (Windows at 32 GiB: 800K faults/s and 3 tokens/s at the worst, with no
/// more read from the drive). Never more than the `banks` and that state need, and less as the
/// system refuses it (it refuses what the free memory cannot back, ERROR_NO_SYSTEM_RESOURCES, as
/// with most banks page-locked at 62 GiB), and never more than leaves [`left_free`] free: what it
/// grants, it keeps from everything else. It is also the most the working set takes: mapped bank pages touched later would otherwise
/// go on joining it with no limit, past [`left_free`] (RTX 5070 Ti, 62 GiB of memory: 3.8 GiB available within a minute of serving, the
/// desktop stalling); past the limit they wait in the standby list, a soft fault away. Best effort: only speed depends on it.
pub fn keep_resident(memory: u64, pinned: u64, banks: u64) {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MAX_ENABLE, QUOTA_LIMITS_HARDWS_MIN_ENABLE};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        use crate::arch::served::{HOST_OTHER, HOST_SHARE};
        let share = ((memory as f64 * HOST_SHARE) as u64).saturating_sub(pinned + HOST_OTHER).min(banks + pinned + HOST_OTHER);
        let available = windows_available().0;
        let was_paging = PAGING.load(std::sync::atomic::Ordering::Relaxed) != 0;
        let paging = drive_busy().unwrap_or_else(|| memory_short(available, memory, was_paging));
        if paging != was_paging {
            PAGING.store(u8::from(paging), std::sync::atomic::Ordering::Relaxed);
            let gib = available as f64 / (1u64 << 30) as f64;
            if paging {
                eprintln!("Straydog host paging: {gib:.1} GiB available; the CPU's experts are read ahead from now");
            } else {
                eprintln!("Straydog host paging: {gib:.1} GiB available; the CPU's experts are no longer read ahead");
            }
        }
        // (not so low that the CPU's experts fault on every page: 32 GiB of memory ran at 3 tokens/s when it gave them up)
        let share = working_set_limit(share, memory / 8 * 3, windows_working_set(), available, memory);
        let mut wanted = share;
        while wanted >= 1 << 30 {
            // SAFETY: the pseudo handle of this process and plain values.
            let done = unsafe { SetProcessWorkingSetSizeEx(GetCurrentProcess(), (wanted / 8 * 7) as usize, wanted as usize, QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE) };
            if done != 0 {
                if wanted < share {
                    eprintln!("a working set of {} GiB was refused; {} GiB taken", share >> 30, wanted >> 30);
                }
                return;
            }
            wanted = wanted / 4 * 3;
        }
        eprintln!("a working set of {} GiB was refused: {}", share >> 30, std::io::Error::last_os_error());
    }
    #[cfg(not(windows))]
    let _ = (memory, pinned, banks);
}

/// `physical` bounded by the memory limits of this process's cgroup and every one above it, up
/// to and including the root of the hierarchy as mounted at `cgroup_root`. In a container (a
/// cgroup namespace) that root is the container's own cgroup, which holds its limit; the host's
/// true root has no limit file. A path that leaves the namespace (`/..`) is not followed: the
/// mounted root alone then bounds it.
#[cfg(target_os = "linux")]
pub fn linux_memory(physical: u64, proc_cgroup: &std::path::Path, cgroup_root: &std::path::Path) -> u64 {
    use std::path::{Component, Path};
    let Ok(text) = std::fs::read_to_string(proc_cgroup) else { return physical };
    // From `directory` up to `root`, both included: the smallest limit in `file`.
    let bounded = |root: &Path, relative: &str, file: &str| -> u64 {
        let relative = Path::new(relative.trim().trim_start_matches('/'));
        let inside = relative.components().all(|component| matches!(component, Component::Normal(_)));
        let mut directory = if inside { root.join(relative) } else { root.to_path_buf() };
        let mut total = physical;
        loop {
            if let Ok(limit) = std::fs::read_to_string(directory.join(file)) {
                // "max" (v2) is no limit; v1 writes a huge number instead, which min() ignores.
                if let Ok(limit) = limit.trim().parse::<u64>() {
                    total = total.min(limit);
                }
            }
            if directory == root || !directory.pop() {
                return total;
            }
        }
    };
    let mut total = physical;
    for line in text.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (fields.next(), fields.next(), fields.next()) else { continue };
        if id == "0" && controllers.is_empty() {
            total = total.min(bounded(cgroup_root, path, "memory.max"));
        } else if controllers.split(',').any(|controller| controller == "memory") {
            total = total.min(bounded(&cgroup_root.join("memory"), path, "memory.limit_in_bytes"));
        }
    }
    total
}

#[cfg(test)]
mod paging {
    const GIB: u64 = 1 << 30;

    #[test]
    fn memory_is_short_from_the_upper_mark_until_a_band_above_it() {
        use super::memory_short;
        // 62 GiB: the marks are 3.1 and 4.7 GiB, the band 1.7.
        let memory = 62 * GIB;
        assert!(!memory_short(30 * GIB, memory, false));
        assert!(!memory_short(5 * GIB, memory, false));
        assert!(memory_short(4 * GIB, memory, false));
        // Once short, it stays so up to the band above the upper mark, and not past it.
        assert!(memory_short(6 * GIB, memory, true));
        assert!(!memory_short(7 * GIB, memory, true));
        // Small memory: the marks are 3 and 4 GiB at the least.
        assert!(memory_short(3 * GIB, 16 * GIB, false));
        assert!(!memory_short(4 * GIB, 16 * GIB, false));
        assert!(memory_short(5 * GIB, 16 * GIB, true));
        assert!(!memory_short(5 * GIB + 1, 16 * GIB, true));
    }

    #[test]
    fn the_drive_is_quiet_after_two_seconds_under_the_reads_of_a_busy_one() {
        use super::{quiet_calls, QUIET_CALLS};
        // 6,250 pages in a quarter of a second are 25,000 a second.
        assert_eq!(quiet_calls(5, 6_250, 0.25), 0);
        assert_eq!(quiet_calls(5, 6_249, 0.25), 6);
        // An idle system reads tens of pages a second.
        let mut quiet = 0;
        for _ in 0..QUIET_CALLS - 1 {
            quiet = quiet_calls(quiet, 10, 0.25);
        }
        assert!(quiet < QUIET_CALLS);
        assert!(quiet_calls(quiet, 10, 0.25) >= QUIET_CALLS);
        assert_eq!(quiet_calls(u32::MAX, 0, 0.25), u32::MAX);
    }
}

#[cfg(test)]
mod priority_tests {
    use super::{Lowered, Priority};

    /// The calling thread's memory priority, which Windows' background mode lowers from 5 to 1 (with the CPU and I/O priority).
    #[cfg(windows)]
    fn memory_priority() -> u32 {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadInformation, ThreadMemoryPriority, MEMORY_PRIORITY_INFORMATION};
        let mut information = MEMORY_PRIORITY_INFORMATION { MemoryPriority: 0 };
        // SAFETY: the calling thread's pseudo handle and an initialized structure of the size passed.
        let read = unsafe {
            GetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, std::ptr::from_mut(&mut information).cast(),
                                 std::mem::size_of::<MEMORY_PRIORITY_INFORMATION>() as u32)
        };
        assert_ne!(read, 0, "{}", std::io::Error::last_os_error());
        information.MemoryPriority
    }

    /// The calling thread's nice value.
    #[cfg(target_os = "linux")]
    fn nice() -> i32 {
        // SAFETY: the calling thread's own id.
        unsafe { libc::getpriority(libc::PRIO_PROCESS, libc::syscall(libc::SYS_gettid) as libc::id_t) }
    }

    /// Decides: the background priority is the calling thread's alone, for as long as it is held on Windows (ended after) and for the thread's life
    /// on Linux, and the normal priority changes nothing. Each runs on a thread of its own, which ends with the lowering.
    #[test]
    fn the_background_priority_is_the_calling_threads_alone() {
        #[cfg(windows)]
        let before = std::thread::spawn(memory_priority).join().unwrap();
        #[cfg(target_os = "linux")]
        let before = std::thread::spawn(nice).join().unwrap();
        #[cfg(any(windows, target_os = "linux"))]
        {
            let other = std::thread::spawn(|| {
                let _normal = Lowered::begin(Priority::Normal);
                #[cfg(windows)]
                let now = memory_priority();
                #[cfg(target_os = "linux")]
                let now = nice();
                now
            });
            assert_eq!(other.join().unwrap(), before, "the normal priority changes nothing");
            let (sender, receiver) = std::sync::mpsc::channel();
            let (lowered, resume) = (std::sync::mpsc::channel::<()>(), std::sync::mpsc::channel::<()>());
            let lowering = std::thread::spawn(move || {
                let held = Lowered::begin(Priority::Background);
                #[cfg(windows)]
                sender.send(memory_priority()).unwrap();
                #[cfg(target_os = "linux")]
                sender.send(nice() as u32).unwrap();
                lowered.0.send(()).unwrap();
                resume.1.recv().unwrap();
                drop(held);
                #[cfg(windows)]
                return memory_priority();
                #[cfg(target_os = "linux")]
                return nice() as u32;
            });
            lowered.1.recv().unwrap();
            let during = receiver.recv().unwrap();
            // (another thread started meanwhile is as it was made)
            #[cfg(windows)]
            assert_eq!(std::thread::spawn(memory_priority).join().unwrap(), before, "no other thread is lowered");
            #[cfg(target_os = "linux")]
            assert_eq!(std::thread::spawn(nice).join().unwrap(), before, "no other thread is lowered");
            #[cfg(windows)]
            assert!(during < before, "memory priority {during} while held against {before}");
            #[cfg(target_os = "linux")]
            assert_eq!(during, 19);
            resume.0.send(()).unwrap();
            let after = lowering.join().unwrap();
            #[cfg(windows)]
            assert_eq!(after, before, "the background mode ends when it is dropped");
            #[cfg(target_os = "linux")]
            assert_eq!(after, 19, "a thread cannot take its nice value back, and ends with the work");
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        let _ = Lowered::begin(Priority::Background);
    }
}

#[cfg(test)]
mod prefetch_tests {
    /// Decides: a hint on the ranges of a mapped file (a bank's experts, joined or apart, one that is not a whole page, and none at all) is taken and
    /// changes nothing of what the mapping reads.
    #[test]
    fn the_os_is_asked_for_ranges_of_a_mapping_and_the_bytes_stay_what_they_were() {
        let path = std::env::temp_dir().join(format!("straydog-prefetch-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..(3 << 20)).map(|at: usize| (at % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: the file is private to this test and stays as written while mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let base = map.as_ptr() as usize;
        super::prefetch(&[]).unwrap();
        super::prefetch(&[(base, 0)]).unwrap();
        super::prefetch(&[(base + 100, 3000), (base + (1 << 20), 1 << 20), (base + (2 << 20) + 4095, 5000)]).unwrap();
        assert_eq!(&map[..], &bytes[..]);
        drop(map);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    #[test]
    fn what_is_held_leaves_an_eighth_of_the_memory_free() {
        use super::{left_free, within_free};
        const GIB: u64 = 1 << 30;
        assert_eq!((left_free(32 * GIB), left_free(62 * GIB)), (4 * GIB, 62 * GIB / 8));
        // 62 GiB with 30 free: the 47 GiB asked becomes what is held and what can go.
        assert_eq!(within_free(47 * GIB, 2 * GIB, 30 * GIB, 62 * GIB), 2 * GIB + 30 * GIB - 62 * GIB / 8);
        assert_eq!(within_free(20 * GIB, 2 * GIB, 30 * GIB, 62 * GIB), 20 * GIB);
        assert_eq!(within_free(47 * GIB, 2 * GIB, GIB, 62 * GIB), 2 * GIB);
        // Between the two marks it stays; under the lower it gives back, down to the least; over the upper it takes more, up to what it wants.
        assert_eq!(super::working_set_limit(47 * GIB, 23 * GIB, 30 * GIB, 4 * GIB, 62 * GIB), 30 * GIB);
        assert_eq!(super::working_set_limit(47 * GIB, 23 * GIB, 30 * GIB, 2 * GIB, 62 * GIB), 32 * GIB - 62 * GIB / 20);
        assert_eq!(super::working_set_limit(47 * GIB, 23 * GIB, 24 * GIB, 0, 62 * GIB), 23 * GIB);
        assert_eq!(super::working_set_limit(47 * GIB, 23 * GIB, 30 * GIB, 10 * GIB, 62 * GIB), 40 * GIB - 62 * GIB / 13);
        assert_eq!(super::working_set_limit(35 * GIB, 23 * GIB, 30 * GIB, 30 * GIB, 62 * GIB), 35 * GIB);
        // Commit bounds the locks where physical memory does not: 34 GiB of commit left, 46 of memory, 4 of VRAM to come.
        assert_eq!(super::lockable_of(46 * GIB, 34 * GIB, 4 * GIB, 62 * GIB), 29 * GIB - 62 * GIB / 8);
        assert_eq!(super::lockable_of(46 * GIB, 3 * GIB, 0, 62 * GIB), 0);
    }

    use super::linux_memory;

    const GIB: u64 = 1 << 30;

    /// A /proc/self/cgroup text and a cgroup mount with these limit files, under a directory of
    /// the test's own.
    fn tree(test: &str, cgroup: &str, limits: &[(&str, &str)]) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("straydog-cgroup-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("fs")).unwrap();
        std::fs::write(root.join("cgroup"), cgroup).unwrap();
        for (path, limit) in limits {
            let file = root.join("fs").join(path);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, limit).unwrap();
        }
        root
    }

    #[test]
    fn a_containers_limit_at_the_namespace_root_counts() {
        // A container's cgroup namespace: its own cgroup is "/", its limit at the mount's root.
        let root = tree("namespace", "0::/\n", &[("memory.max", "1073741824\n")]);
        assert_eq!(linux_memory(125 * GIB, &root.join("cgroup"), &root.join("fs")), GIB);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_smallest_limit_on_the_way_up_counts() {
        let root = tree("way_up", "0::/a/b\n", &[("a/memory.max", "4294967296"), ("a/b/memory.max", "max"), ("memory.max", "8589934592")]);
        assert_eq!(linux_memory(125 * GIB, &root.join("cgroup"), &root.join("fs")), 4 * GIB);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_path_outside_the_namespace_falls_back_to_its_root() {
        let root = tree("outside", "0::/../../elsewhere\n", &[("memory.max", "2147483648")]);
        assert_eq!(linux_memory(125 * GIB, &root.join("cgroup"), &root.join("fs")), 2 * GIB);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cgroup_v1_memory_limits_count() {
        let root = tree("v1", "12:cpu,cpuacct:/x\n5:memory:/docker/abc\n",
                        &[("memory/docker/abc/memory.limit_in_bytes", "3221225472"),
                          ("memory/memory.limit_in_bytes", "9223372036854771712")]);
        assert_eq!(linux_memory(125 * GIB, &root.join("cgroup"), &root.join("fs")), 3 * GIB);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn no_limits_leave_physical_memory() {
        let root = tree("none", "0::/user.slice/session\n", &[]);
        assert_eq!(linux_memory(125 * GIB, &root.join("cgroup"), &root.join("fs")), 125 * GIB);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(windows)]
fn windows_available() -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: an initialized structure with its length set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { (status.ullAvailPhys, status.ullAvailPageFile) } else { (0, 0) }
}

/// The physical memory available and the commit the system has left, where the engine can run out of it: Windows charges page-locked memory
/// and device memory to commit.
pub fn commit_left() -> Option<(u64, u64)> {
    #[cfg(windows)]
    {
        Some(windows_available()).filter(|(_, commit)| *commit > 0)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// With DS41_PROFILE set on Windows: the system's commit left beside this process's own private commit at a step of a start.
/// What the system loses beyond the process's own is charged elsewhere (the display driver's, for device memory).
pub fn commit_note(step: &str) {
    #[cfg(windows)]
    if std::env::var_os("DS41_PROFILE").is_some() {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: the pseudo handle of this process and an initialized structure with its size set.
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        let (physical, commit) = windows_available();
        const GIB: f64 = (1u64 << 30) as f64;
        eprintln!("commit at {step}: {:.2} GiB left, {:.2} private, {:.2} physical left", commit as f64 / GIB,
                  counters.PagefileUsage as f64 / GIB, physical as f64 / GIB);
    }
    #[cfg(not(windows))]
    let _ = step;
}

/// Pages the system has read from the drive to satisfy page faults since it started (all processes): the 32-bit PageReadCount of
/// SystemPerformanceInformation, which `Memory\Pages Input/sec` counts. None where the system does not say.
#[cfg(windows)]
fn windows_page_reads() -> Option<u32> {
    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(class: u32, information: *mut u64, length: u32, returned: *mut u32) -> i32;
    }
    let mut buffer = [0u64; 64];
    let mut returned = 0u32;
    // SAFETY: a buffer of the length given, and a place for the length returned.
    let status = unsafe { NtQuerySystemInformation(2, buffer.as_mut_ptr(), std::mem::size_of_val(&buffer) as u32, &mut returned) };
    // (the counter lies at byte 80, after four 64-bit counters and twelve 32-bit ones, of which AvailablePages at byte 44 tells the layout is
    // the one known: it must be what the memory status says, to a tenth)
    let layout_known = |available_pages: u64| {
        let status = windows_available().0 / 4096;
        status > 0 && available_pages.abs_diff(status) <= status / 10
    };
    (status >= 0 && returned >= 84 && layout_known(buffer[5] >> 32)).then_some((buffer[10] & 0xffff_ffff) as u32)
}

/// Pages a second of the system's reads for page faults from which the drive counts as being read: 25,000 are 100 MB/s, where an idle system
/// reads tens (serving UD-IQ1_M partially resident reads 40,000 on average, in bursts).
#[cfg(any(windows, test))]
const READS_BUSY: f64 = 25_000.0;
/// Calls of [`drive_busy`] (250 ms apart) that stay under [`READS_BUSY`] before the drive counts as quiet: two seconds.
#[cfg(any(windows, test))]
const QUIET_CALLS: u32 = 8;

/// The count of calls in a row that read under [`READS_BUSY`], after one that saw `pages` read in `seconds`.
#[cfg(any(windows, test))]
fn quiet_calls(quiet: u32, pages: u32, seconds: f64) -> u32 {
    if f64::from(pages) / seconds >= READS_BUSY { 0 } else { quiet.saturating_add(1) }
}

/// Whether the drive is being read for page faults: at READS_BUSY pages a second or more over the time since the last call or in the two
/// seconds before. None where the system does not say. The first call only takes the count.
#[cfg(windows)]
fn drive_busy() -> Option<bool> {
    static LAST: std::sync::Mutex<Option<(std::time::Instant, u32, u32)>> = std::sync::Mutex::new(None);
    let count = windows_page_reads()?;
    let now = std::time::Instant::now();
    let mut last = LAST.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some((at, before, quiet)) = *last else {
        *last = Some((now, count, QUIET_CALLS));
        return Some(false);
    };
    let seconds = now.duration_since(at).as_secs_f64();
    // (calls close together, as when a K/V growth sets the limit, add nothing)
    if seconds < 0.1 {
        return Some(quiet < QUIET_CALLS);
    }
    let quiet = quiet_calls(quiet, count.wrapping_sub(before), seconds);
    *last = Some((now, count, quiet));
    Some(quiet < QUIET_CALLS)
}

#[cfg(windows)]
fn windows_working_set() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: the pseudo handle of this process and an initialized structure with its size set.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) } != 0 { counters.WorkingSetSize as u64 } else { 0 }
}

#[cfg(windows)]
fn windows_memory() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: an initialized structure with its length set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { status.ullTotalPhys } else { 0 }
}
