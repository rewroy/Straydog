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

/// The limit for a working set of `held` with `available` free, of what it wants and at least `least`. Under the lower mark it gives back, over the
/// upper it may take more, and between them it stays. The marks are low because giving back is dear: IQ3_XXS on 62 GiB read a 33K prompt in 58 s with
/// 6M page faults at a working set of 40 GiB (2 GiB available), in 93 s with 37M at 32 GiB (7 GiB available), and in 117 s with 63M at 28 GiB.
#[cfg(any(windows, test))]
fn working_set_limit(wanted: u64, least: u64, held: u64, available: u64, memory: u64) -> u64 {
    let (low, high) = ((memory / 20).max(3 << 30), (memory / 13).max(4 << 30));
    let limit = if available < low {
        held.saturating_sub(low - available)
    } else if available > high {
        held + (available - high)
    } else {
        held
    };
    limit.min(wanted).max(least.min(wanted))
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
        // (not so low that the CPU's experts fault on every page: 32 GiB of memory ran at 3 tokens/s when it gave them up)
        let share = working_set_limit(share, memory / 8 * 3, windows_working_set(), windows_available().0, memory);
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
