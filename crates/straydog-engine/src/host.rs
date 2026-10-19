//! Host facts the isolated worker cannot observe itself: the runtime measures them before
//! spawning it (the worker runs in its own cgroup namespace, where its limits are invisible).

/// Physical memory, bounded on Linux by every cgroup v2 `memory.max` above this process; 0 if unknown.
pub fn host_memory_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        linux_memory(std::path::Path::new("/proc/self/cgroup"), std::path::Path::new("/sys/fs/cgroup"))
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

#[cfg(target_os = "linux")]
pub fn linux_memory(proc_cgroup: &std::path::Path, cgroup_root: &std::path::Path) -> u64 {
    // SAFETY: sysconf has no preconditions.
    let (pages, page) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGE_SIZE)) };
    if pages <= 0 || page <= 0 {
        return 0;
    }
    let mut total = pages as u64 * page as u64;
    let Ok(text) = std::fs::read_to_string(proc_cgroup) else { return total };
    let Some(entry) = text.lines().find_map(|line| line.strip_prefix("0::")) else { return total };
    let mut directory = cgroup_root.join(entry.trim().trim_start_matches('/'));
    loop {
        if directory == cgroup_root || !directory.starts_with(cgroup_root) {
            break;
        }
        if let Ok(limit) = std::fs::read_to_string(directory.join("memory.max")) {
            if let Ok(limit) = limit.trim().parse::<u64>() {
                total = total.min(limit);
            }
        }
        if !directory.pop() {
            break;
        }
    }
    total
}

#[cfg(windows)]
fn windows_memory() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: an initialized structure with its length set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { status.ullTotalPhys } else { 0 }
}
