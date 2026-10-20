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
fn windows_memory() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: an initialized structure with its length set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { status.ullTotalPhys } else { 0 }
}
