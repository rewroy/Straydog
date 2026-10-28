//! What `straydog setup` finds out about the machine: its GPUs, its memory, the space where the
//! models would go.

use std::path::{Path, PathBuf};

use super::catalog::GIB;

/// One GPU, as the GPU library reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpu {
    pub ordinal: usize,
    pub name: String,
    pub total: u64,
    /// What is free now: less than the total when something else holds memory.
    pub free: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Machine {
    /// The GPU library this executable was built with.
    pub backend: &'static str,
    pub gpus: Vec<Gpu>,
    /// Why no GPU was found, when none was.
    pub gpu_error: Option<String>,
    /// Host memory as the server measures it (cgroup limits included); 0 if unknown.
    pub ram: u64,
}

/// The GPU library this executable links.
pub const BACKEND: &str = if cfg!(feature = "hip") {
    "ROCm/HIP"
} else if cfg!(feature = "cuda") {
    "CUDA"
} else {
    "no GPU library"
};

impl Machine {
    /// Look at this machine: every GPU (or, with `only`, those ordinals; at most four are used)
    /// and the host memory.
    pub fn probe(only: Option<&[usize]>) -> Machine {
        let mut machine = Machine { backend: BACKEND, gpus: Vec::new(), gpu_error: None, ram: straydog_engine::host::host_memory_bytes() };
        match straydog_gpu::device_count() {
            Ok(count) => {
                let ordinals: Vec<usize> = only.map_or_else(|| (0..count).collect(), <[usize]>::to_vec);
                for ordinal in ordinals.into_iter().take(4) {
                    match straydog_gpu::device_info(ordinal) {
                        Ok((name, free, total)) => machine.gpus.push(Gpu { ordinal, name, total, free }),
                        Err(error) => machine.gpu_error = Some(format!("GPU {ordinal}: {error}")),
                    }
                }
            }
            Err(error) => machine.gpu_error = Some(error.to_string()),
        }
        machine
    }

    /// What was found, in a line or two.
    pub fn describe(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if self.gpus.is_empty() {
            lines.push(format!("No GPU found ({}){}.", self.backend, self.gpu_error.as_ref().map_or_else(String::new, |error| format!(": {error}"))));
        } else {
            let mut groups: Vec<(&Gpu, usize)> = Vec::new();
            for gpu in &self.gpus {
                match groups.iter_mut().find(|(first, _)| first.name == gpu.name && first.total == gpu.total) {
                    Some((_, count)) => *count += 1,
                    None => groups.push((gpu, 1)),
                }
            }
            let parts: Vec<String> = groups
                .iter()
                .map(|(gpu, count)| {
                    let memory = format!("{:.1} GiB", gpu.total as f64 / GIB as f64);
                    if *count > 1 { format!("{count} x {} ({memory} each)", gpu.name) } else { format!("{} ({memory})", gpu.name) }
                })
                .collect();
            lines.push(format!("GPUs ({}): {}.", self.backend, parts.join(", ")));
            // (a GPU another program holds memory of cannot be given its whole total)
            for gpu in self.gpus.iter().filter(|gpu| gpu.free < gpu.total / 10 * 9) {
                lines.push(format!("GPU {} has {:.1} GiB free now: another program is using the rest.", gpu.ordinal, gpu.free as f64 / GIB as f64));
            }
        }
        lines.push(if self.ram == 0 { "Memory: unknown.".to_string() } else { format!("Memory: {:.1} GiB of RAM.", self.ram as f64 / GIB as f64) });
        lines
    }
}

/// The free bytes of the disk `path` lies on (or would lie on: its nearest existing parent).
pub fn free_space(path: &Path) -> Option<u64> {
    let mut at = path.to_path_buf();
    let existing = loop {
        if let Ok(found) = at.canonicalize() {
            break found;
        }
        if !at.pop() {
            return None;
        }
    };
    let disks = sysinfo::Disks::new_with_refreshed_list();
    // (the disk whose mount point is the longest prefix of the path)
    disks.iter().filter(|disk| existing.starts_with(disk.mount_point())).max_by_key(|disk| disk.mount_point().components().count()).map(|disk| disk.available_space())
}

/// `~` in a typed path is the home folder (the shell does not expand a path read from a prompt).
pub fn expand_home(text: &str, home: Option<PathBuf>) -> PathBuf {
    match (text.strip_prefix('~'), home) {
        (Some(""), Some(home)) => home,
        (Some(rest), Some(home)) if rest.starts_with(['/', '\\']) => home.join(&rest[1..]),
        _ => PathBuf::from(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(ordinal: usize, name: &str, gib: u64, free_gib: u64) -> Gpu {
        Gpu { ordinal, name: name.into(), total: gib * GIB, free: free_gib * GIB }
    }

    #[test]
    fn a_machine_is_described_in_a_line_or_two() {
        let two = Machine { backend: "ROCm/HIP", gpus: vec![gpu(0, "R9700", 32, 32), gpu(1, "R9700", 32, 32)], gpu_error: None, ram: 125 * GIB };
        assert_eq!(two.describe(), ["GPUs (ROCm/HIP): 2 x R9700 (32.0 GiB each).", "Memory: 125.0 GiB of RAM."]);
        let mixed = Machine { gpus: vec![gpu(0, "RTX 4070", 12, 3), gpu(1, "RTX 3060", 12, 12)], backend: "CUDA", gpu_error: None, ram: 32 * GIB };
        assert_eq!(mixed.describe(), ["GPUs (CUDA): RTX 4070 (12.0 GiB), RTX 3060 (12.0 GiB).",
                                      "GPU 0 has 3.0 GiB free now: another program is using the rest.", "Memory: 32.0 GiB of RAM."]);
        let none = Machine { backend: "CUDA", gpus: vec![], gpu_error: Some("no driver".into()), ram: 0 };
        assert_eq!(none.describe(), ["No GPU found (CUDA): no driver.", "Memory: unknown."]);
    }

    #[test]
    fn free_space_is_read_from_the_disk_a_path_will_lie_on() {
        let temporary = std::env::temp_dir();
        let here = free_space(&temporary).expect("the temporary directory lies on a disk");
        // a path that does not exist yet lies where its nearest parent does
        assert_eq!(free_space(&temporary.join("straydog-not-yet").join("models")).map(|bytes| bytes / GIB), Some(here / GIB));
    }

    #[test]
    fn a_tilde_is_the_home_folder() {
        let home = Some(PathBuf::from("/home/u"));
        assert_eq!(expand_home("~", home.clone()), PathBuf::from("/home/u"));
        assert_eq!(expand_home("~/models", home.clone()), PathBuf::from("/home/u/models"));
        assert_eq!(expand_home("~other/models", home.clone()), PathBuf::from("~other/models"));
        assert_eq!(expand_home("/data/models", home), PathBuf::from("/data/models"));
        assert_eq!(expand_home("~/models", None), PathBuf::from("~/models"));
    }
}
