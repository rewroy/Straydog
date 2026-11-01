//! The bubblewrap command that contains the worker on Linux.
//!
//! Every namespace is unshared, the network's included (loopback too). The root is empty but
//! for what is mounted read-only here: the system libraries, ROCm, the loader's cache and
//! configuration, the executable and each exposed file; the GPU device nodes are bound, and
//! `/tmp` and `/dev/shm` are private memory file systems. The environment is cleared but for
//! named variables. No general home directory mount, no writable mount, and nothing the server
//! could probe at run time turns any of it off.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::{check_exposable, resolve, Exposure};

/// The system's libraries, the kernel's device information (the GPU runtimes read
/// `/sys/class/kfd` and `/sys/bus/pci`) and packaged ROCm.
const SYSTEM: [&str; 5] = ["/usr", "/sys", "/lib", "/lib64", "/opt/rocm"];

/// The loader's cache and configuration. Packaged ROCm may reach /opt/rocm/lib through
/// alternatives links.
const LOADER: [&str; 4] = ["/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives"];

/// The variables the worker keeps from the server's environment, when set there: the search
/// path, the locale and time zone, and the GPU runtimes' device selection and architecture
/// override, and the DeepSeek-V4.1 engine's switches: its phase timing (DS41_PROFILE), its drafts
/// (DS41_DRAFTS) and DSpark's (DS41_DSPARK, DS41_DSPARK_RESIDENT). Nothing else reaches it: no
/// proxy, token, key or credential the server may hold.
const VARIABLES: [&str; 12] = [
    "PATH",
    "LANG",
    "TZ",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "DS41_PROFILE",
    "STRAYDOG_DIRECT",
    "DS41_DRAFTS",
    "DS41_DSPARK",
    "DS41_DSPARK_RESIDENT",
];

/// GPU device nodes: AMD's, then NVIDIA's (further NVIDIA GPUs are found by name).
const DEVICES: [&str; 7] = [
    "/dev/kfd",
    "/dev/dri",
    "/dev/nvidia0",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-caps",
];

/// The bubblewrap executable on PATH (only absolute entries: never one relative to wherever
/// the server was started).
pub fn program() -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let search = std::env::var_os("PATH").unwrap_or_else(|| "/bin:/usr/bin".into());
    std::env::split_paths(&search)
        .filter(|directory| directory.is_absolute())
        .map(|directory| directory.join("bwrap"))
        .find(|candidate| {
            std::fs::metadata(candidate)
                .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        })
}

/// bubblewrap's arguments that run `exposure.executable worker_args...` contained, after
/// refusing any mount that is, holds or lies inside a secret. The exposure is validated.
pub fn arguments(worker_args: &[OsString], exposure: &Exposure) -> Result<Vec<OsString>, String> {
    let mut mounts: BTreeSet<PathBuf> = SYSTEM.iter().chain(&LOADER).map(PathBuf::from).collect();
    // A system directory mounted where it really is shows everything below it as on the host.
    let real: Vec<PathBuf> =
        mounts.iter().filter(|mount| canonical(mount).as_deref() == Some(mount.as_path())).cloned().collect();
    let covered = |path: &Path| real.iter().any(|mount| path.starts_with(mount));
    // A system directory that is a link (a ROCm installed as /opt/rocm -> /opt/rocm-7.2.0) is
    // mounted at its own path, and its target too: libraries and the loader's cache may name the
    // real directory. /lib -> usr/lib needs nothing more.
    let targets: Vec<PathBuf> =
        mounts.iter().filter_map(|mount| canonical(mount)).filter(|target| !covered(target)).collect();
    mounts.extend(targets);
    // The executable, its file when reached through a link (~/.local/bin/straydog to an
    // installed binary), so a path the executable resolves for itself exists too, and each file.
    let executable = resolve(&exposure.executable);
    let files: Vec<&Path> = [exposure.executable.as_path(), &executable]
        .into_iter()
        .chain(exposure.files.iter().map(PathBuf::as_path))
        .collect();
    mounts.extend(files.iter().filter(|file| !covered(file)).map(|file| file.to_path_buf()));
    let devices = device_nodes();
    let directories: BTreeSet<&Path> =
        exposure.directories.iter().chain([&exposure.working_directory]).map(PathBuf::as_path).collect();
    let exposed = mounts.iter().map(PathBuf::as_path).chain(files).chain(directories.iter().copied());
    check_exposable(&exposed.chain(devices.iter().map(PathBuf::as_path)).collect::<Vec<_>>(), &exposure.secrets)?;

    let mut args: Vec<OsString> = [
        "--clearenv",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/dev/shm",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "TMPDIR",
        "/tmp",
    ]
    .into_iter()
    .map(OsString::from)
    .collect();
    for name in VARIABLES {
        if let Some(value) = std::env::var_os(name) {
            args.extend(["--setenv".into(), name.into(), value]);
        }
    }
    // The directories the worker opens files in exist (empty but for the files bound in them).
    for directory in &directories {
        push(&mut args, "--dir", &[directory]);
    }
    // Sorted by component: a directory is mounted before anything inside it, and the loader's
    // alternatives links (under /etc) before a path that goes through them.
    for mount in mounts.iter().filter(|mount| mount.exists()) {
        push(&mut args, "--ro-bind", &[mount, mount]);
    }
    for device in &devices {
        push(&mut args, "--dev-bind", &[device, device]);
    }
    push(&mut args, "--chdir", &[&exposure.working_directory]);
    push(&mut args, "--", &[&exposure.executable]);
    args.extend(worker_args.iter().cloned());
    Ok(args)
}

fn push<P: AsRef<Path>>(args: &mut Vec<OsString>, flag: &str, paths: &[P]) {
    args.push(flag.into());
    args.extend(paths.iter().map(|path| path.as_ref().as_os_str().to_owned()));
}

fn canonical(path: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(path).ok()
}

/// The GPU device nodes this host has.
fn device_nodes() -> Vec<PathBuf> {
    let mut nodes: Vec<PathBuf> = DEVICES.iter().map(PathBuf::from).filter(|path| path.exists()).collect();
    let mut numbered: Vec<PathBuf> = std::fs::read_dir("/dev")
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.file_name())
        .filter(|name| {
            let name = name.to_string_lossy();
            let digit = name.strip_prefix("nvidia").is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
            digit && name != "nvidia0"
        })
        .map(|name| Path::new("/dev").join(name))
        .collect();
    numbered.sort();
    nodes.extend(numbered);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::Scratch;

    fn exposure(scratch: &Scratch) -> Exposure {
        let model = scratch.0.join("model");
        let drafter = scratch.0.join("mtp");
        let files = vec![scratch.file("model/a.gguf", "a"), scratch.file("mtp/mtp.gguf", "m")];
        Exposure {
            executable: std::env::current_exe().unwrap(),
            files,
            directories: vec![model.clone(), drafter],
            working_directory: model,
            secrets: vec![scratch.0.join("api-key")],
        }
    }

    fn arguments_for(exposure: &Exposure) -> Result<Vec<OsString>, String> {
        exposure.validate()?;
        arguments(&["worker".into(), "--model-root".into(), exposure.working_directory.clone().into()], exposure)
    }

    /// The sources of every mount of `flag`.
    fn sources(args: &[OsString], flag: &str) -> Vec<PathBuf> {
        args.windows(3).filter(|window| window[0] == flag).map(|window| PathBuf::from(&window[1])).collect()
    }

    #[test]
    fn the_worker_has_no_network_and_no_general_home_mount() {
        let scratch = Scratch::new("straydog-bwrap");
        let exposure = exposure(&scratch);
        let args = arguments_for(&exposure).unwrap();
        assert!(args.contains(&"--unshare-all".into()) && !args.contains(&"--share-net".into()));
        assert!(args.contains(&"--die-with-parent".into()) && args.contains(&"--new-session".into()));
        let home = std::env::var_os("HOME").unwrap();
        assert!(!args.contains(&home) && !args.contains(&"--bind".into()));
        assert!(!args.iter().any(|arg| arg.to_string_lossy().ends_with("-try")));
        let mounted = sources(&args, "--ro-bind");
        for file in &exposure.files {
            assert!(mounted.contains(file), "{file:?}");
        }
        // The model and drafter directories exist in the sandbox, but only their pinned files are in them.
        for directory in &exposure.directories {
            assert!(!mounted.contains(directory));
            let at = args.iter().position(|arg| arg == directory.as_os_str()).unwrap();
            assert_eq!(args[at - 1], "--dir");
        }
        assert!(sources(&args, "--dev-bind").iter().all(|device| device.starts_with("/dev")));
        let working: OsString = exposure.working_directory.clone().into();
        let executable: OsString = exposure.executable.clone().into();
        let tail = [
            "--chdir".into(),
            working.clone(),
            "--".into(),
            executable,
            "worker".into(),
            "--model-root".into(),
            working,
        ];
        assert_eq!(&args[args.len() - tail.len()..], &tail);
    }

    #[test]
    fn the_environment_is_cleared_but_for_named_variables() {
        let scratch = Scratch::new("straydog-bwrap-environment");
        let args = arguments_for(&exposure(&scratch)).unwrap();
        assert_eq!(args[0], "--clearenv");
        let set: Vec<(&OsString, &OsString)> =
            args.windows(3).filter(|window| window[0] == "--setenv").map(|window| (&window[1], &window[2])).collect();
        for (name, _) in &set {
            let name = name.to_str().unwrap();
            assert!(["HOME", "TMPDIR"].contains(&name) || VARIABLES.contains(&name), "{name}");
        }
        let path = std::env::var_os("PATH").unwrap();
        assert!(set.contains(&(&"PATH".into(), &path)));
        assert!(set.contains(&(&"HOME".into(), &"/tmp".into())));
    }

    #[test]
    fn every_read_only_mount_is_a_system_path_the_executable_or_an_exposed_file() {
        let scratch = Scratch::new("straydog-bwrap-mounts");
        let exposure = exposure(&scratch);
        let args = arguments_for(&exposure).unwrap();
        let executable = [exposure.executable.clone(), resolve(&exposure.executable)];
        for source in &sources(&args, "--ro-bind") {
            let system = SYSTEM.iter().chain(&LOADER).any(|root| source.starts_with(root));
            let linked = SYSTEM.iter().any(|root| std::fs::canonicalize(root).is_ok_and(|real| real == *source));
            let file = executable.contains(source) || exposure.files.contains(source);
            assert!(system || linked || file, "{source:?}");
        }
    }

    #[test]
    fn what_a_system_directory_already_shows_is_not_mounted_again() {
        // /opt/rocm/bin/rocminfo goes through /etc/alternatives links: binding it before they
        // exist in the sandbox failed ("Can't create file").
        let scratch = Scratch::new("straydog-bwrap-system");
        let mut exposure = exposure(&scratch);
        exposure.executable = PathBuf::from("/usr/bin/env");
        let args = arguments_for(&exposure).unwrap();
        assert!(!sources(&args, "--ro-bind").iter().any(|source| source.starts_with("/usr/bin")));
        let mounted = sources(&args, "--ro-bind");
        let alternatives = mounted.iter().position(|source| source == Path::new("/etc/alternatives"));
        let rocm = mounted.iter().position(|source| source == Path::new("/opt/rocm"));
        if let (Some(alternatives), Some(rocm)) = (alternatives, rocm) {
            assert!(alternatives < rocm);
        }
    }

    #[test]
    fn secrets_are_refused_with_any_directory_holding_one() {
        let scratch = Scratch::new("straydog-bwrap-secret");
        let mut exposure = exposure(&scratch);
        exposure.secrets = vec![scratch.file("model/api-key", "k")];
        assert!(arguments_for(&exposure).unwrap_err().contains("secret"));
        exposure.secrets = vec![exposure.files[0].clone()];
        assert!(arguments_for(&exposure).unwrap_err().contains("secret"));
        // A secret directory that holds the model directory.
        exposure.secrets = vec![scratch.0.clone()];
        assert!(arguments_for(&exposure).unwrap_err().contains("secret"));
        exposure.secrets = vec![scratch.file("elsewhere/api-key", "k")];
        arguments_for(&exposure).unwrap();
    }

    #[test]
    fn an_exposed_file_must_be_a_regular_file_at_an_absolute_path() {
        let scratch = Scratch::new("straydog-bwrap-files");
        let mut exposure = exposure(&scratch);
        let link = scratch.0.join("model/b.gguf");
        std::os::unix::fs::symlink(&exposure.files[0], &link).unwrap();
        exposure.files.push(link);
        assert!(arguments_for(&exposure).unwrap_err().contains("not a regular file"));
        exposure.files.pop();
        exposure.files.push("model/a.gguf".into());
        assert!(arguments_for(&exposure).unwrap_err().contains("absolute"));
    }

    #[test]
    fn a_linked_executable_is_mounted_with_its_file() {
        let scratch = Scratch::new("straydog-bwrap-link");
        let mut exposure = exposure(&scratch);
        let link = scratch.0.join("bin/straydog");
        std::fs::create_dir_all(link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&exposure.executable, &link).unwrap();
        let real = resolve(&exposure.executable);
        exposure.executable = link.clone();
        let args = arguments_for(&exposure).unwrap();
        let mounted = sources(&args, "--ro-bind");
        assert!(mounted.contains(&link) && mounted.contains(&real));
        assert_eq!(args[args.iter().position(|arg| arg == "--").unwrap() + 1], link.as_os_str());
    }
}
