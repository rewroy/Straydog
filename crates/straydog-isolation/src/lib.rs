//! Containment of the GPU worker: bubblewrap on Linux, an AppContainer of its own on
//! Windows. There is no fallback to an unrestricted child.
//!
//! The worker is the server's own executable (`straydog worker ...`) and reads only what an
//! [`Exposure`] names: the executable, the GPU runtime, the pinned model and drafter files and
//! the GPU device nodes. The server's secrets (its API key file) are never exposed, nor any
//! directory holding one.
//!
//! The server builds the contained command with [`worker_command`], configures it with
//! [`prepare`], spawns it with piped standard handles and owns what it started through a
//! [`ProcessTree`]. On Windows that command is the launcher, `straydog contain`
//! (`launcher_main`): it grants the `Straydog.Worker` AppContainer profile read access to
//! what the worker reads, runs the worker in that AppContainer over the launcher's own standard
//! handles and exits with the worker's status.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

mod tree;
pub use tree::{prepare, ProcessTree};

/// The engine's own switches, passed to the worker on both systems when the server's environment sets them: DS41_PROFILE (the engines'
/// phase timing, which a request's log line reports), DeepSeek-V4.1's drafts (DS41_DRAFTS, DS41_DSPARK, DS41_DSPARK_RESIDENT),
/// STRAYDOG_DIRECT (the GPU queues written directly), STRAYDOG_ENGRAM (the engram table Qwen3.8 reads), STRAYDOG_EXPERT_FLOOR (speed
/// first's router weight floor), STRAYDOG_CPU_INT8 (=0: the CPU experts' F32 path), STRAYDOG_CPU_AVX512 (=0: their kernels of a
/// CPU without AVX-512, to measure those on one with it), STRAYDOG_PROMPT_CPU (=0: a prompt batch streams every expert beyond the
/// slots, none to the CPU), STRAYDOG_PEER_DIRECT (=0: two GPUs linked through host memory, as on Windows) and STRAYDOG_HOST_REGISTER_READ_ONLY
/// (=0: the GPUs are taken as unable to register host memory read-only, so the expert banks are page-locked as copies, to try that on a GPU that
/// can); a prompt's layer order (STRAYDOG_LAYER_PREFILL, the rows of its passes STRAYDOG_LAYER_PREFILL_ROWS, where they are held
/// STRAYDOG_LAYER_ROWS_VRAM, and STRAYDOG_LOCK_DURING_PREFILL, the
/// locking behind "ready" going on meanwhile, which the worker's own code reads); STRAYDOG_KV (=f16: Qwen3.8's K/V held as
/// FP16, not Q8_0); STRAYDOG_PREFILL_WIDE (=off: a long read's batches are not widened; N: the rows to try); STRAYDOG_DELTA_SCAN (=token: a long read scans its DeltaNet layers token by token); STRAYDOG_HC (=blas: a long read's HyperConnections keep their two kernels, hipblasGemmEx and the finish); STRAYDOG_PARTS_OVERLAP (=off: a long read's sends between the two GPUs of Qwen3.8's layers in parts stay in the compute's order, none beside it); and the engines' development switches, each the one place
/// the engine reads it: Qwen3.8's STRAYDOG_Q8, STRAYDOG_HC_Q8, STRAYDOG_PARTS, STRAYDOG_MTP_PARTS and STRAYDOG_DRAFT_VOCABULARY, and
/// DeepSeek-V4.1's DS41_ ones. One list for both: a switch named on one system's list alone never reached the other's worker (speed first
/// and the engram table did nothing in a Linux worker), and one on neither never reached a worker at all (the layer order's rows and the K/V's
/// format measured nothing on 2026-11-11): every name the engine reads is here or a test's (the test below).
pub const ENGINE_VARIABLES: [&str; 45] = ["DS41_PROFILE", "DS41_DRAFTS", "DS41_DSPARK", "DS41_DSPARK_RESIDENT", "STRAYDOG_DIRECT",
                                         "STRAYDOG_ENGRAM", "STRAYDOG_EXPERT_FLOOR", "STRAYDOG_CPU_INT8", "STRAYDOG_CPU_AVX512",
                                         "STRAYDOG_PROMPT_CPU", "STRAYDOG_LAYER_PREFILL", "STRAYDOG_PEER_DIRECT",
                                         "STRAYDOG_HOST_REGISTER_READ_ONLY", "STRAYDOG_LAYER_PREFILL_ROWS", "STRAYDOG_LOCK_DURING_PREFILL", "STRAYDOG_KV",
                                         "STRAYDOG_Q8", "STRAYDOG_HC_Q8", "STRAYDOG_PARTS", "STRAYDOG_MTP_PARTS", "STRAYDOG_DRAFT_VOCABULARY",
                                         "DS41_CHECK_SLOTS", "DS41_COUNT_VIEWS", "DS41_EXPANSIONS", "DS41_FP16_CHECKS", "DS41_FULL_DECODER",
                                         "DS41_GATE_UP_FIRST", "DS41_HEADS_FIRST", "DS41_HITS_FIRST", "DS41_LEND", "DS41_LEND_ROWS",
                                         "DS41_LEND_STATE", "DS41_LINK", "DS41_NO_GRAPHS", "DS41_PREDICT", "DS41_PREDICT_DRIVE", "DS41_Q8",
                                         "DS41_ROUTE_READBACK", "DS41_SLOT_WMMA", "STRAYDOG_LAYER_ROWS_VRAM", "STRAYDOG_PREFILL_WIDE", "STRAYDOG_DELTA_SCAN",
                                         "STRAYDOG_HC", "STRAYDOG_PARTS_OVERLAP", "STRAYDOG_VRAM_DIAGNOSTIC"];

#[cfg(any(windows, test))]
mod appcontainer;
#[cfg(windows)]
pub use appcontainer::WORKER_VARIABLES;
#[cfg(target_os = "linux")]
mod bubblewrap;
#[cfg(windows)]
mod win32;

/// What the contained worker may read. Every path is absolute: the sandbox shows each one at
/// the place the server names it.
#[derive(Debug, Clone, Default)]
pub struct Exposure {
    /// The straydog executable the worker runs (the same binary as the server).
    pub executable: PathBuf,
    /// Pinned files the worker reads: model shards and resources, the drafter.
    pub files: Vec<PathBuf>,
    /// Directories the worker opens files in (the model and drafter directories).
    pub directories: Vec<PathBuf>,
    /// The worker's working directory (the model directory).
    pub working_directory: PathBuf,
    /// Never exposed, nor any directory holding one (the API key file).
    pub secrets: Vec<PathBuf>,
}

/// The executable's subcommand that runs `launcher_main` on Windows (`straydog contain`).
pub const LAUNCHER_SUBCOMMAND: &str = "contain";

/// What a line the launcher writes to standard error begins with when it only warns. The server
/// takes the worker's last line of standard error for the reason it failed to start, and passes
/// over these.
pub const LAUNCHER_WARNING: &str = "Straydog warning:";

#[cfg(not(windows))]
const UNAVAILABLE: &str = "Straydog requires Linux bubblewrap or Windows AppContainer GPU isolation";

/// Why this host cannot contain the worker, or None when it can.
pub fn unavailable_reason() -> Option<String> {
    #[cfg(windows)]
    {
        None // AppContainer: every supported Windows (10 and later) provides it
    }
    #[cfg(target_os = "linux")]
    {
        bubblewrap::program().is_none().then(|| UNAVAILABLE.to_owned())
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Some(UNAVAILABLE.to_owned())
    }
}

/// Refuse to expose a secret: no exposed path may be one, hold one or lie inside one.
///
/// Paths are compared resolved, so a link cannot smuggle a secret in, and on Windows without
/// regard to case, as its file systems compare names.
pub fn check_exposable(paths: &[&Path], secrets: &[PathBuf]) -> Result<(), String> {
    let secrets: Vec<Vec<OsString>> = secrets.iter().map(|secret| key(&resolve(secret))).collect();
    for path in paths {
        let resolved = resolve(path);
        let path_key = key(&resolved);
        if secrets.iter().any(|secret| path_key.starts_with(secret) || secret.starts_with(&path_key)) {
            return Err(format!(
                "Straydog never exposes a secret, or a directory holding one, to its worker: {}",
                resolved.display()
            ));
        }
    }
    Ok(())
}

/// The program and arguments that run `executable worker_args...` contained, to be spawned
/// by the server with piped standard handles.
///
/// Linux: bubblewrap. Spawn it from a thread that lives as long as the worker (not a pool
/// thread that may retire): `--die-with-parent`, like [`prepare`], ends the sandbox when the
/// thread that spawned it ends.
///
/// Windows: the executable's `contain` subcommand (`launcher_main`) with the grants to make
/// and the worker command after `--`. The launcher runs the worker in the working directory
/// itself, with only the system's, the loader's and the CUDA runtime's variables of its own
/// environment (and the AppContainer's private folder as TEMP).
///
/// Either way the worker's environment is an allowlist: no proxy, token, key or credential
/// variable of the server's reaches it.
pub fn worker_command(worker_args: &[OsString], exposure: &Exposure) -> Result<(PathBuf, Vec<OsString>), String> {
    #[cfg(target_os = "linux")]
    {
        let program = bubblewrap::program().ok_or(UNAVAILABLE)?;
        exposure.validate()?;
        Ok((program, bubblewrap::arguments(worker_args, exposure)?))
    }
    #[cfg(windows)]
    {
        exposure.validate()?;
        let arguments = appcontainer::launcher_arguments(worker_args, exposure, &cuda_runtime_directories())?;
        Ok((exposure.executable.clone(), arguments))
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = (worker_args, exposure);
        Err(UNAVAILABLE.to_owned())
    }
}

/// Windows: the `straydog contain` entry point that grants access, starts the worker in the
/// AppContainer with this process's standard handles, and returns its exit status.
///
/// `args` are the arguments after `contain`, as [`worker_command`] wrote them. A failure is
/// reported as one line on standard error (the server shows the worker's last stderr line as
/// the start failure) and status 1; malformed arguments give status 2.
#[cfg(windows)]
pub fn launcher_main(args: &[OsString]) -> i32 {
    appcontainer::main(
        args,
        |launch| appcontainer::grant_worker_access(&win32::Acl::new().map_err(|error| error.to_string())?, launch),
        |launch, sid| win32::launch(&launch.command, &sid, launch.working_directory.as_deref()),
    )
}

/// Windows: why the contained worker may have been refused (Win32 error 5, "access denied") the
/// first of `paths` it reads, found as the user, or None when every path looks right.
///
/// Two causes are told apart. A drive that keeps no file permissions (exFAT, FAT32) cannot hold
/// the `Straydog.Worker` AppContainer's read entry, so the worker is refused there however the
/// launcher grants. On a drive that does, the path's permissions may hold no entry for the
/// profile (or for ALL APPLICATION PACKAGES) that allows reading. A path whose permissions
/// cannot be read is passed over, and nothing is created: the profile's SID is derived from its
/// name.
#[cfg(windows)]
pub fn why_refused(paths: &[PathBuf]) -> Option<String> {
    win32::why_refused(paths)
}

/// Windows: the CUDA Toolkit's DLL directories (`bin`, and `bin\x64` since CUDA 13) named by
/// `CUDA_PATH`. Windows has no RUNPATH, so a CUDA build's cuBLAS import resolves from the
/// executable's own directory or from these, which the worker adds to its DLL search; the
/// launcher grants the worker read and execute access to them.
#[cfg(windows)]
pub fn cuda_runtime_directories() -> Vec<PathBuf> {
    appcontainer::cuda_directories(std::env::var_os("CUDA_PATH").as_deref())
}

/// Decode captured console output without mangling localized text.
///
/// The worker and the launcher write UTF-8, but text from Windows itself or from older console
/// tools comes in the OEM code page (cp932 on Japanese Windows) or the ANSI one. UTF-8 is tried
/// strictly first (localized text is essentially never valid UTF-8), then on Windows those code
/// pages, then lossy UTF-8.
pub fn decode_console_output(data: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(data) {
        return text.to_owned();
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Globalization::{CP_ACP, CP_OEMCP};
        if let Some(text) = [CP_OEMCP, CP_ACP].into_iter().find_map(|code_page| win32::decode(data, code_page)) {
            return text;
        }
    }
    String::from_utf8_lossy(data).into_owned()
}

#[cfg(any(windows, target_os = "linux"))]
impl Exposure {
    /// Every path absolute, the executable a file, each file a regular file (not a link: what
    /// the worker authenticates is what it opens) and each directory a directory.
    fn validate(&self) -> Result<(), String> {
        let paths = [&self.executable, &self.working_directory].into_iter();
        for path in paths.chain(&self.files).chain(&self.directories).chain(&self.secrets) {
            if !path.is_absolute() {
                return Err(format!("Straydog isolation needs absolute paths: {}", path.display()));
            }
        }
        if !self.executable.is_file() {
            return Err(format!("Straydog worker executable is missing: {}", self.executable.display()));
        }
        for file in &self.files {
            if !std::fs::symlink_metadata(file).is_ok_and(|metadata| metadata.is_file()) {
                return Err(format!("Straydog worker file is missing or not a regular file: {}", file.display()));
            }
        }
        for directory in self.directories.iter().chain([&self.working_directory]) {
            if !directory.is_dir() {
                return Err(format!("Straydog worker directory is missing: {}", directory.display()));
            }
        }
        Ok(())
    }
}

/// `path` made absolute with every symbolic link on it that exists resolved, like Python's
/// `Path.resolve()`: what does not exist is kept as written, `..` after it lexically.
fn resolve(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(_) => {
                resolved.push(component);
                if let Ok(real) = std::fs::canonicalize(&resolved) {
                    resolved = plain(real);
                }
            }
            Component::Prefix(_) | Component::RootDir => resolved.push(component),
        }
    }
    resolved
}

/// Windows: a canonical path without the `\\?\` prefix canonicalization adds, as written
/// elsewhere (in command lines, messages and the Win32 calls that take it).
#[cfg(windows)]
fn plain(path: PathBuf) -> PathBuf {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    let units: Vec<u16> = path.as_os_str().encode_wide().collect();
    let verbatim: Vec<u16> = r"\\?\".encode_utf16().collect();
    let Some(rest) = units.strip_prefix(verbatim.as_slice()) else { return path };
    let unc: Vec<u16> = r"UNC\".encode_utf16().collect();
    if let Some(share) = rest.strip_prefix(unc.as_slice()) {
        let mut plain: Vec<u16> = r"\\".encode_utf16().collect();
        plain.extend_from_slice(share);
        return PathBuf::from(OsString::from_wide(&plain));
    }
    let disk =
        rest.len() >= 2 && rest[1] == u16::from(b':') && u8::try_from(rest[0]).is_ok_and(|c| c.is_ascii_alphabetic());
    if disk {
        PathBuf::from(OsString::from_wide(rest))
    } else {
        path
    }
}

#[cfg(not(windows))]
fn plain(path: PathBuf) -> PathBuf {
    path
}

/// A path's components as compared: Windows compares names without regard to case, and a
/// drive or share the same whether written plainly or verbatim.
fn key(path: &Path) -> Vec<OsString> {
    #[cfg(windows)]
    {
        use std::path::Prefix;
        path.components()
            .map(|component| {
                let text = match component {
                    Component::Prefix(prefix) => match prefix.kind() {
                        Prefix::Disk(drive) | Prefix::VerbatimDisk(drive) => format!("{}:", drive as char),
                        Prefix::UNC(server, share) | Prefix::VerbatimUNC(server, share) => {
                            format!(r"\\{}\{}", server.to_string_lossy(), share.to_string_lossy())
                        }
                        _ => prefix.as_os_str().to_string_lossy().into_owned(),
                    },
                    other => other.as_os_str().to_string_lossy().into_owned(),
                };
                OsString::from(text.to_lowercase())
            })
            .collect()
    }
    #[cfg(not(windows))]
    {
        path.components().map(|component| component.as_os_str().to_owned()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) struct Scratch(pub PathBuf);

    impl Scratch {
        /// An empty directory of its own under `parent`, removed with its contents on drop.
        pub(crate) fn under(parent: &Path, name: &str) -> Scratch {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let unique = format!("{name}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
            let path = parent.join(unique);
            std::fs::create_dir_all(&path).unwrap();
            Scratch(resolve(&path))
        }

        pub(crate) fn new(name: &str) -> Scratch {
            Scratch::under(&std::env::temp_dir(), name)
        }

        /// A file at `relative` (its directories created) holding `content`.
        pub(crate) fn file(&self, relative: &str, content: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, content).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_secret_and_the_directories_holding_it_are_never_exposed() {
        let scratch = Scratch::new("straydog-secrets");
        let home = scratch.0.join("home");
        let key = home.join(".straydog/api-key");
        for path in [key.clone(), home.join(".straydog"), home.clone()] {
            let error = check_exposable(&[&path], std::slice::from_ref(&key)).unwrap_err();
            assert!(error.contains("secret"), "{error}");
        }
        check_exposable(&[&home.join(".straydog/models/IQ3_S")], std::slice::from_ref(&key)).unwrap();
        let secrets = scratch.0.join("secrets");
        assert!(check_exposable(&[&secrets.join("inside")], &[secrets]).is_err());
        // A sibling whose name merely begins like the secret's is not inside it.
        check_exposable(&[&home.join(".straydog/api-key-backup")], &[key]).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_link_cannot_smuggle_a_secret_in() {
        let scratch = Scratch::new("straydog-link");
        let key = scratch.file("private/api-key", "k");
        let link = scratch.0.join("models");
        std::os::unix::fs::symlink(scratch.0.join("private"), &link).unwrap();
        assert!(check_exposable(&[&link.join("api-key")], std::slice::from_ref(&key)).is_err());
        assert!(check_exposable(&[&link], std::slice::from_ref(&key)).is_err());
        // Resolution keeps what does not exist, and applies `..` after it lexically.
        let missing = scratch.0.join("missing/../models/new");
        assert_eq!(resolve(&missing), scratch.0.join("private/new"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_compares_paths_without_regard_to_case_or_verbatim_prefixes() {
        let scratch = Scratch::new("straydog-case");
        let key = scratch.file("Home/.straydog/api-key", "k");
        let upper = PathBuf::from(scratch.0.to_string_lossy().to_uppercase()).join("HOME");
        assert!(check_exposable(&[&upper], std::slice::from_ref(&key)).is_err());
        let verbatim = std::fs::canonicalize(scratch.0.join("Home")).unwrap();
        assert!(verbatim.to_string_lossy().starts_with(r"\\?\"));
        assert!(check_exposable(&[&verbatim], std::slice::from_ref(&key)).is_err());
        assert!(!resolve(&verbatim).to_string_lossy().starts_with(r"\\?\"));
    }

    #[test]
    fn utf8_console_output_is_kept_and_anything_else_is_decoded_lossily_at_worst() {
        assert_eq!(decode_console_output("アクセスが拒否されました。".as_bytes()), "アクセスが拒否されました。");
        assert_eq!(decode_console_output(b""), "");
        assert!(!decode_console_output(&[0x82, 0xa0, b'x']).is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn localized_console_output_in_a_windows_code_page_is_readable() {
        // "アクセスが拒否されました。" (access is denied) in cp932, as Japanese Windows writes it.
        let cp932 = [
            0x83, 0x41, 0x83, 0x4e, 0x83, 0x5a, 0x83, 0x58, 0x82, 0xaa, 0x8b, 0x91, 0x94, 0xdb, 0x82, 0xb3, 0x82, 0xea,
            0x82, 0xdc, 0x82, 0xb5, 0x82, 0xbd, 0x81, 0x42,
        ];
        assert_eq!(win32::decode(&cp932, 932).as_deref(), Some("アクセスが拒否されました。"));
    }
}

#[cfg(test)]
mod engine_variables_tests {
    use super::ENGINE_VARIABLES;
    use std::path::{Path, PathBuf};

    /// What the engine's code reads that no worker needs, and why: a test's own paths, set by hand where the test runs in-process.
    const TEST_ONLY: [(&str, &str); 5] = [
        ("STRAYDOG_TOKENIZER_JSON", "a test's tokenizer file (text.rs)"),
        ("STRAYDOG_QWEN_MMPROJ", "a test's vision projector (qwen_vision.rs)"),
        ("STRAYDOG_DRAFTER", "the layer-order diagnostic test's drafter (layer_prefill_diagnostics.rs)"),
        ("STRAYDOG_LAYER_MODEL", "the layer-order diagnostic test's model directory (layer_prefill_diagnostics.rs)"),
        ("STRAYDOG_LAYER_DIAGNOSTIC_OUT", "the layer-order diagnostic test's output directory (layer_prefill_diagnostics.rs)"),
    ];

    /// What the worker's own code reads beside the engine's (behind.rs: the locking behind "ready").
    const WORKER_READS: [&str; 2] = ["STRAYDOG_LOCK_DURING_PREFILL", "STRAYDOG_VRAM_DIAGNOSTIC"];

    /// The names given as string literals to `calls` in the files under `root` with one of `extensions`, with the file each is read in.
    fn names_read(root: &Path, extensions: &[&str], calls: &[&str]) -> Vec<(String, PathBuf)> {
        let mut found = Vec::new();
        let mut directories = vec![root.to_path_buf()];
        while let Some(directory) = directories.pop() {
            for entry in std::fs::read_dir(&directory).unwrap().map(Result::unwrap) {
                let path = entry.path();
                if path.is_dir() {
                    directories.push(path);
                    continue;
                }
                if !path.extension().and_then(|extension| extension.to_str()).is_some_and(|extension| extensions.contains(&extension)) {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                for call in calls {
                    for (at, _) in text.match_indices(call) {
                        let rest = &text[at + call.len()..];
                        if let Some(end) = rest.find('"') {
                            found.push((rest[..end].to_string(), path.clone()));
                        }
                    }
                }
            }
        }
        found
    }

    /// Decides: every variable the engine reads by name (its Rust, and its GPU code's getenv) reaches the worker, or is a test's, named here
    /// with why; the worker's own reads are on the list too; and the list names nothing twice, nor a test's. A switch the worker never got
    /// measured nothing: the layer order's rows and the K/V's format on 2026-11-11.
    #[test]
    fn every_variable_the_engine_reads_reaches_the_worker() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut read = names_read(&repository.join("crates/straydog-engine/src"), &["rs"], &["env::var(\"", "env::var_os(\""]);
        read.extend(names_read(&repository.join("gpu"), &["cpp", "hpp", "h"], &["getenv(\""]));
        assert!(read.len() > 20, "the scan found the engine's reads: {read:?}");
        let missing: Vec<&(String, PathBuf)> = read.iter()
            .filter(|(name, _)| !ENGINE_VARIABLES.contains(&name.as_str()) && !TEST_ONLY.iter().any(|(test, _)| test == name))
            .collect();
        assert!(missing.is_empty(), "read by the engine, not passed to the worker (ENGINE_VARIABLES) nor a test's (TEST_ONLY): {missing:?}");
        for name in WORKER_READS {
            assert!(ENGINE_VARIABLES.contains(&name), "{name}");
        }
        for (at, name) in ENGINE_VARIABLES.iter().enumerate() {
            assert!(!ENGINE_VARIABLES[..at].contains(name), "{name} twice");
            assert!(!TEST_ONLY.iter().any(|(test, _)| test == name), "{name} is a test's");
        }
    }
}
