//! The Windows launcher's logic apart from its Win32 calls (`win32`): its arguments, the grants
//! it makes, and the command line and environment it starts the worker with. It builds
//! everywhere so that its tests run everywhere; only Windows runs it.
//!
//! The server starts the launcher (`straydog contain`) as an ordinary child with its private
//! pipes. The launcher grants the dedicated `Straydog.Worker` AppContainer profile read access
//! to exactly what the worker reads: the executable's directory (cuBLAS may sit next to the
//! executable), the CUDA Toolkit's DLL directories, each pinned file and the directories the
//! worker opens them in. It then starts the worker in that AppContainer with the launcher's own
//! standard handles (and no other inheritable handle) and exits with the worker's status.
//!
//! No capability is given, so the network (loopback included) is closed; nothing outside the
//! profile's own folder is writable. A Job Object ends the worker with the launcher. There is
//! no fallback to an unrestricted child.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};

use crate::{check_exposable, resolve, Exposure, LAUNCHER_SUBCOMMAND};

/// The AppContainer profile the worker runs as. It persists, and so do its grants.
pub const PROFILE_NAME: &str = "Straydog.Worker";

/// The access a grant gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Access {
    Read,
    ReadExecute,
}

impl Access {
    /// What a file needs: a DLL or an executable is mapped as code.
    fn for_file(path: &Path) -> Access {
        let code = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dll") || extension.eq_ignore_ascii_case("exe"));
        if code {
            Access::ReadExecute
        } else {
            Access::Read
        }
    }
}

/// What granting needs from the file system's access control: Win32 (`win32::Acl`), or a fake
/// in tests.
pub trait Grants {
    type Sid;
    /// Create the AppContainer profile, or find the existing one, and return its SID.
    fn create_profile(&self, name: &str) -> io::Result<Self::Sid>;
    /// Whether the path's DACL already allows `access` to the profile or to ALL APPLICATION
    /// PACKAGES (by an entry the subtree inherits, when `inherit`). Any failure answers false,
    /// so the grant is written instead: merged into the DACL, a redundant entry is harmless.
    fn has_grant(&self, path: &Path, sid: &Self::Sid, access: Access, inherit: bool) -> bool;
    /// Merge an entry allowing `access` to the profile into the path's DACL, inherited by the
    /// subtree when `inherit` (writing it propagates it there).
    fn grant(&self, path: &Path, sid: &Self::Sid, access: Access, inherit: bool) -> io::Result<()>;
}

/// What the launcher was asked to do.
#[derive(Debug, Default, PartialEq)]
pub struct Launch {
    /// Directories whose whole subtree the worker reads and executes.
    pub trees: Vec<PathBuf>,
    pub files: Vec<PathBuf>,
    /// Directories the worker opens files in (their own entries, not their subtree).
    pub directories: Vec<PathBuf>,
    pub secrets: Vec<PathBuf>,
    /// Where the worker runs; the launcher's own working directory when not given.
    pub working_directory: Option<PathBuf>,
    /// The worker's program and arguments.
    pub command: Vec<OsString>,
}

impl Launch {
    /// The arguments after `contain`: `--tree`, `--file`, `--directory` and `--secret` (each
    /// repeated), `--chdir`, then the worker command after `--`.
    fn parse(args: &[OsString]) -> Result<Launch, String> {
        let mut launch = Launch::default();
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            if arg == "--" {
                launch.command = rest.cloned().collect();
                break;
            }
            let list = match arg.to_str() {
                Some("--tree") => &mut launch.trees,
                Some("--file") => &mut launch.files,
                Some("--directory") => &mut launch.directories,
                Some("--secret") => &mut launch.secrets,
                Some("--chdir") => {
                    let value = rest.next().ok_or("--chdir needs a path")?;
                    launch.working_directory = Some(value.into());
                    continue;
                }
                _ => return Err(format!("unknown argument {}", arg.to_string_lossy())),
            };
            list.push(rest.next().ok_or_else(|| format!("{} needs a path", arg.to_string_lossy()))?.into());
        }
        if launch.command.is_empty() {
            return Err("runs on Windows with a worker command after --".into());
        }
        Ok(launch)
    }
}

/// The launcher's arguments that run `exposure.executable worker_args...` contained: the
/// subcommand, what to grant, the working directory, then the worker command after `--`.
/// `cuda` are the CUDA Toolkit's DLL directories.
pub fn launcher_arguments(
    worker_args: &[OsString],
    exposure: &Exposure,
    cuda: &[PathBuf],
) -> Result<Vec<OsString>, String> {
    let directory = exposure.executable.parent().ok_or("Straydog worker executable has no directory")?;
    let mut trees = vec![directory.to_path_buf()];
    for path in cuda {
        if !trees.contains(path) {
            trees.push(path.clone());
        }
    }
    let exposed = trees.iter().chain(&exposure.files).chain(&exposure.directories).chain([&exposure.working_directory]);
    check_exposable(&exposed.map(PathBuf::as_path).collect::<Vec<_>>(), &exposure.secrets)?;
    let mut args: Vec<OsString> = vec![LAUNCHER_SUBCOMMAND.into()];
    for (flag, paths) in [
        ("--tree", &trees),
        ("--file", &exposure.files),
        ("--directory", &exposure.directories),
        ("--secret", &exposure.secrets),
    ] {
        for path in paths {
            args.extend([flag.into(), path.into()]);
        }
    }
    args.extend([
        "--chdir".into(),
        exposure.working_directory.clone().into(),
        "--".into(),
        exposure.executable.clone().into(),
    ]);
    args.extend(worker_args.iter().cloned());
    Ok(args)
}

/// The CUDA Toolkit's DLL directories under `root` (`CUDA_PATH`): `bin`, and `bin\x64` since
/// CUDA 13.
pub fn cuda_directories(root: Option<&OsStr>) -> Vec<PathBuf> {
    let Some(root) = root.filter(|root| !root.is_empty()) else { return Vec::new() };
    let bin = Path::new(root).join("bin");
    [bin.clone(), bin.join("x64")].into_iter().filter(|path| path.is_dir()).collect()
}

/// Grant the profile read access to what the worker reads and return the profile's SID.
///
/// Idempotent: a grant already present is not written again (the first grant on a tree
/// propagates through it and is slow). Nothing is granted, and no profile created, when an
/// exposed path is, holds or lies inside a secret.
pub fn grant_worker_access<G: Grants>(grants: &G, launch: &Launch) -> Result<G::Sid, String> {
    // A pinned file is a regular file: what the worker authenticates is what it opens.
    for file in &launch.files {
        if !std::fs::symlink_metadata(file).is_ok_and(|metadata| metadata.is_file()) {
            return Err(format!("Straydog worker file is missing or not a regular file: {}", file.display()));
        }
    }
    let resolved = |paths: &[PathBuf]| paths.iter().map(|path| resolve(path)).collect::<Vec<_>>();
    let (trees, files, mut directories) =
        (resolved(&launch.trees), resolved(&launch.files), resolved(&launch.directories));
    if let Some(working) = launch.working_directory.as_deref().map(resolve) {
        if !directories.contains(&working) {
            directories.push(working);
        }
    }
    let exposed: Vec<&Path> = trees.iter().chain(&files).chain(&directories).map(PathBuf::as_path).collect();
    check_exposable(&exposed, &launch.secrets)?;

    let sid = grants
        .create_profile(PROFILE_NAME)
        .map_err(|error| format!("Straydog cannot create its worker's AppContainer profile: {error}"))?;
    let refused = |path: &Path, error: io::Error| {
        format!("Straydog cannot grant its worker read access to {}: {error}", path.display())
    };
    for tree in &trees {
        if let Err(error) = ensure(grants, tree, &sid, Access::ReadExecute, true) {
            // An administrator-owned tree (the CUDA Toolkit under Program Files) cannot take our
            // entry; it is usable when it already lets AppContainers (ALL APPLICATION PACKAGES) read it.
            if grants.has_grant(tree, &sid, Access::ReadExecute, false) {
                continue;
            }
            return Err(refused(tree, error));
        }
        let missing = ungranted_entries(grants, tree, &sid);
        if missing.is_empty() {
            continue;
        }
        eprintln!(
            "Straydog: granting the worker read access to {} new entries in {} (one-time)",
            missing.len(),
            tree.display()
        );
        // Writing the tree's DACL again propagates its inheritable grant to every entry it can
        // reach; an entry that still lacks it (inheritance blocked on it) gets an entry of its own.
        grants.grant(tree, &sid, Access::ReadExecute, true).map_err(|error| refused(tree, error))?;
        for (path, access) in missing {
            if !grants.has_grant(&path, &sid, access, false) {
                grants.grant(&path, &sid, access, false).map_err(|error| refused(&path, error))?;
            }
        }
    }
    for file in &files {
        ensure(grants, file, &sid, Access::for_file(file), false).map_err(|error| refused(file, error))?;
    }
    for directory in &directories {
        // The directory's own entries (the worker opens files in it), not its subtree.
        ensure(grants, directory, &sid, Access::ReadExecute, false).map_err(|error| refused(directory, error))?;
    }
    Ok(sid)
}

fn ensure<G: Grants>(grants: &G, path: &Path, sid: &G::Sid, access: Access, inherit: bool) -> io::Result<()> {
    if grants.has_grant(path, sid, access, inherit) {
        Ok(())
    } else {
        grants.grant(path, sid, access, inherit)
    }
}

/// The files directly in `tree` the profile cannot use, with the access each needs.
///
/// A file copied, moved or hard-linked into a directory keeps the ACL it had rather than
/// inheriting the directory's grant, and the worker then fails to load it ("Access is
/// denied"): a cuBLAS DLL moved next to the executable, a toolkit file installed that way. The
/// directory itself inherits the grant, so it says nothing about the files in it; each is
/// checked (one DACL read each). Only the files directly in a tree: the loader looks for DLLs
/// in these directories, never below them.
fn ungranted_entries<G: Grants>(grants: &G, tree: &Path, sid: &G::Sid) -> Vec<(PathBuf, Access)> {
    let Ok(entries) = std::fs::read_dir(tree) else { return Vec::new() };
    let mut missing: Vec<(PathBuf, Access)> = entries
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .map(|entry| entry.path())
        .map(|path| {
            let access = Access::for_file(&path);
            (path, access)
        })
        .filter(|(path, access)| !grants.has_grant(path, sid, *access, false))
        .collect();
    missing.sort();
    missing
}

/// The launcher: read the arguments, grant, run the worker and return its status. A failure is
/// one line on standard error (the server shows the worker's last stderr line as the start
/// failure) and status 1; malformed arguments give status 2.
pub fn main<S>(
    args: &[OsString],
    grant: impl FnOnce(&Launch) -> Result<S, String>,
    run: impl FnOnce(&Launch, S) -> io::Result<u32>,
) -> i32 {
    let launch = match Launch::parse(args) {
        Ok(launch) => launch,
        Err(message) => {
            eprintln!("straydog {LAUNCHER_SUBCOMMAND}: {message}");
            return 2;
        }
    };
    match grant(&launch).and_then(|sid| run(&launch, sid).map_err(|error| error.to_string())) {
        // An exit status is a 32-bit value (an NTSTATUS for a crash); the process returns the same bits.
        Ok(status) => status as i32,
        Err(message) => {
            eprintln!("Straydog AppContainer launch failed: {message}");
            1
        }
    }
}

/// A Windows command line for `args` (UTF-16, without the final NUL), quoted the way the C
/// runtime and `CommandLineToArgvW` split it again: an argument with a space or a tab, or an
/// empty one, is quoted; a quote is escaped, and so are the backslashes before a quote.
pub fn command_line(args: &[Vec<u16>]) -> Vec<u16> {
    const SPACE: u16 = b' ' as u16;
    const TAB: u16 = b'\t' as u16;
    const QUOTE: u16 = b'"' as u16;
    const BACKSLASH: u16 = b'\\' as u16;
    let mut line = Vec::new();
    for (index, arg) in args.iter().enumerate() {
        if index > 0 {
            line.push(SPACE);
        }
        let quoted = arg.is_empty() || arg.iter().any(|&unit| unit == SPACE || unit == TAB);
        if quoted {
            line.push(QUOTE);
        }
        let mut backslashes = 0;
        for &unit in arg {
            if unit == BACKSLASH {
                backslashes += 1;
                continue;
            }
            // Backslashes are literal unless a quote follows them.
            let escaped = if unit == QUOTE { backslashes * 2 + 1 } else { backslashes };
            line.extend(std::iter::repeat_n(BACKSLASH, escaped));
            line.push(unit);
            backslashes = 0;
        }
        // Before the closing quote, they are doubled too.
        line.extend(std::iter::repeat_n(BACKSLASH, if quoted { backslashes * 2 } else { backslashes }));
        if quoted {
            line.push(QUOTE);
        }
    }
    line
}

/// A Unicode environment block from `variables`, with `overrides` replacing any variable of the
/// same name (Windows ignores the case of names): `NAME=value` strings sorted by name without
/// regard to case, as Windows keeps them, each NUL-terminated, then a final NUL.
pub fn environment_block(variables: Vec<(Vec<u16>, Vec<u16>)>, overrides: Vec<(Vec<u16>, Vec<u16>)>) -> Vec<u16> {
    let upper = |name: &[u16]| -> Vec<u16> {
        name.iter()
            .map(|&unit| if (u16::from(b'a')..=u16::from(b'z')).contains(&unit) { unit - 32 } else { unit })
            .collect()
    };
    let mut sorted = BTreeMap::new();
    for (name, value) in variables.into_iter().chain(overrides) {
        sorted.insert(upper(&name), (name, value));
    }
    let mut block = Vec::new();
    for (name, value) in sorted.into_values() {
        block.extend(name);
        block.push(u16::from(b'='));
        block.extend(value);
        block.push(0);
    }
    if block.is_empty() {
        block.push(0);
    }
    block.push(0);
    block
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeSet;

    use super::*;
    use crate::tests::Scratch;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    fn os(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn the_command_names_trees_files_and_the_worker_after_the_separator() {
        let scratch = Scratch::new("straydog-launcher-command");
        let cuda = scratch.0.join("cuda/bin");
        std::fs::create_dir_all(cuda.join("x64")).unwrap();
        assert_eq!(cuda_directories(Some(scratch.0.join("cuda").as_os_str())), vec![cuda.clone(), cuda.join("x64")]);
        assert!(cuda_directories(None).is_empty() && cuda_directories(Some(OsStr::new(""))).is_empty());
        let model = scratch.0.join("model");
        let exposure = Exposure {
            executable: scratch.file("bin/straydog.exe", "MZ"),
            files: vec![scratch.file("model/a.gguf", "a"), scratch.file("model/tokenizer.json", "{}")],
            directories: vec![model.clone()],
            working_directory: model.clone(),
            secrets: vec![scratch.0.join("key")],
        };
        let worker = os(&["worker", "--model-root", "C:\\models\\IQ3_S", "--file"]);
        let args = launcher_arguments(&worker, &exposure, std::slice::from_ref(&cuda)).unwrap();
        assert_eq!(args[0], "contain");
        let separator = args.iter().position(|arg| arg == "--").unwrap();
        assert_eq!(args[separator + 1], exposure.executable.as_os_str());
        assert_eq!(&args[separator + 2..], &worker);
        let launch = Launch::parse(&args[1..]).unwrap();
        assert_eq!(
            launch,
            Launch {
                trees: vec![scratch.0.join("bin"), cuda],
                files: exposure.files.clone(),
                directories: vec![model.clone()],
                secrets: exposure.secrets.clone(),
                working_directory: Some(model),
                command: args[separator + 1..].to_vec(),
            }
        );
        // The executable's directory must not hold a secret either.
        let exposure = Exposure { secrets: vec![scratch.file("bin/api-key", "k")], ..exposure };
        assert!(launcher_arguments(&worker, &exposure, &[]).unwrap_err().contains("secret"));
    }

    /// ACL calls against paths that lack the grant; a grant on a tree clears what is below it
    /// when inheritance reaches there.
    #[derive(Default)]
    struct Fake {
        ungranted: RefCell<BTreeSet<PathBuf>>,
        /// Paths whose DACL already lets AppContainers read them, but takes no entry of ours.
        administered: BTreeSet<PathBuf>,
        refused: BTreeSet<PathBuf>,
        inheritance_blocked: bool,
        calls: RefCell<Vec<(PathBuf, Access, bool)>>,
        profiles: RefCell<usize>,
    }

    impl Fake {
        fn lacking(paths: &[&PathBuf]) -> Fake {
            Fake { ungranted: RefCell::new(paths.iter().map(|path| path.to_path_buf()).collect()), ..Fake::default() }
        }

        fn calls(&self) -> Vec<(PathBuf, Access, bool)> {
            self.calls.borrow_mut().drain(..).collect()
        }
    }

    impl Grants for Fake {
        type Sid = u32;

        fn create_profile(&self, name: &str) -> io::Result<u32> {
            assert_eq!(name, "Straydog.Worker");
            *self.profiles.borrow_mut() += 1;
            Ok(7)
        }

        fn has_grant(&self, path: &Path, sid: &u32, _access: Access, inherit: bool) -> bool {
            assert_eq!(*sid, 7);
            if self.administered.contains(path) {
                return !inherit;
            }
            !self.ungranted.borrow().contains(path)
        }

        fn grant(&self, path: &Path, _sid: &u32, access: Access, inherit: bool) -> io::Result<()> {
            self.calls.borrow_mut().push((path.to_path_buf(), access, inherit));
            if self.refused.contains(path) || self.administered.contains(path) {
                return Err(io::Error::from(io::ErrorKind::PermissionDenied));
            }
            let mut ungranted = self.ungranted.borrow_mut();
            if inherit && !self.inheritance_blocked {
                ungranted.retain(|entry| !entry.starts_with(path));
            }
            ungranted.remove(path);
            Ok(())
        }
    }

    /// The executable's directory: the executable, cuBLAS (moved in from elsewhere) and a note.
    fn executable_directory(scratch: &Scratch) -> (PathBuf, PathBuf, PathBuf) {
        scratch.file("bin/straydog.exe", "MZ");
        let dll = scratch.file("bin/cublas64_12.dll", "MZ");
        let note = scratch.file("bin/NOTICE.txt", "");
        std::fs::create_dir_all(scratch.0.join("bin/sub")).unwrap();
        (scratch.0.join("bin"), dll, note)
    }

    fn launch(trees: Vec<PathBuf>) -> Launch {
        Launch { trees, command: os(&["straydog.exe", "worker"]), ..Launch::default() }
    }

    #[test]
    fn a_secret_and_the_directories_holding_it_are_never_granted() {
        let scratch = Scratch::new("straydog-launcher-secret");
        let key = scratch.file("home/.straydog/api-key", "k");
        let fake = Fake::default();
        for tree in [scratch.0.join("home"), scratch.0.join("home/.straydog")] {
            let launch = Launch { secrets: vec![key.clone()], ..launch(vec![tree]) };
            assert!(grant_worker_access(&fake, &launch).unwrap_err().contains("secret"));
        }
        let launch = Launch { files: vec![key.clone()], secrets: vec![key.clone()], ..launch(vec![]) };
        assert!(grant_worker_access(&fake, &launch).unwrap_err().contains("secret"));
        assert_eq!(*fake.profiles.borrow(), 0, "profile created before the check");
    }

    #[test]
    fn files_moved_in_without_the_grant_get_it_again() {
        let scratch = Scratch::new("straydog-launcher-moved");
        let (tree, dll, note) = executable_directory(&scratch);
        let fake = Fake::lacking(&[&dll, &note]);
        assert_eq!(ungranted_entries(&fake, &tree, &7), vec![(note, Access::Read), (dll, Access::ReadExecute)]);
        let launch = launch(vec![tree.clone(), scratch.0.join("cuda/bin")]);
        assert_eq!(grant_worker_access(&fake, &launch), Ok(7));
        // Granted again once, on the tree, which propagates; the missing toolkit directory needed nothing.
        assert_eq!(fake.calls(), vec![(tree, Access::ReadExecute, true)]);
        grant_worker_access(&fake, &launch).unwrap();
        assert_eq!(fake.calls(), vec![]);
    }

    #[test]
    fn a_dll_inheritance_cannot_reach_gets_its_own_grant() {
        // A DLL whose own DACL blocks inheritance keeps lacking the tree's grant, and the worker
        // fails to load it: "Access is denied".
        let scratch = Scratch::new("straydog-launcher-blocked");
        let (tree, dll, _) = executable_directory(&scratch);
        let fake = Fake { inheritance_blocked: true, ..Fake::lacking(&[&dll]) };
        grant_worker_access(&fake, &launch(vec![tree.clone()])).unwrap();
        assert_eq!(fake.calls(), vec![(tree, Access::ReadExecute, true), (dll, Access::ReadExecute, false)]);
    }

    #[test]
    fn an_entry_that_refuses_the_grant_names_itself() {
        let scratch = Scratch::new("straydog-launcher-refused");
        let (tree, dll, _) = executable_directory(&scratch);
        let fake = Fake { inheritance_blocked: true, refused: [dll.clone()].into(), ..Fake::lacking(&[&dll]) };
        let error = grant_worker_access(&fake, &launch(vec![tree])).unwrap_err();
        assert!(error.contains("cublas64_12.dll"), "{error}");
    }

    #[test]
    fn an_administrator_owned_tree_is_used_when_app_containers_may_already_read_it() {
        let scratch = Scratch::new("straydog-launcher-administered");
        let (tree, dll, _) = executable_directory(&scratch);
        let toolkit = scratch.0.join("cuda/bin");
        std::fs::create_dir_all(&toolkit).unwrap();
        let fake = Fake { administered: [toolkit.clone()].into(), ..Fake::default() };
        grant_worker_access(&fake, &launch(vec![tree.clone(), toolkit.clone()])).unwrap();
        assert_eq!(fake.calls(), vec![(toolkit.clone(), Access::ReadExecute, true)]);
        // Without that, the grant's failure is the launcher's.
        let fake = Fake { refused: [tree.clone()].into(), ..Fake::lacking(&[&tree, &dll]) };
        let error = grant_worker_access(&fake, &launch(vec![tree.clone()])).unwrap_err();
        assert!(error.contains(&tree.display().to_string()), "{error}");
    }

    #[test]
    fn files_and_directories_get_grants_of_their_own() {
        let scratch = Scratch::new("straydog-launcher-files");
        let shard = scratch.file("model/a.gguf", "a");
        let library = scratch.file("model/helper.dll", "MZ");
        let model = scratch.0.join("model");
        let drafter = scratch.file("mtp/mtp.gguf", "m").parent().unwrap().to_path_buf();
        let fake = Fake::lacking(&[&shard, &library, &model, &drafter]);
        let launch = Launch {
            files: vec![shard.clone(), library.clone()],
            directories: vec![drafter.clone()],
            working_directory: Some(model.clone()),
            ..launch(vec![])
        };
        grant_worker_access(&fake, &launch).unwrap();
        assert_eq!(
            fake.calls(),
            vec![
                (shard, Access::Read, false),
                (library, Access::ReadExecute, false),
                (drafter, Access::ReadExecute, false),
                (model.clone(), Access::ReadExecute, false),
            ]
        );
        let missing = Launch { files: vec![model.join("b.gguf")], ..Launch::default() };
        assert!(grant_worker_access(&fake, &missing).unwrap_err().contains("not a regular file"));
        #[cfg(unix)]
        {
            let link = model.join("c.gguf");
            std::os::unix::fs::symlink(model.join("a.gguf"), &link).unwrap();
            let linked = Launch { files: vec![link], ..Launch::default() };
            assert!(grant_worker_access(&fake, &linked).unwrap_err().contains("not a regular file"));
        }
    }

    #[test]
    fn main_grants_then_launches_the_command_after_the_separator() {
        let args = os(&["--tree", "T", "--file", "F", "--directory", "D", "--secret", "K", "--chdir", "W"]);
        let args = [args, os(&["--", "worker.exe", "-c", "--file"])].concat();
        let seen = RefCell::new(Vec::new());
        let status = main(
            &args,
            |launch| {
                seen.borrow_mut().push(format!("{launch:?}"));
                Ok(7)
            },
            |launch, sid| {
                assert_eq!(sid, 7);
                assert_eq!(launch.command, os(&["worker.exe", "-c", "--file"]));
                Ok(0xC000_0135)
            },
        );
        assert_eq!(status as u32, 0xC000_0135);
        let expected = Launch {
            trees: vec!["T".into()],
            files: vec!["F".into()],
            directories: vec!["D".into()],
            secrets: vec!["K".into()],
            working_directory: Some("W".into()),
            command: os(&["worker.exe", "-c", "--file"]),
        };
        assert_eq!(seen.into_inner(), vec![format!("{expected:?}")]);
        let failing = |_: &Launch| Err::<u32, _>("missing worker file".to_owned());
        assert_eq!(main(&args, failing, |_, _| unreachable!()), 1);
        assert_eq!(main::<u32>(&os(&["--tree", "T"]), |_| unreachable!(), |_, _| unreachable!()), 2);
        assert_eq!(main::<u32>(&os(&["--bogus", "--", "x"]), |_| unreachable!(), |_, _| unreachable!()), 2);
    }

    #[test]
    fn the_command_line_splits_back_into_the_same_arguments() {
        let line = |args: &[&str]| {
            String::from_utf16(&command_line(&args.iter().map(|arg| wide(arg)).collect::<Vec<_>>())).unwrap()
        };
        assert_eq!(line(&[r"C:\Program Files\straydog.exe", "worker"]), r#""C:\Program Files\straydog.exe" worker"#);
        assert_eq!(line(&["", "a b", "tab\there"]), "\"\" \"a b\" \"tab\there\"");
        assert_eq!(line(&[r#"say "hi""#, r"C:\dir\", r"C:\my dir\"]), r#""say \"hi\"" C:\dir\ "C:\my dir\\""#);
        assert_eq!(line(&[r#"a\"b"#, r"\\server\share"]), r#"a\\\"b \\server\share"#);
    }

    #[test]
    fn the_environment_block_is_sorted_without_case_and_overrides_replace_any_case() {
        let pair = |name: &str, value: &str| (wide(name), wide(value));
        let block = environment_block(
            vec![pair("Path", "C:\\bin"), pair("temp", "C:\\Users\\u\\Temp"), pair("ALPHA", "1"), pair("=C:", "C:\\")],
            vec![pair("TEMP", "C:\\container\\Temp"), pair("TMP", "C:\\container\\Temp")],
        );
        let text = String::from_utf16(&block).unwrap();
        assert_eq!(text, "=C:=C:\\\0ALPHA=1\0Path=C:\\bin\0TEMP=C:\\container\\Temp\0TMP=C:\\container\\Temp\0\0");
        assert_eq!(environment_block(vec![], vec![]), vec![0, 0]);
    }
}
