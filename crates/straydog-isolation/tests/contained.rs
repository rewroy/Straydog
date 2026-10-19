//! The worker contained for real: bubblewrap on Linux, the AppContainer launcher on Windows,
//! and process trees ended whole.
//!
//! This executable plays the straydog executable: its `probe` subcommand is the worker, its
//! `contain` subcommand the Windows launcher, and `parent` starts a grandchild for the
//! process-tree tests (`sleeper` is that grandchild); so it runs its tests itself, after
//! dispatching those. The Windows AppContainer test is opt-in (`cargo test -- --ignored`): it
//! grants the persistent `Straydog.Worker` profile read access to a copy of this executable's
//! directory.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use straydog_isolation::{prepare, unavailable_reason, worker_command, Exposure, ProcessTree};

fn main() {
    let args: Vec<OsString> = std::env::args_os().collect();
    match args.get(1).and_then(|arg| arg.to_str()) {
        Some("probe") => probe(&args[2..]),
        Some("parent") => parent(),
        Some("sleeper") => sleeper(&args[2..]),
        #[cfg(windows)]
        Some(straydog_isolation::LAUNCHER_SUBCOMMAND) => {
            std::process::exit(straydog_isolation::launcher_main(&args[2..]))
        }
        _ => run(&args[1..]),
    }
}

type Test = (&'static str, fn(), bool);

/// Run the tests (`--ignored` runs the opt-in ones too); any other argument filters by name.
fn run(args: &[OsString]) {
    let tests: &[Test] = &[
        ("a_child_and_its_grandchild_end_together", a_child_and_its_grandchild_end_together, false),
        ("a_grandchild_that_ignores_the_request_still_ends", a_grandchild_that_ignores_the_request_still_ends, false),
        #[cfg(target_os = "linux")]
        ("the_contained_worker_reads_only_what_is_exposed", the_contained_worker_reads_only_what_is_exposed, false),
        #[cfg(target_os = "linux")]
        ("a_contained_tree_ends_with_its_sandbox", a_contained_tree_ends_with_its_sandbox, false),
        #[cfg(windows)]
        ("the_worker_runs_contained_over_the_server_pipes", the_worker_runs_contained_over_the_server_pipes, true),
    ];
    let args: Vec<String> = args.iter().map(|arg| arg.to_string_lossy().into_owned()).collect();
    let ignored = args.iter().any(|arg| arg == "--ignored" || arg == "--include-ignored");
    let filters: Vec<&String> = args.iter().filter(|arg| !arg.starts_with('-')).collect();
    let mut failed = Vec::new();
    for &(name, test, opt_in) in tests {
        if !filters.is_empty() && !filters.iter().any(|filter| name.contains(filter.as_str())) {
            continue;
        }
        if opt_in && !ignored {
            println!("test {name} ... ignored (opt-in: --ignored)");
            continue;
        }
        let passed = std::panic::catch_unwind(test).is_ok();
        println!("test {name} ... {}", if passed { "ok" } else { "FAILED" });
        if !passed {
            failed.push(name);
        }
    }
    if !failed.is_empty() {
        println!("failed: {}", failed.join(", "));
        std::process::exit(1);
    }
}

/// The worker: echo one line of standard input, then report what it can reach, one `key=value`
/// line each. `args`: a loopback port, then paths to read.
fn probe(args: &[OsString]) {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    println!("echo={}", line.trim_end());
    let port: u16 = args[0].to_str().unwrap().parse().unwrap();
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let network = TcpStream::connect_timeout(&address, Duration::from_secs(3)).is_ok();
    println!("network={}", if network { "open" } else { "closed" });
    println!("cwd={}", std::env::current_dir().map(|path| path.display().to_string()).unwrap_or_default());
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).unwrap_or_default();
    println!("home={}", home.to_string_lossy());
    for (name, directory) in [("write-cwd", PathBuf::from(".")), ("write-home", PathBuf::from(&home))] {
        let path = directory.join("straydog-probe.txt");
        let written = std::fs::write(&path, "x").is_ok();
        let _ = std::fs::remove_file(&path);
        println!("{name}={}", if written { "written" } else { "denied" });
    }
    for path in &args[1..] {
        let content = std::fs::read_to_string(path).unwrap_or_else(|_| "denied".into());
        println!("read {}={content}", path.to_string_lossy());
    }
}

/// Start a grandchild once told to on standard input, print its id once it is ready, and wait
/// for it. The arguments are the grandchild's.
fn parent() {
    std::io::stdin().read_line(&mut String::new()).unwrap();
    let mut sleeper = Command::new(std::env::current_exe().unwrap())
        .arg("sleeper")
        .args(std::env::args_os().skip(2))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    BufReader::new(sleeper.stdout.take().unwrap()).read_line(&mut String::new()).unwrap();
    println!("{}", sleeper.id());
    std::io::stdout().flush().unwrap();
    sleeper.wait().unwrap();
}

/// Sleep for ten minutes, ignoring SIGTERM when asked to (`ignore-term`); say when ready.
fn sleeper(args: &[OsString]) {
    #[cfg(unix)]
    if args.iter().any(|arg| arg == "ignore-term") {
        // SAFETY: ignoring a signal runs no handler code.
        unsafe { libc::signal(libc::SIGTERM, libc::SIG_IGN) };
    }
    #[cfg(not(unix))]
    let _ = args;
    println!("ready");
    std::io::stdout().flush().unwrap();
    std::thread::sleep(Duration::from_secs(600));
}

/// What `probe` reported.
fn report(output: &str) -> Vec<(String, String)> {
    output.lines().filter_map(|line| line.split_once('=')).map(|(key, value)| (key.into(), value.into())).collect()
}

fn value<'a>(report: &'a [(String, String)], key: &str) -> &'a str {
    report
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
        .unwrap_or_else(|| panic!("no {key} in {report:?}"))
}

/// An empty directory of its own under `parent`, removed with its contents on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn under(parent: &Path, name: &str) -> Scratch {
        let path = parent.join(format!("{name}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(std::fs::canonicalize(&path).map(plain).unwrap())
    }

    fn file(&self, relative: &str, content: &str) -> PathBuf {
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

/// A canonical path as Windows writes it elsewhere, without the `\\?\` prefix.
fn plain(path: PathBuf) -> PathBuf {
    match path.to_str().and_then(|text| text.strip_prefix(r"\\?\")) {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => path,
    }
}

fn a_child_and_its_grandchild_end_together() {
    end_a_tree(&[]);
}

/// The request to end is ignored (Unix): the hard kill ends the tree anyway.
fn a_grandchild_that_ignores_the_request_still_ends() {
    end_a_tree(&["ignore-term"]);
}

/// Start `parent` (the grandchild given `args`) as the server starts its worker, then end it.
fn end_a_tree(args: &[&str]) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.arg("parent").args(args).stdin(Stdio::piped()).stdout(Stdio::piped());
    prepare(&mut command);
    let mut child = command.spawn().unwrap();
    let tree = ProcessTree::new(&child).unwrap();
    // The grandchild starts only once the child is in its tree (Windows: in the job).
    child.stdin.as_mut().unwrap().write_all(b"go\n").unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    let grandchild: u32 = line.trim().parse().unwrap();
    assert!(alive(grandchild));
    let status = tree.end(&mut child, Duration::from_secs(2)).unwrap();
    assert!(!status.success());
    assert!(gone_within(grandchild, Duration::from_secs(5)), "the grandchild outlived its tree");
    // Ending it again finds nothing left to end.
    tree.kill().unwrap();
}

fn gone_within(pid: u32, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    while alive(pid) {
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    true
}

/// Whether the process runs (a zombie, ended but not yet reaped by its new parent, does not).
#[cfg(target_os = "linux")]
fn alive(pid: u32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else { return false };
    let state = stat.rsplit_once(") ").and_then(|(_, rest)| rest.chars().next());
    state.is_some_and(|state| state != 'Z' && state != 'X')
}

#[cfg(windows)]
fn alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
    // SAFETY: plain values; the handle, when opened, is closed once.
    unsafe {
        let process = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if process.is_null() {
            return false;
        }
        let running = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
        CloseHandle(process);
        running
    }
}

/// Spawn `program args` the way the server spawns its worker, write `input`, and collect what
/// it prints until it exits.
fn run_contained(program: &Path, args: &[OsString], input: &[u8]) -> (bool, String, String) {
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    prepare(&mut command);
    let mut child = command.spawn().unwrap();
    let _tree = ProcessTree::new(&child).unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let mut errors = String::new();
    let mut stderr = child.stderr.take().unwrap();
    let reader = std::thread::spawn(move || {
        stderr.read_to_string(&mut errors).unwrap();
        errors
    });
    let mut output = String::new();
    child.stdout.take().unwrap().read_to_string(&mut output).unwrap();
    let status = child.wait().unwrap();
    (status.success(), output, reader.join().unwrap())
}

#[cfg(target_os = "linux")]
fn the_contained_worker_reads_only_what_is_exposed() {
    if let Some(reason) = unavailable_reason() {
        panic!("{reason}");
    }
    let home = PathBuf::from(std::env::var_os("HOME").unwrap());
    let scratch = Scratch::under(&home, ".straydog-isolation-test");
    let exposed = scratch.file("model/a.gguf", "exposed");
    let hidden = scratch.file("model/not exposed.txt", "hidden");
    let outside = scratch.file("private.txt", "private");
    let model = scratch.0.join("model");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    TcpStream::connect(("127.0.0.1", port)).expect("the port listens outside the sandbox");
    let exposure = Exposure {
        executable: std::env::current_exe().unwrap(),
        files: vec![exposed.clone()],
        directories: vec![model.clone()],
        working_directory: model.clone(),
        secrets: vec![scratch.file("api-key", "secret")],
    };
    let args: Vec<OsString> = vec![
        "probe".into(),
        port.to_string().into(),
        exposed.clone().into(),
        hidden.clone().into(),
        outside.clone().into(),
    ];
    let (program, args) = worker_command(&args, &exposure).unwrap();
    let (success, output, errors) = run_contained(&program, &args, b"over the private pipe\n");
    assert!(success, "{errors}");
    let report = report(&output);
    assert_eq!(value(&report, "echo"), "over the private pipe");
    assert_eq!(value(&report, "network"), "closed");
    assert_eq!(value(&report, "cwd"), model.display().to_string());
    assert_eq!(value(&report, "home"), "/tmp");
    assert_eq!(value(&report, &format!("read {}", exposed.display())), "exposed");
    assert_eq!(value(&report, &format!("read {}", hidden.display())), "denied");
    assert_eq!(value(&report, &format!("read {}", outside.display())), "denied");
    // The working directory is writable only in the sandbox's own memory: nothing reached the host.
    assert!(!model.join("straydog-probe.txt").exists());
    assert_eq!(std::fs::read_dir(&model).unwrap().count(), 2);
}

#[cfg(target_os = "linux")]
fn a_contained_tree_ends_with_its_sandbox() {
    let scratch = Scratch::under(&std::env::temp_dir(), "straydog-isolation-tree");
    let token = format!("straydog-isolation-token-{}", std::process::id());
    let exposure = Exposure {
        executable: std::env::current_exe().unwrap(),
        working_directory: scratch.0.clone(),
        ..Exposure::default()
    };
    let (program, args) = worker_command(&["parent".into(), token.clone().into()], &exposure).unwrap();
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::piped()).stdout(Stdio::piped());
    prepare(&mut command);
    let mut child = command.spawn().unwrap();
    let tree = ProcessTree::new(&child).unwrap();
    child.stdin.as_mut().unwrap().write_all(b"go\n").unwrap();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut String::new()).unwrap();
    // The sandbox has a process namespace of its own: its processes are found by their token
    // (bubblewrap's own command line holds it too).
    assert_eq!(carrying(&["sleeper", &token]).len(), 1, "the contained grandchild runs");
    tree.end(&mut child, Duration::from_secs(5)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !carrying(&[&token]).is_empty() {
        assert!(Instant::now() < deadline, "contained processes outlived their tree: {:?}", carrying(&[&token]));
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The running processes whose command line holds each of `args`.
#[cfg(target_os = "linux")]
fn carrying(args: &[&str]) -> Vec<u32> {
    let entries = std::fs::read_dir("/proc").unwrap().flatten();
    let pids = entries.filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok());
    pids.filter(|&pid| {
        let line = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
        let held = |arg: &&str| line.split(|&byte| byte == 0).any(|held| held == arg.as_bytes());
        alive(pid) && args.iter().all(held)
    })
    .collect()
}

#[cfg(windows)]
fn the_worker_runs_contained_over_the_server_pipes() {
    assert_eq!(unavailable_reason(), None);
    let scratch = Scratch::under(&std::env::temp_dir(), "straydog-isolation-appcontainer");
    // A copy of this executable in a directory of its own: the launcher grants the whole directory.
    let executable = scratch.0.join("bin/straydog.exe");
    std::fs::create_dir_all(executable.parent().unwrap()).unwrap();
    std::fs::copy(std::env::current_exe().unwrap(), &executable).unwrap();
    let exposed = scratch.file("model/a.gguf", "exposed");
    let hidden = scratch.file("model/not exposed.txt", "hidden");
    let outside = scratch.file("private.txt", "private");
    let model = scratch.0.join("model");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let exposure = Exposure {
        executable,
        files: vec![exposed.clone()],
        directories: vec![model.clone()],
        working_directory: model.clone(),
        secrets: vec![scratch.file("api-key", "secret")],
    };
    let args: Vec<OsString> = vec![
        "probe".into(),
        port.to_string().into(),
        exposed.clone().into(),
        hidden.clone().into(),
        outside.clone().into(),
    ];
    let (program, args) = worker_command(&args, &exposure).unwrap();
    let (success, output, errors) = run_contained(&program, &args, b"over the private pipe\n");
    assert!(success, "{errors}");
    let report = report(&output);
    assert_eq!(value(&report, "echo"), "over the private pipe");
    assert_eq!(value(&report, "network"), "closed");
    assert_eq!(value(&report, "cwd").to_lowercase(), model.display().to_string().to_lowercase());
    assert_eq!(value(&report, "write-cwd"), "denied");
    assert_eq!(value(&report, "write-home"), "denied");
    assert_eq!(value(&report, &format!("read {}", exposed.display())), "exposed");
    assert_eq!(value(&report, &format!("read {}", hidden.display())), "denied");
    assert_eq!(value(&report, &format!("read {}", outside.display())), "denied");
    drop(listener);
}
