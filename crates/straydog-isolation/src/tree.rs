//! Owning the worker's whole process tree, so that no part of it outlives the server or
//! survives being stopped.
//!
//! Linux: the child starts a session of its own, so its process group, whose id is the
//! child's, holds every process it starts that does not leave it (bubblewrap's sandbox does,
//! and bubblewrap ends it when it ends itself). The child is sent SIGKILL when the thread that
//! spawned it ends, which is what ends it with the server.
//!
//! Windows: the child runs without a console window, in a console process group of its own,
//! and right after the spawn joins a Job Object that ends every process in it when its last
//! handle closes: when the [`ProcessTree`] is dropped, or when the server ends in any way.

use std::io;
use std::process::{Child, Command, ExitStatus};
use std::time::Duration;

/// Configure `command` to start the root of a tree a [`ProcessTree`] then owns.
///
/// Linux: a session of its own, and SIGKILL when the spawning thread ends. Spawn from a thread
/// that lives as long as the child (the server's runtime thread), not from a pool thread that
/// may retire. Windows: `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`: a console group of its
/// own and no visible console window, which the child's own children inherit (a console
/// program started without a visible console gets a new, visible one otherwise, redirected
/// output or not).
pub fn prepare(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        #[cfg(target_os = "linux")]
        let parent = std::process::id() as libc::pid_t;
        // SAFETY: the closure runs between fork and exec and calls only async-signal-safe
        // functions (setsid, prctl, getppid); it allocates nothing.
        unsafe {
            command.pre_exec(move || {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                #[cfg(target_os = "linux")]
                {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                        return Err(io::Error::last_os_error());
                    }
                    // The server may have ended before the request: nothing would send the signal then.
                    if libc::getppid() != parent {
                        return Err(io::Error::from_raw_os_error(libc::ESRCH));
                    }
                }
                Ok(())
            });
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::System::Threading::{CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW};
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
}

/// The processes a [`prepare`]d child started, ended together.
///
/// On Windows dropping it ends them (its Job Object kills on close): keep it for as long as the
/// worker runs. On Linux dropping it changes nothing.
#[derive(Debug)]
pub struct ProcessTree {
    #[cfg(unix)]
    group: libc::pid_t,
    #[cfg(windows)]
    job: crate::win32::Job,
}

impl ProcessTree {
    /// The tree of `child`, spawned from a prepared command. On Windows call it right after the
    /// spawn: a process the child starts before joining the job is not in it.
    pub fn new(child: &Child) -> io::Result<ProcessTree> {
        #[cfg(unix)]
        {
            ProcessTree::from_pid(child.id())
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            ProcessTree::from_handle(child.as_raw_handle())
        }
    }

    /// Unix: the tree of the prepared child with process id `pid` (a child that is not a
    /// `std::process::Child`, such as tokio's).
    #[cfg(unix)]
    pub fn from_pid(pid: u32) -> io::Result<ProcessTree> {
        let group = libc::pid_t::try_from(pid).map_err(io::Error::other)?;
        // SAFETY: plain values; getpgid only reads.
        if unsafe { libc::getpgid(group) } != group {
            return Err(io::Error::other(
                "Straydog's worker must be started from a prepared command (a session of its own)",
            ));
        }
        Ok(ProcessTree { group })
    }

    /// Windows: the tree of the prepared child with this process handle (a child that is not a
    /// `std::process::Child`, such as tokio's). The handle needs `PROCESS_SET_QUOTA` and
    /// `PROCESS_TERMINATE` access, which a spawn's handle has.
    #[cfg(windows)]
    pub fn from_handle(process: std::os::windows::io::RawHandle) -> io::Result<ProcessTree> {
        let job = crate::win32::Job::new()?;
        job.assign(process)?;
        Ok(ProcessTree { job })
    }

    /// Ask every process in the tree to end: SIGTERM to the group. Windows has no such request
    /// for processes without a console window: the job is terminated, as by
    /// [`ProcessTree::kill`].
    pub fn terminate(&self) -> io::Result<()> {
        #[cfg(unix)]
        {
            self.signal(libc::SIGTERM)
        }
        #[cfg(windows)]
        {
            self.job.terminate()
        }
    }

    /// End every process in the tree now: SIGKILL to the group; on Windows the job is
    /// terminated (exit status 1).
    pub fn kill(&self) -> io::Result<()> {
        #[cfg(unix)]
        {
            self.signal(libc::SIGKILL)
        }
        #[cfg(windows)]
        {
            self.job.terminate()
        }
    }

    /// End the tree gracefully, then hard: [`terminate`](ProcessTree::terminate), wait up to
    /// `grace` for the child, [`kill`](ProcessTree::kill) whatever remains, and reap the child.
    ///
    /// Call it before the child is reaped. On Unix the child is watched without being reaped,
    /// so the group's id stays its own until the hard kill (an id is reused only once nothing
    /// holds it); when it was already reaped, that kill is skipped rather than risk reaching a
    /// group that took the id over.
    pub fn end(&self, child: &mut Child, grace: Duration) -> io::Result<ExitStatus> {
        self.terminate()?;
        #[cfg(unix)]
        {
            let deadline = std::time::Instant::now() + grace;
            loop {
                match exited(self.group) {
                    Ok(true) => break,
                    Ok(false) if std::time::Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
                    Ok(false) => break,
                    Err(_) => return child.wait(),
                }
            }
            self.kill()?;
        }
        #[cfg(windows)]
        let _ = grace; // terminating the job ended every process at once
        child.wait()
    }

    #[cfg(unix)]
    fn signal(&self, signal: libc::c_int) -> io::Result<()> {
        // SAFETY: plain values.
        if unsafe { libc::killpg(self.group, signal) } == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        // No process is left in the group.
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

/// Whether the child `pid` has exited, leaving it waitable (a zombie keeps its id, and so its
/// group's, from being reused). An error when it is no child of this process or was reaped.
#[cfg(unix)]
fn exited(pid: libc::pid_t) -> io::Result<bool> {
    // SAFETY: an all-zero siginfo_t is a valid value.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let options = libc::WEXITED | libc::WNOHANG | libc::WNOWAIT;
    // SAFETY: a valid siginfo_t to write; WNOWAIT leaves the child waitable.
    if unsafe { libc::waitid(libc::P_PID, pid as libc::id_t, &mut info, options) } == -1 {
        return Err(io::Error::last_os_error());
    }
    // With WNOHANG and no exit yet, si_pid keeps the zero it was given.
    // SAFETY: waitid filled in a SIGCHLD siginfo_t, or left it zeroed.
    Ok(unsafe { info.si_pid() } != 0)
}
