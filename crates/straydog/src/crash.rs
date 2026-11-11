//! What a process that dies of an exception nothing handled says first, on Windows: one line on stderr with the exception, what it
//! touched, and the module and offset it happened at (an access violation in a GPU driver or a library, an illegal instruction). A
//! program that runs Straydog shows its last line: Dogmind's setup said only "setup ended (exit code: 0xc0000005)", which no one could
//! act on, and the system's own record of it is in an event log a user does not know to open.
//!
//! The place alone does not say enough where it is a routine everything calls: a release died "reading 0x0 at straydog.exe+0xD2C3A1",
//! which was the C runtime's `strlen` given a null pointer, and nothing said by whom. So the line names what called it too, and a
//! line before it has the whole way there.

/// The line, from the exception's code, what an access violation touched (its kind, 0 read, 1 write or 8 execute, and the address),
/// and the module it happened in with the offset into it.
pub fn crash_line(code: u32, access: Option<(usize, u64)>, module: &str, offset: u64) -> String {
    let what = match code {
        0xC000_0005 => "access violation",
        0xC000_0006 => "in-page error (a mapped file could not be read)",
        0xC000_001D => "illegal instruction",
        0xC000_0094 => "integer division by zero",
        0xC000_00FD => "stack overflow",
        0xC000_0409 => "fail-fast abort",
        _ => "exception",
    };
    let touched = match access {
        Some((0, address)) => format!(" reading 0x{address:X}"),
        Some((1, address)) => format!(" writing 0x{address:X}"),
        Some((8, address)) => format!(" executing 0x{address:X}"),
        _ => String::new(),
    };
    format!("Straydog crashed: {what} (0x{code:08X}){touched} at {module}+0x{offset:X}")
}

/// How many callers the crash line itself names: it is the last line, which is all a program that runs Straydog shows, and it has to fit there.
pub const CALLERS_IN_THE_LINE: usize = 3;
/// How many the line before it names.
pub const CALLERS_AT_MOST: usize = 24;

/// The places a crash was called from, nearest first, as `module+0xoffset < module+0xoffset`: each is where its call returns to.
pub fn callers_said(callers: &[(String, u64)]) -> String {
    callers.iter().map(|(module, offset)| format!("{module}+0x{offset:X}")).collect::<Vec<_>>().join(" < ")
}

/// What is written for a crash: the whole way there on a line of its own when there is more of it than the crash line holds, then the
/// crash line with its nearest callers, last.
pub fn crash_report(line: &str, callers: &[(String, u64)]) -> String {
    let near = &callers[..callers.len().min(CALLERS_IN_THE_LINE)];
    let whole = if callers.len() > near.len() { format!("Straydog crash stack: {}\n", callers_said(callers)) } else { String::new() };
    let from = if near.is_empty() { String::new() } else { format!(", called from {}", callers_said(near)) };
    format!("\n{whole}{line}{from}\n")
}

/// The file name of a module's path (its folder says nothing of where it failed, and may be a user's name).
pub fn module_name(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

#[cfg(windows)]
static PREVIOUS: std::sync::OnceLock<windows_sys::Win32::System::Diagnostics::Debug::LPTOP_LEVEL_EXCEPTION_FILTER> =
    std::sync::OnceLock::new();

/// Say the line when the process dies of an exception, then let the system end it as before (the same exit code, the same report).
pub fn report_unhandled() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter;
        // SAFETY: the process's filter, set once at its start; the one it replaces is called after ours.
        let previous = unsafe { SetUnhandledExceptionFilter(Some(filter)) };
        let _ = PREVIOUS.set(previous);
    }
}

/// The filter: the line, written straight to the stderr handle (std's stderr takes a lock that the failing thread may hold), then the
/// filter before, else on to the system.
#[cfg(windows)]
unsafe extern "system" fn filter(info: *const windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> i32 {
    use windows_sys::Win32::Storage::FileSystem::WriteFile;
    use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE};
    // SAFETY: the system passes the exception's pointers, live while the filter runs.
    if let Some(record) = unsafe { info.as_ref() }.and_then(|info| unsafe { info.ExceptionRecord.as_ref() }) {
        let code = record.ExceptionCode as u32;
        let address = record.ExceptionAddress as usize;
        let access = ((code == 0xC000_0005 || code == 0xC000_0006) && record.NumberParameters >= 2)
            .then(|| (record.ExceptionInformation[0], record.ExceptionInformation[1] as u64));
        let (module, offset) = module_of(address);
        // (not of a stack that ran out: walking it takes stack)
        // SAFETY: the context the system passes with the exception, live while the filter runs.
        let callers: Vec<(String, u64)> =
            if code == 0xC000_00FD { Vec::new() } else { unsafe { callers(info) }.into_iter().map(module_of).collect() };
        let line = crash_report(&crash_line(code, access, &module, offset), &callers);
        let mut written = 0u32;
        // SAFETY: the process's stderr handle and a buffer of the length given.
        unsafe { WriteFile(GetStdHandle(STD_ERROR_HANDLE), line.as_ptr(), line.len() as u32, &mut written, std::ptr::null_mut()) };
    }
    match PREVIOUS.get().copied().flatten() {
        // SAFETY: the filter the process had before ours, with the same pointers.
        Some(previous) => unsafe { previous(info) },
        // (EXCEPTION_CONTINUE_SEARCH: the system ends the process as it would have)
        None => 0,
    }
}

/// Where the failing code was called from, nearest first: the return addresses, unwound from the exception's own context by the
/// functions' unwind data, as a debugger does it. It stops at the first frame that is not on this thread's stack, so a stack that was
/// overwritten ends the list and not the report.
#[cfg(all(windows, target_arch = "x86_64"))]
unsafe fn callers(info: *const windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> Vec<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlLookupFunctionEntry, RtlVirtualUnwind, CONTEXT};
    use windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits;
    // SAFETY (all of it): the pointers are the system's, checked for null; every read of the stack is within the thread's limits.
    let Some(mut context) = info.as_ref().and_then(|info| info.ContextRecord.as_ref()).map(|context: &CONTEXT| *context) else {
        return Vec::new();
    };
    let (mut low, mut high) = (0usize, 0usize);
    GetCurrentThreadStackLimits(&mut low, &mut high);
    let mut found = Vec::new();
    while found.len() < CALLERS_AT_MOST {
        let stack = context.Rsp as usize;
        if stack < low || stack.saturating_add(8) > high || stack % 8 != 0 {
            break;
        }
        let mut base = 0u64;
        let entry = RtlLookupFunctionEntry(context.Rip, &mut base, std::ptr::null_mut());
        if entry.is_null() {
            // (a function with no unwind data touches no stack: its return address is on top)
            context.Rip = *(stack as *const u64);
            context.Rsp += 8;
        } else {
            let (mut handler_data, mut frame) = (std::ptr::null_mut(), 0u64);
            // (UNW_FLAG_NHANDLER: only the frame is wanted)
            RtlVirtualUnwind(0, base, context.Rip, entry, &mut context, &mut handler_data, &mut frame, std::ptr::null_mut());
        }
        if context.Rip == 0 {
            break;
        }
        found.push(context.Rip as usize);
    }
    found
}

/// (Unwinding is written for x86-64 only: elsewhere the line is as it was.)
#[cfg(all(windows, not(target_arch = "x86_64")))]
unsafe fn callers(_info: *const windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> Vec<usize> {
    Vec::new()
}

/// The module `address` lies in, by its file name, and the offset into it; "?" and the address where none is known.
#[cfg(windows)]
fn module_of(address: usize) -> (String, u64) {
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };
    let mut module = std::ptr::null_mut();
    // SAFETY: FROM_ADDRESS reads the name argument as an address in the module; UNCHANGED_REFCOUNT takes no reference to release.
    let found = unsafe {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, address as *const u16, &mut module)
    };
    if found == 0 || module.is_null() {
        return ("?".to_string(), address as u64);
    }
    let mut path = [0u16; 520];
    // SAFETY: a module handle and a buffer of the length given.
    let length = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&path[..length.min(path.len())]);
    (module_name(&path).to_string(), (address - module as usize) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line says the exception by its name and code, what an access violation touched, and where: a module and an offset.
    #[test]
    fn a_crash_is_said_in_one_line() {
        assert_eq!(crash_line(0xC000_0005, Some((0, 0x10)), "nvcuda64.dll", 0x1A2B3C),
                   "Straydog crashed: access violation (0xC0000005) reading 0x10 at nvcuda64.dll+0x1A2B3C");
        assert_eq!(crash_line(0xC000_0005, Some((1, 0xDEAD0000)), "straydog.exe", 0x40),
                   "Straydog crashed: access violation (0xC0000005) writing 0xDEAD0000 at straydog.exe+0x40");
        assert_eq!(crash_line(0xC000_001D, None, "straydog.exe", 0x1234), "Straydog crashed: illegal instruction (0xC000001D) at straydog.exe+0x1234");
        assert_eq!(crash_line(0xE06D_7363, None, "?", 0x7FF0), "Straydog crashed: exception (0xE06D7363) at ?+0x7FF0");
    }

    /// The crash line stays the last line and names its nearest callers; the whole way there is on a line before it, and only when
    /// there is more of it than the crash line holds.
    #[test]
    fn a_crash_says_what_called_it_and_the_crash_line_is_still_last() {
        let line = "Straydog crashed: access violation (0xC0000005) reading 0x0 at straydog.exe+0xD2C3A1";
        assert_eq!(crash_report(line, &[]), format!("\n{line}\n"), "nothing known of the stack: as it was");
        let frame = |module: &str, offset: u64| (module.to_string(), offset);
        let two = [frame("straydog.exe", 0x4A1B2C), frame("nvcuda64.dll", 0x12)];
        assert_eq!(crash_report(line, &two), format!("\n{line}, called from straydog.exe+0x4A1B2C < nvcuda64.dll+0x12\n"));
        let five: Vec<(String, u64)> = (1..=5).map(|at| frame("straydog.exe", at * 0x10)).collect();
        let said = crash_report(line, &five);
        let lines: Vec<&str> = said.trim().lines().collect();
        assert_eq!(lines.len(), 2, "{said}");
        assert_eq!(lines[0], "Straydog crash stack: straydog.exe+0x10 < straydog.exe+0x20 < straydog.exe+0x30 < straydog.exe+0x40 < straydog.exe+0x50");
        assert_eq!(lines[1], format!("{line}, called from straydog.exe+0x10 < straydog.exe+0x20 < straydog.exe+0x30"), "the last line is what a program shows");
    }

    /// Only the file name of a module's path is said: its folder may hold the user's name.
    #[test]
    fn a_module_is_named_by_its_file() {
        assert_eq!(module_name(r"C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64\nvcuda64.dll"), "nvcuda64.dll");
        assert_eq!(module_name(r"C:\Users\someone\AppData\Local\Programs\Dogmind\straydog.exe"), "straydog.exe");
        assert_eq!(module_name("straydog"), "straydog");
    }
}
