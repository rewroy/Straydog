//! The Win32 calls of containment: Job Objects, the AppContainer profile, file access grants
//! and starting the worker in the AppContainer.

use std::ffi::{c_void, OsStr, OsString};
use std::io;
use std::mem::{size_of, size_of_val, zeroed};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::path::{Path, PathBuf};
use std::ptr::{addr_of, addr_of_mut, null, null_mut};

use windows_sys::Win32::Foundation::{
    LocalFree, SetHandleInformation, ERROR_ALREADY_EXISTS, HANDLE, HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Globalization::{MultiByteToWideChar, MB_ERR_INVALID_CHARS};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSidToSidW, GetNamedSecurityInfoW, SetEntriesInAclW, SetNamedSecurityInfoW,
    EXPLICIT_ACCESS_W, GRANT_ACCESS, NO_MULTIPLE_TRUSTEE, SE_FILE_OBJECT, TRUSTEE_IS_SID, TRUSTEE_IS_WELL_KNOWN_GROUP,
    TRUSTEE_W,
};
use windows_sys::Win32::Security::Isolation::{
    CreateAppContainerProfile, DeriveAppContainerSidFromAppContainerName, GetAppContainerFolderPath,
};
use windows_sys::Win32::Security::{
    AclSizeInformation, EqualSid, FreeSid, GetAce, GetAclInformation, ACCESS_ALLOWED_ACE, ACE_HEADER, ACL,
    ACL_SIZE_INFORMATION, CONTAINER_INHERIT_ACE, DACL_SECURITY_INFORMATION, INHERIT_ONLY_ACE, NO_INHERITANCE,
    OBJECT_INHERIT_ACE, PSECURITY_DESCRIPTOR, PSID, SECURITY_CAPABILITIES, SUB_CONTAINERS_AND_OBJECTS_INHERIT,
};
use windows_sys::Win32::Storage::FileSystem::{
    GetVolumeInformationW, GetVolumePathNameW, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ,
};
use windows_sys::Win32::System::Com::CoTaskMemFree;
use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE};
use windows_sys::Win32::System::Diagnostics::Debug::{SetErrorMode, SEM_FAILCRITICALERRORS, SEM_NOOPENFILEERRORBOX};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
    TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::SystemServices::ACCESS_ALLOWED_ACE_TYPE;
use windows_sys::Win32::System::Threading::{
    CreateProcessW, DeleteProcThreadAttributeList, GetExitCodeProcess, InitializeProcThreadAttributeList, ResumeThread,
    UpdateProcThreadAttribute, WaitForSingleObject, CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT,
    EXTENDED_STARTUPINFO_PRESENT, INFINITE, LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION,
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROC_THREAD_ATTRIBUTE_JOB_LIST, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
    STARTF_USESTDHANDLES, STARTUPINFOEXW,
};

use crate::appcontainer::{
    command_line, environment_block, permissionless_volume, ungranted_path, Access, Grants, PROFILE_NAME,
};
use crate::resolve;

/// ALL APPLICATION PACKAGES: an entry for it lets every AppContainer in, ours included.
const ALL_APPLICATION_PACKAGES: &str = "S-1-15-2-1";
/// The HRESULT of ERROR_ALREADY_EXISTS: the profile was created by an earlier start.
const PROFILE_EXISTS: i32 = (0x8007_0000 | ERROR_ALREADY_EXISTS) as i32;
/// WAIT_FAILED.
const WAIT_FAILED: u32 = u32::MAX;

/// A Job Object that ends every process in it when its last handle closes, or when one of them
/// has an unhandled exception (no error report dialog then waits for anyone). No process in it
/// may break away, so none outlives it.
#[derive(Debug)]
pub struct Job(OwnedHandle);

impl Job {
    pub fn new() -> io::Result<Job> {
        // SAFETY: no security attributes and no name: a new anonymous job, not inheritable.
        let handle = unsafe { CreateJobObjectW(null(), null()) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a handle just created, owned by nothing else.
        let job = Job(unsafe { OwnedHandle::from_raw_handle(handle) });
        // SAFETY: an all-zero structure is a valid value.
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { zeroed() };
        limits.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        // SAFETY: the structure the information class names, with its size.
        let set = unsafe {
            SetInformationJobObject(
                job.handle(),
                JobObjectExtendedLimitInformation,
                addr_of!(limits).cast(),
                size_of_val(&limits) as u32,
            )
        };
        if set == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(job)
    }

    fn handle(&self) -> HANDLE {
        self.0.as_raw_handle()
    }

    /// Put a running process in the job.
    pub fn assign(&self, process: RawHandle) -> io::Result<()> {
        // SAFETY: two open handles; the call only reads them.
        if unsafe { AssignProcessToJobObject(self.handle(), process) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// End every process in the job, with exit status 1.
    pub fn terminate(&self) -> io::Result<()> {
        // SAFETY: an open job handle.
        if unsafe { TerminateJobObject(self.handle(), 1) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// A SID this process allocated, freed the way its allocator requires.
pub struct Sid {
    pointer: PSID,
    /// Allocated with LocalAlloc (a converted string SID), not by the security subsystem.
    local: bool,
}

impl Drop for Sid {
    fn drop(&mut self) {
        // SAFETY: the SID was allocated as `local` says and is freed once, here.
        unsafe {
            if self.local {
                LocalFree(self.pointer);
            } else {
                FreeSid(self.pointer);
            }
        }
    }
}

/// The access control lists of files, for the launcher's grants.
pub struct Acl {
    all_application_packages: Sid,
}

impl Acl {
    pub fn new() -> io::Result<Acl> {
        let name = wide(OsStr::new(ALL_APPLICATION_PACKAGES));
        let mut sid: PSID = null_mut();
        // SAFETY: a NUL-terminated string SID; the SID returned is allocated with LocalAlloc.
        if unsafe { ConvertStringSidToSidW(name.as_ptr(), &mut sid) } == 0 || sid.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Acl { all_application_packages: Sid { pointer: sid, local: true } })
    }

    /// Whether the descriptor's DACL holds an entry that allows `access` to the profile or to ALL
    /// APPLICATION PACKAGES (by an entry the subtree inherits, when `inherit`). A NULL DACL holds
    /// none, though it lets everyone in: [`Grants::has_grant`] decides what that means to it.
    fn lists(&self, descriptor: &Descriptor, sid: &Sid, access: Access, inherit: bool) -> bool {
        if descriptor.dacl.is_null() {
            return false;
        }
        // SAFETY: an all-zero structure is a valid value.
        let mut size: ACL_SIZE_INFORMATION = unsafe { zeroed() };
        // SAFETY: the DACL lives in the descriptor; the structure the class names, with its size.
        let read = unsafe {
            GetAclInformation(descriptor.dacl, addr_of_mut!(size).cast(), size_of_val(&size) as u32, AclSizeInformation)
        };
        if read == 0 {
            return false;
        }
        let required = if inherit { OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE } else { 0 };
        let mask = mask(access);
        for index in 0..size.AceCount {
            let mut entry: *mut c_void = null_mut();
            // SAFETY: an index below the DACL's entry count; the entry points into the DACL.
            if unsafe { GetAce(descriptor.dacl, index, &mut entry) } == 0 || entry.is_null() {
                continue;
            }
            // SAFETY: every entry starts with its header.
            let header = unsafe { &*entry.cast::<ACE_HEADER>() };
            let flags = u32::from(header.AceFlags);
            // An inherit-only entry grants nothing on the path itself.
            let allowing = u32::from(header.AceType) == ACCESS_ALLOWED_ACE_TYPE;
            if !allowing || flags & INHERIT_ONLY_ACE != 0 || flags & required != required {
                continue;
            }
            // SAFETY: an access-allowed entry, whose SID starts at SidStart.
            let allowed = unsafe { &*entry.cast::<ACCESS_ALLOWED_ACE>() };
            if allowed.Mask & mask != mask {
                continue;
            }
            let trustee: PSID = addr_of!(allowed.SidStart).cast_mut().cast();
            // SAFETY: two valid SIDs.
            let ours = unsafe {
                EqualSid(sid.pointer, trustee) != 0 || EqualSid(self.all_application_packages.pointer, trustee) != 0
            };
            if ours {
                return true;
            }
        }
        false
    }

    /// The SID of the AppContainer profile `name`, derived from the name: nothing is created, and
    /// the profile need not exist.
    fn profile_sid(name: &str) -> io::Result<Sid> {
        let name = wide(OsStr::new(name));
        let mut sid: PSID = null_mut();
        // SAFETY: a NUL-terminated name; the SID returned is freed with FreeSid.
        let result = unsafe { DeriveAppContainerSidFromAppContainerName(name.as_ptr(), &mut sid) };
        if result != 0 {
            return Err(hresult_error("DeriveAppContainerSidFromAppContainerName", result));
        }
        Ok(Sid { pointer: sid, local: false })
    }

    /// Why the profile `sid` may be refused `path`, as the user sees it: its volume keeps no file
    /// permissions, or its permissions hold no entry that lets the profile read it. None when it
    /// looks right, and when its permissions cannot be read.
    fn refusal(&self, path: &Path, sid: &Sid) -> Option<String> {
        let path = resolve(path);
        let volume = Volume::of(&path);
        self.refusal_on(&path, sid, volume.as_ref())
    }

    /// [`Acl::refusal`] for a path on `volume`. The volume comes first: one that keeps no file
    /// permissions gives every path a NULL DACL, which is no sign of a grant.
    fn refusal_on(&self, path: &Path, sid: &Sid, volume: Option<&Volume>) -> Option<String> {
        if let Some(volume) = volume {
            if let Some(sentence) = permissionless_volume(&volume.root, &volume.file_system, volume.flags, path) {
                return Some(sentence);
            }
        }
        let descriptor = Descriptor::read(path).ok()?;
        (!self.lists(&descriptor, sid, Access::Read, false)).then(|| ungranted_path(path))
    }
}

impl Grants for Acl {
    type Sid = Sid;

    /// The profile persists once created: a later start derives its SID from its name.
    fn create_profile(&self, name: &str) -> io::Result<Sid> {
        let [name, display, description] =
            [name, "Straydog worker", "The contained Straydog inference worker"].map(|text| wide(OsStr::new(text)));
        let mut sid: PSID = null_mut();
        // SAFETY: NUL-terminated strings and no capabilities; the SID returned is freed with FreeSid.
        let result = unsafe {
            CreateAppContainerProfile(name.as_ptr(), display.as_ptr(), description.as_ptr(), null(), 0, &mut sid)
        };
        if result == PROFILE_EXISTS {
            // SAFETY: as above.
            let result = unsafe { DeriveAppContainerSidFromAppContainerName(name.as_ptr(), &mut sid) };
            if result != 0 {
                return Err(hresult_error("DeriveAppContainerSidFromAppContainerName", result));
            }
        } else if result != 0 {
            return Err(hresult_error("CreateAppContainerProfile", result));
        }
        Ok(Sid { pointer: sid, local: false })
    }

    fn has_grant(&self, path: &Path, sid: &Sid, access: Access, inherit: bool) -> bool {
        let Ok(descriptor) = Descriptor::read(path) else { return false };
        if descriptor.dacl.is_null() {
            return true; // a NULL DACL: everyone has full access
        }
        self.lists(&descriptor, sid, access, inherit)
    }

    fn permissionless(&self, path: &Path) -> Option<String> {
        let volume = Volume::of(path)?;
        permissionless_volume(&volume.root, &volume.file_system, volume.flags, path)
    }

    /// The existing DACL is the merge base: an ACL built from nothing would replace the path's
    /// explicit entries, and on a protected DACL lock its owner out of their own files.
    fn grant(&self, path: &Path, sid: &Sid, access: Access, inherit: bool) -> io::Result<()> {
        let descriptor = Descriptor::read(path)?;
        if descriptor.dacl.is_null() {
            // A NULL DACL: everyone, AppContainers included, already has full access, and any
            // real ACL written would restrict the owner.
            return Ok(());
        }
        let entry = EXPLICIT_ACCESS_W {
            grfAccessPermissions: mask(access),
            grfAccessMode: GRANT_ACCESS,
            grfInheritance: if inherit { SUB_CONTAINERS_AND_OBJECTS_INHERIT } else { NO_INHERITANCE },
            Trustee: TRUSTEE_W {
                pMultipleTrustee: null_mut(),
                MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
                TrusteeForm: TRUSTEE_IS_SID,
                TrusteeType: TRUSTEE_IS_WELL_KNOWN_GROUP,
                ptstrName: sid.pointer.cast(),
            },
        };
        let mut merged: *mut ACL = null_mut();
        // SAFETY: one entry and the path's DACL; the merged ACL is allocated with LocalAlloc.
        let error = unsafe { SetEntriesInAclW(1, &entry, descriptor.dacl, &mut merged) };
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error as i32));
        }
        let name = wide(path.as_os_str());
        // SAFETY: a NUL-terminated name and the merged ACL, freed right after.
        let error = unsafe {
            let error = SetNamedSecurityInfoW(
                name.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                merged,
                null(),
            );
            LocalFree(merged.cast());
            error
        };
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error as i32));
        }
        Ok(())
    }
}

/// Why the worker may have been refused the first of `paths`; see [`crate::why_refused`].
pub fn why_refused(paths: &[PathBuf]) -> Option<String> {
    why_refused_as(PROFILE_NAME, paths)
}

fn why_refused_as(profile: &str, paths: &[PathBuf]) -> Option<String> {
    let acl = Acl::new().ok()?;
    let sid = Acl::profile_sid(profile).ok()?;
    paths.iter().find_map(|path| acl.refusal(path, &sid))
}

/// The volume a path is on.
struct Volume {
    /// Its root: a drive (`D:\`), or the mount point of a volume or a share.
    root: String,
    /// The name of its file system (NTFS, exFAT, FAT32, ReFS...).
    file_system: String,
    /// Its file system's flags (FILE_PERSISTENT_ACLS among them).
    flags: u32,
}

impl Volume {
    /// The volume holding `path` (which need not exist), or None when Windows will not say.
    fn of(path: &Path) -> Option<Volume> {
        let name = wide(path.as_os_str());
        let mut root = vec![0u16; 1024];
        // SAFETY: a NUL-terminated name and a buffer of the length given.
        if unsafe { GetVolumePathNameW(name.as_ptr(), root.as_mut_ptr(), root.len() as u32) } == 0 {
            return None;
        }
        let mut file_system = vec![0u16; 261];
        let mut flags = 0;
        // SAFETY: the NUL-terminated root just written, and a buffer of the length given for the file system's name.
        let read = unsafe {
            GetVolumeInformationW(
                root.as_ptr(),
                null_mut(),
                0,
                null_mut(),
                null_mut(),
                &mut flags,
                file_system.as_mut_ptr(),
                file_system.len() as u32,
            )
        };
        if read == 0 {
            return None;
        }
        let text = |units: &[u16]| {
            String::from_utf16_lossy(&units[..units.iter().position(|&unit| unit == 0).unwrap_or(units.len())])
        };
        Some(Volume { root: text(&root), file_system: text(&file_system), flags })
    }
}

fn mask(access: Access) -> u32 {
    match access {
        Access::Read => FILE_GENERIC_READ,
        Access::ReadExecute => FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
    }
}

/// A file's security descriptor and the DACL inside it (null for a NULL DACL).
struct Descriptor {
    descriptor: PSECURITY_DESCRIPTOR,
    dacl: *mut ACL,
}

impl Descriptor {
    fn read(path: &Path) -> io::Result<Descriptor> {
        let name = wide(path.as_os_str());
        let (mut dacl, mut descriptor) = (null_mut(), null_mut());
        // SAFETY: a NUL-terminated name; the descriptor is allocated with LocalAlloc and holds the DACL.
        let error = unsafe {
            GetNamedSecurityInfoW(
                name.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                &mut dacl,
                null_mut(),
                &mut descriptor,
            )
        };
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error as i32));
        }
        Ok(Descriptor { descriptor, dacl })
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        // SAFETY: allocated by GetNamedSecurityInfoW, freed once.
        unsafe { LocalFree(self.descriptor) };
    }
}

/// The AppContainer's own folder (its private TEMP lives there), when Windows reports it.
fn profile_folder(sid: &Sid) -> Option<PathBuf> {
    let mut text = null_mut();
    // SAFETY: a valid SID; the string is allocated with LocalAlloc.
    if unsafe { ConvertSidToStringSidW(sid.pointer, &mut text) } == 0 {
        return None;
    }
    let mut folder = null_mut();
    // SAFETY: a NUL-terminated string SID, freed right after; the path is allocated with CoTaskMemAlloc.
    let result = unsafe {
        let result = GetAppContainerFolderPath(text, &mut folder);
        LocalFree(text.cast());
        result
    };
    if result != 0 || folder.is_null() {
        return None;
    }
    // SAFETY: a NUL-terminated string from the call above, freed once after it is copied.
    unsafe {
        let length = (0..).take_while(|&index| *folder.add(index) != 0).count();
        let path = PathBuf::from(OsString::from_wide(std::slice::from_raw_parts(folder, length)));
        CoTaskMemFree(folder.cast());
        Some(path)
    }
}

/// An initialized process and thread attribute list. The values its attributes point to must
/// outlive the process creation that uses it.
struct AttributeList(Vec<usize>);

impl AttributeList {
    fn new(count: u32) -> io::Result<AttributeList> {
        let mut size = 0;
        // SAFETY: a null list only asks for the size it needs (and fails, as expected).
        unsafe { InitializeProcThreadAttributeList(null_mut(), count, 0, &mut size) };
        if size == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut list = AttributeList(vec![0; size.div_ceil(size_of::<usize>())]);
        // SAFETY: a buffer of at least `size` bytes, aligned for the pointers it holds.
        if unsafe { InitializeProcThreadAttributeList(list.pointer(), count, 0, &mut size) } == 0 {
            // Not initialized: nothing to delete.
            let error = io::Error::last_os_error();
            std::mem::forget(list);
            return Err(error);
        }
        Ok(list)
    }

    fn pointer(&mut self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.0.as_mut_ptr().cast()
    }

    /// Set `attribute` to the `size` bytes at `value`.
    ///
    /// # Safety
    /// `value` must stay valid, unmoved, until the process is created.
    unsafe fn update(&mut self, attribute: u32, value: *const c_void, size: usize) -> io::Result<()> {
        // SAFETY: an initialized list; the caller keeps `value` alive.
        if unsafe { UpdateProcThreadAttribute(self.pointer(), 0, attribute as usize, value, size, null_mut(), null()) }
            == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Drop for AttributeList {
    fn drop(&mut self) {
        // SAFETY: an initialized list, deleted once.
        unsafe { DeleteProcThreadAttributeList(self.pointer()) };
    }
}

/// Run `command` in the AppContainer with this process's standard handles and no other
/// inheritable handle, in a job that ends it with this process; return its exit status.
///
/// No capability is given: no network, loopback included. The environment holds only named
/// variables (`appcontainer::environment_block`); TEMP and TMP name the AppContainer's own
/// folder, the one place it may write.
pub fn launch(command: &[OsString], sid: &Sid, working_directory: Option<&Path>) -> io::Result<u32> {
    // SAFETY: plain values.
    let standard = [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE].map(|which| unsafe { GetStdHandle(which) });
    if standard.iter().any(|&handle| handle.is_null() || handle == INVALID_HANDLE_VALUE) {
        return Err(io::Error::other("the Straydog launcher needs standard input, output and error handles"));
    }
    let mut inherited: Vec<HANDLE> = Vec::new();
    for handle in standard {
        if !inherited.contains(&handle) {
            inherited.push(handle);
        }
    }
    for &handle in &inherited {
        // SAFETY: a handle of this process.
        if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let units = |text: &OsStr| text.encode_wide().collect::<Vec<u16>>();
    // Only the variables the system, the loader and the CUDA runtime read pass from this
    // environment, which holds whatever the server's had.
    let variables = std::env::vars_os().map(|(name, value)| (units(&name), units(&value))).collect();
    let temp = profile_folder(sid).map(|folder| units(folder.join("Temp").as_os_str()));
    let environment = environment_block(variables, temp);
    let job = Job::new()?; // kill on close: the worker never outlives the launcher

    let mut attributes = AttributeList::new(3)?;
    let capabilities = SECURITY_CAPABILITIES {
        AppContainerSid: sid.pointer,
        Capabilities: null_mut(),
        CapabilityCount: 0, // no internetClient, nor any other capability
        Reserved: 0,
    };
    let jobs = [job.handle()];
    // SAFETY: each value lives until the end of this function, past the process creation.
    unsafe {
        let security = addr_of!(capabilities).cast();
        attributes.update(PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, security, size_of_val(&capabilities))?;
        attributes.update(PROC_THREAD_ATTRIBUTE_JOB_LIST, jobs.as_ptr().cast(), size_of_val(&jobs))?;
        attributes.update(
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            inherited.as_ptr().cast(),
            size_of_val(inherited.as_slice()),
        )?;
    }
    // SAFETY: an all-zero structure is a valid value.
    let mut startup: STARTUPINFOEXW = unsafe { zeroed() };
    startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    [startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError] = standard;
    startup.lpAttributeList = attributes.pointer();

    let program = command.first().ok_or_else(|| io::Error::other("no worker command"))?;
    let application = wide(program);
    let mut line = command_line(&command.iter().map(|arg| units(arg)).collect::<Vec<_>>());
    line.push(0);
    let directory = working_directory.map(|path| wide(path.as_os_str()));
    // A missing or broken DLL then fails the worker instead of opening a dialog nobody can
    // dismiss; the worker inherits this error mode.
    // SAFETY: plain values.
    unsafe { SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX) };
    // SAFETY: an all-zero structure is a valid value.
    let mut information: PROCESS_INFORMATION = unsafe { zeroed() };
    let flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW;
    // SAFETY: NUL-terminated strings, a writable command line, a Unicode environment block and
    // an extended startup structure whose attribute list and values are alive.
    let created = unsafe {
        CreateProcessW(
            application.as_ptr(),
            line.as_mut_ptr(),
            null(),
            null(),
            1,
            flags,
            environment.as_ptr().cast(),
            directory.as_ref().map_or(null(), |directory| directory.as_ptr()),
            addr_of!(startup).cast(),
            &mut information,
        )
    };
    if created == 0 {
        let error = io::Error::last_os_error();
        return Err(io::Error::new(
            error.kind(),
            format!("CreateProcessW for the contained Straydog worker failed: {error}"),
        ));
    }
    // SAFETY: handles just created for this process, owned by nothing else.
    let (process, thread) = unsafe {
        (OwnedHandle::from_raw_handle(information.hProcess), OwnedHandle::from_raw_handle(information.hThread))
    };
    // SAFETY: open handles. A worker left suspended on failure ends with the job, when it is dropped.
    unsafe {
        if ResumeThread(thread.as_raw_handle()) == u32::MAX
            || WaitForSingleObject(process.as_raw_handle(), INFINITE) == WAIT_FAILED
        {
            return Err(io::Error::last_os_error());
        }
        let mut status = 0;
        if GetExitCodeProcess(process.as_raw_handle(), &mut status) == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(status)
    }
}

/// `data` decoded from a Windows code page, or None when it is not valid text there.
pub fn decode(data: &[u8], code_page: u32) -> Option<String> {
    let length = i32::try_from(data.len()).ok()?;
    // SAFETY: the input with its length; no output asks for the length needed.
    let needed = unsafe { MultiByteToWideChar(code_page, MB_ERR_INVALID_CHARS, data.as_ptr(), length, null_mut(), 0) };
    let mut text = vec![0u16; usize::try_from(needed).ok().filter(|&needed| needed > 0)?];
    // SAFETY: an output buffer of the length just reported.
    let written = unsafe {
        MultiByteToWideChar(code_page, MB_ERR_INVALID_CHARS, data.as_ptr(), length, text.as_mut_ptr(), needed)
    };
    text.truncate(usize::try_from(written).ok().filter(|&written| written > 0)?);
    String::from_utf16(&text).ok()
}

/// A NUL-terminated UTF-16 string.
fn wide(text: &OsStr) -> Vec<u16> {
    text.encode_wide().chain([0]).collect()
}

/// A failed HRESULT, with the system's message for a Win32 error code.
fn hresult_error(call: &str, result: i32) -> io::Error {
    let code = result as u32;
    if code >> 16 == 0x8007 {
        let error = io::Error::from_raw_os_error((code & 0xFFFF) as i32);
        return io::Error::new(error.kind(), format!("{call} failed ({code:#010x}): {error}"));
    }
    io::Error::other(format!("{call} failed ({code:#010x})"))
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::Security::Isolation::DeleteAppContainerProfile;
    use windows_sys::Win32::System::SystemServices::FILE_PERSISTENT_ACLS;

    use super::*;
    use crate::tests::Scratch;

    /// A profile of the tests' own, deleted afterwards (its grants stay on the deleted test files only).
    struct Profile(&'static str);

    impl Drop for Profile {
        fn drop(&mut self) {
            let name = wide(OsStr::new(self.0));
            // SAFETY: a NUL-terminated name.
            unsafe { DeleteAppContainerProfile(name.as_ptr()) };
        }
    }

    fn string_sid(sid: &Sid) -> String {
        let mut text = null_mut();
        // SAFETY: a valid SID; the string is allocated with LocalAlloc and freed once after it is copied.
        unsafe {
            assert_ne!(ConvertSidToStringSidW(sid.pointer, &mut text), 0);
            let length = (0..).take_while(|&index| *text.add(index) != 0).count();
            let value = String::from_utf16(std::slice::from_raw_parts(text, length)).unwrap();
            LocalFree(text.cast());
            value
        }
    }

    #[test]
    fn the_profile_is_created_once_then_found_by_name() {
        // Every start after the first finds the persistent profile (ERROR_ALREADY_EXISTS) and must still get its SID.
        let profile = Profile("Straydog.Worker.Test.Profile");
        let acl = Acl::new().unwrap();
        let first = acl.create_profile(profile.0).unwrap();
        let again = acl.create_profile(profile.0).unwrap();
        assert_eq!(string_sid(&first), string_sid(&again));
        assert!(string_sid(&first).starts_with("S-1-15-2-"));
        assert!(profile_folder(&first).is_some());
    }

    #[test]
    fn a_grant_is_found_once_written_and_inherited_by_new_files() {
        let profile = Profile("Straydog.Worker.Test.Grants");
        let scratch = Scratch::new("straydog-acl");
        let file = scratch.file("model/a.gguf", "a");
        let directory = scratch.0.join("bin");
        std::fs::create_dir_all(&directory).unwrap();
        let acl = Acl::new().unwrap();
        let sid = acl.create_profile(profile.0).unwrap();
        assert!(!acl.has_grant(&file, &sid, Access::Read, false));
        acl.grant(&file, &sid, Access::Read, false).unwrap();
        assert!(acl.has_grant(&file, &sid, Access::Read, false));
        assert!(!acl.has_grant(&file, &sid, Access::ReadExecute, false));
        acl.grant(&directory, &sid, Access::ReadExecute, true).unwrap();
        assert!(acl.has_grant(&directory, &sid, Access::ReadExecute, true));
        let created = scratch.file("bin/cublas64_12.dll", "MZ");
        assert!(acl.has_grant(&created, &sid, Access::ReadExecute, false));
    }

    #[test]
    fn a_path_without_the_profiles_entry_is_named_and_one_with_it_is_not() {
        let profile = Profile("Straydog.Worker.Test.Refusal");
        let scratch = Scratch::new("straydog-refusal");
        let denied = scratch.file("model/denied.gguf", "a");
        let granted = scratch.file("model/granted.gguf", "b");
        let shared = scratch.file("model/shared.gguf", "c");
        let directory = scratch.0.join("bin");
        std::fs::create_dir_all(&directory).unwrap();
        let acl = Acl::new().unwrap();
        let sid = acl.create_profile(profile.0).unwrap();
        acl.grant(&granted, &sid, Access::Read, false).unwrap();
        acl.grant(&directory, &sid, Access::ReadExecute, false).unwrap();
        // ALL APPLICATION PACKAGES lets the profile in as well.
        acl.grant(&shared, &acl.all_application_packages, Access::Read, false).unwrap();

        // (the path as resolved: with the platform's separators)
        let sentence = |path: &PathBuf| {
            format!(
                "Straydog's worker has no read permission on {} (no Straydog.Worker entry in its permissions)",
                resolve(path).display()
            )
        };
        assert_eq!(why_refused_as(profile.0, std::slice::from_ref(&denied)), Some(sentence(&denied)));
        // The first path without an entry, in the order given.
        let other = scratch.file("model/other.gguf", "d");
        assert_eq!(why_refused_as(profile.0, &[granted.clone(), other.clone(), denied.clone()]), Some(sentence(&other)));
        assert_eq!(why_refused_as(profile.0, &[denied.clone(), other.clone()]), Some(sentence(&denied)));
        // After a grant, nothing is named; nor is a directory with one, a path ALL APPLICATION
        // PACKAGES may read, or a path that is gone (its permissions cannot be read).
        assert_eq!(why_refused_as(profile.0, &[granted.clone(), directory, shared]), None);
        assert_eq!(why_refused_as(profile.0, &[scratch.0.join("model/missing.gguf")]), None);
        acl.grant(&denied, &sid, Access::Read, false).unwrap();
        assert_eq!(why_refused_as(profile.0, &[denied]), None);
        // The profile's SID is derived from its name: another name has no entry on any of them.
        assert!(why_refused_as("Straydog.Worker.Test.Someone.Else", &[granted]).is_some());
    }

    #[test]
    fn a_null_dacl_is_a_grant_to_the_launcher_but_no_entry_to_the_hint() {
        let profile = Profile("Straydog.Worker.Test.NullDacl");
        let scratch = Scratch::new("straydog-null-dacl");
        let file = scratch.file("model/a.gguf", "a");
        let name = wide(file.as_os_str());
        // SAFETY: a NUL-terminated name; no DACL given with DACL_SECURITY_INFORMATION sets a NULL one.
        let error = unsafe {
            SetNamedSecurityInfoW(
                name.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                null(),
                null(),
            )
        };
        assert_eq!(error, 0);
        assert!(Descriptor::read(&file).unwrap().dacl.is_null());
        let acl = Acl::new().unwrap();
        let sid = acl.create_profile(profile.0).unwrap();
        // What the launcher relies on is unchanged, and it writes nothing there.
        assert!(acl.has_grant(&file, &sid, Access::Read, false));
        acl.grant(&file, &sid, Access::Read, false).unwrap();
        assert!(Descriptor::read(&file).unwrap().dacl.is_null());
        // A NULL DACL says nothing about an AppContainer: it is named.
        let named = why_refused_as(profile.0, std::slice::from_ref(&file)).unwrap();
        assert!(named.contains("no Straydog.Worker entry") && named.contains(&resolve(&file).display().to_string()), "{named}");
    }

    #[test]
    fn a_drive_without_file_permissions_is_named_before_the_permissions_of_its_paths() {
        let profile = Profile("Straydog.Worker.Test.Drive");
        let scratch = Scratch::new("straydog-drive");
        let granted = scratch.file("model/granted.gguf", "a");
        let denied = scratch.file("model/denied.gguf", "b");
        let acl = Acl::new().unwrap();
        let sid = acl.create_profile(profile.0).unwrap();
        acl.grant(&granted, &sid, Access::Read, false).unwrap();
        let drive = |file_system: &str, flags| Volume { root: r"D:\".into(), file_system: file_system.into(), flags };
        let (exfat, ntfs) = (drive("exFAT", 0x7), drive("NTFS", 0x7 | FILE_PERSISTENT_ACLS));
        // On exFAT every path reads as a NULL DACL (or a granted one): the drive is what is named, whatever the path.
        for path in [&granted, &denied] {
            let told = acl.refusal_on(path, &sid, Some(&exfat)).unwrap();
            let start = r"D:\ is formatted exFAT, which keeps no file permissions, so Straydog's contained worker may not read ";
            assert!(told.starts_with(start) && told.ends_with(" there; put it on an NTFS drive"), "{told}");
        }
        // On a drive that keeps them, the path's own entry decides; and where Windows will not say, so does it.
        assert_eq!(acl.refusal_on(&granted, &sid, Some(&ntfs)), None);
        assert_eq!(acl.refusal_on(&granted, &sid, None), None);
        for volume in [Some(&ntfs), None] {
            assert!(acl.refusal_on(&denied, &sid, volume).unwrap().contains("no Straydog.Worker entry"));
        }
    }

    #[test]
    fn the_drive_of_the_scratch_files_keeps_file_permissions() {
        assert_eq!(FILE_PERSISTENT_ACLS, crate::appcontainer::PERSISTENT_ACLS);
        let scratch = Scratch::new("straydog-volume");
        let file = scratch.file("a.gguf", "a");
        let volume = Volume::of(&file).unwrap();
        assert_ne!(volume.flags & FILE_PERSISTENT_ACLS, 0, "{} is {}", volume.root, volume.file_system);
        assert!(!volume.file_system.is_empty() && volume.root.ends_with('\\'), "{:?}", (&volume.root, &volume.file_system));
        assert_eq!(Acl::new().unwrap().permissionless(&file), None);
        // The path need not exist.
        assert_eq!(Volume::of(&scratch.0.join("not/yet/there")).unwrap().root, volume.root);
        // Were its flags without the permissions, the sentence would name its root and file system.
        let told = permissionless_volume(&volume.root, &volume.file_system, 0, &file).unwrap();
        assert!(told.starts_with(&format!("{} is formatted {}, which keeps no file permissions", volume.root, volume.file_system)));
    }

    #[test]
    fn a_job_ends_its_processes_when_terminated() {
        let mut child = std::process::Command::new("ping")
            .args(["-n", "30", "127.0.0.1"])
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let job = Job::new().unwrap();
        job.assign(child.as_raw_handle()).unwrap();
        job.terminate().unwrap();
        assert_eq!(child.wait().unwrap().code(), Some(1));
    }
}
