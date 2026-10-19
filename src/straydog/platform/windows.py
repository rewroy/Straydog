"""Windows primitives for the contained worker: job objects, AppContainer process
creation, the AppContainer profile and file access grants.

Windows only: importing it elsewhere fails at the first Win32 call, not at import.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from ctypes import wintypes

# ---- Job objects
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


_JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400


_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800


_JobObjectExtendedLimitInformation = 9


def _kernel32():
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)


def _create_job_object_for_child() -> int:
    """Create a Job Object configured for tree-kill on close. Returns
    the HANDLE as an int (suitable for ``ctypes`` use)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        | _JOB_OBJECT_LIMIT_BREAKAWAY_OK
    )

    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    ok = kernel32.SetInformationJobObject(
        job,
        _JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        err = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(err, "SetInformationJobObject failed")
    return int(job)


def _assign_to_job(job_handle: int, process_handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    ok = kernel32.AssignProcessToJobObject(
        wintypes.HANDLE(job_handle), wintypes.HANDLE(process_handle)
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")


def _close_handle(handle: int) -> None:
    from ctypes import wintypes

    kernel32 = _kernel32()
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(handle))


# ---- Process creation in an AppContainer
CREATE_SUSPENDED = 0x00000004


EXTENDED_STARTUPINFO_PRESENT = 0x00080000


CREATE_UNICODE_ENVIRONMENT = 0x00000400


CREATE_NO_WINDOW = 0x08000000


STARTF_USESTDHANDLES = 0x00000100


INFINITE = 0xFFFFFFFF


WAIT_OBJECT_0 = 0x00000000


HANDLE_FLAG_INHERIT = 0x00000001


PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009


PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", STARTUPINFOW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", wintypes.DWORD),
    ]


class SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.POINTER(SID_AND_ATTRIBUTES)),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


def _alloc_attribute_list(attribute_count: int) -> tuple[ctypes.c_void_p, int]:
    """Allocate and initialize a ProcThreadAttributeList for the given
    attribute count. Returns (pointer, size). Caller must call
    ``DeleteProcThreadAttributeList`` and free the memory."""
    kernel32 = _kernel32()

    kernel32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL

    size = ctypes.c_size_t(0)
    # First call to learn required size — expected to fail with ERROR_INSUFFICIENT_BUFFER.
    kernel32.InitializeProcThreadAttributeList(None, attribute_count, 0, ctypes.byref(size))

    buffer = ctypes.create_string_buffer(size.value)
    ok = kernel32.InitializeProcThreadAttributeList(
        ctypes.cast(buffer, ctypes.c_void_p),
        attribute_count,
        0,
        ctypes.byref(size),
    )
    if not ok:
        raise OSError(
            ctypes.get_last_error(),
            "InitializeProcThreadAttributeList failed",
        )
    return ctypes.cast(buffer, ctypes.c_void_p), buffer  # keep buffer alive


def _update_attribute(
    attr_list: ctypes.c_void_p,
    attribute_id: int,
    value_ptr: ctypes.c_void_p,
    value_size: int,
) -> None:
    kernel32 = _kernel32()
    kernel32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
    ok = kernel32.UpdateProcThreadAttribute(
        attr_list,
        0,  # dwFlags reserved
        attribute_id,
        value_ptr,
        value_size,
        None,
        None,
    )
    if not ok:
        raise OSError(
            ctypes.get_last_error(),
            f"UpdateProcThreadAttribute({attribute_id:#x}) failed",
        )


def _delete_attribute_list(attr_list: ctypes.c_void_p) -> None:
    kernel32 = _kernel32()
    kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    kernel32.DeleteProcThreadAttributeList.restype = None
    kernel32.DeleteProcThreadAttributeList(attr_list)


def _make_env_block(env: dict[str, str]) -> ctypes.Array[ctypes.c_wchar]:
    """Encode env as a double-NUL-terminated UTF-16 block."""
    parts = [f"{k}={v}" for k, v in env.items()]
    body = "\0".join(parts) + "\0\0"
    return ctypes.create_unicode_buffer(body)


def _resume_thread(thread_handle: int) -> None:
    kernel32 = _kernel32()
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.ResumeThread(wintypes.HANDLE(thread_handle))


def _wait_for(process_handle: int, timeout_ms: int) -> bool:
    """True if the process exited; False if the wait timed out."""
    kernel32 = _kernel32()
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    rc = kernel32.WaitForSingleObject(wintypes.HANDLE(process_handle), timeout_ms)
    return rc == WAIT_OBJECT_0


def _get_exit_code(process_handle: int) -> int:
    kernel32 = _kernel32()
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    rc = wintypes.DWORD(0)
    kernel32.GetExitCodeProcess(wintypes.HANDLE(process_handle), ctypes.byref(rc))
    return int(rc.value)


# ---- AppContainer profile and file grants
logger = logging.getLogger(__name__)


SE_FILE_OBJECT = 1


DACL_SECURITY_INFORMATION = 0x00000004


GRANT_ACCESS = 1


SUB_CONTAINERS_AND_OBJECTS_INHERIT = 3


NO_INHERITANCE = 0


TRUSTEE_IS_SID = 0


TRUSTEE_IS_WELL_KNOWN_GROUP = 5


FILE_GENERIC_READ = 0x00120089


FILE_GENERIC_EXECUTE = 0x001200A0


_RX = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE


ACCESS_ALLOWED_ACE_TYPE = 0


OBJECT_INHERIT_ACE = 0x01


CONTAINER_INHERIT_ACE = 0x02


INHERIT_ONLY_ACE = 0x08


AclSizeInformation = 2


_ALL_APPLICATION_PACKAGES_SID = "S-1-15-2-1"


class TRUSTEEW(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", wintypes.DWORD),
        ("TrusteeForm", wintypes.DWORD),
        ("TrusteeType", wintypes.DWORD),
        ("ptstrName", ctypes.c_void_p),
    ]


class EXPLICIT_ACCESSW(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", wintypes.DWORD),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", TRUSTEEW),
    ]


class ACE_HEADER(ctypes.Structure):
    _fields_ = [
        ("AceType", ctypes.c_ubyte),
        ("AceFlags", ctypes.c_ubyte),
        ("AceSize", wintypes.WORD),
    ]


class ACCESS_ALLOWED_ACE(ctypes.Structure):
    _fields_ = [
        ("Header", ACE_HEADER),
        ("Mask", wintypes.DWORD),
        ("SidStart", wintypes.DWORD),
    ]


class ACL_SIZE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("AceCount", wintypes.DWORD),
        ("AclBytesInUse", wintypes.DWORD),
        ("AclBytesFree", wintypes.DWORD),
    ]


def _userenv():
    return ctypes.WinDLL("userenv", use_last_error=True)


def _advapi32():
    return ctypes.WinDLL("advapi32", use_last_error=True)


_profile_lock = threading.Lock()


_granted_paths: dict[tuple[str, int, bool], int] = {}


_all_app_packages_sid: int | None = None


def _ensure_path_grant(path: str, sid_ptr: int, access_mask: int, *, inherit: bool = True) -> None:
    """Grant ``access_mask`` on ``path`` to the profile SID, once per
    process per path.

    Grants persist on disk, so a matching ACE left by an earlier start (or
    an ALL APPLICATION PACKAGES ACE) is detected and the grant skipped: the
    full-tree ACL propagation is the expensive part and would otherwise be
    re-paid for the Python runtime on every server start."""
    # A grant cached for one profile cannot authorize a different profile,
    # nor can a non-inheriting ACE satisfy a subtree grant.
    with _profile_lock:
        if _granted_paths.get((path, sid_ptr, inherit), 0) & access_mask == access_mask:
            return
    canonical = os.path.realpath(path)
    key = (canonical, sid_ptr, inherit)
    with _profile_lock:
        granted_mask = _granted_paths.get(key, 0)
        if granted_mask & access_mask == access_mask:
            return
    combined_mask = granted_mask | access_mask
    if not _path_has_grant(canonical, sid_ptr, combined_mask, inherit=inherit):
        started = time.monotonic()
        _grant_path_access(canonical, sid_ptr, combined_mask, inherit=inherit)
        elapsed = time.monotonic() - started
        if elapsed >= 1.0:
            logger.info("Granted the Straydog worker access to %s in %.1fs (one-time)", canonical, elapsed)
    with _profile_lock:
        _granted_paths[key] = combined_mask


_HRESULT_ALREADY_EXISTS = ctypes.c_int32(0x800700B7).value


def _hresult_hex(hr: int) -> str:
    return f"0x{hr & 0xFFFFFFFF:08x}"


def _create_app_container_profile(name: str) -> int:
    """Create the profile (or reuse the existing one) and return the
    AppContainer SID pointer.

    Caller owns the SID and must FreeSid() it (via ``LocalFree`` per
    Microsoft docs — ``DeriveAppContainerSidFromAppContainerName``
    allocates via the COM allocator equivalent).

    The userenv calls use a plain ``c_int32`` restype, not
    ``ctypes.HRESULT``: the latter raises ``OSError`` on every failure
    code, which made the ERROR_ALREADY_EXISTS branch unreachable — the
    persistent profile then failed to load in every process after the
    one that first created it, and the sandbox probe failed for good.
    """
    userenv = _userenv()
    userenv.CreateAppContainerProfile.argtypes = [
        wintypes.LPCWSTR,  # pszAppContainerName
        wintypes.LPCWSTR,  # pszDisplayName
        wintypes.LPCWSTR,  # pszDescription
        ctypes.c_void_p,  # pCapabilities (SID_AND_ATTRIBUTES*)
        wintypes.DWORD,  # dwCapabilityCount
        ctypes.POINTER(ctypes.c_void_p),  # ppSidAppContainerSid
    ]
    userenv.CreateAppContainerProfile.restype = ctypes.c_int32

    sid_out = ctypes.c_void_p(0)
    hr = userenv.CreateAppContainerProfile(
        name,
        "Straydog worker",
        "The contained Straydog inference worker",
        None,
        0,
        ctypes.byref(sid_out),
    )
    if hr != 0:
        # Profile may already exist (ERROR_ALREADY_EXISTS = 0x800700B7).
        # Derive the SID directly in that case.
        if hr == _HRESULT_ALREADY_EXISTS:
            userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
                wintypes.LPCWSTR,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_int32
            hr2 = userenv.DeriveAppContainerSidFromAppContainerName(name, ctypes.byref(sid_out))
            if hr2 != 0:
                raise OSError(
                    hr2,
                    f"DeriveAppContainerSidFromAppContainerName failed ({_hresult_hex(hr2)})",
                )
        else:
            raise OSError(hr, f"CreateAppContainerProfile failed ({_hresult_hex(hr)})")
    return sid_out.value


def _string_sid(value: str) -> int:
    """SID pointer for a string SID; caller frees it with ``_free_sid``."""
    advapi = _advapi32()
    advapi.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi.ConvertStringSidToSidW.restype = wintypes.BOOL
    sid = ctypes.c_void_p(0)
    if not advapi.ConvertStringSidToSidW(value, ctypes.byref(sid)) or not sid.value:
        raise OSError(ctypes.get_last_error(), f"ConvertStringSidToSidW({value}) failed")
    return sid.value


def _all_application_packages_sid() -> int:
    global _all_app_packages_sid
    with _profile_lock:
        if _all_app_packages_sid is None:
            _all_app_packages_sid = _string_sid(_ALL_APPLICATION_PACKAGES_SID)
        return _all_app_packages_sid


def _path_has_grant(
    path: str,
    sid_ptr: int,
    access_mask: int,
    *,
    inherit: bool,
    include_app_packages: bool = True,
) -> bool:
    """True when ``path``'s DACL already allows ``access_mask`` to the
    sandbox SID or ALL APPLICATION PACKAGES (inheritable for directories).

    Any failure answers False, so the caller writes the grant instead
    (which merges into the existing DACL, so a redundant ACE is harmless)."""
    try:
        advapi = _advapi32()
        advapi.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.c_int]
        advapi.GetAclInformation.restype = wintypes.BOOL
        advapi.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
        advapi.GetAce.restype = wintypes.BOOL
        advapi.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        advapi.EqualSid.restype = wintypes.BOOL

        trustees = [sid_ptr, _all_application_packages_sid()] if include_app_packages else [sid_ptr]
        dacl, descriptor = _read_dacl(path)
        try:
            if not dacl.value:
                return True  # NULL DACL: everyone has full access
            info = ACL_SIZE_INFORMATION()
            if not advapi.GetAclInformation(dacl, ctypes.byref(info), ctypes.sizeof(info), AclSizeInformation):
                return False
            required_flags = OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE if inherit else 0
            sid_offset = ACCESS_ALLOWED_ACE.SidStart.offset
            for index in range(info.AceCount):
                ace_ptr = ctypes.c_void_p(0)
                if not advapi.GetAce(dacl, index, ctypes.byref(ace_ptr)) or not ace_ptr.value:
                    continue
                ace = ctypes.cast(ace_ptr, ctypes.POINTER(ACCESS_ALLOWED_ACE)).contents
                flags = ace.Header.AceFlags
                if ace.Header.AceType != ACCESS_ALLOWED_ACE_TYPE or flags & INHERIT_ONLY_ACE:
                    continue
                if flags & required_flags != required_flags or ace.Mask & access_mask != access_mask:
                    continue
                ace_sid = ace_ptr.value + sid_offset
                if any(advapi.EqualSid(trustee, ace_sid) for trustee in trustees):
                    return True
            return False
        finally:
            if descriptor.value:
                _kernel32().LocalFree(descriptor)
    except OSError:
        logger.debug("ACL inspection failed for %s", path, exc_info=True)
        return False


def _read_dacl(path: str) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
    """(dacl, security_descriptor) of ``path``; free the descriptor with
    ``LocalFree``. The DACL pointer lives inside the descriptor and may be
    NULL (a NULL DACL grants everyone full access)."""
    advapi = _advapi32()
    advapi.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,  # pObjectName
        wintypes.DWORD,  # ObjectType
        wintypes.DWORD,  # SecurityInfo
        ctypes.POINTER(ctypes.c_void_p),  # ppsidOwner
        ctypes.POINTER(ctypes.c_void_p),  # ppsidGroup
        ctypes.POINTER(ctypes.c_void_p),  # ppDacl
        ctypes.POINTER(ctypes.c_void_p),  # ppSacl
        ctypes.POINTER(ctypes.c_void_p),  # ppSecurityDescriptor
    ]
    advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    dacl = ctypes.c_void_p(0)
    descriptor = ctypes.c_void_p(0)
    err = advapi.GetNamedSecurityInfoW(
        path,
        SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if err != 0:
        raise OSError(err, f"GetNamedSecurityInfoW({path}) failed")
    return dacl, descriptor


def _grant_path_access(
    path: str,
    sid_ptr: int,
    access_mask: int,
    *,
    inherit: bool = True,
    access_mode: int = GRANT_ACCESS,
) -> None:
    """Merge an ALLOW ACE granting ``access_mask`` to the AppContainer SID
    into ``path``'s existing DACL (``REVOKE_ACCESS`` removes the SID's
    ACEs instead). Inheritable for directories so subdirs/files inherit it;
    a file grant covers just that file.

    The existing DACL is the merge base: building the new ACL from NULL
    would replace the folder's explicit ACEs — on a protected DACL that
    locks the owner out of their own files.
    """
    advapi = _advapi32()
    advapi.SetEntriesInAclW.argtypes = [
        wintypes.ULONG,  # cCountOfExplicitEntries
        ctypes.POINTER(EXPLICIT_ACCESSW),  # pListOfExplicitEntries
        ctypes.c_void_p,  # OldAcl
        ctypes.POINTER(ctypes.c_void_p),  # NewAcl
    ]
    advapi.SetEntriesInAclW.restype = wintypes.DWORD

    advapi.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,  # pObjectName
        wintypes.DWORD,  # ObjectType
        wintypes.DWORD,  # SecurityInfo
        ctypes.c_void_p,  # psidOwner
        ctypes.c_void_p,  # psidGroup
        ctypes.c_void_p,  # pDacl
        ctypes.c_void_p,  # pSacl
    ]
    advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD

    ea = EXPLICIT_ACCESSW()
    ea.grfAccessPermissions = access_mask
    ea.grfAccessMode = access_mode
    ea.grfInheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT if inherit else NO_INHERITANCE
    ea.Trustee.pMultipleTrustee = None
    ea.Trustee.MultipleTrusteeOperation = 0
    ea.Trustee.TrusteeForm = TRUSTEE_IS_SID
    ea.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP
    ea.Trustee.ptstrName = sid_ptr

    old_dacl, descriptor = _read_dacl(path)
    new_acl = ctypes.c_void_p(0)
    try:
        if not old_dacl.value:
            # NULL DACL: everyone (AppContainers included) already has full
            # access, and any real ACL we wrote would restrict the owner.
            return
        err = advapi.SetEntriesInAclW(1, ctypes.byref(ea), old_dacl, ctypes.byref(new_acl))
        if err != 0:
            raise OSError(err, "SetEntriesInAclW failed")
        err = advapi.SetNamedSecurityInfoW(
            path,
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            None,
            None,
            new_acl,
            None,
        )
        if err != 0:
            raise OSError(err, f"SetNamedSecurityInfoW({path}) failed")
    finally:
        if new_acl.value:
            _kernel32().LocalFree(new_acl)
        if descriptor.value:
            _kernel32().LocalFree(descriptor)
