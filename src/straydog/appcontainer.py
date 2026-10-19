"""Windows containment for the Straydog worker: its own AppContainer, no capabilities.

The Windows counterpart of the bubblewrap command in straydog.isolation. The
server starts this launcher as an ordinary child with its private pipes; the
launcher grants the dedicated ``Straydog.Worker`` AppContainer profile read
access to exactly what the worker reads (the Python runtime, Straydog's package,
the CUDA toolkit's DLLs, the native artifacts and each pinned model file), then
starts the worker inside that AppContainer with the launcher's own standard
handles (and no other inheritable handle), and exits with the worker's status.

No capability is given, so the network (loopback included) is closed; nothing
outside the profile's own folder is writable. A Job Object
ends the worker with the launcher. There is no fallback to an unrestricted child.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from collections.abc import Sequence
from ctypes import wintypes
from pathlib import Path

PROFILE_NAME = "Straydog.Worker"
_STD_HANDLES = (-10, -11, -12)   # STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002


def appcontainer_worker_command(
    command: list[str], *, files: list[Path], directories: list[Path], secrets: Sequence[Path] = (),
) -> list[str]:
    """The launcher invocation that runs ``command`` contained; paths are resolved by the caller."""
    from straydog.library import cuda_runtime_directories

    # The interpreter, the venv (its site-packages) and the directory holding the straydog package.
    trees = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(), Path(__file__).resolve().parents[1]}
    trees.update(path.resolve() for path in cuda_runtime_directories())
    arguments = [sys.executable, "-I", "-m", "straydog.appcontainer"]
    for flag, paths in (("--tree", sorted(trees)), ("--file", files), ("--directory", directories),
                        ("--secret", secrets)):
        for path in paths:
            arguments += [flag, str(path)]
    return arguments + ["--", *command]


def _ungranted_entries(tree: Path, sid: int) -> list[tuple[str, int]]:
    """Every entry of the tree's site-packages the profile cannot use, with the access it needs.

    uv links packages into an environment as hard links to its cache, so their files keep
    the cache's ACL rather than inheriting the environment's grant: Python silently skips an
    unreadable .pth file, and an extension module or DLL the worker cannot read and execute
    fails to load ("DLL load failed ...: Access is denied"). A package directory is created in
    place and inherits the grant, so it says nothing about the files below it; every entry is
    checked (one DACL read each).
    """
    from straydog.platform.windows import _RX, FILE_GENERIC_READ, _path_has_grant

    site = tree / "Lib" / "site-packages"
    missing: list[tuple[str, int]] = []
    pending = [str(site)] if site.is_dir() else []
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    directory = entry.is_dir()
                    executable = directory or entry.name.lower().endswith((".pyd", ".dll", ".exe"))
                    mask = _RX if executable else FILE_GENERIC_READ
                    if not _path_has_grant(entry.path, sid, mask, inherit=False):
                        missing.append((entry.path, mask))
                    if directory:
                        pending.append(entry.path)
        except OSError:
            continue
    return missing


def grant_worker_access(
    *, trees: list[Path], files: list[Path], directories: list[Path], secrets: Sequence[Path] = (),
) -> int:
    """Grant the Straydog profile read access (idempotent; the first grant on a tree is slow) and return its SID."""
    from straydog.platform.windows import (
        _RX,
        FILE_GENERIC_READ,
        _create_app_container_profile,
        _ensure_path_grant,
        _grant_path_access,
        _path_has_grant,
    )

    from straydog.isolation import check_exposable

    check_exposable([*trees, *files, *directories], secrets)
    sid = _create_app_container_profile(PROFILE_NAME)
    for tree in trees:
        try:
            _ensure_path_grant(str(tree), sid, _RX)
        except OSError:
            # An administrator-owned tree (the CUDA toolkit under Program Files) cannot take our ACE;
            # it is usable when it already lets AppContainers (ALL APPLICATION PACKAGES) read it.
            if not _path_has_grant(str(tree), sid, _RX, inherit=False):
                raise
            continue
        missing = _ungranted_entries(tree, sid)
        if missing:
            print(f"Straydog: granting the worker read access to {len(missing)} new entries in {tree} (one-time)",
                  file=sys.stderr)
            # Writing the tree's DACL again propagates its inheritable grant to every entry it can reach;
            # an entry that still lacks it (inheritance blocked on it) gets an ACE of its own.
            _grant_path_access(str(tree), sid, _RX)
            for path, mask in missing:
                if not _path_has_grant(path, sid, mask, inherit=False):
                    try:
                        _grant_path_access(path, sid, mask, inherit=False)
                    except OSError as exc:
                        raise ValueError(f"Straydog cannot grant its worker read access to {path}: {exc}") from exc
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Straydog worker file is missing or not a regular file: {path}")
        _ensure_path_grant(str(path), sid, _RX if path.suffix.lower() in {".dll", ".exe"} else FILE_GENERIC_READ,
                           inherit=False)
    for directory in directories:
        # The directory's own entries (its working directory), not its subtree.
        _ensure_path_grant(str(directory), sid, _RX, inherit=False)
    return sid


def _profile_folder(sid: int) -> str | None:
    """The AppContainer's own writable folder (its private TEMP), when Windows reports it."""
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
    userenv.GetAppContainerFolderPath.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.LPWSTR)]
    userenv.GetAppContainerFolderPath.restype = ctypes.c_int32
    text, folder = wintypes.LPWSTR(), wintypes.LPWSTR()
    if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        return None
    try:
        if userenv.GetAppContainerFolderPath(text.value, ctypes.byref(folder)) != 0:
            return None
        try:
            return folder.value
        finally:
            ctypes.windll.ole32.CoTaskMemFree(folder)
    finally:
        ctypes.windll.kernel32.LocalFree(text)


def launch(command: list[str], sid: int) -> int:
    """Run ``command`` in the AppContainer with this process's standard handles; return its exit code."""
    from straydog.platform.windows import _create_job_object_for_child
    from straydog.platform.windows import (
        CREATE_NO_WINDOW,
        CREATE_SUSPENDED,
        CREATE_UNICODE_ENVIRONMENT,
        EXTENDED_STARTUPINFO_PRESENT,
        HANDLE_FLAG_INHERIT,
        INFINITE,
        PROC_THREAD_ATTRIBUTE_JOB_LIST,
        PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
        PROCESS_INFORMATION,
        SECURITY_CAPABILITIES,
        STARTF_USESTDHANDLES,
        STARTUPINFOEXW,
        _alloc_attribute_list,
        _close_handle,
        _delete_attribute_list,
        _get_exit_code,
        _make_env_block,
        _resume_thread,
        _update_attribute,
        _wait_for,
    )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
    kernel32.SetHandleInformation.restype = wintypes.BOOL
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
        ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(STARTUPINFOEXW), ctypes.POINTER(PROCESS_INFORMATION),
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL

    standard = [kernel32.GetStdHandle(code) for code in _STD_HANDLES]
    if not all(standard):
        raise OSError("the Straydog launcher needs standard input, output and error handles")
    inherited = list(dict.fromkeys(int(handle) for handle in standard))
    for handle in inherited:
        if not kernel32.SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT):
            raise OSError(ctypes.get_last_error(), "SetHandleInformation failed")

    env = dict(os.environ)
    folder = _profile_folder(sid)
    if folder:
        env["TEMP"] = env["TMP"] = str(Path(folder) / "Temp")
    job = _create_job_object_for_child()   # kill on close: the worker never outlives the launcher
    attributes = buffer = None
    process = PROCESS_INFORMATION()
    try:
        attributes, buffer = _alloc_attribute_list(3)
        capabilities = SECURITY_CAPABILITIES()
        capabilities.AppContainerSid = sid
        capabilities.Capabilities = None
        capabilities.CapabilityCount = 0   # no internetClient or any other capability
        _update_attribute(attributes, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                          ctypes.cast(ctypes.pointer(capabilities), ctypes.c_void_p), ctypes.sizeof(capabilities))
        jobs = (wintypes.HANDLE * 1)(wintypes.HANDLE(job))
        _update_attribute(attributes, PROC_THREAD_ATTRIBUTE_JOB_LIST, ctypes.cast(jobs, ctypes.c_void_p),
                          ctypes.sizeof(jobs))
        handles = (wintypes.HANDLE * len(inherited))(*inherited)
        _update_attribute(attributes, _PROC_THREAD_ATTRIBUTE_HANDLE_LIST, ctypes.cast(handles, ctypes.c_void_p),
                          ctypes.sizeof(handles))
        startup = STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = standard
        startup.lpAttributeList = attributes
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        environment = _make_env_block(env)
        flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW
        if not kernel32.CreateProcessW(command[0], command_line, None, None, True, flags,
                                       ctypes.cast(environment, ctypes.c_void_p), os.getcwd(),
                                       ctypes.byref(startup), ctypes.byref(process)):
            raise OSError(ctypes.get_last_error(), "CreateProcessW for the contained Straydog worker failed")
        _resume_thread(process.hThread)
        _wait_for(process.hProcess, INFINITE)
        return _get_exit_code(process.hProcess)
    finally:
        _close_handle(int(process.hThread or 0))
        _close_handle(int(process.hProcess or 0))
        if attributes is not None:
            _delete_attribute_list(attributes)
        del buffer
        _close_handle(job)


def main() -> int:
    """Grant, then run the worker command after ``--`` inside the Straydog AppContainer."""
    parser = argparse.ArgumentParser(prog="straydog-appcontainer")
    parser.add_argument("--tree", type=Path, action="append", default=[])
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument("--directory", type=Path, action="append", default=[])
    parser.add_argument("--secret", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if sys.platform != "win32" or not command:
        parser.error("runs on Windows with a worker command after --")
    try:
        sid = grant_worker_access(trees=args.tree, files=args.file, directories=args.directory, secrets=args.secret)
        return launch(command, sid)
    except (OSError, ValueError) as exc:
        # The server reports the worker's last stderr line as the start failure.
        print(f"Straydog AppContainer launch failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
