"""Start a child in a group of its own and end the whole group.

POSIX: a new session (``start_new_session``) and ``killpg``. Windows: a job object
created right after the spawn; terminating the job ends every process in it.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)


def detached_spawn_kwargs() -> dict[str, Any]:
    """Keyword arguments for Popen / create_subprocess_exec."""
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW: its own console group, and a hidden
        # console its children inherit.
        return {"creationflags": 0x00000200 | 0x08000000}
    return {"start_new_session": True}


def register_child_for_tree_kill(proc: Any) -> None:
    """Remember how to end ``proc`` and its descendants (call right after spawning it)."""
    if sys.platform == "win32":
        from straydog.platform.windows import _assign_to_job, _close_handle, _create_job_object_for_child

        job = _create_job_object_for_child()
        try:
            _assign_to_job(job, _native_handle(proc))
        except Exception:
            _close_handle(job)
            raise
        proc._straydog_job_handle = job
        return
    # The child's PID is its process group ID; kept because the group can outlive its leader.
    if getattr(proc, "pid", None) is not None:
        proc._straydog_pgid = proc.pid


def kill_tree(proc: Any, *, signal_name: str = "term") -> None:
    """End ``proc`` and its descendants: ``term`` politely, ``kill`` at once."""
    if sys.platform == "win32":
        job = getattr(proc, "_straydog_job_handle", None)
        if job is None:
            try:
                proc.kill() if signal_name == "kill" else proc.terminate()
            except Exception:
                logger.debug("kill_tree fallback failed", exc_info=True)
            return
        from straydog.platform.windows import _close_handle, _kernel32

        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = _kernel32()
            kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            if not kernel32.TerminateJobObject(wintypes.HANDLE(job), 0 if signal_name == "term" else 1):
                raise OSError(ctypes.get_last_error(), "TerminateJobObject failed")
        finally:
            _close_handle(job)
            proc._straydog_job_handle = None
        return
    sig = signal.SIGKILL if signal_name == "kill" else signal.SIGTERM
    pgid = getattr(proc, "_straydog_pgid", None)
    if pgid is None:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        logger.warning("killpg(%d) denied; ending the child alone", pgid)
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            return


def _native_handle(proc: Any) -> int:
    """The Win32 process handle of a Popen or asyncio subprocess (opened by PID when hidden)."""
    handle = getattr(proc, "_handle", None)
    if handle is None:
        inner = getattr(getattr(proc, "_transport", None), "_proc", None)
        handle = getattr(inner, "_handle", None)
    if handle is not None:
        return int(handle)
    import ctypes
    from ctypes import wintypes

    from straydog.platform.windows import _kernel32

    kernel32 = _kernel32()
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    opened = kernel32.OpenProcess(0x0100 | 0x0001, False, int(proc.pid))   # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if not opened:
        raise OSError(ctypes.get_last_error(), f"OpenProcess({proc.pid}) failed")
    return int(opened)
