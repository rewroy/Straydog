"""Host facts the isolated Straydog worker cannot observe itself.

The worker runs in its own cgroup namespace (bubblewrap), where its memory limits are
invisible; the runtime measures them before spawning it and passes the result on.
"""

from __future__ import annotations

import ctypes as ct
import os
import sys
from pathlib import Path


def _windows_memory_bytes() -> int:
    class _MemoryStatus(ct.Structure):
        _fields_ = [("length", ct.c_uint32), ("load", ct.c_uint32)] + [
            (name, ct.c_uint64) for name in ("total", "available", "page_total", "page_available",
                                             "virtual_total", "virtual_available", "extended")]

    status = _MemoryStatus(length=ct.sizeof(_MemoryStatus))
    return status.total if ct.WinDLL("kernel32").GlobalMemoryStatusEx(ct.byref(status)) else 0


def host_memory_bytes(*, proc_cgroup: Path = Path("/proc/self/cgroup"),
                      cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Physical memory, bounded on Linux by every cgroup v2 ``memory.max`` above this process; 0 if unknown."""
    if sys.platform == "win32":
        return _windows_memory_bytes()
    if not hasattr(os, "sysconf"):
        return 0
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return 0
    if total <= 0 or sys.platform != "linux":
        return max(total, 0)
    try:
        entry = next(line[3:] for line in proc_cgroup.read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return total
    group = Path(os.path.normpath(cgroup_root / entry.strip().lstrip("/")))
    for directory in (group, *group.parents):
        if directory == cgroup_root or not directory.is_relative_to(cgroup_root):
            break
        try:
            limit = (directory / "memory.max").read_text().strip()
        except OSError:
            continue
        if limit.isdigit():
            total = min(total, int(limit))
    return total
