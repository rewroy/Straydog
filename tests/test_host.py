"""Host memory for Straydog policies: physical memory bounded by the cgroup limits above the process."""

from __future__ import annotations

import os
import sys

import pytest

from straydog.host import host_memory_bytes
from straydog.weights import lock_budget

linux = pytest.mark.skipif(not hasattr(os, "sysconf") or sys.platform != "linux", reason="Linux cgroup v2")


def _tree(tmp_path, limits: dict[str, str]):
    root = tmp_path / "cgroup"
    for group, limit in limits.items():
        directory = root / group
        directory.mkdir(parents=True, exist_ok=True)
        if limit is not None:
            (directory / "memory.max").write_text(limit + "\n")
    (root / "memory.max").write_text("1\n")   # the root group never limits (and has no memory.max)
    proc = tmp_path / "cgroup-self"
    return root, proc


@linux
def test_the_tightest_ancestor_limit_bounds_physical_memory(tmp_path):
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    root, proc = _tree(tmp_path, {"user.slice": "max", "user.slice/app.slice": str(32 * 1024**3),
                                  "user.slice/app.slice/lab.scope": "max"})
    proc.write_text("0::/user.slice/app.slice/lab.scope\n")
    assert host_memory_bytes(proc_cgroup=proc, cgroup_root=root) == min(physical, 32 * 1024**3)
    proc.write_text("0::/user.slice\n")
    assert host_memory_bytes(proc_cgroup=proc, cgroup_root=root) == physical


@linux
def test_unreadable_or_foreign_cgroup_data_falls_back_to_physical_memory(tmp_path):
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    root, proc = _tree(tmp_path, {"a": str(1024)})
    assert host_memory_bytes(proc_cgroup=proc, cgroup_root=root) == physical   # no /proc/self/cgroup
    proc.write_text("12:memory:/a\n")   # cgroup v1 only
    assert host_memory_bytes(proc_cgroup=proc, cgroup_root=root) == physical
    proc.write_text("0::/../../outside\n")
    assert host_memory_bytes(proc_cgroup=proc, cgroup_root=root) == physical
    proc.write_text("0::/a\n")
    assert host_memory_bytes(proc_cgroup=proc, cgroup_root=root) == 1024


@pytest.mark.skipif(sys.platform != "win32", reason="Windows")
def test_windows_reports_the_installed_memory():
    assert host_memory_bytes() >= 1 << 30


def test_lock_budget_locks_whole_banks_only_with_room_to_spare():
    gib = 1024**3
    assert lock_budget(0, 33 * gib) == 0   # unknown memory: nothing is page-locked
    assert lock_budget(125 * gib, 33 * gib) == 33 * gib
    assert lock_budget(48 * gib, 33 * gib) == 24 * gib   # half the memory: whole layers, the rest streams
    assert lock_budget(32 * gib, 33 * gib) == 0   # too tight: the rest would thrash in what is left
    assert lock_budget(62 * gib, 33 * gib, pinned_state=13 * gib) == 18 * gib
    assert lock_budget(16 * gib, 33 * gib, pinned_state=13 * gib) == 0
