"""The native library loader resolves GPU runtime dependencies without searching PATH."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from straydog import library as loader


class _Kernel32:
    """SetThreadErrorMode as Windows has it: returns the previous mode through the pointer."""

    def __init__(self, events):
        self.mode, self.events = 0x10, events

    def SetThreadErrorMode(self, mode, previous):  # noqa: N802 (the Win32 name)
        if previous is not None:
            previous._obj.value = self.mode
        self.mode = mode
        self.events.append(("mode", mode))
        return 1


@pytest.fixture
def kernel32(monkeypatch):
    events = []
    monkeypatch.setattr(loader.ct, "WinDLL", lambda name, use_last_error=False: _Kernel32(events), raising=False)
    return events


def test_cuda_runtime_directories_come_from_cuda_path_only(tmp_path):
    (tmp_path / "bin/x64").mkdir(parents=True)
    assert loader.cuda_runtime_directories({"CUDA_PATH": str(tmp_path)}) == [tmp_path / "bin", tmp_path / "bin/x64"]
    (tmp_path / "bin/x64").rmdir()
    assert loader.cuda_runtime_directories({"CUDA_PATH": str(tmp_path)}) == [tmp_path / "bin"]
    assert loader.cuda_runtime_directories({"CUDA_PATH": str(tmp_path / "absent")}) == []
    assert loader.cuda_runtime_directories({"PATH": str(tmp_path / "bin")}) == []


def test_windows_load_adds_the_toolkit_once_and_loads_the_explicit_path(tmp_path, monkeypatch, kernel32):
    (tmp_path / "bin").mkdir()
    added, loaded = [], []
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    monkeypatch.setattr(loader.sys, "platform", "win32")
    monkeypatch.setattr(loader, "_dll_directories", {})
    monkeypatch.setattr(loader, "cublas_wheel_directories", lambda: [])   # no wheel: the toolkit alone
    monkeypatch.setattr(loader.os, "add_dll_directory", lambda path: added.append(path) or object(), raising=False)
    monkeypatch.setattr(loader.ct, "CDLL", lambda path: loaded.append(path) or path)
    library = tmp_path / "build/straydog_gpu.dll"
    assert loader.load_library(library) == str(library)
    assert loader.load_library(library) == str(library)
    assert added == [str(tmp_path / "bin")] and loaded == [str(library)] * 2


def test_other_platforms_load_without_adding_directories(monkeypatch):
    def searched():
        raise AssertionError("only Windows adds DLL directories")

    monkeypatch.setattr(loader.sys, "platform", "linux")
    monkeypatch.setattr(loader, "cuda_runtime_directories", searched)
    monkeypatch.setattr(loader.ct, "CDLL", lambda path: path)
    library = Path("/opt/straydog/libstraydog_gpu.so")
    assert loader.load_library(library) == str(library)


def test_windows_load_adds_the_cublas_wheel_without_the_toolkit(tmp_path, monkeypatch, kernel32):
    """The packaged Windows build finds cuBLAS in the straydog extra's nvidia-cublas-cu12 wheel."""
    (tmp_path / "nvidia/cublas/bin").mkdir(parents=True)
    added = []
    monkeypatch.delenv("CUDA_PATH", raising=False)
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path / "nvidia/cublas")])
    monkeypatch.setattr(loader.importlib.util, "find_spec", lambda name: spec if name == "nvidia.cublas" else None)
    monkeypatch.setattr(loader.sys, "platform", "win32")
    monkeypatch.setattr(loader, "_dll_directories", {})
    monkeypatch.setattr(loader.os, "add_dll_directory", lambda path: added.append(path) or object(), raising=False)
    monkeypatch.setattr(loader.ct, "CDLL", lambda path: path)
    assert loader.cublas_wheel_directories() == [tmp_path / "nvidia/cublas/bin"]
    loader.load_library(tmp_path / "straydog_gpu.dll")
    assert added == [str(tmp_path / "nvidia/cublas/bin")]


def test_windows_load_suppresses_error_dialogs_only_while_loading(tmp_path, monkeypatch, kernel32):
    """A broken DLL must fail as an OSError: the loader's modal "bad image" box would wait for a click."""
    monkeypatch.setattr(loader.sys, "platform", "win32")
    monkeypatch.setattr(loader, "cublas_wheel_directories", lambda: [])
    monkeypatch.setattr(loader, "cuda_runtime_directories", lambda: [])

    def broken(path):
        kernel32.append(("load", path))
        raise OSError("[WinError 193] %1 is not a valid Win32 application")

    monkeypatch.setattr(loader.ct, "CDLL", broken)
    with pytest.raises(OSError, match="WinError 193"):
        loader.load_library(tmp_path / "straydog_gpu.dll")
    assert kernel32 == [("mode", 0x8001), ("load", str(tmp_path / "straydog_gpu.dll")), ("mode", 0x10)]
