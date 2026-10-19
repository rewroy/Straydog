"""Loading the owner-supplied Straydog native library.

The library itself is always an explicit absolute path; PATH is never searched
for it. Windows has no RUNPATH, and Python's DLL search excludes PATH, so a CUDA
build's cuBLAS import (cublas64_<major>.dll) would not resolve on its own: the
``nvidia-cublas-cu12`` wheel of the straydog extra (so the packaged Windows build
needs no CUDA Toolkit) and the CUDA Toolkit named by ``CUDA_PATH`` are added as
dependency directories. Either provides the same cuBLAS major version.
"""

from __future__ import annotations

import contextlib
import ctypes as ct
import importlib.util
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

# Handles returned by os.add_dll_directory; closing one would remove its directory.
_dll_directories: dict[Path, object] = {}
# SetThreadErrorMode flags: no critical-error or open-file message boxes.
_SEM_FAILCRITICALERRORS, _SEM_NOOPENFILEERRORBOX = 0x0001, 0x8000


def cuda_runtime_directories(environ: Mapping[str, str] | None = None) -> list[Path]:
    """The ``CUDA_PATH`` toolkit's DLL directories (``bin``, and ``bin/x64`` in CUDA 13)."""
    root = (os.environ if environ is None else environ).get("CUDA_PATH")
    if not root:
        return []
    return [path for path in (Path(root) / "bin", Path(root) / "bin" / "x64") if path.is_dir()]


def cublas_wheel_directories() -> list[Path]:
    """The ``bin`` directories of an installed ``nvidia-cublas-cu12`` wheel (Windows DLLs)."""
    try:
        spec = importlib.util.find_spec("nvidia.cublas")
    except ModuleNotFoundError:
        return []
    locations = getattr(spec, "submodule_search_locations", None)
    return [Path(location) / "bin" for location in locations or [] if (Path(location) / "bin").is_dir()]


@contextlib.contextmanager
def _no_error_dialogs() -> Iterator[None]:
    """Windows: a broken or foreign file fails as an OSError, not as a modal "bad image" box.

    The loader shows that box, and waits, unless the thread's error mode says otherwise; init's
    load check and the server's worker have no one to dismiss it.
    """
    kernel32 = ct.WinDLL("kernel32", use_last_error=True)
    previous = ct.c_uint32()
    kernel32.SetThreadErrorMode(_SEM_FAILCRITICALERRORS | _SEM_NOOPENFILEERRORBOX, ct.byref(previous))
    try:
        yield
    finally:
        kernel32.SetThreadErrorMode(previous.value, None)


def load_library(library: Path) -> ct.CDLL:
    """Load ``library`` with its GPU runtime dependencies resolvable on this platform."""
    if sys.platform == "win32":
        for directory in [*cublas_wheel_directories(), *cuda_runtime_directories()]:
            if directory not in _dll_directories:
                _dll_directories[directory] = os.add_dll_directory(str(directory))
        with _no_error_dialogs():
            return ct.CDLL(str(library))
    return ct.CDLL(str(library))
