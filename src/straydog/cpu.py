"""Straydog's CPU expert engine for partially resident models.

The native implementation lives in native/crates/inference/gpu/cpu_experts.cpp and
is part of the owner-built Straydog library. It evaluates routed experts (Q2_0 and
the IQ/Q8_0 formats of the IQ3 bundles) in F32 directly from the authenticated
read-only model mapping.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

from straydog.formats import CPU_EXPERT_KINDS, load_iq_tables, matrix_bytes
from straydog.library import load_library

_PTR = ct.c_void_p
_F32_PTR = ct.POINTER(ct.c_float)


class CpuExperts:
    """A persistent CPU worker pool bound to one native library."""

    def __init__(self, library: Path, *, threads: int = 0) -> None:
        if not library.is_absolute() or not library.is_file():
            raise ValueError("Straydog requires an explicit native library file")
        if isinstance(threads, bool) or not 0 <= threads <= 256:
            raise ValueError("invalid Straydog CPU expert thread count")
        self._lib = load_library(library)
        signatures = {
            "sd_cpu_error": ([], ct.c_char_p),
            "sd_cpu_experts_create": ([ct.c_uint32], _PTR),
            "sd_cpu_experts_destroy": ([_PTR], None),
            "sd_cpu_experts_isa": ([_PTR], ct.c_char_p),
            "sd_cpu_experts_stats": ([_PTR, ct.POINTER(ct.c_uint64)], ct.c_int),
            "sd_cpu_experts_run": (
                [_PTR, ct.POINTER(ct.c_int), ct.POINTER(_PTR), ct.POINTER(_PTR), ct.POINTER(_PTR), _F32_PTR,
                 ct.c_uint32, ct.c_uint64, ct.c_uint64, _F32_PTR, _F32_PTR], ct.c_int,
            ),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = args, result
        load_iq_tables(self._lib)
        self._handle = self._lib.sd_cpu_experts_create(threads)
        if not self._handle:
            raise RuntimeError("Straydog CPU experts: " + self._lib.sd_cpu_error().decode("utf-8", "replace"))

    @property
    def handle(self) -> int:
        """Native engine pointer, shared with GPU devices that delegate misses."""
        if not self._handle:
            raise RuntimeError("Straydog CPU experts are closed")
        return self._handle

    @property
    def isa(self) -> str:
        """Instruction set selected at run time for this host."""
        return self._lib.sd_cpu_experts_isa(self.handle).decode()

    def stats(self) -> dict[str, int]:
        """Cumulative calls, experts and per-phase nanoseconds of the native engine."""
        values = (ct.c_uint64 * 5)()
        if self._lib.sd_cpu_experts_stats(self.handle, values):
            raise RuntimeError("Straydog CPU experts: " + self._lib.sd_cpu_error().decode("utf-8", "replace"))
        return dict(zip(("calls", "experts", "projection_ns", "product_ns", "down_ns"), values, strict=True))

    def run(self, experts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], weights: np.ndarray, x: np.ndarray,
            *, inter: int, kinds: tuple[int, int, int] = (42, 42, 42)) -> np.ndarray:
        """Return sum_i weights[i] * down_i(silu(gate_i x) * up_i x) for packed expert slices.

        Each expert is a tuple of uint8 views (gate, up, down) of GGML ``kinds`` that must
        stay alive for the call.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        hidden = x.size
        if (x.ndim != 1 or not 1 <= len(experts) <= 64 or weights.shape != (len(experts),)
                or not np.isfinite(x).all() or not np.isfinite(weights).all() or inter <= 0
                or len(kinds) != 3 or not set(kinds) <= CPU_EXPERT_KINDS):
            raise ValueError("invalid Straydog CPU expert request")
        sizes = (matrix_bytes(kinds[0], inter, hidden), matrix_bytes(kinds[1], inter, hidden),
                 matrix_bytes(kinds[2], hidden, inter))
        for parts in experts:
            if len(parts) != 3 or any(
                not isinstance(part, np.ndarray) or part.dtype != np.uint8 or part.nbytes != size
                or not part.flags.c_contiguous for part, size in zip(parts, sizes, strict=True)
            ):
                raise ValueError("invalid Straydog CPU expert slice")
        count = len(experts)
        pointers = [(_PTR * count)(*(parts[role].ctypes.data for parts in experts)) for role in range(3)]
        output = np.empty(hidden, np.float32)
        status = self._lib.sd_cpu_experts_run(
            self.handle, (ct.c_int * 3)(*kinds), *pointers, weights.ctypes.data_as(_F32_PTR), count, hidden, inter,
            x.ctypes.data_as(_F32_PTR), output.ctypes.data_as(_F32_PTR),
        )
        if status:
            raise RuntimeError("Straydog CPU experts: " + self._lib.sd_cpu_error().decode("utf-8", "replace"))
        return output

    def close(self) -> None:
        """Stop the worker pool."""
        if self._handle:
            self._lib.sd_cpu_experts_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> CpuExperts:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
