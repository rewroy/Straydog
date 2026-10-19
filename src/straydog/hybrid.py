"""Partial-residency (hybrid) expert placement for one Straydog GPU backend.

Dense operands, router and shared experts are pinned; routed experts use a bounded
device slot pool filled from routing statistics. Absent experts are evaluated by
the CPU expert engine during decode and streamed through staging slots in prefill.
The ABI is implemented in native/crates/inference/gpu/hybrid_*.hpp.
"""

from __future__ import annotations

import ctypes as ct
from typing import Any

import numpy as np

from straydog.formats import CPU_EXPERT_KINDS, matrix_bytes

_PTR = ct.c_void_p
_U64 = ct.c_uint64

HYBRID_SIGNATURES = {
    "sd_hybrid_moe_create": (
        [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), _U64, ct.POINTER(_PTR), _PTR, _U64], _PTR,
    ),
    "sd_expert_slots": ([_PTR, _U64, _U64, _U64], ct.c_int),
    "sd_hybrid_supported": ([_PTR], ct.c_int),
    "sd_hybrid_attach_cpu": ([_PTR, _PTR], ct.c_int),
    "sd_hybrid_rebalance": ([_PTR, _U64], ct.c_int),
    "sd_hybrid_drain": ([_PTR], ct.c_int),
    "sd_hybrid_counts": ([_PTR] + [ct.POINTER(_U64)] * 4, ct.c_int),
    "sd_hybrid_timing": ([_PTR] + [ct.POINTER(_U64)] * 4, ct.c_int),
    "sd_hybrid_assist": ([_PTR, ct.c_uint32, ct.POINTER(_U64)], ct.c_int),
    "sd_host_register": ([_PTR, _U64], ct.c_int),
    "sd_host_unregister": ([_PTR], ct.c_int),
}

def expert_bytes(kinds: tuple[int, int, int]) -> int:
    """One routed expert: gate and up (640x2560) plus down (2560x640) of the given kinds."""
    return (matrix_bytes(kinds[0], 640, 2560) + matrix_bytes(kinds[1], 640, 2560)
            + matrix_bytes(kinds[2], 2560, 640))


class HybridBackendMixin:
    """GPU backend operations for partially resident routed experts."""

    # Provided by GpuBackend.
    _lib: Any
    _handle: Any
    _states: dict
    _resident_plans: dict
    _resident_weights: set
    _prepare: Any
    _check: Any
    _error: Any
    memory: Any

    def pin_matrices(self, matrices: list[tuple]) -> None:
        """Upload dense operands now and exclude them from cache eviction."""
        pinned = {item[0] for item in matrices}
        for key, factory, rows, cols in matrices:
            self._prepare(key, factory, rows, cols, pinned)
        self._resident_weights.update(pinned)

    def register_hybrid_moe(self, prefix, matrices, banks, *, experts: int, kinds: tuple[int, int, int] = (42, 42, 42),
                            ) -> None:
        """Pin shared/router operands and bind the authenticated host expert banks of one layer."""
        shapes = [(640, 2560), (640, 2560), (2560, 640), (experts, 2560), (1, 2560)]
        if len(kinds) != 3 or not set(kinds) <= CPU_EXPERT_KINDS:
            raise ValueError("unsupported Straydog hybrid expert format")
        sizes = (experts * matrix_bytes(kinds[0], 640, 2560), experts * matrix_bytes(kinds[1], 640, 2560),
                 experts * matrix_bytes(kinds[2], 2560, 640))
        if (not 10 <= experts <= 512 or len(matrices) != 5 or len(banks) != 3 or prefix in self._resident_plans
                or any(item[2:] != shape for item, shape in zip(matrices, shapes, strict=True))
                or any(not isinstance(bank, np.ndarray) or bank.dtype != np.uint8 or bank.nbytes != size
                       for bank, size in zip(banks, sizes, strict=True))):
            raise ValueError("invalid Straydog hybrid MoE geometry")
        workspace = self._states["@moe:workspace"]
        capacity = (workspace[2] - 65536) // (11 * 4480 + 22)
        pinned = {item[0] for item in matrices}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        pointers = (_PTR * 5)(*(entry[0] for entry in entries))
        dense_kinds = (ct.c_int * 5)(*(entry[1] for entry in entries))
        host = (_PTR * 3)(*(bank.ctypes.data for bank in banks))
        routed = (ct.c_int * 3)(*kinds)
        plan = self._lib.sd_hybrid_moe_create(self._handle, pointers, dense_kinds, routed, experts, host, workspace[0],
                                              capacity)
        if not plan:
            raise RuntimeError(self._error())
        self._resident_plans[prefix] = plan
        self._slot_bytes = max(getattr(self, "_slot_bytes", 0), expert_bytes(tuple(kinds)))
        self._resident_weights.update(pinned)
        self._host_banks = getattr(self, "_host_banks", {})
        self._host_banks[prefix] = banks

    def create_expert_slots(self, *, stage: int = 64, margin_bytes: int = 192 * 1024**2) -> int:
        """Give the remaining owned budget to expert cache slots; return the cache slot count."""
        # A slot holds the largest expert of any partially resident layer on this device.
        slot = -(-getattr(self, "_slot_bytes", expert_bytes((42, 42, 42))) // 256) * 256
        used, budget = self.memory()
        free = budget - used - margin_bytes - stage * slot - 8192 * 4
        cache = max(0, free // slot)
        self._check(self._lib.sd_expert_slots(self._handle, cache, stage, slot))
        return cache

    def hybrid_supported(self) -> bool:
        """Partial residency needs the grouped FP16 prefill path: RDNA4 WMMA or CUDA tensor cores (sm_70+).

        GEMM precision is set before weights load.
        """
        result = self._lib.sd_hybrid_supported(self._handle)
        if result < 0:
            raise RuntimeError(self._error())
        return bool(result)

    def attach_cpu(self, engine: int) -> None:
        """Delegate absent routed experts to a native CPU expert engine."""
        self._check(self._lib.sd_hybrid_attach_cpu(self._handle, engine))

    def rebalance(self, max_swaps: int = 0) -> None:
        """Cache the highest-scoring experts by recent routing; 0 refreshes everything and waits."""
        self._check(self._lib.sd_hybrid_rebalance(self._handle, max_swaps))

    def hybrid_counts(self) -> dict[str, int]:
        """Routed experts served by GPU slots, the CPU and prefill streaming."""
        values = [_U64() for _ in range(4)]
        self._check(self._lib.sd_hybrid_counts(self._handle, *(ct.byref(value) for value in values)))
        timers = [_U64(), _U64(), _U64(), _U64()]
        self._check(self._lib.sd_hybrid_timing(self._handle, *(ct.byref(value) for value in timers)))
        return dict(zip(("gpu_experts", "cpu_experts", "streamed_experts", "cached_experts", "wait_ns", "cpu_ns",
                         "swapped_experts", "assisted_experts"),
                        [value.value for value in values + timers], strict=True))

    def enable_assist(self, routes: int) -> int:
        """Let the GPU read up to ``routes`` absent experts per layer from page-locked banks."""
        mapped = _U64()
        self._check(self._lib.sd_hybrid_assist(self._handle, routes, ct.byref(mapped)))
        return mapped.value

    def register_host(self, array: np.ndarray) -> bool:
        """Page-lock an authenticated mapping range for direct DMA; False if refused."""
        if self._lib.sd_host_register(array.ctypes.data, array.nbytes):
            return False
        self._registered = getattr(self, "_registered", [])
        self._registered.append(array)
        return True

    def unregister_hosts(self) -> None:
        """Finish background slot copies, then release page locks and mapping views before the mapping is closed.

        Native plans keep raw bank pointers; they are not used after the weights close.
        """
        if getattr(self, "_handle", None):
            self._check(self._lib.sd_hybrid_drain(self._handle))
        for array in getattr(self, "_registered", []):
            self._lib.sd_host_unregister(array.ctypes.data)
        self._registered = []
        self._host_banks = {}
