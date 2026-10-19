"""Straydog's bounded HIP/CUDA computational backend; no inference-engine dependency.

The ABI is implemented in native/crates/inference/gpu/kernels.cpp. The library
path is supplied by the owner; loading never searches PATH or downloads code.
"""

from __future__ import annotations

import ctypes as ct
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np

from straydog.formats import BLOCKS, load_iq_tables
from straydog.hybrid import HYBRID_SIGNATURES, HybridBackendMixin
from straydog.library import load_library
from straydog.speculative import SPECULATIVE_SIGNATURES, SpeculativeBackendMixin

_PTR = ct.c_void_p
_U64 = ct.c_uint64
_F32_PTR = ct.POINTER(ct.c_float)


_ALLOCATION_TAIL = 64   # the native library pads every VRAM allocation (allocation_tail in kernels.cpp)


def _decoder_workspace_bytes(capacity: int) -> int:
    """The workspace of the fully resident decoder's spans for a prefill capacity."""
    return (capacity * 15364 + 1024 * 2564 + 8) * 4


class GpuBackend(HybridBackendMixin, SpeculativeBackendMixin):
    """One device, one stream and an explicit LRU weight-cache budget."""

    def __init__(self, library: Path, *, device: int = 0, budget_bytes: int) -> None:
        if not library.is_absolute() or not library.is_file():
            raise ValueError("Straydog requires an explicit GPU library file")
        self._lib = load_library(library)
        signatures = {
            "sd_error": ([], ct.c_char_p),
            "sd_device_count": ([], ct.c_int),
            "sd_device_info": ([ct.c_int, _PTR, ct.c_size_t, ct.POINTER(_U64), ct.POINTER(_U64)], ct.c_int),
            "sd_create": ([ct.c_int, _U64], _PTR),
            "sd_destroy": ([_PTR], None),
            "sd_gemm_options": ([_PTR, _U64, ct.c_int], ct.c_int),
            "sd_gemm_counts": ([_PTR, ct.POINTER(_U64), ct.POINTER(_U64)], ct.c_int),
            "sd_hyperconnection": (
                [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64, ct.c_int, _F32_PTR, _F32_PTR],
                ct.c_int,
            ),
            "sd_gdn_sequence": (
                [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _PTR, _PTR, _U64, _F32_PTR, _F32_PTR],
                ct.c_int,
            ),
            "sd_moe": (
                [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64, _PTR, _U64,
                 ct.POINTER(ct.c_int32), _F32_PTR, _F32_PTR, _F32_PTR], ct.c_int,
            ),
            "sd_qsa_sequence": (
                [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _PTR, _PTR, _PTR,
                 _U64, _U64, _U64, _F32_PTR, _F32_PTR], ct.c_int,
            ),
            "sd_resident_moe_create": (
                [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64, _PTR, _U64], _PTR,
            ),
            "sd_resident_moe_part": (
                [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64, _U64, _U64, ct.c_int, _PTR, _U64], _PTR,
            ),
            "sd_helper_moe": ([_PTR, _U64, _PTR], ct.c_int),
            "sd_peer_attach": ([_PTR, _PTR, _U64], ct.c_int),
            "sd_peer_direct": ([_PTR], ct.c_int),
            "sd_resident_moe": ([_PTR, _PTR, _U64, _F32_PTR, _F32_PTR], ct.c_int),
            "sd_execution_counts": ([_PTR, ct.POINTER(_U64), ct.POINTER(_U64), ct.POINTER(_U64)], ct.c_int),
            "sd_ple": (
                [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _PTR, _U64, _F32_PTR, _F32_PTR, _F32_PTR], ct.c_int,
            ),
            "sd_decoder_layer": (
                [_PTR, _U64, _U64, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64,
                 ct.POINTER(_PTR), _PTR, _PTR], ct.c_int,
            ),
            "sd_decode_span": ([_PTR, _U64, _U64, _U64, _F32_PTR, _F32_PTR], ct.c_int),
            "sd_prefill_span": (
                [_PTR, _U64, _U64, _U64, _U64, _F32_PTR, _F32_PTR, _U64, ct.POINTER(ct.c_double)], ct.c_int,
            ),
            "sd_prefill_ple": ([_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _PTR, _U64, _F32_PTR, _U64], ct.c_int),
            "sd_memory": ([_PTR, ct.POINTER(_U64), ct.POINTER(_U64)], ct.c_int),
            "sd_zero": ([_PTR, _PTR], ct.c_int),
            "sd_download": ([_PTR, _PTR, _PTR, _U64], ct.c_int),
            "sd_attention_sequence": ([_PTR, _PTR, _U64, _U64, _U64, _U64, _F32_PTR, _U64, _F32_PTR, _U64], ct.c_int),
            "sd_delta_sequence": ([_PTR, _PTR, _U64, _U64, _U64, _F32_PTR, _U64, _F32_PTR, _U64], ct.c_int),
            "sd_upload": ([_PTR, _PTR, _U64], _PTR),
            "sd_release": ([_PTR, _PTR], ct.c_int),
            "sd_host_state": ([_PTR, _U64], _PTR),
            "sd_checkpoint_buffer": ([_PTR, _U64], _PTR),
            "sd_checkpoint_copy": ([_PTR, ct.c_int, ct.POINTER(_PTR), _U64, _PTR], ct.c_int),
            "sd_host_state_supported": ([_PTR], ct.c_int),
            "sd_kv_stage": ([_PTR, _PTR, _U64], ct.c_int),
            "sd_matvec": ([_PTR, _PTR, ct.c_int, _U64, _U64, _F32_PTR, _F32_PTR], ct.c_int),
            "sd_matvec_batch": (
                [
                    _PTR,
                    ct.POINTER(_PTR),
                    ct.POINTER(ct.c_int),
                    ct.POINTER(_U64),
                    ct.POINTER(_U64),
                    _U64,
                    _F32_PTR,
                    _U64,
                    _F32_PTR,
                    _U64,
                ],
                ct.c_int,
            ),
        }
        signatures.update(HYBRID_SIGNATURES)
        signatures.update(SPECULATIVE_SIGNATURES)
        for name, (args, result) in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = args, result
        self.library = library
        if isinstance(budget_bytes, bool) or not 2**21 <= budget_bytes < 2**63:
            raise ValueError("Straydog VRAM budget must be at least 2 MiB")
        load_iq_tables(self._lib)   # each device uploads the grids when it is created
        self._handle = self._lib.sd_create(device, budget_bytes)
        self._cache: OrderedDict[str, tuple[int, int, int, int]] = OrderedDict()
        self._experts: OrderedDict[str, None] = OrderedDict()
        self._resident_weights: set[str] = set()
        self._resident_plans: dict[str, int] = {}
        self._helper_plans: dict[str, int] = {}   # this device's expert parts for layers its peer owns
        self._decoder_layers: set[int] = set()
        if not self._handle:
            raise RuntimeError(self._error())
        self.device = device
        self.stats: dict[str, float] | None = None
        self._states: dict[str, tuple] = {}
        self._checkpoints: list[int] = []
        self._checkpoint_states = None
        self.batch_capacity = 256 if budget_bytes >= 256 * 1024**2 else 32
        self.logical_capacity = 4096 if budget_bytes >= 2 * 1024**3 else 1024
        self.output_capacity = 262144 * (32 if self.batch_capacity == 256 else 1)
        if budget_bytes >= 2 * 1024**3:
            self.output_capacity *= 2

    def _error(self) -> str:
        return "Straydog GPU: " + self._lib.sd_error().decode("utf-8", "replace")

    def _check(self, status: int) -> None:
        if status != 0:
            raise RuntimeError(self._error())

    def memory(self) -> tuple[int, int]:
        """Observed allocation bytes and the enforced budget, including workspace."""
        used, budget = _U64(), _U64()
        self._check(self._lib.sd_memory(self._handle, ct.byref(used), ct.byref(budget)))
        return used.value, budget.value

    def info(self) -> dict[str, object]:
        """Query the selected device without assuming its advertised memory is free."""
        name, free, total = ct.create_string_buffer(256), _U64(), _U64()
        self._check(self._lib.sd_device_info(self.device, name, len(name), ct.byref(free), ct.byref(total)))
        used, budget = self.memory()
        gemms, vectors = _U64(), _U64()
        self._check(self._lib.sd_gemm_counts(self._handle, ct.byref(gemms), ct.byref(vectors)))
        graphs, replays, plans = _U64(), _U64(), _U64()
        self._check(self._lib.sd_execution_counts(self._handle, ct.byref(graphs), ct.byref(replays), ct.byref(plans)))
        return {
            "name": name.value.decode(),
            "free_bytes": free.value,
            "total_bytes": total.value,
            "allocated_bytes": used,
            "budget_bytes": budget,
            "device": self.device,
            "gemm_calls": gemms.value,
            "gemm_vectors": vectors.value,
            "cached_graphs": graphs.value,
            "graph_replays": replays.value,
            "resident_moe_plans": plans.value,
            "decoder_layers": len(self._decoder_layers),
            **self.hybrid_counts(),
        }

    def _prepare(self, key, matrix, rows, cols, pinned):
        stats = self.stats
        entry = self._cache.get(key)
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["hits"] = stats.get("hits", 0) + int(entry is not None)
        if entry is not None:
            if entry[2:] != (rows, cols):
                raise ValueError("cached Straydog matrix shape mismatch")
            self._cache.move_to_end(key)
            if key in self._experts:
                self._experts.move_to_end(key)
            return entry
        data, kind = matrix()
        data = np.ascontiguousarray(data)
        block, size = BLOCKS.get(kind, (0, 0))
        if not block or cols % block:
            raise ValueError("unsupported Straydog GPU weight format")
        if data.nbytes != rows * cols // block * size:
            raise ValueError("Straydog GPU weight size mismatch")
        used, budget = self.memory()
        while used + data.nbytes > budget:
            # Keep repeatedly used dense projections resident before cold experts.
            # This is a soft priority: dense weights remain evictable when a batch
            # could not otherwise fit. Semantic state is never an eviction target.
            victim = next(
                (name for name in self._experts if name not in pinned and name not in self._resident_weights), None
            )
            if victim is None:
                victim = next((name for name in self._cache if name not in pinned and name not in self._states
                               and name not in self._resident_weights), None)
            if victim is None:
                raise ValueError("matrix batch cannot fit in the Straydog VRAM budget")
            previous = self._cache.pop(victim)
            self._experts.pop(victim, None)
            self._check(self._lib.sd_release(self._handle, previous[0]))
            if stats is not None:
                stats["evictions"] = stats.get("evictions", 0) + 1
            used, budget = self.memory()
        upload_started = perf_counter() if stats is not None else 0
        pointer = self._lib.sd_upload(self._handle, data.ctypes.data, data.nbytes)
        if not pointer:
            raise RuntimeError(self._error())
        if stats is not None:
            stats["upload_seconds"] = stats.get("upload_seconds", 0) + perf_counter() - upload_started
            stats["upload_bytes"] = stats.get("upload_bytes", 0) + data.nbytes
        entry = (pointer, kind, rows, cols)
        self._cache[key] = entry
        if "_exps.weight:" in key:
            self._experts[key] = None
        return entry

    def project(
        self, key: str, matrix: Callable[[], tuple[np.ndarray, int]], rows: int, cols: int, vector: np.ndarray
    ) -> np.ndarray:
        """Multiply one checked matrix using the shared bounded batch path."""
        return self.project_many([(key, matrix, rows, cols, vector)])[0]

    def project_many(self, requests: list[tuple]) -> list[np.ndarray]:
        """Fuse a bounded set of projections into one upload/launch/readback.

        Pin every requested cache key throughout preparation: loading the last
        expert must never evict a pointer belonging to an earlier batch member.
        """
        if not self._handle:
            raise RuntimeError("Straydog GPU is closed")
        if (
            not 1 <= len(requests) <= self.batch_capacity
            or sum(request[2] for request in requests) > self.output_capacity
        ):
            raise ValueError("Straydog matrix batch exceeds workspace limits")
        stats = self.stats
        started = perf_counter() if stats is not None else 0
        vectors = []
        for _, _, rows, cols, vector in requests:
            if not 0 < rows <= 262144 or not 0 < cols <= 16384:
                raise ValueError("matrix dimensions exceed Straydog GPU limits")
            vector = np.ascontiguousarray(vector, dtype=np.float32)
            if vector.shape != (cols,):
                raise ValueError("invalid Straydog input vector")
            vectors.append(vector)
        inputs = np.concatenate(vectors)
        if not np.isfinite(inputs).all():
            raise ValueError("invalid Straydog input vector")
        pinned = {request[0] for request in requests}
        prepared_entries, entries = {}, []
        duplicate_hits = 0
        for key, matrix, rows, cols, _ in requests:
            entry = prepared_entries.get(key)
            if entry is None:
                entry = self._prepare(key, matrix, rows, cols, pinned)
                prepared_entries[key] = entry
            else:
                if entry[2:] != (rows, cols):
                    raise ValueError("cached Straydog matrix shape mismatch")
                self._cache.move_to_end(key)
                duplicate_hits += 1
            entries.append(entry)
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + duplicate_hits
            stats["hits"] = stats.get("hits", 0) + duplicate_hits
        count = len(entries)
        pointers = (_PTR * count)(*(entry[0] for entry in entries))
        kinds = (ct.c_int * count)(*(entry[1] for entry in entries))
        rows = (_U64 * count)(*(entry[2] for entry in entries))
        cols = (_U64 * count)(*(entry[3] for entry in entries))
        output = np.empty(sum(rows), np.float32)
        prepared = perf_counter() if stats is not None else 0
        if stats is not None:
            stats["prepare_seconds"] = stats.get("prepare_seconds", 0) + prepared - started
            stats["batches"] = stats.get("batches", 0) + 1
        self._check(
            self._lib.sd_matvec_batch(
                self._handle,
                pointers,
                kinds,
                rows,
                cols,
                count,
                inputs.ctypes.data_as(_F32_PTR),
                inputs.size,
                output.ctypes.data_as(_F32_PTR),
                output.size,
            )
        )
        if stats is not None:
            stats["matvec_seconds"] = stats.get("matvec_seconds", 0) + perf_counter() - prepared
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite values")
        offsets = np.cumsum([0] + list(rows))
        return [output[offsets[i] : offsets[i + 1]] for i in range(count)]

    def configure_gemm(self, *, minimum_batch: int = 8, precision: str = "fp32") -> None:
        """Select measured GEMM crossover and explicit FP16-input/FP32-accumulation mode."""
        if not 2 <= minimum_batch <= 257 or precision not in {"fp32", "fp16"}:
            raise ValueError("invalid Straydog GEMM configuration")
        self._check(self._lib.sd_gemm_options(self._handle, minimum_batch, int(precision == "fp16")))

    def hyperconnection(self, prefix: str, values: np.ndarray, norm: np.ndarray, matrices: list[tuple]):
        """Keep normalization, three projections and residual mixing on one GPU stream."""
        values, norm = np.ascontiguousarray(values, dtype=np.float32), np.ascontiguousarray(norm, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != 10240
            or not 1 <= len(values) <= 4096
            or norm.shape != (10240,)
            or len(matrices) not in {2, 3}
            or not np.isfinite(values).all()
            or not np.isfinite(norm).all()
        ):
            raise ValueError("invalid Straydog HyperConnection geometry")
        physical = 1024 if self.logical_capacity == 4096 else 256
        if len(values) > physical:
            pieces = [
                self.hyperconnection(prefix, values[first : first + physical], norm, matrices)
                for first in range(0, len(values), physical)
            ]
            return (
                np.concatenate([piece[0] for piece in pieces]),
                np.concatenate([piece[1] for piece in pieces]) if len(matrices) == 3 else None,
            )
        pinned = {item[0] for item in matrices} | {"@norm:" + prefix}
        entries = [self._prepare("@norm:" + prefix, lambda: (norm, 0), 1, 10240, pinned)]
        entries += [self._prepare(key, matrix, rows, cols, pinned) for key, matrix, rows, cols in matrices]
        pointers = (_PTR * 4)(*(entry[0] for entry in entries), *([None] if len(entries) == 3 else []))
        kinds = (ct.c_int * 4)(*(entry[1] for entry in entries), *([0] if len(entries) == 3 else []))
        inject = len(matrices) == 3
        output = np.empty(len(values) * (2564 if inject else 2560), np.float32)
        started = perf_counter() if self.stats is not None else 0
        self._check(
            self._lib.sd_hyperconnection(
                self._handle,
                pointers,
                kinds,
                len(values),
                int(inject),
                values.ctypes.data_as(_F32_PTR),
                output.ctypes.data_as(_F32_PTR),
            )
        )
        if self.stats is not None:
            self.stats["hyperconnection_seconds"] = (
                self.stats.get("hyperconnection_seconds", 0) + perf_counter() - started
            )
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite HyperConnection values")
        split = len(values) * 2560
        return output[:split].reshape(-1, 2560), output[split:].reshape(-1, 4) if inject else None

    def _state(self, key: str, rows: int, cols: int, *, host: bool = False):
        entry = self._states.get(key)
        if entry is None:
            if host:
                # Mapped, page-locked host memory: kernels use it in place; outside the VRAM budget.
                pointer = self._lib.sd_host_state(self._handle, rows * cols * 4)
                if not pointer:
                    raise RuntimeError(self._error())
                entry = (pointer, 0, rows, cols)
            else:
                entry = self._prepare(key, lambda: (np.zeros(rows * cols, np.float32), 0), rows, cols,
                                      set(self._states))
            self._states[key] = entry
        elif entry[2:] != (rows, cols):
            raise ValueError("Straydog state geometry changed")
        return entry

    def host_state_supported(self) -> bool:
        """Whether sequence state can live in mapped host memory that the kernels read over the bus."""
        result = self._lib.sd_host_state_supported(self._handle)
        if result < 0:
            raise RuntimeError(self._error())
        return bool(result)

    def _model_state_plan(
        self, context: int, *, attention: bool, layers=range(48), prefill: int = 1024, drafter: bool = False,
        host_attention: bool = False,
    ) -> list[tuple[str, int, int, bool]]:
        """(key, rows, cols, in host memory) for every sequence state of a placement."""
        if drafter:
            layers = [*layers, 48]
        plan = []
        if self.batch_capacity == 256:
            plan.append(("@moe:workspace", min(context, prefill) * (11 * 4480 + 22) + 65536, 1, False))
            if attention:
                plan.append(("@scratch:qsa", 256 * (context // 4 + 1), 1, False))
        for layer in layers:
            if layer == 1 and self.output_capacity >= 256 * 33280:
                plan.append(("@ple:blk.1.", 10, 10240, False))
            if layer % 4 != 3 and layer < 48:
                plan.append((f"@delta:blk.{layer}.", 48 * 128, 128, False))
                if self.batch_capacity == 256:
                    plan.append((f"@conv:blk.{layer}.", 4, 10240, False))
            elif attention:
                plan.append((f"@qsa:blk.{layer}.", context * 4, 256, host_attention))
                plan.append((f"@indexer:blk.{layer}.", (context // 4 + 1) * 128 + 512, 1, False))
        if attention and host_attention and self.batch_capacity == 256:
            # Reuse up to a sixteenth of the budget: one layer's rows during prefill,
            # or equal per-layer prefix mirrors during decode. The allocation size is unchanged.
            rows = min(context, self.memory()[1] // 16 // 4096)
            if rows >= 256:
                plan.append(("@kv:stage", rows * 4, 256, False))
        return plan

    def reserve_model_state(
        self, context: int, *, attention: bool, layers=range(48), prefill: int = 1024, drafter: bool = False,
        host_attention: bool = False,
    ) -> None:
        """Allocate and touch all sequence state (and the MTP layer's) before advertising model capacity.

        ``host_attention`` places the attention K/V in mapped host memory (it does not fit the VRAM budget).
        """
        self._prefill_capacity = min(context, prefill)
        for key, rows, cols, host in self._model_state_plan(
            context, attention=attention, layers=layers, prefill=prefill, drafter=drafter,
            host_attention=host_attention,
        ):
            entry = self._state(key, rows, cols, host=host)
            if key == "@kv:stage":
                self._check(self._lib.sd_kv_stage(self._handle, entry[0], rows // 4))

    def planned_state_bytes(
        self, context: int, *, attention: bool, layers=range(48), prefill: int = 1024, drafter: bool = False,
        host_attention: bool = False,
    ) -> int:
        """VRAM a placement would still allocate: its sequence state, then (fully resident) the decoder workspace."""
        plan = self._model_state_plan(context, attention=attention, layers=layers, prefill=prefill, drafter=drafter,
                                      host_attention=host_attention)
        states = sum(rows * cols * 4 + _ALLOCATION_TAIL for key, rows, cols, host in plan
                     if not host and key not in self._states)
        return states + _decoder_workspace_bytes(min(context, prefill)) + _ALLOCATION_TAIL

    def sparse_attention(self, prefix, values, matrices, vectors, *, position, context):
        """Keep QSA norms, RoPE, causal pooling and exact top-512 selection on the GPU."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        expected = [(12288, 2560), (512, 2560), (512, 2560), (512, 2560), (128, 2560), (2560, 6144)]
        sizes = [256, 256, 128, 128]
        arrays = [np.ascontiguousarray(value, dtype=np.float32) for _key, value in vectors]
        if (
            values.ndim != 2 or values.shape[1] != 2560 or not 1 <= len(values) <= 4096
            or not 1 <= context <= 262144 or not 0 <= position <= context - len(values)
            or len(matrices) != 6 or len(vectors) != 4 or not np.isfinite(values).all()
            or any(item[2:] != shape for item, shape in zip(matrices, expected, strict=True))
            or any(array.shape != (size,) or not np.isfinite(array).all()
                   for array, size in zip(arrays, sizes, strict=True))
        ):
            raise ValueError("invalid Straydog fused QSA geometry")
        kv = self._state("@qsa:" + prefix, context * 4, 256)
        index = self._state("@indexer:" + prefix, (context // 4 + 1) * 128 + 512, 1)
        scratch = self._state("@scratch:qsa", 256 * (context // 4 + 1), 1)
        pinned = {item[0] for item in matrices} | {"@vector:" + item[0] for item in vectors}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        for (key, _value), array in zip(vectors, arrays, strict=True):
            entries.append(self._prepare("@vector:" + key, lambda array=array: (array, 0), 1, array.size, pinned))
        pointers = (_PTR * 10)(*(entry[0] for entry in entries))
        kinds = (ct.c_int * 10)(*(entry[1] for entry in entries))
        output = np.empty_like(values)
        started = perf_counter() if self.stats is not None else 0
        for first in range(0, len(values), 256):
            incoming, target = values[first : first + 256], output[first : first + 256]
            self._check(self._lib.sd_qsa_sequence(
                self._handle, pointers, kinds, kv[0], index[0], scratch[0], context,
                position + first, len(incoming), incoming.ctypes.data_as(_F32_PTR), target.ctypes.data_as(_F32_PTR),
            ))
        if self.stats is not None:
            self.stats["qsa_seconds"] = self.stats.get("qsa_seconds", 0) + perf_counter() - started
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite QSA output")
        return output

    def can_pin_model(self, requirements: dict[str, int], reserved: int | None = None) -> bool:
        """Allow full residency when the estimated weights fit with what is still to be allocated.

        ``reserved`` is VRAM a placement still needs (planned_state_bytes); by default, after the sequence
        state, the decoder workspace. The slack, max(512 MiB, 3 %), leaves room on the card for what the
        runtime allocates outside the budget (driver, BLAS, graphs: ~0.9 GiB on an R9700) when the budget
        is set close to the card's size.
        """
        used, budget = self.memory()
        if reserved is None:
            reserved = 0 if "@scratch:decoder" in self._states else _decoder_workspace_bytes(
                getattr(self, "_prefill_capacity", 1024))
        slack = max(512 * 1024**2, budget * 3 // 100)
        return self.batch_capacity == 256 and sum(requirements.values()) + reserved <= budget - used - slack

    def register_resident_moe(self, prefix, matrices, *, experts: int, first: int = 0, count: int | None = None,
                              shared: bool = True, helper: bool = False) -> None:
        """Pin three packed expert banks and create a bounded immutable native execution plan.

        Expert parallelism: the banks hold experts [first, first + count) only.
        The layer's owner keeps the shared expert. ``helper`` registers the part this device
        evaluates while its linked peer owns the layer (no shared expert; its three matrices
        are only described, not uploaded).
        """
        count = experts if count is None else count
        shapes = [(count * 640, 2560), (count * 640, 2560), (count * 2560, 640),
                  (640, 2560), (640, 2560), (2560, 640), (experts, 2560), (1, 2560)]
        plans = self._helper_plans if helper else self._resident_plans
        if (not 10 <= experts <= 512 or len(matrices) != 8 or prefix in plans or not 0 < count <= experts - first
                or first < 0 or helper == shared
                or any(item[2:] != shape for item, shape in zip(matrices, shapes, strict=True))):
            raise ValueError("invalid Straydog resident MoE geometry")
        workspace = self._states.get("@moe:workspace")
        if workspace is None:
            workspace = self._state("@moe:workspace", 1024 * (11 * 4480 + 22) + 65536, 1)
        capacity = (workspace[2] - 65536) // (11 * 4480 + 22)
        held = [item for index, item in enumerate(matrices) if shared or not 3 <= index < 6]
        pinned = {item[0] for item in held}
        entries = [self._prepare(key, factory, rows, cols, pinned) if shared or not 3 <= index < 6
                   else (None, factory()[1], rows, cols)
                   for index, (key, factory, rows, cols) in enumerate(matrices)]
        pointers = (_PTR * 8)(*(entry[0] for entry in entries))
        kinds = (ct.c_int * 8)(*(entry[1] for entry in entries))
        if first == 0 and count == experts and shared:
            plan = self._lib.sd_resident_moe_create(self._handle, pointers, kinds, experts, workspace[0], capacity)
        else:
            plan = self._lib.sd_resident_moe_part(self._handle, pointers, kinds, experts, first, count, int(shared),
                                                  workspace[0], capacity)
        if not plan:
            raise RuntimeError(self._error())
        if helper:
            self._check(self._lib.sd_helper_moe(self._handle, int(prefix.split(".")[1]), plan))
        plans[prefix] = plan
        self._resident_weights.update(pinned)

    def link_peer(self, peer: GpuBackend, rows: int) -> bool:
        """Link this device and ``peer`` for expert parallelism; whether they write each other's VRAM."""
        self._check(self._lib.sd_peer_attach(self._handle, peer._handle, rows))
        return self._lib.sd_peer_direct(self._handle) == 1

    def resident_moe(self, prefix, values) -> np.ndarray:
        """Route and evaluate the full top ten on the GPU using pinned packed banks."""
        if not self._handle:
            raise RuntimeError("Straydog GPU is closed")
        values = np.ascontiguousarray(values, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != 2560 or not 1 <= len(values) <= 4096
                or not np.isfinite(values).all()):
            raise ValueError("invalid Straydog resident MoE input")
        output = np.empty_like(values)
        started = perf_counter() if self.stats is not None else 0
        self._check(self._lib.sd_resident_moe(
            self._handle, self._resident_plans[prefix], len(values),
            values.ctypes.data_as(_F32_PTR), output.ctypes.data_as(_F32_PTR),
        ))
        if self.stats is not None:
            self.stats["resident_moe_seconds"] = self.stats.get("resident_moe_seconds", 0) + perf_counter() - started
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite resident MoE output")
        return output

    def ple(self, prefix, values, embeddings, matrices, vectors):
        """Preserve causal PLE state while computing its four streams on the owning GPU."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != 10240 or not 1 <= len(values) <= 4096
                or embeddings.shape != (len(values), 2560) or not np.isfinite(values).all()
                or not np.isfinite(embeddings).all() or len(matrices) != 2 or len(vectors) != 4
                or matrices[0][2:] != (10240, 2560) or matrices[1][2:] != (2560, 2560)):
            raise ValueError("invalid Straydog PLE geometry")
        arrays = [np.ascontiguousarray(value, dtype=np.float32) for _key, value in vectors]
        if any(array.shape != shape or not np.isfinite(array).all()
               for array, shape in zip(arrays, [(10240,), (10240,), (10240,), (10240, 4)], strict=True)):
            raise ValueError("invalid Straydog PLE vectors")
        state = self._state("@ple:" + prefix, 10, 10240)
        pinned = {item[0] for item in matrices} | {"@vector:" + key for key, _value in vectors}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        for (key, _value), array in zip(vectors, arrays, strict=True):
            rows, cols = array.shape if array.ndim == 2 else (1, array.size)
            entries.append(self._prepare("@vector:" + key, lambda array=array: (array, 0), rows, cols, pinned))
        pointers = (_PTR * 6)(*(entry[0] for entry in entries))
        kinds = (ct.c_int * 6)(*(entry[1] for entry in entries))
        output = np.empty_like(values)
        started = perf_counter() if self.stats is not None else 0
        for first in range(0, len(values), 256):
            incoming = values[first:first + 256]
            embedded, target = embeddings[first:first + 256], output[first:first + 256]
            self._check(self._lib.sd_ple(self._handle, pointers, kinds, state[0], len(incoming),
                                       embedded.ctypes.data_as(_F32_PTR), incoming.ctypes.data_as(_F32_PTR),
                                       target.ctypes.data_as(_F32_PTR)))
        if self.stats is not None:
            self.stats["ple_seconds"] = self.stats.get("ple_seconds", 0) + perf_counter() - started
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite PLE state")
        return output

    def register_decoder_layer(self, layer, matrices, *, context):
        """Pin the fixed layer operands for GPU-only decode and prefill spans."""
        linear = layer % 4 != 3 and layer < 48   # layer 48: the MTP draft layer (full attention)
        if (isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer <= 48
                or layer in self._decoder_layers or len(matrices) != (17 if linear else 18)):
            raise ValueError("invalid Straydog decoder layer")
        prefix = f"blk.{layer}."
        plan = self._resident_plans[prefix]
        workspace = self._state(
            "@scratch:decoder", _decoder_workspace_bytes(getattr(self, "_prefill_capacity", 1024)) // 4, 1)
        if linear:
            states = [self._state("@conv:" + prefix, 4, 10240)[0],
                      self._state("@delta:" + prefix, 48 * 128, 128)[0], None]
        else:
            states = [self._state("@qsa:" + prefix, context * 4, 256)[0],
                      self._state("@indexer:" + prefix, (context // 4 + 1) * 128 + 512, 1)[0],
                      self._state("@scratch:qsa", 256 * (context // 4 + 1), 1)[0]]
        pinned = {item[0] for item in matrices}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        pointers = (_PTR * len(entries))(*(entry[0] for entry in entries))
        kinds = (ct.c_int * len(entries))(*(entry[1] for entry in entries))
        self._check(self._lib.sd_decoder_layer(
            self._handle, layer, context, pointers, kinds, len(entries), (_PTR * 3)(*states), workspace[0], plan,
        ))
        self._resident_weights.update(pinned)
        self._decoder_layers.add(layer)

    def decoder_spans(self, start=1):
        """A single GPU owns the remaining contiguous decoder span."""
        return [(start, 48)]

    def decode_span(self, begin, end, values, *, position):
        """Execute contiguous resident layers with only the span boundary transferred to the CPU."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != 10240
                or not 1 <= len(values) <= getattr(self, "_prefill_capacity", 1024) or not np.isfinite(values).all()
                or not 0 <= begin < end <= 49 or not 0 <= position <= 262144 - len(values)
                or (len(values) > 1 and end - begin > 4)):
            raise ValueError("invalid Straydog decoder span")
        output = np.empty_like(values)
        started = perf_counter() if self.stats is not None else 0
        operands = (values.ctypes.data_as(_F32_PTR), output.ctypes.data_as(_F32_PTR))
        if len(values) == 1:
            self._check(self._lib.sd_decode_span(self._handle, begin, end, position, *operands))
        else:
            self._prefill(begin, end, position, len(values), operands[0], operands[1], len(values))
        if self.stats is not None:
            self.stats["decoder_span_seconds"] = self.stats.get("decoder_span_seconds", 0) + perf_counter() - started
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite decoder state")
        return output

    def _prefill(self, begin, end, position, steps, source, target, rows) -> None:
        timings = (ct.c_double * 4)() if self.stats is not None else None
        self._check(self._lib.sd_prefill_span(self._handle, begin, end, position, steps, source, target, rows, timings))
        if timings is not None:
            for category, value in zip(("hc", "gdn", "qsa", "moe"), timings, strict=True):
                key = "prefill_" + category + "_device_seconds"
                self.stats[key] = self.stats.get(key, 0) + value

    def prefill_layers(self, begin, end, values, *, position, steps, output_rows=0):
        """Advance device-resident prefill rows; transfer only explicit inputs and trailing outputs.

        ``values`` uploads new rows; ``None`` continues from the previous span on this GPU.
        The native span rejects non-finite state before returning control for cancellation.
        """
        if (not 0 <= begin < end <= 49 or end - begin > 4 or not 2 <= steps <= getattr(self, "_prefill_capacity", 1024)
                or not 0 <= position <= 262144 - steps or not 0 <= output_rows <= steps):
            raise ValueError("invalid Straydog resident prefill span")
        source = None
        if values is not None:
            values = np.ascontiguousarray(values, dtype=np.float32)
            if values.shape != (steps, 10240) or not np.isfinite(values).all():
                raise ValueError("invalid Straydog resident prefill input")
            source = values.ctypes.data_as(_F32_PTR)
        output = np.empty((output_rows, 10240), np.float32) if output_rows else None
        started = perf_counter() if self.stats is not None else 0
        self._prefill(begin, end, position, steps, source, output.ctypes.data_as(_F32_PTR) if output_rows else None,
                      output_rows)
        if self.stats is not None:
            self.stats["decoder_span_seconds"] = self.stats.get("decoder_span_seconds", 0) + perf_counter() - started
        return output

    def prefill_ple(self, prefix, embeddings, matrices, vectors, *, layer):
        """Apply PLE in place to the rows resident in ``layer``'s decoder workspace."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if (embeddings.ndim != 2 or embeddings.shape[1] != 2560 or not 2 <= len(embeddings) <= 4096
                or not np.isfinite(embeddings).all() or len(matrices) != 2 or len(vectors) != 4
                or matrices[0][2:] != (10240, 2560) or matrices[1][2:] != (2560, 2560)):
            raise ValueError("invalid Straydog PLE geometry")
        arrays = [np.ascontiguousarray(value, dtype=np.float32) for _key, value in vectors]
        if any(array.shape != shape or not np.isfinite(array).all()
               for array, shape in zip(arrays, [(10240,), (10240,), (10240,), (10240, 4)], strict=True)):
            raise ValueError("invalid Straydog PLE vectors")
        state = self._state("@ple:" + prefix, 10, 10240)
        pinned = {item[0] for item in matrices} | {"@vector:" + key for key, _value in vectors}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        for (key, _value), array in zip(vectors, arrays, strict=True):
            rows, cols = array.shape if array.ndim == 2 else (1, array.size)
            entries.append(self._prepare("@vector:" + key, lambda array=array: (array, 0), rows, cols, pinned))
        pointers = (_PTR * 6)(*(entry[0] for entry in entries))
        kinds = (ct.c_int * 6)(*(entry[1] for entry in entries))
        started = perf_counter() if self.stats is not None else 0
        self._check(self._lib.sd_prefill_ple(self._handle, pointers, kinds, state[0], len(embeddings),
                                             embeddings.ctypes.data_as(_F32_PTR), layer))
        if self.stats is not None:
            self.stats["ple_seconds"] = self.stats.get("ple_seconds", 0) + perf_counter() - started

    def moe(self, prefix, values, selected, probabilities, shared_mix, matrices) -> np.ndarray:
        """Evaluate every selected expert and the shared expert without host intermediates."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        selected = np.ascontiguousarray(selected, dtype=np.int32)
        probabilities = np.ascontiguousarray(probabilities, dtype=np.float32)
        shared_mix = np.ascontiguousarray(shared_mix, dtype=np.float32)
        experts = len(matrices) // 3 - 1
        if (
            values.ndim != 2 or values.shape[1] != 2560 or not 1 <= len(values) <= 4096
            or not 10 <= experts <= 512 or len(matrices) != 3 * experts + 3
            or selected.shape != (len(values), 10) or probabilities.shape != selected.shape
            or shared_mix.shape != (len(values), 1)
            or any(not np.isfinite(array).all() for array in (values, probabilities, shared_mix))
            or np.any(selected < 0) or np.any(selected >= experts)
            or np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0)
            or any(item[2:] != ((2560, 640) if i % 3 == 2 else (640, 2560))
                   for i, item in enumerate(matrices))
        ):
            raise ValueError("invalid Straydog MoE geometry or selection")
        workspace = self._states.get("@moe:workspace")
        if workspace is None:
            workspace = self._state("@moe:workspace", max(1024, len(values)) * (11 * 4480 + 22) + 65536, 1)
        if workspace[2] < len(values) * (11 * 4480 + 22) + 65536:
            raise ValueError("Straydog MoE exceeds reserved context")
        pinned = {item[0] for item in matrices}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        pointers = (_PTR * len(entries))(*(entry[0] for entry in entries))
        kinds = (ct.c_int * len(entries))(*(entry[1] for entry in entries))
        packed = np.ascontiguousarray(np.concatenate((probabilities, shared_mix), axis=1))
        output = np.empty_like(values)
        started = perf_counter() if self.stats is not None else 0
        self._check(self._lib.sd_moe(
            self._handle, pointers, kinds, experts, workspace[0], len(values),
            selected.ctypes.data_as(ct.POINTER(ct.c_int32)), packed.ctypes.data_as(_F32_PTR),
            values.ctypes.data_as(_F32_PTR), output.ctypes.data_as(_F32_PTR),
        ))
        if self.stats is not None:
            self.stats["moe_seconds"] = self.stats.get("moe_seconds", 0) + perf_counter() - started
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite MoE values")
        return output

    def gated_delta(self, prefix: str, values: np.ndarray, matrices: list[tuple], vectors: list[tuple]) -> np.ndarray:
        """Run causal convolution, recurrence, norm/gates and projections in bounded GPU chunks."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        expected = [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560), (2560, 6144)]
        shapes = [(10240, 4), (48,), (48,), (128,)]
        arrays = [np.ascontiguousarray(value, dtype=np.float32) for _key, value in vectors]
        if (
            values.ndim != 2
            or values.shape[1] != 2560
            or not 1 <= len(values) <= 4096
            or len(matrices) != 5
            or len(vectors) != 4
            or not np.isfinite(values).all()
            or any(item[2:] != shape for item, shape in zip(matrices, expected, strict=True))
            or any(
                array.shape != shape or not np.isfinite(array).all()
                for array, shape in zip(arrays, shapes, strict=True)
            )
        ):
            raise ValueError("invalid Straydog fused DeltaNet geometry")
        conv, recurrent = self._state("@conv:" + prefix, 4, 10240), self._state("@delta:" + prefix, 48 * 128, 128)
        pinned = {item[0] for item in matrices} | {"@vector:" + item[0] for item in vectors}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        for (key, _value), array in zip(vectors, arrays, strict=True):
            rows, cols = (array.shape[0], array.shape[1]) if array.ndim == 2 else (1, array.size)
            entries.append(self._prepare("@vector:" + key, lambda array=array: (array, 0), rows, cols, pinned))
        pointers = (_PTR * 9)(*(entry[0] for entry in entries))
        kinds = (ct.c_int * 9)(*(entry[1] for entry in entries))
        output = np.empty_like(values)
        started = perf_counter() if self.stats is not None else 0
        physical = 512 if self.logical_capacity == 4096 else 128
        for first in range(0, len(values), physical):
            inputs, target = values[first : first + physical], output[first : first + physical]
            self._check(
                self._lib.sd_gdn_sequence(
                    self._handle,
                    pointers,
                    kinds,
                    conv[0],
                    recurrent[0],
                    len(inputs),
                    inputs.ctypes.data_as(_F32_PTR),
                    target.ctypes.data_as(_F32_PTR),
                )
            )
        if self.stats is not None:
            self.stats["gated_delta_seconds"] = self.stats.get("gated_delta_seconds", 0) + perf_counter() - started
            self.stats["delta_tokens"] = self.stats.get("delta_tokens", 0) + len(values)
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite DeltaNet values")
        return output

    def delta_sequence(
        self, state_key: str, query: np.ndarray, key: np.ndarray, value: np.ndarray, decay: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Scan up to 4096 DeltaNet tokens with persistent GPU state and bounded I/O."""
        if not self._handle:
            raise RuntimeError("Straydog GPU is closed")
        arrays = [np.ascontiguousarray(array, dtype=np.float32) for array in (query, key, value, decay, beta)]
        query, key, value, decay, beta = arrays
        if value.ndim != 3 or query.ndim != 3:
            raise ValueError("invalid Straydog DeltaNet geometry")
        steps, heads, dimension = value.shape
        if (
            not 1 <= steps <= 4096
            or not 1 <= heads <= 64
            or not 1 <= dimension <= 128
            or query.shape != key.shape
            or query.shape[0] != steps
            or query.shape[2] != dimension
            or not query.shape[1]
            or heads % query.shape[1]
            or decay.shape != (steps, heads)
            or beta.shape != (steps, heads)
            or any(not np.isfinite(array).all() for array in arrays)
        ):
            raise ValueError("invalid Straydog DeltaNet geometry")
        query = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
        key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
        query = np.tile(query, (1, heads // query.shape[1], 1)) / np.float32(np.sqrt(dimension))
        key = np.tile(key, (1, heads // key.shape[1], 1))
        factors = np.exp(decay)
        cache_key = "@delta:" + state_key
        entry = self._state(cache_key, heads * dimension, dimension)
        stats = self.stats
        started = perf_counter() if stats is not None else 0
        output = np.empty(value.shape, np.float32)
        chunk = 128 if self.batch_capacity == 256 else 16
        for first in range(0, steps, chunk):
            last = min(first + chunk, steps)
            packed = np.concatenate([array[first:last].reshape(-1) for array in (query, key, value, factors, beta)])
            target = output[first:last]
            self._check(
                self._lib.sd_delta_sequence(
                    self._handle,
                    entry[0],
                    heads,
                    dimension,
                    last - first,
                    packed.ctypes.data_as(_F32_PTR),
                    packed.size,
                    target.ctypes.data_as(_F32_PTR),
                    target.size,
                )
            )
        if stats is not None:
            stats["delta_seconds"] = stats.get("delta_seconds", 0) + perf_counter() - started
            stats["delta_tokens"] = stats.get("delta_tokens", 0) + steps
            stats["delta_state_bytes"] = sum(
                item[2] * item[3] * 4 for name, item in self._states.items() if name.startswith("@delta:")
            )
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite recurrent values")
        return output

    def attention_sequence(
        self,
        state_key: str,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        selected: list[np.ndarray],
        *,
        position: int,
        context: int,
    ) -> np.ndarray:
        """Run causal GQA over explicit QSA selections in persistent GPU K/V buffers."""
        arrays = [np.ascontiguousarray(array, dtype=np.float32) for array in (query, key, value)]
        query, key, value = arrays
        steps = len(query)
        if (
            query.shape != (steps, 24, 256)
            or key.shape != (steps, 2, 256)
            or key.shape != value.shape
            or not 1 <= steps <= 4096
            or len(selected) != steps
            or not 1 <= context <= 262144
            or not 0 <= position <= context - steps
            or any(not np.isfinite(array).all() for array in arrays)
        ):
            raise ValueError("invalid Straydog attention geometry")
        counts = []
        for step, indices in enumerate(selected):
            if (
                not isinstance(indices, np.ndarray)
                or indices.ndim != 1
                or not 1 <= len(indices) <= 2052
                or indices.dtype.kind not in "iu"
                or np.any(indices < 0)
                or np.any(indices > position + step)
            ):
                raise ValueError("invalid Straydog attention selection")
            counts.append(len(indices))
        if steps > 256:
            return np.concatenate(
                [
                    self.attention_sequence(
                        state_key,
                        query[first : first + 256],
                        key[first : first + 256],
                        value[first : first + 256],
                        selected[first : first + 256],
                        position=position + first,
                        context=context,
                    )
                    for first in range(0, steps, 256)
                ]
            )
        width = max(counts)
        padded = np.zeros((steps, width), np.float32)
        for step, indices in enumerate(selected):
            padded[step, : len(indices)] = indices
        packed = np.concatenate(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1), padded.reshape(-1), np.array(counts, np.float32)]
        )
        cache_key = "@qsa:" + state_key
        entry = self._state(cache_key, context * 4, 256)
        output = np.empty(query.shape, np.float32)
        stats = self.stats
        started = perf_counter() if stats is not None else 0
        self._check(
            self._lib.sd_attention_sequence(
                self._handle,
                entry[0],
                context,
                position,
                steps,
                width,
                packed.ctypes.data_as(_F32_PTR),
                packed.size,
                output.ctypes.data_as(_F32_PTR),
                output.size,
            )
        )
        if stats is not None:
            stats["attention_seconds"] = stats.get("attention_seconds", 0) + perf_counter() - started
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite attention values")
        return output

    def reset_state(self) -> None:
        """Clear every owned recurrent state without transferring it through host RAM."""
        for key, entry in self._states.items():
            # MoE scratch is completely overwritten on every call; it carries no sequence state.
            if not key.startswith(("@moe:", "@scratch:")):
                self._check(self._lib.sd_zero(self._handle, entry[0]))

    def reserve_checkpoints(self, count: int) -> int:
        """Allocate ``count`` page-locked host copies of the owned recurrent sequence state; their bytes.

        DeltaNet state, its convolution and PLE history. Attention K/V and the QSA indexer are
        indexed by position and need no copy: a checkpoint is taken at a multiple of four tokens,
        where the indexer's partial block holds nothing yet, and rows past a restored position are
        rewritten (their K/V mirrors invalidated) before use.
        """
        keys = [key for key in self._states if key.startswith(("@delta:", "@conv:", "@ple:"))]
        self._checkpoint_states = (_PTR * len(keys))(*[self._states[key][0] for key in keys])
        size = sum(self._states[key][2] * self._states[key][3] * 4 for key in keys)
        self._checkpoints = []
        for _slot in range(count if keys else 0):
            buffer = self._lib.sd_checkpoint_buffer(self._handle, size)
            if not buffer:
                raise RuntimeError(self._error())
            self._checkpoints.append(buffer)
        return size * len(self._checkpoints)

    def save_checkpoint(self, slot: int) -> None:
        if self._checkpoints:
            self._check(self._lib.sd_checkpoint_copy(
                self._handle, 1, self._checkpoint_states, len(self._checkpoint_states), self._checkpoints[slot]))

    def load_checkpoint(self, slot: int) -> None:
        if self._checkpoints:
            self._check(self._lib.sd_checkpoint_copy(
                self._handle, 0, self._checkpoint_states, len(self._checkpoint_states), self._checkpoints[slot]))

    def _read_delta_state(self, state_key: str) -> np.ndarray:
        entry = self._states["@delta:" + state_key]
        dim = entry[3]
        output = np.empty((entry[2] // dim, dim, dim), np.float32)
        self._check(self._lib.sd_download(self._handle, entry[0], output.ctypes.data, output.nbytes))
        return output

    def close(self) -> None:
        """Release every owned allocation and the GPU stream."""
        if self._handle:
            self._lib.sd_destroy(self._handle)
            self._handle = None
            self._cache.clear()
            self._experts.clear()
            self._resident_plans.clear()
            self._helper_plans.clear()
            self._resident_weights.clear()
            self._decoder_layers.clear()
            self._states.clear()

    def __enter__(self) -> GpuBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
