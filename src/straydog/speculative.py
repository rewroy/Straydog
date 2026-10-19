"""Speculative decoding operations for one Straydog GPU backend.

Up to four consecutive tokens are verified per call with the one-token arithmetic,
so accepted tokens match token-by-token decode bitwise; DeltaNet state is restored
and replayed for a partial acceptance. The MTP draft layer reuses the decoder layer
machinery as layer 48. The ABI is implemented in native/crates/inference/gpu/verify_span.hpp.
"""

from __future__ import annotations

import ctypes as ct
from typing import Any

import numpy as np

_PTR = ct.c_void_p
_U64 = ct.c_uint64
_F32_PTR = ct.POINTER(ct.c_float)

VERIFY_MAX = 4
VOCABULARY = 248320

SPECULATIVE_SIGNATURES = {
    "sd_verify_span": ([_PTR, _U64, _U64, _U64, _U64, _F32_PTR, _F32_PTR], ct.c_int),
    "sd_verify_commit": ([_PTR, _U64, _U64, _U64], ct.c_int),
    "sd_output_head": (
        [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64, _U64, _F32_PTR, ct.POINTER(ct.c_int32), _F32_PTR],
        ct.c_int,
    ),
    "sd_output_head_split": (
        [_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64, _U64, _U64, _PTR, _F32_PTR, ct.POINTER(ct.c_int32),
         _F32_PTR],
        ct.c_int,
    ),
    "sd_mtp_prologue": ([_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _U64, _F32_PTR, _F32_PTR, _F32_PTR], ct.c_int),
    "sd_ple_rows": ([_PTR, ct.POINTER(_PTR), ct.POINTER(ct.c_int), _PTR, _U64, _F32_PTR, _F32_PTR, _F32_PTR], ct.c_int),
    "sd_copy_allocation": ([_PTR, _PTR, _PTR], ct.c_int),
}


def _rows(values: np.ndarray, width: int) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    if (values.ndim != 2 or values.shape[1] != width or not 1 <= len(values) <= VERIFY_MAX
            or not np.isfinite(values).all()):
        raise ValueError("invalid Straydog speculative rows")
    return values


class SpeculativeBackendMixin:
    """GPU backend operations for MTP drafting and exact multi-token verification."""

    # Provided by GpuBackend.
    _lib: Any
    _handle: Any
    _states: dict
    _resident_weights: set
    _prepare: Any
    _state: Any
    _check: Any

    def _operands(self, matrices: list[tuple]) -> tuple[Any, Any]:
        pinned = {item[0] for item in matrices}
        entries = [self._prepare(key, factory, rows, cols, pinned) for key, factory, rows, cols in matrices]
        self._resident_weights.update(pinned)
        pointers = (_PTR * len(entries))(*(entry[0] for entry in entries))
        return pointers, (ct.c_int * len(entries))(*(entry[1] for entry in entries))

    def verify_span(self, begin: int, end: int, values: np.ndarray, *, position: int) -> np.ndarray:
        """Run up to four consecutive tokens through resident layers [begin, end)."""
        values = _rows(values, 10240)
        if not 0 <= begin < end <= 49 or not 0 <= position <= 262144 - len(values):
            raise ValueError("invalid Straydog verification span")
        output = np.empty_like(values)
        self._check(self._lib.sd_verify_span(self._handle, begin, end, position, len(values),
                                             values.ctypes.data_as(_F32_PTR), output.ctypes.data_as(_F32_PTR)))
        if not np.isfinite(output).all():
            raise RuntimeError("Straydog GPU produced non-finite decoder state")
        return output

    def verify_commit(self, begin: int, end: int, kept: int) -> None:
        """Keep the first ``kept`` tokens of the last verification of this span."""
        self._check(self._lib.sd_verify_commit(self._handle, begin, end, kept))

    def output_head(self, matrices: list[tuple], values: np.ndarray, *, logits: bool = False):
        """HyperConnection mix and vocabulary head; returns each row's first maximum (and logits)."""
        values = _rows(values, 10240)
        if len(matrices) != 4 or matrices[3][3] != 2560 or not 1 <= matrices[3][2] <= VOCABULARY:
            raise ValueError("invalid Straydog output head operands")
        vocabulary = matrices[3][2]
        pointers, kinds = self._operands(matrices)
        tokens = np.empty(len(values), np.int32)
        scores = np.empty((len(values), vocabulary), np.float32) if logits else None
        self._check(self._lib.sd_output_head(
            self._handle, pointers, kinds, len(values), vocabulary, values.ctypes.data_as(_F32_PTR),
            tokens.ctypes.data_as(ct.POINTER(ct.c_int32)), None if scores is None else scores.ctypes.data_as(_F32_PTR),
        ))
        return tokens if scores is None else (tokens, scores)

    def output_head_split(self, matrices: list[tuple], values: np.ndarray, *, split: int, peer_head: int,
                          logits: bool = False):
        """The same head with rows [0, split) evaluated by the linked peer from ``peer_head``."""
        values = _rows(values, 10240)
        if len(matrices) != 4 or matrices[3][3] != 2560 or not 0 < split < matrices[3][2] <= VOCABULARY:
            raise ValueError("invalid Straydog output head operands")
        vocabulary = matrices[3][2]
        pointers, kinds = self._operands(matrices)
        tokens = np.empty(len(values), np.int32)
        scores = np.empty((len(values), vocabulary), np.float32) if logits else None
        self._check(self._lib.sd_output_head_split(
            self._handle, pointers, kinds, len(values), vocabulary, split, peer_head, values.ctypes.data_as(_F32_PTR),
            tokens.ctypes.data_as(ct.POINTER(ct.c_int32)), None if scores is None else scores.ctypes.data_as(_F32_PTR),
        ))
        return tokens if scores is None else (tokens, scores)

    def mtp_prologue(self, matrices: list[tuple], embeddings: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """Fuse token embeddings with backbone hidden rows into MTP layer inputs."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        if (embeddings.ndim != 2 or embeddings.shape[1] != 2560 or hidden.shape != (len(embeddings), 10240)
                or not 1 <= len(hidden) <= 512 or len(matrices) != 4
                or not np.isfinite(embeddings).all() or not np.isfinite(hidden).all()):
            raise ValueError("invalid Straydog MTP prologue")
        pointers, kinds = self._operands(matrices)
        output = np.empty_like(hidden)
        self._check(self._lib.sd_mtp_prologue(self._handle, pointers, kinds, len(hidden),
                                              embeddings.ctypes.data_as(_F32_PTR), hidden.ctypes.data_as(_F32_PTR),
                                              output.ctypes.data_as(_F32_PTR)))
        return output

    def ple_rows(self, prefix, values, embeddings, matrices, vectors) -> np.ndarray:
        """PLE for up to four rows with the one-row arithmetic (speculative verification)."""
        values, embeddings = _rows(values, 10240), _rows(embeddings, 2560)
        state = self._states["@ple:" + prefix]
        arrays = [np.ascontiguousarray(value, dtype=np.float32) for _key, value in vectors]
        operands = list(matrices) + [
            ("@vector:" + key, lambda array=array: (array, 0), *(array.shape if array.ndim == 2 else (1, array.size)))
            for (key, _value), array in zip(vectors, arrays, strict=True)
        ]
        pointers, kinds = self._operands(operands)
        output = np.empty_like(values)
        self._check(self._lib.sd_ple_rows(self._handle, pointers, kinds, state[0], len(values),
                                          embeddings.ctypes.data_as(_F32_PTR), values.ctypes.data_as(_F32_PTR),
                                          output.ctypes.data_as(_F32_PTR)))
        return output

    def copy_state(self, target: str, source: str) -> None:
        """Copy one owned state allocation onto another of the same geometry."""
        source_entry = self._states[source]
        target_entry = self._states.get(target) or self._state(target, *source_entry[2:])
        self._check(self._lib.sd_copy_allocation(self._handle, target_entry[0], source_entry[0]))
