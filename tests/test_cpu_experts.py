"""CPU expert evaluation is compared with an independent NumPy dequantization."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.cpu import CpuExperts
from straydog.library import load_library
from straydog.numeric import silu
from straydog.weights import dequantize
from tests.blocks import set_scales


def _bank(rng, rows, cols):
    raw = rng.integers(0, 256, (rows * cols // 64, 18), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(rng.uniform(0.002, 0.03, rows * cols // 64).astype(np.float16).tobytes(),
                               np.uint8).reshape(-1, 2)
    return raw.reshape(-1)


@pytest.fixture
def library():
    path = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not path:
        pytest.skip("set STRAYDOG_GPU_LIBRARY to the compiled Straydog library")
    return Path(path)


@pytest.mark.parametrize("threads,count", [(1, 1), (3, 7), (0, 10)])
def test_cpu_experts_match_independent_reference(library, threads, count):
    rng = np.random.default_rng(31 + count)
    hidden, inter = 256, 128
    experts = [(_bank(rng, inter, hidden), _bank(rng, inter, hidden), _bank(rng, hidden, inter)) for _ in range(count)]
    weights = rng.uniform(0.01, 0.3, count).astype(np.float32)
    x = rng.standard_normal(hidden).astype(np.float32)
    expected = np.zeros(hidden, np.float32)
    for (gate, up, down), weight in zip(experts, weights, strict=True):
        g = dequantize(gate, 42).reshape(inter, hidden) @ x
        u = dequantize(up, 42).reshape(inter, hidden) @ x
        expected += weight * (dequantize(down, 42).reshape(hidden, inter) @ (silu(g) * u))
    with CpuExperts(library, threads=threads) as engine:
        assert engine.isa in {"avx512-f32", "avx2-fma", "scalar"}
        np.testing.assert_allclose(engine.run(experts, weights, x, inter=inter), expected, rtol=2e-4, atol=2e-5)
        with pytest.raises(ValueError, match="slice"):
            engine.run([(experts[0][0][:-1], experts[0][1], experts[0][2])], weights[:1], x, inter=inter)
    with pytest.raises(RuntimeError, match="closed"):
        engine.run(experts, weights, x, inter=inter)


@pytest.mark.parametrize("count", [1, 3, 7, 10])
@pytest.mark.parametrize("hidden,inter", [(192, 320), (2560, 640)])
def test_q2_expert_row_tails_and_partitions_are_bitwise_identical(library, count, hidden, inter):
    """Odd worker slices and expert counts keep each row and the expert sum in the same order."""
    rng = np.random.default_rng(512 + count)
    experts = [(_bank(rng, inter, hidden), _bank(rng, inter, hidden), _bank(rng, hidden, inter))
               for _ in range(count)]
    weights = rng.uniform(-0.1, 0.3, count).astype(np.float32)
    x = rng.normal(0, 0.5, hidden).astype(np.float32)
    with CpuExperts(library, threads=1) as engine:
        expected = engine.run(experts, weights, x, inter=inter).tobytes()
    for threads in (3, 6, 8):
        with CpuExperts(library, threads=threads) as engine:
            for activation in (x, -x, np.zeros_like(x)):
                # Re-run through one worker as well: zeros exercise signed-zero products.
                actual = engine.run(experts, weights, activation, inter=inter)
                if activation is x:
                    assert actual.tobytes() == expected
                else:
                    with CpuExperts(library, threads=1) as reference:
                        assert actual.tobytes() == reference.run(experts, weights, activation, inter=inter).tobytes()
            stats = engine.stats()
            assert stats["calls"] == 3 and stats["experts"] == 3 * count


def _packed(rng, kind, rows, cols):
    from straydog.formats import BLOCKS

    values, size = BLOCKS[kind]
    raw = rng.integers(0, 256, (rows * cols // values, size), dtype=np.uint8)
    return set_scales(raw, kind, rng.uniform(0.0005, 0.004, len(raw))).reshape(-1)


@pytest.mark.parametrize("kinds", [(16, 16, 20), (17, 17, 42), (22, 22, 20), (18, 18, 8), (21, 21, 20), (23, 23, 42),
                                   (29, 29, 42), (22, 22, 42)])
def test_cpu_experts_decode_iq_bundle_formats(library, kinds):
    """IQ1/IQ2/IQ3/IQ4 gate-up with IQ4_NL / Q2_0 / Q8_0 down, as in the IQ2_XS and IQ3 bundles."""
    rng = np.random.default_rng(sum(kinds))
    hidden, inter = 512, 128
    experts = [(_packed(rng, kinds[0], inter, hidden), _packed(rng, kinds[1], inter, hidden),
                _packed(rng, kinds[2], hidden, inter)) for _ in range(3)]
    weights = rng.uniform(0.05, 0.3, 3).astype(np.float32)
    x = rng.standard_normal(hidden).astype(np.float32)
    expected = np.zeros(hidden, np.float64)
    for (gate, up, down), weight in zip(experts, weights, strict=True):
        g = dequantize(gate, kinds[0]).reshape(inter, hidden).astype(np.float64) @ x
        u = dequantize(up, kinds[1]).reshape(inter, hidden).astype(np.float64) @ x
        product = (g / (1 + np.exp(-g))) * u
        expected += weight * (dequantize(down, kinds[2]).reshape(hidden, inter).astype(np.float64) @ product)
    with CpuExperts(library, threads=2) as engine:
        actual = engine.run(experts, weights, x, inter=inter, kinds=kinds)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5 * np.abs(expected).max())



@pytest.mark.parametrize("kind", [8, 16, 17, 18, 20, 21, 22, 23, 29])
def test_wide_block_decoders_match_portable_bitwise(library, kind):
    import ctypes as ct

    from straydog.formats import BLOCKS, load_iq_tables

    lib = load_library(library)
    load_iq_tables(lib)
    lib.sd_cpu_decode.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
    values, size = BLOCKS[kind]
    raw = _packed(np.random.default_rng(kind), kind, 64, values)   # 64 blocks
    raw[0:2] = 0   # a zero scale: signed zeros must agree too
    for block in raw.reshape(-1, size):
        portable, wide = np.empty(values, np.float32), np.empty(values, np.float32)
        assert lib.sd_cpu_decode(kind, 0, block.ctypes.data, portable.ctypes.data) == 0
        if lib.sd_cpu_decode(kind, 1, block.ctypes.data, wide.ctypes.data):
            pytest.skip("no AVX-512 on this host")
        assert portable.tobytes() == wide.tobytes()
        np.testing.assert_array_equal(portable, dequantize(block, kind))


@pytest.mark.parametrize("kind", [8, 16, 17, 18, 20, 21, 22, 23, 29])
def test_fused_row_kernels_repeat_the_decoded_rows_bitwise(library, kind):
    """AVX-512 / AVX2 multi-row kernels: per row the same values, FMA order and lane reduction as the scalar path."""
    import ctypes as ct

    from straydog.formats import BLOCKS, load_iq_tables

    lib = load_library(library)
    load_iq_tables(lib)
    lib.sd_cpu_rows.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_uint64, ct.c_uint32, ct.c_void_p, ct.c_uint64,
                                ct.c_void_p]
    rng = np.random.default_rng(100 + kind)
    rows, cols = 11, 512   # rows run as 4 + 4 + 3
    raw = _packed(rng, kind, rows, cols).reshape(rows, -1)
    raw[1, :BLOCKS[kind][1]] = 0   # a zero block: signed zeros must agree too
    x = rng.standard_normal(cols).astype(np.float32)
    portable = np.empty(rows, np.float32)
    assert lib.sd_cpu_rows(kind, 0, raw.ctypes.data, raw.shape[1], rows, x.ctypes.data, cols, portable.ctypes.data) == 0
    for level in (1, 2):
        fused = np.empty(rows, np.float32)
        if lib.sd_cpu_rows(kind, level, raw.ctypes.data, raw.shape[1], rows, x.ctypes.data, cols, fused.ctypes.data):
            continue   # that instruction set is absent here
        assert fused.tobytes() == portable.tobytes(), level
    expected = dequantize(raw.reshape(-1), kind).reshape(rows, cols).astype(np.float64) @ x
    np.testing.assert_allclose(portable, expected, rtol=1e-4, atol=1e-5 * np.abs(expected).max())
