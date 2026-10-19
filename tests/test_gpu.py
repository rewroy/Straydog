from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend
from straydog.weights import dequantize
from tests.blocks import set_scales

pytestmark = pytest.mark.gpu


@pytest.fixture
def backend():
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("set STRAYDOG_GPU_LIBRARY to the compiled Straydog HIP/CUDA library")
    with GpuBackend(Path(library), budget_bytes=8 * 1024**2) as device:
        yield device


def test_gpu_float_formats_match_numpy_reference(backend) -> None:
    rng = np.random.default_rng(712)
    matrix = rng.normal(size=(17, 128)).astype(np.float32)
    vector = rng.normal(size=128).astype(np.float32)
    for kind in (0, 1, 30):
        if kind == 0:
            raw = matrix.view(np.uint8)
        elif kind == 1:
            raw = matrix.astype(np.float16).view(np.uint8)
        else:
            raw = (matrix.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
        reference = dequantize(raw.reshape(-1), kind).reshape(matrix.shape) @ vector
        result = backend.project(str(kind), lambda r=raw, k=kind: (r, k), 17, 128, vector)
        np.testing.assert_allclose(result, reference, rtol=3e-5, atol=1e-5)
    assert backend.info()["total_bytes"] > 0
    used, budget = backend.memory()
    assert 0 < used <= budget


def test_gpu_q2_signed_codes_match_scalar_reference(backend) -> None:
    rng = np.random.default_rng(23)
    raw = rng.integers(0, 256, (19, 2, 18), dtype=np.uint8)
    raw[:, :, :2] = np.array([0.25], dtype=np.float16).view(np.uint8)
    vector = rng.normal(size=128).astype(np.float32)
    result = backend.project("q2", lambda: (raw, 42), 19, 128, vector)
    reference = dequantize(raw.reshape(-1), 42).reshape(19, 128) @ vector
    np.testing.assert_allclose(result, reference, rtol=3e-5, atol=1e-5)

    # A hit must not re-read the weights from the host.
    def unavailable():
        raise AssertionError("weight cache was missed")

    np.testing.assert_array_equal(backend.project("q2", unavailable, 19, 128, vector), result)


def test_gpu_lru_budget_and_closed_state(backend) -> None:
    matrix = np.ones((512, 1024), np.float32)
    vector = np.ones(1024, np.float32)
    for index in range(5):
        result = backend.project(str(index), lambda: (matrix, 0), 512, 1024, vector)
        np.testing.assert_array_equal(result, np.full(512, 1024))
        used, budget = backend.memory()
        assert used <= budget
    with pytest.raises(ValueError, match="size mismatch"):
        backend.project("broken", lambda: (np.ones(2, np.float32), 0), 512, 1024, vector)
    with pytest.raises(ValueError, match="shape mismatch"):
        backend.project("4", lambda: (matrix, 0), 511, 1024, vector)
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.project("4", lambda: (matrix, 0), 512, 1024, vector)


def test_dense_cache_priority_is_soft_and_preserves_vram_bound(backend):
    matrix = np.ones((512, 1024), np.float32)
    vector = np.ones(1024, np.float32)
    backend.project("dense:None", lambda: (matrix, 0), 512, 1024, vector)
    for expert in range(5):
        backend.project(f"blk.0.ffn_gate_exps.weight:{expert}", lambda: (matrix, 0), 512, 1024, vector)

    def unavailable():
        raise AssertionError("hot dense matrix was evicted before a cold expert")

    np.testing.assert_array_equal(backend.project("dense:None", unavailable, 512, 1024, vector), np.full(512, 1024))
    large = np.ones((1024, 1024), np.float32)
    backend.project("large:None", lambda: (large, 0), 1024, 1024, vector)
    # Dense priority must not turn into hard pinning that prevents legal batches.
    backend.project("blk.1.ffn_up_exps.weight:0", lambda: (large, 0), 1024, 1024, vector)
    assert backend.memory()[0] <= backend.memory()[1]


@pytest.mark.parametrize(
    "kind,block,size,scale_offset",
    [
        (2, 32, 18, 0),
        (6, 32, 22, 0),
        (8, 32, 34, 0),
        (20, 32, 18, 0),
        (11, 256, 110, 108),
        (12, 256, 144, 0),
        (13, 256, 176, 0),
        (14, 256, 210, 208),
        (23, 256, 136, 0),
        (16, 256, 66, 0),
        (17, 256, 74, 0),
        (18, 256, 98, 0),
        (21, 256, 110, 0),
        (22, 256, 82, 0),
        (29, 256, 56, 0),
    ],
)
def test_mixed_packed_kernels_match_independent_standard_decoder(backend, kind, block, size, scale_offset):
    from straydog.weights import dequantize

    rows, cols = 7, 512
    rng = np.random.default_rng(123 + kind)
    raw = rng.integers(0, 256, (rows * cols // block, size), dtype=np.uint8)
    raw[:, scale_offset : scale_offset + 2] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
    if kind == 29:
        set_scales(raw, kind, 0.01)
    if kind in {12, 13}:
        raw[:, 2:4] = np.frombuffer(np.float16(0.015).tobytes(), np.uint8)
    vector = rng.standard_normal(cols).astype(np.float32)
    expected = dequantize(raw.reshape(-1), kind).reshape(rows, cols) @ vector
    observed = backend.project(str(kind), lambda: (raw, kind), rows, cols, vector)
    np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=1e-4)


def test_worker_namespace_blocks_unmounted_files_and_network(tmp_path):
    import json
    import subprocess
    import sys

    from straydog.isolation import isolated_worker_command
    from straydog.platform.spawn import no_window_kwargs

    if sys.platform != "linux":
        pytest.skip("Linux namespace acceptance")
    outside = tmp_path / "unmounted-owner-file"
    outside.write_text("synthetic sentinel")
    root = tmp_path / "model"
    root.mkdir()
    script = (
        "import json,pathlib; print(json.dumps(dict("
        f"outside=pathlib.Path({str(outside)!r}).exists(),"
        "interfaces=pathlib.Path('/proc/net/dev').read_text())))"
    )
    command = isolated_worker_command(
        [sys.executable, "-I", "-c", script], model=root, inspector=root / "inspect", library=root / "gpu"
    )
    result = subprocess.run(command, capture_output=True, timeout=30, **no_window_kwargs())
    assert result.returncode == 0, result.stderr.decode()
    observed = json.loads(result.stdout)
    assert not observed["outside"]
    assert [row.split(":")[0].strip() for row in observed["interfaces"].splitlines() if ":" in row] == ["lo"]


def test_heterogeneous_batch_matches_single_projections_and_preserves_pins(backend):
    rng = np.random.default_rng(919)
    requests, expected = [], []
    for index, (rows, cols, kind) in enumerate([(9, 128, 0), (7, 256, 1), (3, 128, 42)]):
        if kind == 42:
            raw = rng.integers(0, 256, (rows, cols // 64, 18), dtype=np.uint8)
            raw[:, :, :2] = np.frombuffer(np.float16(0.25).tobytes(), np.uint8)
        else:
            raw = rng.normal(size=(rows, cols)).astype(np.float32 if kind == 0 else np.float16)
        vector = rng.normal(size=cols).astype(np.float32)
        requests.append((f"group-{index}", lambda raw=raw, kind=kind: (raw, kind), rows, cols, vector))
        expected.append(dequantize(raw.view(np.uint8).reshape(-1), kind).reshape(rows, cols) @ vector)
    backend.stats = {}
    observed = backend.project_many(requests)
    for actual, reference in zip(observed, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=2e-5)
    assert backend.stats["batches"] == 1 and backend.stats["calls"] == 3
    for actual, reference in zip(backend.project_many(requests), observed, strict=True):
        np.testing.assert_array_equal(actual, reference)
    assert backend.stats["hits"] == 3
    # More than the available cache must fail before launch, never reuse an evicted pointer.
    matrix = np.ones((512, 1024), np.float32)
    too_big = [(f"large-{i}", lambda: (matrix, 0), 512, 1024, np.ones(1024, np.float32)) for i in range(4)]
    with pytest.raises(ValueError, match="VRAM budget"):
        backend.project_many(too_big)
    assert backend.memory()[0] <= backend.memory()[1]
    with pytest.raises(ValueError, match="workspace"):
        backend.project_many([])


@pytest.mark.parametrize("steps,heads,dimension", [(1, 6, 8), (8, 48, 128), (32, 48, 128), (273, 6, 8)])
def test_gpu_delta_sequence_matches_scalar_reference_and_reset(backend, steps, heads, dimension):
    from straydog.numeric import delta_step

    rng = np.random.default_rng(341)
    queries = rng.normal(size=(steps, heads // 3, dimension)).astype(np.float32)
    keys = rng.normal(size=queries.shape).astype(np.float32)
    values = rng.normal(size=(steps, heads, dimension)).astype(np.float32)
    decay = -rng.uniform(0.01, 1, (steps, heads)).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, (steps, heads)).astype(np.float32)
    state = np.zeros((heads, dimension, dimension), np.float32)
    expected = np.stack(
        [delta_step(q, k, v, d, b, state) for q, k, v, d, b in zip(queries, keys, values, decay, beta, strict=True)]
    )
    observed = backend.delta_sequence("state", queries, keys, values, decay, beta)
    np.testing.assert_allclose(observed, expected, rtol=3e-4, atol=2e-6)
    np.testing.assert_allclose(backend._read_delta_state("state"), state, rtol=3e-4, atol=2e-6)
    # Weight-cache churn must not evict recurrent state.
    matrix = np.ones((256, 1024), np.float32)
    for i in range(4):
        backend.project(f"churn-{i}", lambda: (matrix, 0), 256, 1024, np.ones(1024, np.float32))
    np.testing.assert_allclose(backend._read_delta_state("state"), state, rtol=3e-4, atol=2e-6)
    backend.reset_state()
    assert not backend._read_delta_state("state").any()
    np.testing.assert_allclose(
        backend.delta_sequence("state", queries, keys, values, decay, beta), expected, rtol=3e-4, atol=2e-6
    )
    with pytest.raises(ValueError, match="geometry"):
        backend.delta_sequence("state", queries, keys, values[:, :, :1], decay, beta)


@pytest.mark.parametrize(
    "kind,block,size,scale_offset",
    [(0, 1, 4, None), (30, 1, 2, None), (42, 64, 18, 0), (11, 256, 110, 108), (13, 256, 176, 0), (23, 256, 136, 0),
     (16, 256, 66, 0), (17, 256, 74, 0), (18, 256, 98, 0), (21, 256, 110, 0), (22, 256, 82, 0), (20, 32, 18, 0),
     (8, 32, 34, 0)],
)
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_true_gemm_matches_cpu_and_mixed_fallback(kind, block, size, scale_offset, precision):
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(521)
    rows, cols = 19, 256
    if kind in {0, 30}:
        matrix = rng.normal(size=(rows, cols)).astype(np.float32)
        raw = matrix if kind == 0 else (matrix.view(np.uint32) >> 16).astype(np.uint16)
    else:
        raw = rng.integers(0, 256, (rows * cols // block, size), dtype=np.uint8)
        raw[:, scale_offset : scale_offset + 2] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
        if kind == 13:
            raw[:, 2:4] = np.frombuffer(np.float16(0.015).tobytes(), np.uint8)
    reference = dequantize(raw.view(np.uint8).reshape(-1), kind).reshape(rows, cols)
    vectors = rng.normal(size=(7, cols)).astype(np.float32)
    requests = [("shared", lambda: (raw, kind), rows, cols, vector) for vector in vectors]
    small = np.ones((3, 64), np.float32)
    requests.insert(3, ("fallback", lambda: (small, 0), 3, 64, np.ones(64, np.float32)))
    with GpuBackend(Path(library), budget_bytes=256 * 1024**2) as gpu:
        gpu.configure_gemm(minimum_batch=2, precision=precision)
        outputs = gpu.project_many(requests)
        if precision == "fp16":
            reference = reference.astype(np.float16).astype(np.float32)
            vectors = vectors.astype(np.float16).astype(np.float32)
        expected = [reference @ vector for vector in vectors]
        expected.insert(3, np.full(3, 64, np.float32))
        for actual, want in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(actual, want, rtol=3e-5, atol=2e-5)
        assert gpu.info()["gemm_calls"] == 2 and gpu.info()["gemm_vectors"] == 7
        assert gpu.memory()[0] <= gpu.memory()[1]


def test_gpu_sparse_attention_sequence_is_causal_and_preserves_kv():
    from straydog.numeric import softmax

    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(872)
    query = rng.normal(size=(7, 24, 256)).astype(np.float32) * 0.2
    keys = rng.normal(size=(7, 2, 256)).astype(np.float32) * 0.2
    values = rng.normal(size=(7, 2, 256)).astype(np.float32) * 0.2

    def reference(first, selections):
        output = np.empty((len(selections), 24, 256), np.float32)
        for i, indices in enumerate(selections):
            for head in range(24):
                scores = keys[indices, head // 12] @ query[first + i, head] / np.float32(16)
                output[i, head] = softmax(scores) @ values[indices, head // 12]
        return output

    initial = [np.arange(i + 1) for i in range(5)]
    continuation = [np.array([0, 2, 5]), np.array([1, 4, 6])]
    with GpuBackend(Path(library), budget_bytes=512 * 1024**2) as gpu:
        gpu.reserve_model_state(1024, attention=True, layers=[0, 3])
        before = gpu.memory()[0]
        observed = gpu.attention_sequence("blk.3.", query[:5], keys[:5], values[:5], initial, position=0, context=1024)
        np.testing.assert_allclose(observed, reference(0, initial), rtol=3e-5, atol=2e-6)
        result = gpu.attention_sequence(
            "blk.3.", query[5:], keys[5:], values[5:], continuation, position=5, context=1024
        )
        np.testing.assert_allclose(result, reference(5, continuation), rtol=3e-5, atol=2e-6)
        assert before == gpu.memory()[0]
        gpu.reset_state()
        future_keys, future_values = keys[:5].copy(), values[:5].copy()
        future_keys[1:] += 100
        future_values[1:] -= 100
        future = gpu.attention_sequence(
            "blk.3.", query[:5], future_keys, future_values, initial, position=0, context=1024
        )
        np.testing.assert_array_equal(future[0], observed[0])
        with pytest.raises(ValueError, match="selection"):
            gpu.attention_sequence("blk.3.", query[:1], keys[:1], values[:1], [np.array([1])], position=0, context=1024)


def test_transfers_larger_than_the_staging_arena_arrive_intact():
    # Host memory reaches the GPU through a 32 MiB page-locked arena; 40 MiB of weights take three
    # alternating halves, and a 40 MiB readback takes two rounds.
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(4096, 2560)).astype(np.float32)
    vector = rng.normal(size=2560).astype(np.float32)
    with GpuBackend(Path(library), budget_bytes=512 * 1024**2) as gpu:
        result = gpu.project("big", lambda: (matrix, 0), 4096, 2560, vector)
        np.testing.assert_allclose(result, matrix.astype(np.float64) @ vector, rtol=1e-4, atol=1e-3)
        np.testing.assert_array_equal(gpu.project("big", lambda: (matrix, 0), 4096, 2560, vector), result)
        pointer = gpu._prepare("big", lambda: (matrix, 0), 4096, 2560, set())[0]
        readback = np.empty_like(matrix)
        gpu._check(gpu._lib.sd_download(gpu._handle, pointer, readback.ctypes.data, readback.nbytes))
        np.testing.assert_array_equal(readback, matrix)


def test_large_gemm_exceeds_old_descriptor_and_output_limits():
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(117)
    matrix = rng.normal(size=(1536, 64)).astype(np.float32)
    vectors = rng.normal(size=(256, 64)).astype(np.float32)
    with GpuBackend(Path(library), budget_bytes=512 * 1024**2) as gpu:
        assert gpu.batch_capacity == 256
        gpu.configure_gemm(minimum_batch=8, precision="fp16")
        actual = gpu.project_many([("wide", lambda: (matrix, 0), 1536, 64, vector) for vector in vectors])
        expected = np.einsum(
            "rc,tc->tr", matrix.astype(np.float16).astype(np.float32), vectors.astype(np.float16).astype(np.float32)
        )
        np.testing.assert_allclose(np.stack(actual), expected, rtol=1e-4, atol=2e-5)
        assert gpu.info()["gemm_vectors"] == 256
        assert gpu.memory()[0] <= gpu.memory()[1]


@pytest.mark.parametrize(
    "steps,precision,inject",
    [
        (1, "fp32", True),
        (8, "fp32", True),
        (8, "fp16", True),
        (256, "fp16", True),
        (273, "fp16", True),
        (1025, "fp16", True),
        (1, "fp32", False),
    ],
)
def test_fused_hyperconnection_matches_independent_cpu_equations(steps, precision, inject):
    from straydog.numeric import rms_norm, sigmoid, silu

    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(541)
    values = rng.normal(size=(steps, 10240)).astype(np.float32)
    norm = rng.uniform(0.5, 1.5, 10240).astype(np.float32)
    down = rng.normal(scale=0.01, size=(320, 10240)).astype(np.float32)
    up = rng.normal(scale=0.01, size=(10240, 320)).astype(np.float32)
    injection = rng.normal(scale=0.01, size=(4, 10240)).astype(np.float32)
    raw = [(weight.view(np.uint32) >> 16).astype(np.uint16) for weight in (down, up, injection)]
    matrices = [
        (str(i), lambda i=i: (raw[i], 30), weight.shape[0], weight.shape[1])
        for i, weight in enumerate((down, up, injection))
        if i < 2 or inject
    ]
    down, up, injection = [(weight.astype(np.uint32) << 16).view(np.float32) for weight in raw]
    normalized = rms_norm(values, norm, 2560)

    def projection(weight, inputs):
        pieces = []
        physical = 1024 if steps > 273 else 256
        for first in range(0, len(inputs), physical):
            part, projected_weight = inputs[first:first + physical], weight
            if precision == "fp16" and len(part) >= 8:
                part = part.astype(np.float16).astype(np.float32)
                projected_weight = weight.astype(np.float16).astype(np.float32)
            pieces.append(part @ projected_weight.T)
        return np.concatenate(pieces)

    low = silu(projection(down, normalized) / np.float32(4))
    gate = sigmoid(projection(up, low)).reshape(-1, 4, 2560)
    expected = np.mean(gate * normalized.reshape(-1, 4, 2560), axis=1)
    with GpuBackend(Path(library), budget_bytes=2 * 1024**3 if steps > 273 else 512 * 1024**2) as gpu:
        gpu.configure_gemm(precision=precision)
        mixed, observed_injection = gpu.hyperconnection("blk.0.hc_attn", values, norm, matrices)
        np.testing.assert_allclose(mixed, expected, rtol=5e-4, atol=3e-5)
        if inject:
            np.testing.assert_allclose(
                observed_injection, 2 * sigmoid(projection(injection, normalized) / 4), rtol=5e-4, atol=3e-5
            )
        else:
            assert observed_injection is None
        assert gpu.memory()[0] <= gpu.memory()[1]
        with pytest.raises(ValueError, match="geometry"):
            gpu.hyperconnection("blk.0.hc_attn", values[:, :2], norm, matrices)


def test_attention_large_logical_batch_preserves_chunk_boundary_and_causality():
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    steps = 273
    queries = np.zeros((steps, 24, 256), np.float32)
    keys = np.zeros((steps, 2, 256), np.float32)
    values = np.broadcast_to(np.arange(steps, dtype=np.float32)[:, None, None], keys.shape).copy()
    selections = [np.array([0, i], np.int32) if i else np.array([0], np.int32) for i in range(steps)]
    with GpuBackend(Path(library), budget_bytes=512 * 1024**2) as gpu:
        actual = gpu.attention_sequence("blk.3.", queries, keys, values, selections, position=0, context=512)
        np.testing.assert_allclose(actual, np.broadcast_to((np.arange(steps) / 2)[:, None, None], actual.shape),
                                   rtol=1e-6, atol=1e-6)
        # Validate the complete selection list before any native chunk mutates K/V.
        selections[-1] = np.array([steps], np.int32)
        with pytest.raises(ValueError, match="selection"):
            gpu.attention_sequence("blk.3.", queries, keys, values, selections, position=0, context=512)


@pytest.mark.parametrize("kind,block,size", [(8, 32, 34), (20, 32, 18), (16, 256, 66), (17, 256, 74), (18, 256, 98),
                                             (21, 256, 110), (22, 256, 82), (29, 256, 56)])
def test_multi_vector_gemv_matches_single_vector_and_reference(kind, block, size):
    """Verification batches (2-4 vectors) of the IQ bundle formats repeat each vector's one-vector arithmetic."""
    from straydog.formats import BLOCKS

    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(kind)
    rows = 1000
    raw = rng.integers(0, 256, (rows * 2560 // block, size), dtype=np.uint8)
    set_scales(raw, kind, rng.uniform(0.0005, 0.004, len(raw)))
    assert BLOCKS[kind] == (block, size)

    def norm(name, width):
        return (name, lambda: (np.ones((1, width), np.float32), 0), 1, width)

    def dense(name, r, c, scale):
        values = rng.normal(scale=scale, size=(r, c)).astype(np.float16)
        return (name, lambda: (values, 1), r, c)

    head = [norm("n", 10240), dense("d", 320, 10240, 0.01), dense("u", 10240, 320, 0.01),
            ("v", lambda: (raw, kind), rows, 2560)]
    values = rng.normal(scale=0.05, size=(4, 10240)).astype(np.float32)
    with GpuBackend(Path(library), budget_bytes=512 * 1024**2) as gpu:
        gpu.configure_gemm(minimum_batch=2, precision="fp32")
        for count in (2, 3, 4):
            _tokens, logits = gpu.output_head(head, values[:count], logits=True)
            for i in range(count):
                _token, single = gpu.output_head(head, values[i:i + 1], logits=True)
                assert np.array_equal(single[0], logits[i])



@pytest.mark.parametrize("kind,size,scale_offset", [(11, 110, 108), (12, 144, 0), (13, 176, 0),
                                                   (14, 210, 208), (23, 136, 0)])
def test_decode_output_projection_geometry_matches_independent_decoder(kind, size, scale_offset):
    """Exercise the 2560x6144 decode shape, including CUDA's whole-warp FP32 row sums."""
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rows, cols = 2560, 6144
    rng = np.random.default_rng(921 + kind)
    raw = rng.integers(0, 256, (rows * cols // 256, size), dtype=np.uint8)
    raw[:, scale_offset:scale_offset + 2] = np.frombuffer(np.float16(0.001).tobytes(), np.uint8)
    if kind in {12, 13}:
        raw[:, 2:4] = np.frombuffer(np.float16(0.0015).tobytes(), np.uint8)
    vector = rng.normal(scale=0.1, size=cols).astype(np.float32)
    expected = dequantize(raw.reshape(-1), kind).reshape(rows, cols) @ vector
    with GpuBackend(Path(library), budget_bytes=512 * 1024**2) as gpu:
        observed = gpu.project("output", lambda: (raw, kind), rows, cols, vector)
        np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=1e-4)
        np.testing.assert_array_equal(gpu.project("output", lambda: (raw, kind), rows, cols, vector), observed)


@pytest.mark.gpu
def test_conversation_checkpoints_restore_recurrent_state_exactly_outside_the_budget() -> None:
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(7)

    def read(gpu, key):
        entry = gpu._states[key]
        output = np.empty(entry[2] * entry[3], np.float32)
        gpu._check(gpu._lib.sd_download(gpu._handle, entry[0], output.ctypes.data, output.nbytes))
        return output

    def fill(gpu, keys, tag):
        values = {}
        for key in keys:
            rows, cols = gpu._states[key][2:]
            values[key] = rng.standard_normal(rows * cols).astype(np.float32)
            gpu._states[f"@{tag}{key}"] = gpu._prepare(f"@{tag}{key}", lambda array=values[key]: (array, 0),
                                                        rows, cols, set())
            gpu.copy_state(key, f"@{tag}{key}")
        return values

    with GpuBackend(Path(library), budget_bytes=512 * 1024**2) as gpu:
        gpu.reserve_model_state(1024, attention=True, layers=[0, 1, 3])
        used = gpu.memory()[0]
        keys = [key for key in gpu._states if key.startswith(("@delta:", "@conv:", "@ple:"))]
        assert "@delta:blk.0." in keys and "@delta:blk.1." in keys
        size = sum(gpu._states[key][2] * gpu._states[key][3] * 4 for key in keys)
        assert gpu.reserve_checkpoints(2) == 2 * size
        assert gpu.memory()[0] == used   # page-locked host buffers, not VRAM
        first = fill(gpu, keys, "a")
        gpu.save_checkpoint(0)
        second = fill(gpu, keys, "b")
        gpu.save_checkpoint(1)
        gpu.load_checkpoint(0)
        assert all(np.array_equal(read(gpu, key), first[key]) for key in keys)
        gpu.reset_state()   # the sequence restarts; checkpoints outlive it
        assert not any(read(gpu, key).any() for key in keys)
        gpu.load_checkpoint(1)
        assert all(np.array_equal(read(gpu, key), second[key]) for key in keys)


@pytest.mark.gpu
def test_planned_state_is_what_reservation_allocates() -> None:
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    from straydog.gpu import _ALLOCATION_TAIL, _decoder_workspace_bytes

    for host in (False, True):
        with GpuBackend(Path(library), budget_bytes=2 * 1024**3) as gpu:
            kwargs = {"attention": True, "layers": [0, 1, 3], "prefill": 512, "host_attention": host}
            planned = gpu.planned_state_bytes(4096, **kwargs)
            before = gpu.memory()[0]
            gpu.reserve_model_state(4096, **kwargs)
            # The decoder workspace comes later, under full residency.
            workspace = _decoder_workspace_bytes(512) + _ALLOCATION_TAIL
            assert gpu.memory()[0] - before == planned - workspace
            assert gpu.planned_state_bytes(4096, **kwargs) == workspace
