"""Fused native DeltaNet is checked against independent CPU state arithmetic."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend
from straydog.numeric import causal_conv, delta_step, rms_norm, sigmoid, silu

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("steps,precision", [(3, "fp32"), (9, "fp16"), (513, "fp16")])
def test_fused_delta_preserves_convolution_recurrence_gates_and_reset(steps, precision):
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(772)
    shapes = [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560), (2560, 6144)]
    raw = [rng.normal(scale=0.01, size=shape).astype(np.float16) for shape in shapes]
    values = rng.normal(scale=0.2, size=(steps, 2560)).astype(np.float32)
    conv = rng.normal(scale=0.1, size=(10240, 4)).astype(np.float32)
    bias = rng.normal(scale=0.1, size=48).astype(np.float32)
    decay = -rng.uniform(0.1, 1, 48).astype(np.float32)
    norm = rng.uniform(0.5, 1.5, 128).astype(np.float32)
    matrices = [(str(i), lambda i=i: (raw[i], 1), rows, cols) for i, (rows, cols) in enumerate(shapes)]
    vectors = [("conv", conv), ("bias", bias), ("decay", decay), ("norm", norm)]
    history, state = np.zeros((4, 10240), np.float32), np.zeros((48, 128, 128), np.float32)

    def reference(incoming):
        def project(index, inputs):
            pieces = []
            physical = 512 if steps > 128 else 128
            for first in range(0, len(inputs), physical):
                part = inputs[first:first + physical]
                if precision == "fp16" and len(part) >= 8:
                    part = part.astype(np.float16).astype(np.float32)
                pieces.append(part @ raw[index].astype(np.float32).T)
            return np.concatenate(pieces)

        qkv, gate, beta, alpha = [project(index, incoming) for index in range(4)]
        qkv = silu(causal_conv(qkv, history, conv))
        alpha = decay * np.logaddexp(np.float32(0), alpha + bias)
        beta = sigmoid(beta)
        result = np.stack(
            [
                delta_step(
                    row[:2048].reshape(16, 128),
                    row[2048:4096].reshape(16, 128),
                    row[4096:].reshape(48, 128),
                    d,
                    b,
                    state,
                )
                for row, d, b in zip(qkv, alpha, beta, strict=True)
            ]
        )
        result = rms_norm(result, norm) * sigmoid(gate.reshape(-1, 48, 128))
        return project(4, result.reshape(-1, 6144))

    expected = reference(values)
    with GpuBackend(Path(library), budget_bytes=(2 if steps > 128 else 1) * 1024**3) as gpu:
        gpu.configure_gemm(precision=precision)
        actual = gpu.gated_delta("blk.0.", values, matrices, vectors)
        np.testing.assert_allclose(actual, expected, rtol=8e-4, atol=4e-5)
        np.testing.assert_allclose(gpu._read_delta_state("blk.0."), state, rtol=8e-4, atol=4e-6)
        continuation = values[:1] * np.float32(0.9)
        np.testing.assert_allclose(
            gpu.gated_delta("blk.0.", continuation, matrices, vectors), reference(continuation), rtol=8e-4, atol=4e-5
        )
        gpu.reset_state()
        history.fill(0)
        state.fill(0)
        np.testing.assert_allclose(
            gpu.gated_delta("blk.0.", values, matrices, vectors), reference(values), rtol=8e-4, atol=4e-5
        )
        assert gpu.memory()[0] <= gpu.memory()[1]
        with pytest.raises(ValueError, match="geometry"):
            gpu.gated_delta("blk.0.", values[:, :1], matrices, vectors)


@pytest.mark.parametrize("kind,size,scale_offset", [(11, 110, 108), (12, 144, 0), (13, 176, 0),
                                                   (14, 210, 208), (23, 136, 0)])
@pytest.mark.parametrize("mixed", [False, True, pytest.param("q3", id="q3-input"),
                                   pytest.param("q3-q2", id="q2-gate")])
def test_quantized_delta_output_projection_matches_cpu_and_refreshes_input(kind, size, scale_offset, mixed):
    """Wide output decode, including the shared F32 input kernels, matches independent CPU equations."""
    from straydog.weights import dequantize

    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(421 + kind)
    shapes = [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560)]
    raw = [rng.normal(scale=0.01, size=shape).astype(np.float16) for shape in shapes]
    input_kinds = [1] * 4
    decoded_inputs = [w.astype(np.float32) for w in raw]
    if mixed:
        input_kinds = [11 if mixed in {"q3", "q3-q2"} else 23, 42 if mixed == "q3-q2" else 12, 30, 30]
        for i, (rows, cols) in enumerate(shapes):
            if i < 2:
                block = 64 if input_kinds[i] == 42 else 256
                input_size = {11: 110, 12: 144, 23: 136, 42: 18}[input_kinds[i]]
                raw[i] = rng.integers(0, 256, (rows * cols // block, input_size), dtype=np.uint8)
                offset = 108 if input_kinds[i] == 11 else 0
                scale = np.float16(0.01 if input_kinds[i] == 42 else 0.0001)
                raw[i][:, offset:offset + 2] = np.frombuffer(scale.tobytes(), np.uint8)
                if input_kinds[i] == 12:
                    raw[i][:, 2:4] = np.frombuffer(np.float16(0.00015).tobytes(), np.uint8)
            else:
                raw[i] = (raw[i].astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            decoded_inputs[i] = dequantize(raw[i].view(np.uint8).reshape(-1), input_kinds[i]).reshape(rows, cols)
    packed = rng.integers(0, 256, (2560 * 6144 // 256, size), dtype=np.uint8)
    packed[:, scale_offset:scale_offset + 2] = np.frombuffer(np.float16(0.001).tobytes(), np.uint8)
    if kind in {12, 13}:
        packed[:, 2:4] = np.frombuffer(np.float16(0.0015).tobytes(), np.uint8)
    decoded = dequantize(packed.reshape(-1), kind).reshape(2560, 6144)
    matrices = [(str(i), lambda i=i: (raw[i], input_kinds[i]), *shape) for i, shape in enumerate(shapes)]
    matrices.append(("out", lambda: (packed, kind), 2560, 6144))
    conv = rng.normal(scale=0.1, size=(10240, 4)).astype(np.float32)
    bias, decay, norm = np.zeros(48, np.float32), -np.ones(48, np.float32), np.ones(128, np.float32)
    vectors = [("conv", conv), ("bias", bias), ("decay", decay), ("norm", norm)]
    incoming = rng.normal(scale=0.2, size=(1, 2560)).astype(np.float32)

    def reference(x):
        qkv, gate, beta, alpha = [x @ w.T for w in decoded_inputs]
        qkv = silu(causal_conv(qkv, np.zeros((4, 10240), np.float32), conv))[0]
        state = np.zeros((48, 128, 128), np.float32)
        delta = delta_step(qkv[:2048].reshape(16, 128), qkv[2048:4096].reshape(16, 128),
                           qkv[4096:].reshape(48, 128), decay * np.logaddexp(np.float32(0), alpha[0] + bias),
                           sigmoid(beta[0]), state)
        projected = rms_norm(delta, norm) * sigmoid(gate.reshape(48, 128))
        return projected.reshape(1, 6144) @ decoded.T

    with GpuBackend(Path(library), budget_bytes=1024**3) as gpu:
        gpu.configure_gemm(precision="fp16", minimum_batch=2)
        for factor in (1.0, 0.8):
            gpu.reset_state()
            x = incoming * np.float32(factor)
            expected = reference(x)
            observed = gpu.gated_delta("blk.0.", x, matrices, vectors)
            np.testing.assert_allclose(observed, expected, rtol=8e-4, atol=4e-5)
