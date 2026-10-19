"""Packed-bank resident routing and graph reuse must match independent CPU arithmetic."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend
from straydog.numeric import sigmoid, silu, softmax
from straydog.weights import dequantize
from tests.blocks import set_scales

pytestmark = pytest.mark.gpu


# Routed gate/up/down formats: Q2_0, and the IQ2_XS / IQ3 bundles' IQ1/IQ2/IQ3 gate-up with IQ4_NL / Q2_0 / Q8_0 down.
BANK_KINDS = [(42, 42, 42), (21, 21, 20), (16, 16, 42), (17, 17, 20), (18, 18, 8), (22, 22, 20), (29, 29, 42)]


def test_grouped_q2_signed_subnormal_scales_and_fp16_input_rounding():
    """Amplify signed subnormal Q2 scales with large, non-FP16 input values.

    Identical experts and a uniform router make the oracle independent of the
    top-ten tie order. Sixty-five rows exercise both halves and a partial tile.
    """
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng, experts, steps = np.random.default_rng(7319), 10, 65
    shapes = [(640, 2560), (640, 2560), (2560, 640)]
    packed, decoded = [], []
    for role, (rows, cols) in enumerate(shapes):
        raw = rng.integers(0, 256, (rows * cols // 64, 18), dtype=np.uint8)
        if role < 2:
            # Include both signed zeros and FP16 subnormals; F32 decoding must
            # happen before rounding, even for the dedicated Q2_0 quarter path.
            scales = (rng.integers(-8, 9, len(raw)) * 2.0**-24).astype(np.float16)
            scales[::17] = np.float16(-0.0)
            raw[:, :2] = scales.view(np.uint8).reshape(-1, 2)
        else:
            set_scales(raw, 42, 0.01)
        decoded.append(dequantize(raw.reshape(-1), 42).reshape(rows, cols).astype(np.float16).astype(np.float32))
        packed.append(np.tile(raw, (experts, 1)))
    shared = [np.zeros(shape, np.float16) for shape in shapes]
    router, mix = np.zeros((experts, 2560), np.float16), np.zeros((1, 2560), np.float32)
    matrices = [(f"bank{role}", lambda role=role: (packed[role], 42), rows * experts, cols)
                for role, (rows, cols) in enumerate(shapes)]
    matrices += [(f"shared{role}", lambda role=role: (shared[role], 1), rows, cols)
                 for role, (rows, cols) in enumerate(shapes)]
    matrices += [("router", lambda: (router, 1), experts, 2560), ("mix", lambda: (mix, 0), 1, 2560)]
    incoming = rng.uniform(-16000, 16000, (steps, 2560)).astype(np.float32)
    rounded = incoming.astype(np.float16).astype(np.float32)
    product = (silu(rounded @ decoded[0].T) * (rounded @ decoded[1].T)).astype(np.float16).astype(np.float32)
    expected = product @ decoded[2].T
    with GpuBackend(Path(library), budget_bytes=1024**3) as gpu:
        gpu.configure_gemm(minimum_batch=2, precision="fp16")
        gpu.register_resident_moe("blk.0.", matrices, experts=experts)
        np.testing.assert_allclose(gpu.resident_moe("blk.0.", incoming), expected, rtol=2e-3, atol=3e-5)


@pytest.mark.parametrize("kinds", BANK_KINDS)
@pytest.mark.parametrize("steps,precision", [(1, "fp32"), (9, "fp16"), (33, "fp16"), (65, "fp16"), (273, "fp16")])
def test_resident_routing_banks_and_replay_match_cpu(steps, precision, kinds):
    from straydog.formats import BLOCKS

    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng, experts = np.random.default_rng(628), 11
    shapes = [(640, 2560), (640, 2560), (2560, 640)]
    packed, decoded = [], []
    for expert in range(experts):
        for (rows, cols), kind in zip(shapes, kinds, strict=True):
            block, size = BLOCKS[kind]
            raw = rng.integers(0, 256, (rows * cols // block, size), dtype=np.uint8)
            scale = (0.004 + expert * 0.001) if kind == 42 else (0.0004 + expert * 0.0001)
            set_scales(raw, kind, scale)
            packed.append(raw)
            decoded.append(dequantize(raw.reshape(-1), kind).reshape(rows, cols))
    banks = [np.concatenate(packed[role::3]) for role in range(3)]
    shared = [rng.normal(scale=0.01, size=shape).astype(np.float16) for shape in shapes]
    router = (rng.normal(scale=0.001, size=(experts, 2560)) + np.arange(experts)[:, None] * 0.003).astype(np.float16)
    mix = rng.normal(scale=0.01, size=(1, 2560)).astype(np.float32)
    matrices = [("bank" + str(role), lambda role=role: (banks[role], kinds[role]), rows * experts, cols)
                for role, (rows, cols) in enumerate(shapes)]
    matrices += [("shared" + str(role), lambda role=role: (shared[role], 1), rows, cols)
                 for role, (rows, cols) in enumerate(shapes)]
    matrices += [("router", lambda: (router, 1), experts, 2560), ("mix", lambda: (mix, 0), 1, 2560)]
    incoming = rng.uniform(0.01, 0.2, (steps, 2560)).astype(np.float32)

    def reference(values):
        # FP16 mode evaluates the whole batch (router and every expert group) in FP16 from 8 rows on.
        half = precision == "fp16" and len(values) >= 8
        route_input = values.astype(np.float16).astype(np.float32) if half else values
        logits = route_input @ router.astype(np.float32).T
        selected = np.argsort(-logits, axis=1)[:, :10]
        probabilities = softmax(np.take_along_axis(logits, selected, axis=1), axis=1)
        outputs = np.empty((len(values), 10, 2560), np.float32)

        def evaluate(weights, x):
            def project(role, data):
                weight = weights[role].astype(np.float32)
                if half:
                    data = data.astype(np.float16).astype(np.float32)
                    weight = weight.astype(np.float16).astype(np.float32)
                return data @ weight.T
            return project(2, silu(project(0, x)) * project(1, x))

        for expert in range(experts):
            tokens, ranks = np.where(selected == expert)
            if len(tokens):
                outputs[tokens, ranks] = evaluate(decoded[expert * 3 : expert * 3 + 3], values[tokens])
        result = np.zeros_like(values)
        for rank in range(10):
            result += probabilities[:, rank, None] * outputs[:, rank]
        return result + sigmoid(values @ mix.T) * evaluate(shared, values)

    expected = reference(incoming)
    # In FP16 mode a one-ulp difference of an FP16 SiLU product (from a different F32
    # summation order) is scaled by the down weights; bound it relative to the output.
    atol = max(3e-5, 2.5e-4 * float(np.abs(expected).max())) if precision == "fp16" and steps >= 8 else 3e-5
    with GpuBackend(Path(library), budget_bytes=1024**3) as gpu:
        gpu.configure_gemm(precision=precision)
        assert gpu.can_pin_model({"small": 1024})
        assert not gpu.can_pin_model({"large": 2 * 1024**3})
        gpu.register_resident_moe("blk.0.", matrices, experts=experts)
        np.testing.assert_allclose(gpu.resident_moe("blk.0.", incoming), expected, rtol=2e-3, atol=atol)
        before = gpu.info()
        np.testing.assert_allclose(gpu.resident_moe("blk.0.", incoming), expected, rtol=2e-3, atol=atol)
        if before["cached_graphs"]:
            assert gpu.info()["graph_replays"] > before["graph_replays"]
        changed = -incoming[:1]
        np.testing.assert_allclose(gpu.resident_moe("blk.0.", changed), reference(changed), rtol=2e-3, atol=3e-5)
        gpu.reset_state()
        np.testing.assert_allclose(gpu.resident_moe("blk.0.", changed), reference(changed), rtol=2e-3, atol=3e-5)
        assert gpu._lib.sd_release(gpu._handle, gpu._cache["bank0"][0]) != 0
        assert gpu.info()["resident_moe_plans"] == 1
        assert gpu.memory()[0] <= gpu.memory()[1]
        with pytest.raises(ValueError, match="geometry"):
            gpu.register_resident_moe("blk.0.", matrices, experts=experts)
        gpu.close()
        with pytest.raises(RuntimeError, match="closed"):
            gpu.resident_moe("blk.0.", changed)
