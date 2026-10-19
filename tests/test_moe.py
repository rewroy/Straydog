"""Full routed MoE matches CPU arithmetic, including shared and reordered experts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend
from straydog.numeric import silu, softmax
from straydog.weights import dequantize

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("steps,precision", [(1, "fp32"), (9, "fp16"), (33, "fp16"), (273, "fp16")])
def test_full_moe_matches_grouped_reference_and_has_no_sequence_state(steps, precision):
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(892)
    experts = 11
    shapes = [(640, 2560), (640, 2560), (2560, 640)] * (experts + 1)
    raw, decoded, matrices = [], [], []
    for index, (rows, cols) in enumerate(shapes):
        kind = 42 if index < experts * 3 else 1
        if kind == 42:
            data = rng.integers(0, 256, (rows * cols // 64, 18), dtype=np.uint8)
            data[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
        else:
            data = rng.normal(scale=0.01, size=(rows, cols)).astype(np.float16)
        raw.append(data)
        decoded.append(dequantize(data.view(np.uint8).reshape(-1), kind).reshape(rows, cols))
        matrices.append((str(index), lambda data=data, kind=kind: (data, kind), rows, cols))
    values = rng.normal(scale=0.2, size=(steps, 2560)).astype(np.float32)
    selected = np.stack([rng.permutation(experts)[:10] for _ in values]).astype(np.int32)
    probabilities = softmax(rng.normal(size=(steps, 10)).astype(np.float32), axis=1)
    shared = rng.uniform(0, 1, (steps, 1)).astype(np.float32)

    def reference(incoming, chosen, probs, mix):
        routed = np.empty((len(incoming), 10, 2560), np.float32)

        def evaluate(expert, x):
            def project(role, data):
                if precision == "fp16" and len(x) >= 8:
                    data = data.astype(np.float16).astype(np.float32)
                return data @ decoded[expert * 3 + role].T

            product = silu(project(0, x)) * project(1, x)
            return project(2, product)

        for expert in range(experts):
            tokens, ranks = np.where(chosen == expert)
            if len(tokens):
                routed[tokens, ranks] = evaluate(expert, incoming[tokens])
        output = np.zeros_like(incoming)
        for rank in range(10):
            output += probs[:, rank, None] * routed[:, rank]
        return output + mix * evaluate(experts, incoming)

    expected = reference(values, selected, probabilities, shared)
    with GpuBackend(Path(library), budget_bytes=1024**3) as gpu:
        gpu.configure_gemm(precision=precision)
        actual = gpu.moe("blk.0.", values, selected, probabilities, shared, matrices)
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=3e-5)
        gpu.reset_state()
        again = gpu.moe("blk.0.", values[:1], selected[:1], probabilities[:1], shared[:1], matrices)
        np.testing.assert_allclose(
            again, reference(values[:1], selected[:1], probabilities[:1], shared[:1]), rtol=2e-3, atol=3e-5
        )
        assert gpu.memory()[0] <= gpu.memory()[1]
        duplicate = selected.copy()
        duplicate[:, 1] = duplicate[:, 0]
        with pytest.raises(ValueError, match="selection"):
            gpu.moe("blk.0.", values, duplicate, probabilities, shared, matrices)
