"""Native PLE must match CPU gates, grouped norms and dilated causal history."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend
from straydog.numeric import causal_conv, rms_norm, sigmoid, silu

pytestmark = pytest.mark.gpu


def test_ple_chunks_continuation_and_reset_match_cpu():
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(373)
    steps = 273
    raw = [rng.normal(scale=0.01, size=(rows, 2560)).astype(np.float16) for rows in (10240, 2560)]
    matrices = [("projection" + str(i), lambda i=i: (raw[i], 1), raw[i].shape[0], 2560) for i in range(2)]
    norms = [rng.uniform(0.7, 1.3, size).astype(np.float32) for size in [10240, 10240, 10240]]
    coefficients = rng.normal(scale=0.1, size=(10240, 4)).astype(np.float32)
    vectors = [("norm" + str(i), norm) for i, norm in enumerate(norms)] + [("conv", coefficients)]
    hidden = rng.normal(scale=0.2, size=(steps, 10240)).astype(np.float32)
    embeddings = rng.normal(scale=0.2, size=(steps, 2560)).astype(np.float32)
    history = np.zeros((10, 10240), np.float32)

    def reference(values, embedded):
        # Physical chunks retain full F32 matvec for singleton tails.
        projections = []
        for weight in raw:
            parts = []
            for first in range(0, len(embedded), 256):
                part = embedded[first:first + 256]
                incoming = part.astype(np.float16).astype(np.float32) if len(part) >= 2 else part
                parts.append(incoming @ weight.astype(np.float32).T)
            projections.append(np.concatenate(parts))
        key = rms_norm(projections[0], norms[0], 2560).reshape(-1, 4, 2560)
        query = rms_norm(values, norms[1], 2560).reshape(-1, 4, 2560)
        gate = np.sum(key * query, axis=2) / np.sqrt(np.float32(2560))
        gate = np.sign(gate) * np.sqrt(np.maximum(np.abs(gate), np.float32(1e-6)))
        gated = (sigmoid(gate)[:, :, None] * projections[1][:, None, :]).reshape(-1, 10240)
        normalized = rms_norm(gated, norms[2], 2560)
        return values + gated + silu(causal_conv(normalized, history, coefficients, dilation=3))

    expected = reference(hidden, embeddings)
    following = reference(hidden[:1] * 0.8, embeddings[:1])
    with GpuBackend(Path(library), budget_bytes=2 * 1024**3) as gpu:
        gpu.configure_gemm(precision="fp16", minimum_batch=2)
        np.testing.assert_allclose(gpu.ple("blk.1.", hidden, embeddings, matrices, vectors), expected,
                                   rtol=2e-3, atol=5e-5)
        np.testing.assert_allclose(gpu.ple("blk.1.", hidden[:1] * 0.8, embeddings[:1], matrices, vectors), following,
                                   rtol=2e-3, atol=5e-5)
        gpu.reset_state()
        np.testing.assert_allclose(gpu.ple("blk.1.", hidden, embeddings, matrices, vectors), expected,
                                   rtol=2e-3, atol=5e-5)
        with pytest.raises(ValueError, match="PLE geometry"):
            gpu.ple("blk.1.", hidden[:, :1], embeddings, matrices, vectors)
        assert gpu.memory()[0] <= gpu.memory()[1]
