"""Resident spans must preserve the independently verified component contracts."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("layer,position", [(0, 0), (3, 2051)])
def test_prefill_span_chunk_boundaries_decode_replay_and_reset(layer, position):
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(219)
    steps, context, experts = 273, 4096, 10
    prefix = f"blk.{layer}."

    def matrix(key, rows, cols, *, norm=False):
        raw = (rng.uniform(0.7, 1.3, (rows, cols)) if norm else rng.normal(scale=0.004, size=(rows, cols)))
        raw = raw.astype(np.float32 if norm else np.float16)
        return (prefix + key, lambda raw=raw: (raw, 0 if norm else 1), rows, cols)

    def hc(name):
        return [matrix(name + str(i), rows, cols, norm=i == 0)
                for i, (rows, cols) in enumerate([(1, 10240), (320, 10240), (10240, 320), (4, 10240)])]

    before, after = hc("hc_attn"), hc("hc_ffn")
    expert_shapes = [(640, 2560), (640, 2560), (2560, 640)]
    moe = [matrix("bank" + str(i), rows * experts, cols) for i, (rows, cols) in enumerate(expert_shapes)]
    moe += [matrix("shared" + str(i), rows, cols) for i, (rows, cols) in enumerate(expert_shapes)]
    moe += [matrix("router", experts, 2560), matrix("mix", 1, 2560)]
    if layer % 4 != 3:
        shapes = [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560), (2560, 6144)]
        vectors = [matrix("conv", 10240, 4, norm=True), matrix("bias", 1, 48, norm=True),
                   matrix("decay", 1, 48, norm=True), matrix("norm", 1, 128, norm=True)]
        # Stable decay keeps the reference recurrence finite across all chunks.
        decay = -np.ones((1, 48), np.float32)
        vectors[2] = (prefix + "decay", lambda: (decay, 0), 1, 48)
    else:
        shapes = [(12288, 2560), (512, 2560), (512, 2560), (512, 2560), (128, 2560), (2560, 6144)]
        vectors = [matrix("norm" + str(i), 1, size, norm=True) for i, size in enumerate([256, 256, 128, 128])]
    attention = [matrix("attention" + str(i), *shape) for i, shape in enumerate(shapes)]
    incoming = rng.normal(scale=0.05, size=(steps, 10240)).astype(np.float32)
    device_outputs = []
    for host_attention in ([False, True] if layer == 3 else [False]):
        with GpuBackend(Path(library), budget_bytes=1024**3) as gpu:
            gpu.configure_gemm(precision="fp16", minimum_batch=2)
            if host_attention and not gpu.host_state_supported():
                continue   # Keep the original VRAM assertions live on HIP.
            gpu.reserve_model_state(context, attention=True, layers=[layer], prefill=steps,
                                    host_attention=host_attention)
            if host_attention:
                stage = gpu._state("@test:stage", 512 * 4, 256)
                gpu._check(gpu._lib.sd_kv_stage(gpu._handle, stage[0], 512))
            gpu.register_resident_moe(prefix, moe, experts=experts)
            gpu.register_decoder_layer(layer, before + attention + vectors + after, context=context)

            def components(values, base):
                def mix(entries, x):
                    return gpu.hyperconnection(entries[0][0][:-1], x, entries[0][1]()[0].reshape(-1), entries[1:])
                mixed, injection = mix(before, values)
                raw_vectors = [(key, factory()[0]) for key, factory, _rows, _cols in vectors]
                raw_vectors = [(key, value if i == 0 and layer % 4 != 3 else value.reshape(-1))
                               for i, (key, value) in enumerate(raw_vectors)]
                branch = (gpu.gated_delta(prefix, mixed, attention, raw_vectors) if layer % 4 != 3
                          else gpu.sparse_attention(prefix, mixed, attention, raw_vectors,
                                                    position=base, context=context))
                values = values + (injection[:, :, None] * branch[:, None, :]).reshape(-1, 10240)
                mixed, injection = mix(after, values)
                return values + (injection[:, :, None] * gpu.resident_moe(prefix, mixed)[:, None, :]).reshape(-1, 10240)

            expected = components(incoming, position)
            continuation = incoming[:1] * np.float32(0.8)
            expected_next = components(continuation, position + steps)
            gpu.reset_state()
            gpu.stats = {}
            actual = gpu.decode_span(layer, layer + 1, incoming, position=position)
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=3e-5)
            assert gpu.stats["prefill_hc_device_seconds"] > 0
            assert gpu.stats["prefill_moe_device_seconds"] > 0
            assert gpu.stats["prefill_gdn_device_seconds" if layer == 0 else "prefill_qsa_device_seconds"] > 0
            np.testing.assert_allclose(
                gpu.decode_span(layer, layer + 1, continuation, position=position + steps),
                expected_next, rtol=2e-3, atol=3e-5,
            )
            gpu.reset_state()
            np.testing.assert_allclose(gpu.decode_span(layer, layer + 1, incoming, position=position), expected,
                                       rtol=2e-3, atol=3e-5)
            with pytest.raises(ValueError, match="span"):
                gpu.decode_span(layer, layer + 1, incoming[:, :1], position=position)
            assert gpu.memory()[0] <= gpu.memory()[1]

            # Compare the captured and replayed decode graphs bitwise across storage
            # locations. Refill after prefill, rewind, stage mutation and owner zero.
            def record(values):
                if host_attention:
                    assert values.tobytes() == device_outputs.pop(0).tobytes()
                else:
                    device_outputs.append(values.copy())

            for base in (position + steps, position + steps + 1, 7, position + steps + 2):
                record(gpu.decode_span(layer, layer + 1, continuation, position=base))
            with pytest.raises(RuntimeError, match="span"):
                gpu.decode_span(layer, layer + 2, continuation, position=8)
            if host_attention:
                gpu._check(gpu._lib.sd_zero(gpu._handle, stage[0]))
            record(gpu.decode_span(layer, layer + 1, continuation, position=position + steps + 3))
            gpu.reset_state()
            record(gpu.decode_span(layer, layer + 1, continuation, position=0))
            record(gpu.decode_span(layer, layer + 1, continuation, position=1))
            gpu.reset_state()
            record(gpu.decode_span(layer, layer + 1, incoming, position=position))
            record(gpu.decode_span(layer, layer + 1, continuation, position=position + steps))
            if host_attention:
                gpu._states.pop("@test:stage")
                gpu._cache.pop("@test:stage", None)
                gpu._check(gpu._lib.sd_release(gpu._handle, stage[0]))
            record(gpu.decode_span(layer, layer + 1, continuation, position=position + steps + 1))
