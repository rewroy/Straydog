"""Fused causal QSA is compared with CPU norms, pooling, selection and attention."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend
from straydog.numeric import rms_norm, rotary, sigmoid, softmax

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("steps,precision,base,mixed", [(1, "fp32", 0, False), (9, "fp16", 0, False),
                                                        (273, "fp16", 0, False), (3, "fp32", 2051, False),
                                                        (1, "fp32", 0, True)])
def test_fused_qsa_matches_cpu_selection_and_causal_state(steps, precision, base, mixed):
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(1531)
    context = 4096 if base else 512
    pool = context // 4 + 1
    shapes = [(12288, 2560), (512, 2560), (512, 2560), (512, 2560), (128, 2560), (2560, 6144)]
    raw = [rng.normal(scale=0.01, size=shape).astype(np.float16) for shape in shapes]
    kinds = [1] * 6
    decoded = [w.astype(np.float32) for w in raw]
    if mixed:
        from straydog.weights import dequantize

        kinds = [23, 14, 13, 30, 30, 1]
        for i, (rows, cols) in enumerate(shapes[:5]):
            if i < 3:
                size = [136, 210, 176][i]
                raw[i] = rng.integers(0, 256, (rows * cols // 256, size), dtype=np.uint8)
                offset = 208 if i == 1 else 0
                raw[i][:, offset:offset + 2] = np.frombuffer(np.float16(0.0001).tobytes(), np.uint8)
                if i == 2:
                    raw[i][:, 2:4] = np.frombuffer(np.float16(0.00015).tobytes(), np.uint8)
            else:
                raw[i] = (raw[i].astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            decoded[i] = dequantize(raw[i].view(np.uint8).reshape(-1), kinds[i]).reshape(rows, cols)
    if base:
        raw[3] = np.full(shapes[3], 0.002, np.float16)
        decoded[3] = raw[3].astype(np.float32)
    matrices = [(str(i), lambda i=i: (raw[i], kinds[i]), rows, cols) for i, (rows, cols) in enumerate(shapes)]
    norms = [rng.uniform(0.7, 1.3, size).astype(np.float32) for size in [256, 256, 128, 128]]
    vectors = [("norm" + str(i), value) for i, value in enumerate(norms)]
    values = rng.uniform(0.01, 0.2, (steps, 2560)).astype(np.float32)
    keys, vals = np.zeros((context, 2, 256), np.float32), np.zeros((context, 2, 256), np.float32)
    blocks, history = np.zeros((pool, 128), np.float32), np.zeros((4, 128), np.float32)
    if base:
        keys[:base] = rng.normal(scale=0.1, size=keys[:base].shape)
        vals[:base] = rng.normal(scale=0.1, size=vals[:base].shape)
        blocks[:base // 4] = np.arange(1, base // 4 + 1)[:, None] * np.float32(0.002)
        history[:] = rng.normal(scale=0.1, size=history.shape)
    initial = [array.copy() for array in (keys, vals, blocks, history)]

    def reference(incoming, start):
        def project(index, x):
            if precision == "fp16" and len(incoming) >= 8:
                x = x.astype(np.float16).astype(np.float32)
            return x @ decoded[index].T

        projections = [project(i, incoming) for i in range(5)]
        attended = []
        for offset, (q, k, v, iq, ik) in enumerate(zip(*projections, strict=True)):
            pos = start + offset
            gate = q.reshape(24, 512)
            query = rotary(rms_norm(gate[:, :256], norms[0]), pos)
            keys[pos] = rotary(rms_norm(k.reshape(2, 256), norms[1]), pos)
            vals[pos] = v.reshape(2, 256)
            index_query = rotary(rms_norm(iq.reshape(4, 128), norms[2]), pos)
            history[pos % 4] = ik
            complete = (pos + 1) // 4
            if (pos + 1) % 4 == 0:
                blocks[complete - 1] = rotary(rms_norm(np.mean(history, axis=0), norms[3]), pos - 3)
            if complete > 512:
                scores = np.maximum(blocks[:complete] @ index_query.T, 0).sum(axis=1)
                chosen = np.argpartition(scores, -512)[-512:]
            else:
                chosen = np.arange(complete)
            selected = np.concatenate(
                ((chosen[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(complete * 4, pos + 1))
            )
            output = []
            for head in range(24):
                scores = keys[selected, head // 12] @ query[head] / np.float32(16)
                output.append(softmax(scores) @ vals[selected, head // 12])
            attended.append((np.stack(output) * sigmoid(gate[:, 256:])).reshape(-1))
        return project(5, np.stack(attended))

    def kv_packed():
        return np.concatenate((keys.transpose(1, 0, 2).reshape(-1), vals.transpose(1, 0, 2).reshape(-1)))

    with GpuBackend(Path(library), budget_bytes=1024**3) as gpu:
        gpu.configure_gemm(precision=precision)
        if base:
            for name, data, rows, cols in [
                ("@qsa:blk.3.", kv_packed(), context * 4, 256),
                ("@indexer:blk.3.", np.concatenate((blocks.T.reshape(-1), history.reshape(-1))), pool * 128 + 512, 1),
            ]:
                gpu._states[name] = gpu._prepare(name, lambda data=data: (data, 0), rows, cols, set())
        expected = reference(values, base)
        actual = gpu.sparse_attention("blk.3.", values, matrices, vectors, position=base, context=context)
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=4e-5)
        packed = np.empty(pool * 128 + 512, np.float32)
        entry = gpu._states["@indexer:blk.3."]
        gpu._check(gpu._lib.sd_download(gpu._handle, entry[0], packed.ctypes.data, packed.nbytes))
        np.testing.assert_allclose(
            packed, np.concatenate((blocks.T.reshape(-1), history.reshape(-1))), rtol=3e-3, atol=4e-5
        )
        continuation = values[:1]
        expected = reference(continuation, base + steps)
        np.testing.assert_allclose(
            gpu.sparse_attention("blk.3.", continuation, matrices, vectors, position=base + steps, context=context),
            expected, rtol=3e-3, atol=4e-5,
        )
        gpu.reset_state()
        for target, original in zip((keys, vals, blocks, history), initial, strict=True):
            target[:] = original if not base else 0
        np.testing.assert_allclose(
            gpu.sparse_attention("blk.3.", values[:1], matrices, vectors, position=0, context=context),
            reference(values[:1], 0), rtol=3e-3, atol=4e-5,
        )
        assert gpu.memory()[0] <= gpu.memory()[1]
        with pytest.raises(ValueError, match="geometry"):
            gpu.sparse_attention("blk.3.", values, matrices, vectors, position=context, context=context)


@pytest.mark.parametrize("stage_rows", [512, 8192])
def test_host_mapped_kv_runs_the_same_kernels_bitwise(stage_rows):
    """Long-context K/V in mapped host memory: identical results, outside the VRAM budget, reset like VRAM state."""
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(7)
    context = 16384
    budget = 768 * 1024**2   # two VRAM K/V owners plus dense projections must fit in the reference run
    reserved_stage_rows = min(context, budget // 16 // 4096)
    shapes = [(12288, 2560), (512, 2560), (512, 2560), (512, 2560), (128, 2560), (2560, 6144)]
    raw = [rng.normal(scale=0.01, size=shape).astype(np.float16) for shape in shapes]
    matrices = [(str(i), lambda i=i: (raw[i], 1), rows, cols) for i, (rows, cols) in enumerate(shapes)]
    vectors = [("norm" + str(i), rng.uniform(0.7, 1.3, size).astype(np.float32))
               for i, size in enumerate([256, 256, 128, 128])]
    prompt = rng.uniform(0.01, 0.2, (2100, 2560)).astype(np.float32)   # past 2048: sparse selection
    following = rng.uniform(0.01, 0.2, (3, 2560)).astype(np.float32)
    later = rng.uniform(0.01, 0.2, (300, 2560)).astype(np.float32)

    def run(host: bool):
        with GpuBackend(Path(library), budget_bytes=budget) as gpu:
            gpu.configure_gemm(precision="fp16")
            if host and not gpu.host_state_supported():
                pytest.skip("mapped host state is not enabled for this backend")
            used = gpu.memory()[0]
            gpu.reserve_model_state(context, attention=True, layers=[3, 7], prefill=256, host_attention=host)
            assert ("@kv:stage" in gpu._states) == host
            outputs = [gpu.memory()[0] - used]
            if host:
                stage = gpu._state("@test:stage", stage_rows * 4, 256)
                gpu._check(gpu._lib.sd_kv_stage(gpu._handle, stage[0], stage_rows))

            def attend(prefix, values, position):
                if prefix == "blk.7.":
                    values = values * np.float32(0.9)
                outputs.append(gpu.sparse_attention(prefix, values, matrices, vectors,
                                                    position=position, context=context))

            # Two owners, staged/direct prefill, decode within/beyond the per-owner mirror,
            # then prefill again. The small stage splits a sparse selection across VRAM/host.
            for prefix in ("blk.3.", "blk.7."):
                attend(prefix, following[:1], 0)
            for prefix in ("blk.3.", "blk.7."):
                attend(prefix, prompt, 0)
            for offset, row in enumerate(following):
                for prefix in ("blk.3.", "blk.7."):
                    attend(prefix, row[None], len(prompt) + offset)
            if host:
                # Prove that both owners occupy the existing stage, rather than
                # merely comparing two implementations that both read host K/V.
                packed = np.empty(stage_rows * 1024, np.float32)
                gpu._check(gpu._lib.sd_download(gpu._handle, stage[0], packed.ctypes.data, packed.nbytes))
                rows = stage_rows // 2
                valid = min(len(prompt) + len(following), rows)
                copies = [part[:, :valid].tobytes() for part in packed.reshape(2, 4, rows, 256)]
                originals = []
                for prefix in ("blk.3.", "blk.7."):
                    source = np.empty((4, context, 256), np.float32)
                    state = gpu._states["@qsa:" + prefix]
                    gpu._check(gpu._lib.sd_download(gpu._handle, state[0], source.ctypes.data, source.nbytes))
                    originals.append(source[:, :valid].tobytes())
                assert sorted(copies) == sorted(originals)
            for prefix in ("blk.3.", "blk.7."):
                attend(prefix, later, len(prompt) + len(following))
                attend(prefix, following[:1], len(prompt) + len(following) + len(later))
                # Rewrite before the valid end, without resetting the rest of the state.
                attend(prefix, following[1:2], 7)
                attend(prefix, following[2:3], len(prompt) + len(following) + len(later) + 1)
            # Mutating the stage itself and copying/zeroing an owner must invalidate mirrors.
            if host:
                gpu._check(gpu._lib.sd_zero(gpu._handle, stage[0]))
            attend("blk.3.", following[:1], 2405)
            attend("blk.7.", following[:1], 2405)
            if host:
                gpu._state("@test:empty-stage", stage_rows * 4, 256)
                gpu.copy_state("@test:stage", "@test:empty-stage")
            attend("blk.7.", following[:1], 2406)
            gpu.copy_state("@qsa:blk.3.", "@qsa:blk.7.")
            attend("blk.3.", following[:1], 2406)
            gpu._check(gpu._lib.sd_zero(gpu._handle, gpu._states["@qsa:blk.7."][0]))
            attend("blk.7.", following[:1], 2407)
            if host:
                # Re-registering the same stage resets its layout and all valid ends.
                gpu._check(gpu._lib.sd_kv_stage(gpu._handle, stage[0], stage_rows))
            attend("blk.3.", following[:1], 2408)
            outputs.append(gpu.attention_sequence(
                "blk.3.", np.ones((1, 24, 256), np.float32),
                np.full((1, 2, 256), 0.3, np.float32), np.full((1, 2, 256), 0.7, np.float32),
                [np.array([5], np.int32)], position=5, context=context,
            ))
            # Below 2048 all causal rows are selected: the rewritten row 5 is
            # guaranteed to be read, including from the small mirror's prefix.
            attend("blk.3.", following[:1], 512)
            for prefix in ("blk.3.", "blk.7."):
                entry = gpu._states.pop("@qsa:" + prefix)
                gpu._cache.pop("@qsa:" + prefix, None)
                gpu._check(gpu._lib.sd_release(gpu._handle, entry[0]))
                gpu._state("@qsa:" + prefix, context * 4, 256, host=host)
            if host:
                gpu._check(gpu._lib.sd_kv_stage(gpu._handle, stage[0], stage_rows))
            attend("blk.3.", following[:1], 2410)
            gpu.reset_state()
            attend("blk.3.", following[:1], 0)
            if host:
                # Releasing the stage must leave a safe direct-host path.
                gpu._states.pop("@test:stage")
                gpu._cache.pop("@test:stage", None)
                gpu._check(gpu._lib.sd_release(gpu._handle, stage[0]))
            attend("blk.3.", following[:1], 1)
            return outputs

    device, host = run(False), run(True)
    # Both K/V allocations (including tails) leave VRAM; only the original bounded stage enters it.
    assert device[0] - host[0] == (2 * context - reserved_stage_rows) * 4 * 256 * 4 + 64
    for index, (in_vram, in_host) in enumerate(zip(device[1:], host[1:], strict=True)):
        assert in_vram.tobytes() == in_host.tobytes(), f"output {index}, stage rows {stage_rows}"
