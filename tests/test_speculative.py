"""Speculative decoding: exact multi-token verification, rollback, drafting helpers and the drafter asset."""
from __future__ import annotations

import ctypes as ct
import hashlib
import io
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from straydog.library import load_library
from straydog.numeric import softmax
from straydog.model import StraydogModel, draft_length


def _library() -> Path:
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    return Path(library)


@pytest.mark.gpu
@pytest.mark.parametrize("layer,position", [(0, 0), (3, 2051), (48, 5)])
def test_verification_matches_token_by_token_decode_and_rolls_back(layer, position):
    from straydog.gpu import GpuBackend

    library = _library()
    rng = np.random.default_rng(311)
    context, experts = 4096, 10
    prefix = f"blk.{layer}."
    linear = layer % 4 != 3 and layer < 48

    def matrix(key, rows, cols, *, norm=False):
        raw = (rng.uniform(0.7, 1.3, (rows, cols)) if norm else rng.normal(scale=0.004, size=(rows, cols)))
        raw = raw.astype(np.float32 if norm else np.float16)
        return (prefix + key, lambda raw=raw: (raw, 0 if norm else 1), rows, cols)

    def hc(name):
        return [matrix(name + str(i), rows, cols, norm=i == 0)
                for i, (rows, cols) in enumerate([(1, 10240), (320, 10240), (10240, 320), (4, 10240)])]

    expert_shapes = [(640, 2560), (640, 2560), (2560, 640)]
    moe = [matrix("bank" + str(i), rows * experts, cols) for i, (rows, cols) in enumerate(expert_shapes)]
    moe += [matrix("shared" + str(i), rows, cols) for i, (rows, cols) in enumerate(expert_shapes)]
    moe += [matrix("router", experts, 2560), matrix("mix", 1, 2560)]
    if linear:
        shapes = [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560), (2560, 6144)]
        decay = -np.ones((1, 48), np.float32)
        vectors = [matrix("conv", 10240, 4, norm=True), matrix("bias", 1, 48, norm=True),
                   (prefix + "decay", lambda: (decay, 0), 1, 48), matrix("norm", 1, 128, norm=True)]
    else:
        shapes = [(12288, 2560), (512, 2560), (512, 2560), (512, 2560), (128, 2560), (2560, 6144)]
        vectors = [matrix("norm" + str(i), 1, size, norm=True) for i, size in enumerate([256, 256, 128, 128])]
    attention = [matrix("attention" + str(i), *shape) for i, shape in enumerate(shapes)]
    rows = rng.normal(scale=0.05, size=(4, 10240)).astype(np.float32)
    with GpuBackend(library, budget_bytes=1024**3) as gpu:
        gpu.configure_gemm(precision="fp16", minimum_batch=2)
        gpu.reserve_model_state(context, attention=True, layers=[layer], prefill=16)
        gpu.register_resident_moe(prefix, moe, experts=experts)
        gpu.register_decoder_layer(layer, hc("hc_attn") + attention + vectors + hc("hc_ffn"), context=context)
        warmup = rng.normal(scale=0.05, size=(1, 10240)).astype(np.float32)

        def fresh():
            gpu.reset_state()
            gpu.decode_span(layer, layer + 1, warmup, position=position)

        fresh()
        sequential = [gpu.decode_span(layer, layer + 1, rows[i:i + 1], position=position + 1 + i)[0] for i in range(4)]
        fresh()
        verified = gpu.verify_span(layer, layer + 1, rows, position=position + 1)
        assert np.array_equal(verified, np.stack(sequential))
        gpu.verify_commit(layer, layer + 1, 2)
        assert np.array_equal(gpu.decode_span(layer, layer + 1, rows[2:3], position=position + 3)[0], sequential[2])
        with pytest.raises(RuntimeError, match="commit"):
            gpu.verify_commit(layer, layer + 1, 1)

        head = [matrix("head_norm", 1, 10240, norm=True), matrix("head_down", 320, 10240),
                matrix("head_up", 10240, 320), matrix("vocabulary", 1000, 2560)]
        tokens, logits = gpu.output_head(head, rows[:3], logits=True)
        for i in range(3):
            single, single_logits = gpu.output_head(head, rows[i:i + 1], logits=True)
            assert np.array_equal(single_logits[0], logits[i]) and single[0] == tokens[i] == np.argmax(logits[i])


@pytest.mark.gpu
def test_mtp_prologue_matches_reference():
    from straydog.gpu import GpuBackend

    rng = np.random.default_rng(5)
    norms = [rng.uniform(0.7, 1.3, size).astype(np.float32) for size in (2560, 10240)]
    fcs = [rng.normal(scale=0.02, size=(2560, 2560)).astype(np.float16) for _ in range(2)]
    operands = [("norm_e", lambda: (norms[0][None], 0), 1, 2560), ("fc_e", lambda: (fcs[0], 1), 2560, 2560),
                ("norm_h", lambda: (norms[1][None], 0), 1, 10240), ("fc_h", lambda: (fcs[1], 1), 2560, 2560)]
    for steps in (1, 3, 9):
        embeddings = rng.normal(size=(steps, 2560)).astype(np.float32)
        hidden = rng.normal(size=(steps, 10240)).astype(np.float32)

        def norm(x, w):
            return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w

        projected = norm(embeddings, norms[0]) @ fcs[0].astype(np.float32).T
        streams = (norm(hidden, norms[1]).reshape(-1, 2560) @ fcs[1].astype(np.float32).T).reshape(steps, 4, 2560)
        expected = (streams + projected[:, None, :]).reshape(steps, 10240)
        with GpuBackend(_library(), budget_bytes=1024**3) as gpu:
            gpu.configure_gemm(precision="fp32", minimum_batch=2)
            np.testing.assert_allclose(gpu.mtp_prologue(operands, embeddings, hidden), expected, rtol=2e-3, atol=2e-4)


def _q2_0_reference(bits: np.ndarray) -> bytes:
    out = bytearray()
    for block in (bits.astype(np.uint32) << 16).view(np.float32).reshape(-1, 64):
        amax = np.float32(np.abs(block).max())
        best, best_scale, best_codes = np.float32(np.inf), np.float16(0), np.zeros(64, np.int64)
        for step in range(33):
            fraction = np.float32(0.2) + np.float32(0.8) * np.float32(step) / np.float32(32)
            for sign in (1, -1):
                scale16 = np.float16(np.float32(sign) * amax * fraction / np.float32(2))
                scale = np.float32(scale16)
                if amax == 0 or scale == 0:
                    continue
                codes = np.clip(np.rint(block / scale), -1, 2)
                error = np.float32(0)
                for code, value in zip(codes, block, strict=True):
                    delta = np.float32(np.float32(code) * scale - value)
                    error = np.float32(error + np.float32(delta * delta))
                if error < best:
                    best, best_scale, best_codes = error, scale16, codes.astype(np.int64)
        out += np.array([best_scale], np.float16).tobytes()
        packed = (best_codes + 1).reshape(16, 4) << np.array([0, 2, 4, 6])
        out += packed.sum(axis=1).astype(np.uint8).tobytes()
    return bytes(out)


def test_native_q2_0_quantizer_matches_reference():
    lib = load_library(_library())
    lib.sd_quantize_q2_0.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p]
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(scale=0.02, size=64 * 6), np.zeros(64), np.full(64, 0.5)]).astype(np.float32)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    output = np.empty(len(bits) // 64 * 18, np.uint8)
    assert lib.sd_quantize_q2_0(bits.ctypes.data, bits.size, output.ctypes.data) == 0
    assert output.tobytes() == _q2_0_reference(bits)


def test_draft_length_balances_acceptance_and_cost():
    assert draft_length(0.95, 3) == 3
    assert draft_length(0.2, 3) == 0
    assert draft_length(0.95, 1) == 1
    lengths = [draft_length(a, 3) for a in (0.3, 0.5, 0.7, 0.9)]
    assert lengths == sorted(lengths)


def test_acceptance_is_greedy_exact_and_samples_the_residual():
    targets = np.array([5, 6, 7, 8])
    assert StraydogModel._accept([5, 6, 9], targets, None, 0, None) == (2, 7)
    assert StraydogModel._accept([5, 6, 7], targets, None, 0, None) == (3, 8)
    scores = np.full((2, 4), -30.0, np.float32)
    scores[0, 1], scores[1, 2] = 30.0, 30.0   # target puts all mass on 1, then 2
    rng = np.random.default_rng(0)
    assert StraydogModel._accept([1], None, scores, 1.0, rng) == (1, 2)
    accepted, token = StraydogModel._accept([3], None, scores, 1.0, rng)
    assert accepted == 0 and token == 1   # draft 3 has no target mass; the residual excludes it
    assert softmax(scores[0])[3] < 1e-20


def test_drafter_fetch_authenticates_each_tensor(tmp_path):
    from straydog.mtp import fetch_mtp_source

    payload = {"mtp.a.weight": np.arange(8, dtype=np.uint16).tobytes(),
               "mtp.b.weight": np.ones(4, np.uint16).tobytes()}
    header, offset = {}, 0
    for name, data in payload.items():
        header[name] = {"dtype": "BF16", "shape": [len(data) // 2], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header).encode()
    shard = struct.pack("<Q", len(blob)) + blob + b"".join(payload.values())

    def opener(request, timeout):
        first, last = map(int, request.headers["Range"].split("=")[1].split("-"))
        return io.BytesIO(shard[first:last + 1])

    pins = {name: {"shard": "model.safetensors", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payload.items()}
    drafter = {"repository": "Qwen/Qwen3.8-Flash-Next", "revision": "0" * 40, "source_tensors": pins}
    fetch_mtp_source(drafter, tmp_path / "source.safetensors", opener=opener)
    size = struct.unpack("<Q", (tmp_path / "source.safetensors").read_bytes()[:8])[0]
    assert set(json.loads((tmp_path / "source.safetensors").read_bytes()[8:8 + size])) == set(payload)
    pins["mtp.b.weight"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authentication"):
        fetch_mtp_source(drafter, tmp_path / "tampered.safetensors", opener=opener)
    assert not (tmp_path / "tampered.safetensors").exists()


def test_drafter_fetch_splits_tensors_into_pieces_on_several_connections(tmp_path, monkeypatch):
    from straydog import mtp as straydog_mtp

    monkeypatch.setattr(straydog_mtp, "_PIECE", 6)
    payload = {"mtp.a.weight": np.arange(13, dtype=np.uint16).tobytes(),
               "mtp.b.weight": np.ones(4, np.uint16).tobytes()}
    header, offset = {}, 0
    for name, data in payload.items():
        header[name] = {"dtype": "BF16", "shape": [len(data) // 2], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header).encode()
    shard = struct.pack("<Q", len(blob)) + blob + b"".join(payload.values())
    ranges = []

    def opener(request, timeout):
        first, last = map(int, request.headers["Range"].split("=")[1].split("-"))
        ranges.append(last + 1 - first)
        return io.BytesIO(shard[first:last + 1])

    pins = {name: {"shard": "model.safetensors", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payload.items()}
    drafter = {"repository": "Qwen/Qwen3.8-Flash-Next", "revision": "0" * 40, "source_tensors": pins}
    counted = []
    straydog_mtp.fetch_mtp_source(drafter, tmp_path / "source.safetensors", opener=opener, progress=counted.append)
    written = (tmp_path / "source.safetensors").read_bytes()
    size = struct.unpack("<Q", written[:8])[0]
    # The same layout a single sequential fetch wrote: the header, then the tensors in name order.
    assert written[8 + size:] == payload["mtp.a.weight"] + payload["mtp.b.weight"]
    assert sum(counted) == 34 and max(ranges[2:]) == 6 and sorted(ranges[2:]) == [2, 2, 6, 6, 6, 6, 6]
    assert not (tmp_path / "source.safetensors.part").exists()


def test_drafter_experts_quantized_in_groups_match_one_call(tmp_path, monkeypatch):
    from straydog import mtp as straydog_mtp

    rng = np.random.default_rng(1)
    meta, blobs, offset = {}, [], 0
    for origin, transform in dict.fromkeys(straydog_mtp.TENSOR_PLAN.values()):
        if transform.startswith("experts:"):
            shape = [3, 2560, 64]
        elif transform.startswith("rows:"):
            shape = [640, 64]
        else:
            shape = [64] if transform in {"norm", "vector"} else [8, 64]
        if origin in meta:
            continue
        data = (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        meta[origin] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    blob = json.dumps(meta).encode()
    source = tmp_path / "source.safetensors"
    source.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))
    # A stand-in quantizer that depends on every value and its order, 18 bytes per 64 values like Q2_0.
    library = SimpleNamespace(sd_quantize_q2_0=SimpleNamespace())
    monkeypatch.setattr(straydog_mtp, "load_library", lambda path: library)
    monkeypatch.setattr(straydog_mtp, "_quantize_q2_0", lambda bits, lib: b"".join(
        hashlib.sha256(block.tobytes()).digest()[:18] for block in np.ascontiguousarray(bits).reshape(-1, 64)))
    outputs, steps = [], []
    for group in (1, 512):
        monkeypatch.setattr(straydog_mtp, "_EXPERT_GROUP", group)
        target = tmp_path / f"drafter-{group}.gguf"
        straydog_mtp.convert_mtp(source, target, tmp_path / "lib", revision="r",
                                 progress=lambda done, total: steps.append((group, done, total)))
        outputs.append(target.read_bytes())
    assert outputs[0] == outputs[1]
    total = 3 * 64 * (640 + 640 + 2560)
    assert [step for step in steps if step[0] == 1][-1] == (1, total, total)
    assert len([step for step in steps if step[0] == 1]) == 9 and len([step for step in steps if step[0] == 512]) == 3
