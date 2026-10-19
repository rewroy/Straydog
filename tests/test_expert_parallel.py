"""Two GPUs evaluating each layer's experts at once: the same layer as one GPU, within tolerance,
and verification still bitwise equal to token-by-token decode."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.devices import GpuGroup
from straydog.gpu import GpuBackend

pytestmark = pytest.mark.gpu


def _library() -> Path:
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    try:
        with GpuBackend(Path(library), device=1, budget_bytes=64 * 1024**2):
            pass
    except (RuntimeError, ValueError):
        pytest.skip("two GPUs required")
    return Path(library)


def _layer(layer: int, experts: int):
    rng = np.random.default_rng(613 + layer)
    prefix = f"blk.{layer}."

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
    # Router rows scaled up so the top ten mixes experts from both halves.
    moe += [matrix("router", experts, 2560), matrix("mix", 1, 2560)]
    if layer % 4 != 3:
        shapes = [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560), (2560, 6144)]
        decay = -np.ones((1, 48), np.float32)
        vectors = [matrix("conv", 10240, 4, norm=True), matrix("bias", 1, 48, norm=True),
                   (prefix + "decay", lambda: (decay, 0), 1, 48), matrix("norm", 1, 128, norm=True)]
    else:
        shapes = [(12288, 2560), (512, 2560), (512, 2560), (512, 2560), (128, 2560), (2560, 6144)]
        vectors = [matrix("norm" + str(i), 1, size, norm=True) for i, size in enumerate([256, 256, 128, 128])]
    attention = [matrix("attention" + str(i), *shape) for i, shape in enumerate(shapes)]
    return moe, hc("hc_attn") + attention + vectors + hc("hc_ffn")


@pytest.mark.parametrize("layer,position", [(0, 0), (3, 2051)])
def test_expert_parallel_layer_matches_one_gpu_and_verification_stays_exact(layer, position):
    library = _library()
    context, experts, steps = 4096, 20, 273
    moe, operands = _layer(layer, experts)
    rng = np.random.default_rng(97)
    prompt = rng.normal(scale=0.05, size=(steps, 10240)).astype(np.float32)
    rows = rng.normal(scale=0.05, size=(4, 10240)).astype(np.float32)
    results = {}
    for name in ("one", "parallel"):
        if name == "one":
            gpu = GpuBackend(library, budget_bytes=1024**3)
        else:
            gpu = GpuGroup(library, devices=[0, 1], budget_bytes=1024**3)
        with gpu:
            gpu.configure_gemm(precision="fp16", minimum_batch=2)
            if name == "parallel":
                assert gpu.expert_parallel and gpu.device_of(layer) == 0
                gpu.reserve_model_state(context, attention=True, prefill=steps)
            else:
                gpu.reserve_model_state(context, attention=True, layers=[layer], prefill=steps)
            gpu.register_resident_moe(f"blk.{layer}.", moe, experts=experts)
            gpu.register_decoder_layer(layer, operands, context=context)
            prefilled = gpu.decode_span(layer, layer + 1, prompt, position=position)
            base = position + steps
            sequential = [gpu.decode_span(layer, layer + 1, rows[i:i + 1], position=base + i)[0] for i in range(4)]
            results[name] = (prefilled, np.stack(sequential))
            if name == "parallel":
                gpu.reset_state()
                gpu.decode_span(layer, layer + 1, prompt, position=position)
                verified = gpu.verify_span(layer, layer + 1, rows, position=base)
                assert np.array_equal(verified, results[name][1])   # verification == decode, bitwise
                gpu.verify_commit(layer, layer + 1, 4)
                # The links' message counts stay in step across graph replays.
                again = gpu.decode_span(layer, layer + 1, rows[:1], position=base + 4)
                assert np.isfinite(again).all()
    for one, parallel in zip(results["one"], results["parallel"], strict=True):
        # The two partial sums are added in a different grouping than one GPU's single combine.
        np.testing.assert_allclose(parallel, one, rtol=2e-3, atol=3e-5)


def test_split_vocabulary_head_equals_one_gpus_head():
    library = _library()
    rng = np.random.default_rng(29)
    norm = rng.uniform(0.7, 1.3, (1, 10240)).astype(np.float32)
    down, up = (rng.normal(scale=0.01, size=shape).astype(np.float16) for shape in [(320, 10240), (10240, 320)])
    vocabulary = rng.normal(scale=0.01, size=(1001, 2560)).astype(np.float16)
    rows = rng.normal(scale=0.05, size=(4, 10240)).astype(np.float32)

    def head(weights):
        return [("head_norm", lambda: (norm, 0), 1, 10240), ("head_down", lambda: (down, 1), 320, 10240),
                ("head_up", lambda: (up, 1), 10240, 320),
                (f"vocabulary{id(weights)}", lambda: (weights, 1), 1001, 2560)]

    def one_gpu(weights):
        with GpuBackend(library, budget_bytes=512 * 1024**2) as gpu:
            return gpu.output_head(head(weights), rows, logits=True)

    tokens, _ = one_gpu(vocabulary)
    # The first row's maximum, copied to the other half of the vocabulary, makes a tie across the halves.
    tied = vocabulary.copy()
    tied[3 if tokens[0] >= 500 else 507] = vocabulary[tokens[0]]
    for weights in (vocabulary, tied):
        expected_tokens, expected_logits = one_gpu(weights)
        with GpuGroup(library, devices=[0, 1], budget_bytes=512 * 1024**2) as group:
            group.configure_gemm(precision="fp16", minimum_batch=2)
            assert group.expert_parallel
            group.reserve_model_state(1024, attention=True, prefill=16)
            actual_tokens, actual_logits = group.output_head(head(weights), rows, logits=True)
            assert np.array_equal(actual_tokens, expected_tokens) and np.array_equal(actual_logits, expected_logits)
            for steps in (1, 3):
                assert np.array_equal(group.output_head(head(weights), rows[:steps]), expected_tokens[:steps])
    assert expected_tokens[0] == min(tokens[0], 3 if tokens[0] >= 500 else tokens[0])
