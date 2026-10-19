"""Partially resident experts match the fully resident plan through streaming and caching."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from straydog.gpu import GpuBackend

pytestmark = pytest.mark.gpu


def _q2(rng, rows, cols):
    raw = rng.integers(0, 256, (rows * cols // 64, 18), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(rng.uniform(0.002, 0.02, rows * cols // 64).astype(np.float16).tobytes(),
                               np.uint8).reshape(-1, 2)
    return raw.reshape(-1)


def test_hybrid_prefill_streams_and_caches_like_resident_banks(tmp_path):
    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    experts, steps = 16, 64
    rng = np.random.default_rng(97)
    banks = [_q2(rng, experts * 640, 2560), _q2(rng, experts * 640, 2560), _q2(rng, experts * 2560, 640)]
    shared = [_q2(rng, 640, 2560), _q2(rng, 640, 2560), _q2(rng, 2560, 640)]
    router = rng.normal(scale=0.05, size=(experts, 2560)).astype(np.float16)
    mix = rng.normal(scale=0.05, size=2560).astype(np.float32)
    values = rng.normal(scale=0.5, size=(steps, 2560)).astype(np.float32)
    dense = [(f"shared:{i}", lambda i=i: (shared[i], 42), *shape)
             for i, shape in enumerate([(640, 2560), (640, 2560), (2560, 640)])]
    dense += [("router", lambda: (router, 1), experts, 2560), ("mix", lambda: (mix, 0), 1, 2560)]

    def backend():
        gpu = GpuBackend(Path(library), budget_bytes=2 * 1024**3)
        gpu.configure_gemm(minimum_batch=2, precision="fp16")
        gpu.reserve_model_state(1024, attention=False, layers=[0], prefill=steps)
        return gpu

    with backend() as gpu:
        bank_matrices = [(f"bank:{role}", lambda role=role: (banks[role], 42), rows, cols)
                         for role, (rows, cols) in enumerate([(experts * 640, 2560), (experts * 640, 2560),
                                                              (experts * 2560, 640)])]
        gpu.register_resident_moe("blk.0.", bank_matrices + dense, experts=experts)
        try:
            expected = gpu.resident_moe("blk.0.", values)
        except RuntimeError as exc:
            pytest.skip(f"grouped prefill unavailable on this GPU: {exc}")
    with GpuBackend(Path(library), budget_bytes=2 * 1024**3) as gpu:
        gpu.reserve_model_state(1024, attention=False, layers=[0], prefill=steps)
        assert not gpu.hybrid_supported()  # FP32 GEMM: the model keeps the bounded expert cache
        with pytest.raises(RuntimeError, match="fp16 on an RDNA4 or CUDA"):
            gpu.register_hybrid_moe("blk.0.", dense, banks, experts=experts)
    with backend() as gpu:
        if not gpu.hybrid_supported():
            pytest.skip("partial residency needs the grouped FP16 prefill path")
        gpu.register_hybrid_moe("blk.0.", dense, banks, experts=experts)
        assert gpu.create_expert_slots(stage=4, margin_bytes=0) > 0
        streamed = gpu.resident_moe("blk.0.", values)
        counts = gpu.hybrid_counts()
        assert counts["streamed_experts"] > 0 and counts["cached_experts"] == 0
        np.testing.assert_allclose(streamed, expected, rtol=1e-5, atol=1e-6)
        gpu.rebalance()
        assert gpu.hybrid_counts()["cached_experts"] > 0
        cached = gpu.resident_moe("blk.0.", values)
        np.testing.assert_allclose(cached, expected, rtol=1e-5, atol=1e-6)
        assert gpu.hybrid_counts()["streamed_experts"] - counts["streamed_experts"] < counts["streamed_experts"]
        with pytest.raises(RuntimeError, match="hybrid decoder"):
            gpu.resident_moe("blk.0.", values[:1])
        with pytest.raises(ValueError, match="hybrid MoE geometry"):
            gpu.register_hybrid_moe("blk.1.", dense, banks[:2], experts=experts)
    # With a CPU engine attached, the CPU workers gather absent experts that are not in the page cache
    # into page-locked staging (more than one double-buffered chunk of the four staging slots), others
    # copy directly: the same bytes, the same result. The banks are mapped from a file whose pages are
    # dropped first.
    from straydog.cpu import CpuExperts

    path = tmp_path / "banks"
    with path.open("wb") as file:
        for bank in banks:
            file.write(bank.tobytes())
        file.flush()
        os.fsync(file.fileno())
    mapped, offset = [], 0
    for bank in banks:
        mapped.append(np.memmap(path, np.uint8, "r", offset=offset, shape=(bank.nbytes,)))
        offset += bank.nbytes
    if hasattr(os, "posix_fadvise"):
        descriptor = os.open(path, os.O_RDONLY)
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(descriptor)
    cpu = CpuExperts(Path(library), threads=3)
    try:
        with backend() as gpu:
            gpu.register_hybrid_moe("blk.0.", dense, mapped, experts=experts)
            gpu.create_expert_slots(stage=4, margin_bytes=0)
            gpu.attach_cpu(cpu.handle)
            gathered = gpu.resident_moe("blk.0.", values)
            assert gpu.hybrid_counts()["streamed_experts"] > 4
            assert gathered.tobytes() == streamed.tobytes()
    finally:
        cpu.close()


def test_hybrid_decode_cpu_residual_matches_resident_across_layers_and_graph_replays():
    """Mapped CPU results must reach every stream, both between layers and at the span tail."""
    from straydog.cpu import CpuExperts
    from straydog.hybrid import expert_bytes

    library = os.environ.get("STRAYDOG_GPU_LIBRARY")
    if not library:
        pytest.skip("explicit GPU library required")
    rng = np.random.default_rng(137)
    experts = 10
    layers = []
    for layer in range(2):
        prefix = f"blk.{layer}."

        def matrix(name, rows, cols, norm=False):
            raw = (rng.uniform(0.7, 1.3, (rows, cols)) if norm
                   else rng.normal(scale=0.004, size=(rows, cols))).astype(np.float32 if norm else np.float16)
            return (prefix + name, lambda raw=raw: (raw, 0 if norm else 1), rows, cols)

        def hc(name):
            return [matrix(name + str(i), rows, cols, norm=i == 0)
                    for i, (rows, cols) in enumerate([(1, 10240), (320, 10240), (10240, 320), (4, 10240)])]

        attention = [matrix("attention" + str(i), *shape) for i, shape in enumerate(
            [(10240, 2560), (6144, 2560), (48, 2560), (48, 2560), (2560, 6144)])]
        vectors = [matrix("conv", 10240, 4, norm=True), matrix("bias", 1, 48, norm=True),
                   matrix("decay", 1, 48, norm=True), matrix("norm", 1, 128, norm=True)]
        decay = -np.ones((1, 48), np.float32)
        vectors[2] = (prefix + "decay", lambda decay=decay: (decay, 0), 1, 48)
        operands = hc("before") + attention + vectors + hc("after")
        shapes = [(640, 2560), (640, 2560), (2560, 640)]
        banks = [_q2(rng, experts * rows, cols) for rows, cols in shapes]
        bank_matrices = [(prefix + f"bank{i}", lambda i=i, banks=banks: (banks[i], 42), experts * rows, cols)
                         for i, (rows, cols) in enumerate(shapes)]
        shared = [_q2(rng, rows, cols) for rows, cols in shapes]
        dense = [(prefix + f"shared{i}", lambda i=i, shared=shared: (shared[i], 42), rows, cols)
                 for i, (rows, cols) in enumerate(shapes)]
        dense += [matrix("router", experts, 2560), matrix("mix", 1, 2560)]
        layers.append((prefix, operands, bank_matrices, dense, banks))
    incoming = rng.normal(scale=0.05, size=(3, 10240)).astype(np.float32)
    results = []
    for hybrid in (False, True):
        with (
            CpuExperts(Path(library), threads=2) as cpu,
            GpuBackend(Path(library), budget_bytes=1024**3) as gpu,
        ):
            gpu.configure_gemm(minimum_batch=2, precision="fp16")
            gpu.reserve_model_state(64, attention=False, layers=[0, 1], prefill=1)
            for layer, (prefix, operands, bank_matrices, dense, banks) in enumerate(layers):
                if hybrid:
                    gpu.register_hybrid_moe(prefix, dense, banks, experts=experts)
                else:
                    gpu.register_resident_moe(prefix, bank_matrices + dense, experts=experts)
                gpu.register_decoder_layer(layer, operands, context=64)
            if hybrid:
                gpu.attach_cpu(cpu.handle)
                slot = -(-expert_bytes((42, 42, 42)) // 256) * 256
                used, budget = gpu.memory()
                assert gpu.create_expert_slots(stage=2, margin_bytes=budget - used - 3 * slot - 8192 * 4) == 1
            rows = [gpu.decode_span(0, 2, incoming[t:t + 1], position=t) for t in range(3)]
            results.append(np.concatenate(rows))
            if hybrid:
                assert gpu.hybrid_counts()["cpu_experts"] >= 50
    np.testing.assert_allclose(results[1], results[0], rtol=2e-4, atol=2e-5)
