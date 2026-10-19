"""Chunked prefill must preserve the sequential graph, recurrence, PLE and QSA."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from straydog.model import StraydogModel


class _Weights:
    context_limit = 262144
    integer_arrays = {
        "qwen4exp.ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
        "qwen4exp.ple.head_offsets": [i * 1009 for i in range(16)],
        "qwen4exp.ple.head_vocab_sizes": [1009] * 16,
    }

    def row(self, name, token):
        size = 160 if name.startswith("per_layer") else 2560
        return (np.sin(np.arange(size, dtype=np.float32) * 0.01 + token % 37) * 0.1).astype(np.float32)

    def vector(self, name):
        if "conv1d" in name:
            return np.full((10240, 4), 0.1, np.float32)
        if "ssm_dt.bias" in name:
            return np.zeros(48, np.float32)
        if "ssm_a" in name:
            return np.full(48, -0.1, np.float32)
        size = 10240
        if name.endswith(("ssm_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight")):
            size = 128
        elif name.endswith(("attn_q_norm.weight", "attn_k_norm.weight")):
            size = 256
        return np.ones(size, np.float32)

    def project(self, name, value, expert=None):
        suffix = name.split(".", 2)[-1] if name.startswith("blk.") else name
        rows = {
            "attn_qkv.weight": 10240,
            "attn_gate.weight": 6144,
            "ssm_beta.weight": 48,
            "ssm_alpha.weight": 48,
            "ssm_out.weight": 2560,
            "attn_q.weight": 12288,
            "attn_k.weight": 512,
            "attn_v.weight": 512,
            "attn_output.weight": 2560,
            "indexer.q_proj.weight": 512,
            "indexer.k_proj.weight": 128,
            "ple_key.weight": 10240,
            "ple_value.weight": 2560,
            "ffn_gate_inp.weight": 512,
            "ffn_gate_exps.weight": 640,
            "ffn_up_exps.weight": 640,
            "ffn_down_exps.weight": 2560,
            "ffn_gate_shexp.weight": 640,
            "ffn_up_shexp.weight": 640,
            "ffn_down_shexp.weight": 2560,
            "ffn_gate_inp_shexp.weight": 1,
            "output.weight": 128,
        }.get(suffix)
        if rows is None:
            rows = 320 if "_down.weight" in name else 4 if "_inject.weight" in name else 10240
        phase = np.float32(np.sum(value, dtype=np.float64) * 0.00001 + sum(name.encode()) % 19 + (expert or 0) * 0.01)
        return np.sin(np.arange(rows, dtype=np.float32) * 0.01 + phase) * np.float32(0.005)

    def project_many(self, requests):
        return [self.project(*request) for request in requests]

    def check_unchanged(self):
        return None


def test_chunked_prefill_matches_sequential_states_and_continuation(monkeypatch):
    monkeypatch.setattr(StraydogModel, "_validate_geometry", lambda self: None)
    sequential = StraydogModel(_Weights(), context=32, prefill_batch_size=1)
    batched = StraydogModel(_Weights(), context=32, prefill_batch_size=4)
    tokens = [21, 248044, 3, 4]
    for token in tokens:
        expected = sequential.step(token)
    observed = batched._forward_tokens(tokens, logits=True)
    np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=1e-7)
    assert sequential.history == batched.history == tokens
    np.testing.assert_array_equal(sequential._ple_history, batched._ple_history)
    for layer, arrays in sequential._linear.items():
        for actual, reference in zip(batched._linear[layer], arrays, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=1e-7)
    for layer, arrays in sequential._attention.items():
        for actual, reference in zip(batched._attention[layer], arrays, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(batched.step(5), sequential.step(5), rtol=2e-5, atol=1e-7)
    assert batched._forward_tokens([6, 7], logits=True, cancelled=lambda: True) is None
    assert batched.history == []


def test_prefill_batch_configuration_is_bounded(monkeypatch):
    monkeypatch.setattr(StraydogModel, "_validate_geometry", lambda self: None)
    for batch in (0, 4097, True):
        with pytest.raises(ValueError, match="prefill batch"):
            StraydogModel(_Weights(), prefill_batch_size=batch)


def test_benchmark_cli_and_cases_are_explicit(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark.py"
    spec = importlib.util.spec_from_file_location("straydog_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    assert [case["name"] for case in benchmark._cases("both")] == ["short", "tools"]
    assert benchmark._cases("tools")[0]["max_tokens"] == 64
    assert [case["name"] for case in benchmark._cases("all")] == ["short", "tools", "long"]
    assert [(case["name"], case["max_tokens"]) for case in benchmark._cases("generate")] == [("generate", 128)]
    assert "STRAYDOG-7319" in benchmark._cases("long")[0]["messages"][0]["content"]
    diverse = benchmark._cases("diverse")[0]["messages"][0]["content"]
    assert diverse == benchmark._cases("diverse")[0]["messages"][0]["content"] and diverse.count("\n") == 121
    assert [case["name"] for case in benchmark._cases("ssd")] == ["short", "tools", "long", "diverse"]
    monkeypatch.setattr("sys.argv", [str(path), "--help"])
    with pytest.raises(SystemExit) as result:
        benchmark.main()
    assert result.value.code == 0


def test_native_moe_maps_unique_authenticated_experts_and_shared_weights():
    from types import SimpleNamespace

    import numpy as np

    from straydog.weights import StraydogWeights

    weights = StraydogWeights.__new__(StraydogWeights)
    projected, calls = [], []

    def projection(name, value, expert):
        projected.append((name, expert))
        return (name, None, 1, 1, None)

    weights._projection = projection
    weights.gpu = SimpleNamespace(moe=lambda *args: calls.append(args) or np.zeros((2, 2560)))
    selected = np.array([list(range(30, 40)), list(range(39, 29, -1))])
    result = weights.moe("blk.2.", np.ones((2, 2560)), selected, np.ones((2, 10)), np.ones((2, 1)))
    assert result.shape == (2, 2560)
    assert len(projected) == 33 and {expert for _, expert in projected[:-3]} == set(range(30, 40))
    assert projected[-3:] == [("blk.2.ffn_" + role + "_shexp.weight", None) for role in ("gate", "up", "down")]
    np.testing.assert_array_equal(calls[0][2], selected - 30)



def test_native_qsa_uses_authenticated_norms_and_projections():
    from types import SimpleNamespace

    from straydog.weights import StraydogWeights

    weights = StraydogWeights.__new__(StraydogWeights)
    projected, vectors, calls = [], [], []
    weights._projection = lambda name, value, expert: projected.append(name) or (name, None, 1, 1, None)
    weights.vector = lambda name: vectors.append(name) or np.ones(128, np.float32)
    weights.gpu = SimpleNamespace(sparse_attention=lambda *args, **kwargs: calls.append((args, kwargs)) or 7)
    assert weights.sparse_attention("blk.3.", np.zeros((1, 2560)), position=14, context=32) == 7
    assert len(projected) == 6 and projected[-1] == "blk.3.attn_output.weight"
    assert len(vectors) == 4 and vectors[-1] == "blk.3.indexer.k_norm.weight"
    assert calls[0][1] == {"position": 14, "context": 32}


def test_resident_preflight_counts_all_weights_and_prepares_complete_native_layers():
    from types import SimpleNamespace

    from straydog.weights import StraydogWeights

    weights = StraydogWeights.__new__(StraydogWeights)
    weights._tensors = {
        "token_embd.weight": (0, {"dimensions": [2560, 248320], "bytes": 10**9}),
        "per_layer_token_embd.weight": (0, {"dimensions": [160, 320001536], "bytes": 30 * 10**9}),
        "blk.0.norm": (0, {"dimensions": [128], "bytes": 256}),
        "output.weight": (0, {"dimensions": [2560, 248320], "bytes": 10**8}),
    }
    seen, prepared, layers = [], [], []
    weights.gpu = SimpleNamespace(can_pin_model=lambda requirements: seen.append(requirements) or False)
    assert not weights.prepare_resident_moe()
    assert seen == [{"blk.0.norm": 512, "output.weight": 10**8}]
    weights.gpu.can_pin_model = lambda requirements: True
    weights.gpu.register_resident_moe = lambda prefix, matrices, **kwargs: prepared.append((prefix, matrices, kwargs))
    weights.gpu.register_decoder_layer = lambda layer, matrices, **kwargs: layers.append((layer, matrices, kwargs))
    weights.gpu.resident_moe = lambda prefix, values: values
    weights.gpu.decode_span = lambda begin, end, values, **kwargs: values
    ple_calls = []
    weights.gpu.ple = lambda prefix, values, *args: ple_calls.append((prefix, args)) or values
    weights.shape = lambda name: (640, 2560, 512) if "down_exps" in name else (2560, 640, 512)
    weights._projection = lambda name, value, expert: (name, None, 1, 1, None)
    weights.vector = lambda name: np.ones((10240, 4) if "conv1d" in name else (10240,), np.float32)
    assert weights.prepare_resident_moe()
    assert len(prepared) == 48 and len(prepared[0][1]) == 8
    assert prepared[0][1][0][2:] == (512 * 640, 2560)
    assert weights.prepare_decoder(8192)
    assert len(layers) == 48 and len(layers[0][1]) == 17 and len(layers[3][1]) == 18
    values = np.ones((1, 10240), np.float32)
    assert weights.resident_moe("blk.0.", values) is values
    assert weights.decode_span(1, 24, values, position=10) is values
    assert weights.ple(values, np.ones((1, 2560), np.float32)) is values
    assert ple_calls[0][0] == "blk.1." and len(ple_calls[0][1][1]) == 2 and len(ple_calls[0][1][2]) == 4


