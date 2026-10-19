"""Device placement adds cache residency without creating another runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from straydog.devices import GpuGroup


class _Device:
    def __init__(self, library, *, device, budget_bytes):
        self.device, self.budget, self.stats = device, budget_bytes, None
        self.closed, self.resets = False, 0
        self.batch_capacity, self.output_capacity = 256, 8388608
        self.precision, self.layers = None, []
        self.logical_capacity = 4096
        self.resident, self.decoder = [], []

    def project(self, *args):
        return np.array([self.device])

    def project_many(self, requests):
        return [self.project() for _ in requests]

    def delta_sequence(self, *args):
        return self.project()

    def hyperconnection(self, *args):
        return self.project()

    def gated_delta(self, *args):
        return self.project()

    def moe(self, *args):
        return self.project()

    def sparse_attention(self, *args, **kwargs):
        return self.project()

    def ple(self, *args):
        return self.project()

    def can_pin_model(self, requirements, reserved=0):
        self.pinned = dict(requirements)
        return sum(requirements.values()) + reserved <= (self.budget - 1) * 0.9

    def hybrid_supported(self):
        return self.grouped_prefill

    grouped_prefill = False

    def link_peer(self, peer, rows):
        self.peer, self.peer_rows = peer, rows
        return True

    def register_resident_moe(self, prefix, matrices, **kwargs):
        self.resident.append(prefix)
        self.parts = getattr(self, "parts", []) + [(prefix, [item[:1] + item[2:] for item in matrices], kwargs)]

    def resident_moe(self, prefix, values):
        return self.project()

    def register_decoder_layer(self, layer, matrices, **kwargs):
        self.decoder.append(layer)

    def decode_span(self, *args, **kwargs):
        return self.project()

    def configure_gemm(self, **kwargs):
        self.precision = kwargs["precision"]

    def reserve_model_state(self, context, *, attention, layers, prefill=1024):
        self.layers = layers

    def planned_state_bytes(self, context, *, attention, prefill=1024, host_attention=False, layers=()):
        return 100

    def attention_sequence(self, *args, **kwargs):
        return self.project()

    def reset_state(self):
        self.resets += 1

    def reserve_checkpoints(self, count):
        self.checkpoints = [None] * count
        return 10 * count

    def save_checkpoint(self, slot):
        self.checkpoints[slot] = "saved"

    def load_checkpoint(self, slot):
        self.loaded = slot

    def memory(self):
        return 1, self.budget

    def info(self):
        return {
            "name": f"GPU {self.device}",
            "device": self.device,
            "allocated_bytes": 1,
            "budget_bytes": self.budget,
            "free_bytes": 100,
            "total_bytes": 200,
            "gemm_calls": 2,
            "gemm_vectors": 4,
        }

    def close(self):
        self.closed = True


def test_layer_placement_order_telemetry_state_and_lifecycle(monkeypatch):
    monkeypatch.setattr("straydog.devices.GpuBackend", _Device)
    with GpuGroup(Path("/explicit/library"), devices=[0, 1], budget_bytes=6) as gpu:
        backends = gpu._backends.copy()
        assert gpu.project("blk.0.test", None, 1, 1, None).item() == 0
        assert gpu.project("blk.24.test", None, 1, 1, None).item() == 1
        assert gpu.project("output.weight", None, 1, 1, None).item() == 1
        requests = [("blk.30.test",), ("blk.5.test",), ("blk.47.test",)]
        assert [out.item() for out in gpu.project_many(requests)] == [1, 0, 1]
        assert gpu.delta_sequence("blk.1.").item() == 0
        assert gpu.hyperconnection("blk.30.hc_attn").item() == 1
        assert gpu.gated_delta("blk.30.").item() == 1
        assert gpu.moe("blk.30.").item() == 1
        assert gpu.sparse_attention("blk.30.").item() == 1
        assert gpu.ple("blk.1.").item() == 0
        assert gpu.logical_capacity == 4096
        assert gpu.can_pin_model({"blk.0.test": 2, "blk.30.test": 2, "output.weight": 1})
        assert not gpu.can_pin_model({"blk.0.test": 100})
        gpu.register_resident_moe("blk.30.", [], experts=512)
        assert backends[1].resident == ["blk.30."]
        assert gpu.resident_moe("blk.30.", None).item() == 1
        gpu.register_decoder_layer(30, [], context=1024)
        assert backends[1].decoder == [30]
        assert gpu.decoder_spans() == [(1, 24), (24, 48)]
        assert gpu.decode_span(24, 48, None, position=0).item() == 1
        with pytest.raises(ValueError, match="crosses devices"):
            gpu.decode_span(0, 48, None, position=0)
        assert gpu.attention_sequence("blk.30.").item() == 1
        gpu.configure_gemm(precision="fp16")
        assert all(device.precision == "fp16" for device in backends)
        gpu.reserve_model_state(1024, attention=True)
        assert backends[0].layers == list(range(24)) and backends[1].layers == list(range(24, 48))
        assert gpu.batch_capacity == 256 and gpu.output_capacity == 8388608
        gpu.reset_state()
        assert all(backend.resets == 1 for backend in backends)
        # Each device keeps host copies of the recurrent state it owns.
        assert gpu.reserve_checkpoints(4) == 80
        gpu.save_checkpoint(2)
        gpu.load_checkpoint(2)
        assert all(backend.checkpoints[2] == "saved" and backend.loaded == 2 for backend in backends)
        assert gpu.memory() == (2, 12)
        assert gpu.info()["budget_bytes"] == 12 and len(gpu.info()["devices"]) == 2
        assert gpu.stats is None
        gpu.stats = {}
        backends[0].stats["calls"] = 3
        backends[1].stats["calls"] = 4
        assert gpu.stats == {"calls": 7}
        gpu.stats = None
        assert gpu.stats is None
    assert all(backend.closed for backend in backends)
    with pytest.raises(RuntimeError, match="closed"):
        gpu.project("blk.0.test", None, 1, 1, None)
    gpu.close()


def test_invalid_devices_and_partial_startup_release(monkeypatch):
    created = []

    def create(library, *, device, budget_bytes):
        if device == 1:
            raise RuntimeError("device unavailable")
        result = _Device(library, device=device, budget_bytes=budget_bytes)
        created.append(result)
        return result

    monkeypatch.setattr("straydog.devices.GpuBackend", create)
    for ordinals in ([], [0, 0], [-1], [True], [0, 1, 2, 3, 4]):
        with pytest.raises(ValueError, match="ordinals"):
            GpuGroup(Path("/explicit/library"), devices=ordinals, budget_bytes=6)
    with pytest.raises(RuntimeError, match="unavailable"):
        GpuGroup(Path("/explicit/library"), devices=[0, 1], budget_bytes=6)
    assert created[0].closed


def test_layers_balance_by_weight_bytes_before_placement(monkeypatch):
    monkeypatch.setattr("straydog.devices.GpuBackend", _Device)
    with GpuGroup(Path("/explicit/library"), devices=[0, 1], budget_bytes=6) as gpu:
        sizes, tail = [10] * 24 + [12] * 24, 30
        gpu.balance_layers(sizes, tail)
        owners = [gpu.device_of(layer) for layer in range(48)]
        split = owners.index(1)
        assert owners == [0] * split + [1] * (48 - split) and split > 24   # the head moves layers forward
        loads = [max(sum(sizes[:s]), sum(sizes[s:]) + tail) for s in range(1, 48)]
        assert max(sum(sizes[:split]), sum(sizes[split:]) + tail) == min(loads)
        assert gpu._index("blk.48.") == 1 and gpu._index("output.weight") == 1
        assert gpu.decoder_spans() == [(1, split), (split, 48)]


def _expert_parallel_group(monkeypatch, split_mode="experts"):
    monkeypatch.setattr("straydog.devices.GpuBackend", _Device)
    monkeypatch.setattr(_Device, "grouped_prefill", True)
    return GpuGroup(Path("/explicit/library"), devices=[0, 1], budget_bytes=10**9, split_mode=split_mode)


def test_two_gpus_split_every_layers_experts_and_link_once(monkeypatch):
    with _expert_parallel_group(monkeypatch) as gpu:
        assert gpu.expert_parallel
        first, second = gpu._backends
        # Planning: each GPU holds half of every layer's routed expert banks, wherever the layer lives,
        # plus its inbox; the shared expert and dense weights stay with the layer's owner.
        requirements = {"blk.3.ffn_gate_exps.weight": 1000, "blk.3.ffn_gate_shexp.weight": 10,
                        "blk.40.ffn_down_exps.weight": 2000, "blk.48.ffn_up_exps.weight": 300, "output.weight": 50}
        assert gpu.can_pin_model(requirements, reserved=[0, 0])
        # The first GPU also holds the lower half of the vocabulary head it evaluates.
        assert first.pinned == {"blk.3.ffn_gate_exps.weight": 500, "blk.3.ffn_gate_shexp.weight": 10,
                                "blk.40.ffn_down_exps.weight": 1000, "output.weight:rows0": 25}
        assert second.pinned == {"blk.3.ffn_gate_exps.weight": 500, "blk.40.ffn_down_exps.weight": 1000,
                                 "blk.48.ffn_up_exps.weight": 300, "output.weight": 50}
        assert gpu.planned_state_bytes(8192, attention=True, prefill=4096) == [100 + 4096 * 2560 * 8 + 256] * 2
        with pytest.raises(RuntimeError, match="linked"):
            gpu.register_resident_moe("blk.3.", [], experts=512)
        gpu.reserve_model_state(8192, attention=True, prefill=4096)
        gpu.reserve_model_state(8192, attention=True, prefill=4096)
        assert first.peer is second and first.peer_rows == 4096 and not hasattr(second, "peer")

        bank = np.arange(512 * 6, dtype=np.uint8)
        matrices = [(f"blk.3.ffn_{role}_exps.weight:bank", lambda: (bank, 21), 512 * 640, 2560)
                    for role in ("gate", "up", "down")]
        matrices += [(f"blk.3.ffn_{role}_shexp.weight", None, 640, 2560) for role in ("gate", "up", "down")]
        matrices += [("blk.3.ffn_gate_inp.weight", None, 512, 2560), ("blk.3.ffn_gate_inp_shexp.weight", None, 1, 2560)]
        gpu.register_resident_moe("blk.3.", matrices, experts=512)
        (owner_prefix, owner_items, owner_kwargs), = first.parts
        (helper_prefix, helper_items, helper_kwargs), = second.parts
        assert owner_kwargs == {"experts": 512, "first": 0, "count": 256, "shared": True, "helper": False}
        assert helper_kwargs == {"experts": 512, "first": 256, "count": 256, "shared": False, "helper": True}
        assert owner_items[0] == ("blk.3.ffn_gate_exps.weight:bank:0+256", 256 * 640, 2560)
        assert helper_items[0] == ("blk.3.ffn_gate_exps.weight:bank:256+256", 256 * 640, 2560)
        assert owner_items[3:] == helper_items[3:]


def test_bank_slices_are_each_halfs_experts():
    from straydog.devices import _bank_slice

    bank = np.arange(512 * 6, dtype=np.uint8)
    low, kind = _bank_slice(lambda: (bank, 21), 0, 256, 512)()
    high, _ = _bank_slice(lambda: (bank, 21), 256, 256, 512)()
    assert kind == 21 and np.array_equal(np.concatenate([low, high]), bank) and len(low) == 256 * 6


def test_the_mtp_layer_and_the_layer_split_keep_whole_experts(monkeypatch):
    with _expert_parallel_group(monkeypatch) as gpu:
        gpu.reserve_model_state(8192, attention=True, prefill=1024)
        gpu.register_resident_moe("blk.48.", [("x", None, 1, 1)] * 8, experts=512)
        assert gpu._backends[1].parts[0][2] == {"experts": 512} and not hasattr(gpu._backends[0], "parts")
    with _expert_parallel_group(monkeypatch, split_mode="layer") as gpu:
        assert not gpu.expert_parallel
        gpu.reserve_model_state(8192, attention=True, prefill=1024)
        assert not hasattr(gpu._backends[0], "peer")
        gpu.register_resident_moe("blk.3.", [("x", None, 1, 1)] * 8, experts=512)
        assert gpu._backends[0].parts[0][2] == {"experts": 512}
