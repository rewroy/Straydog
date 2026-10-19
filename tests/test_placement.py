"""Straydog places the K/V and the prefill workspace so that every weight can stay on the GPUs."""

from __future__ import annotations

from types import SimpleNamespace

from straydog.model import StraydogModel


def placement(fits, *, vram_kv=False, drafting=True, prefill=4096):
    """``fits(host, batch, drafter)`` says whether full residency fits that placement."""
    planned = []
    gpu = SimpleNamespace(planned_state_bytes=lambda context, **kwargs: planned.append(kwargs) or [kwargs])
    weights = SimpleNamespace(resident_fits=lambda reserved, drafter: fits(
        reserved[0]["host_attention"], reserved[0]["prefill"], drafter))
    chosen = StraydogModel._resident_placement(weights, gpu, 262144, prefill, drafting, vram_kv=vram_kv,
                                               default=("default", prefill))
    return chosen, planned


def test_vram_kv_and_the_configured_batch_come_first():
    chosen, planned = placement(lambda host, batch, drafter: True, vram_kv=True)
    assert chosen == (False, 4096) and len(planned) == 1


def test_host_kv_then_smaller_prefill_workspaces_until_every_weight_fits():
    chosen, planned = placement(lambda host, batch, drafter: host and batch <= 2048)
    assert chosen == (True, 2048)
    assert [(item["host_attention"], item["prefill"]) for item in planned] == [(True, 4096), (True, 2048)]
    assert all(item["drafter"] for item in planned)   # the MTP layer's state is planned too


def test_keeping_the_drafter_wins_over_a_larger_batch_without_it():
    chosen, _planned = placement(lambda host, batch, drafter: batch <= 1024 or not drafter)
    assert chosen == (True, 1024)


def test_no_full_residency_keeps_the_default_placement():
    chosen, planned = placement(lambda host, batch, drafter: False, vram_kv=True)
    assert chosen == ("default", 4096)
    assert [(item["host_attention"], item["prefill"]) for item in planned][:4] == [
        (False, 4096), (True, 4096), (True, 2048), (True, 1024)]
