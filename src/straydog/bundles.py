"""Pinned Straydog model bundles.

``assets.json`` pins the default Q2_0 shards and, under ``bundles``, the
IQ2_XS, IQ3_XXS and IQ3_S shards of the same upstream revision. All bundles share the
tokenizer, template, config and MTP drafter pins (and the second shard's bytes).
"""

from __future__ import annotations

import json
from pathlib import Path

QUANTIZATIONS = ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S")
MANIFEST = Path(__file__).with_name("assets.json")


def model_lock(quantization: str = "Q2_0", manifest: Path = MANIFEST) -> dict:
    """The manifest as one bundle: its quantization and shard pins, shared resources."""
    lock = json.loads(manifest.read_text())
    bundles = lock.pop("bundles", {})
    if quantization != lock.get("quantization"):
        if quantization not in QUANTIZATIONS or quantization not in bundles:
            raise ValueError(f"unsupported Straydog model quantization: {quantization}")
        lock["quantization"], lock["assets"] = quantization, bundles[quantization]["assets"]
    return lock
