"""Straydog's MTP drafter asset: the base model's own multi-token-prediction layer.

The GSQ-RCO GGUF bundles omit the MTP layer, so its tensors are range-fetched from the
pinned official ``Qwen/Qwen3.8-Flash-Next`` checkpoint (each tensor's SHA256 is
pinned in ``assets.json``) and converted into one GGUF named like the
decoder layers (``blk.48.*`` plus ``mtp.*`` prologue/mixer operands). Gemma norm
weights gain their +1 offset like the main bundle; routed experts are encoded as
Q2_0 by a deterministic native quantizer, everything else stays BF16/F32. Drafts
are always verified by the main model, so the drafter's precision affects only
the acceptance rate, never the generated tokens.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import mmap
import os
import struct
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from straydog.library import load_library

MTP_FILE = "Qwen3.8-Flash-Next-MTP-Q2_0.gguf"
SOURCE_FILE = "mtp-bf16.safetensors"
_ALIGNMENT = 32
# Download pieces and parallel connections (one connection reaches ~42 MB/s from the Hub, four ~57),
# and experts per quantization call (progress steps; the Q2_0 bytes do not depend on it).
_PIECE, _CONNECTIONS, _EXPERT_GROUP = 64 * 1024**2, 4, 32
_F32, _BF16, _Q2_0 = 0, 30, 42
_PREFIX = "mtp.layers.0."

# GGUF name -> (source tensor, transform). "norm" adds the Gemma +1 offset (F32),
# "rows:a:b" slices source rows, "experts:a:b" slices each expert's rows (Q2_0).
TENSOR_PLAN: dict[str, tuple[str, str]] = {
    "mtp.embed_norm.weight": ("mtp.pre_fc_norm_embedding.weight", "norm"),
    "mtp.fc_embedding.weight": ("mtp.fc_embedding.weight", "bf16"),
    "mtp.hidden_norm.weight": ("mtp.pre_fc_norm_hidden.weight", "norm"),
    "mtp.fc_hidden.weight": ("mtp.fc_hidden.weight", "bf16"),
    "mtp.hc_out_norm.weight": ("mtp.hyper_connection_mixer.hc_norm.weight", "norm"),
    "mtp.hc_out_down.weight": ("mtp.hyper_connection_mixer.input_mix_weight_down.weight", "bf16"),
    "mtp.hc_out_up.weight": ("mtp.hyper_connection_mixer.input_mix_weight_up.weight", "bf16"),
    "blk.48.attn_q.weight": (_PREFIX + "self_attn.q_proj.weight", "bf16"),
    "blk.48.attn_k.weight": (_PREFIX + "self_attn.k_proj.weight", "bf16"),
    "blk.48.attn_v.weight": (_PREFIX + "self_attn.v_proj.weight", "bf16"),
    "blk.48.attn_output.weight": (_PREFIX + "self_attn.o_proj.weight", "bf16"),
    "blk.48.attn_q_norm.weight": (_PREFIX + "self_attn.q_norm.weight", "norm"),
    "blk.48.attn_k_norm.weight": (_PREFIX + "self_attn.k_norm.weight", "norm"),
    "blk.48.indexer.q_proj.weight": (_PREFIX + "self_attn.indexer.index_qk_proj.weight", "rows:0:512"),
    "blk.48.indexer.k_proj.weight": (_PREFIX + "self_attn.indexer.index_qk_proj.weight", "rows:512:640"),
    "blk.48.indexer.q_norm.weight": (_PREFIX + "self_attn.indexer.q_layernorm.weight", "norm"),
    "blk.48.indexer.k_norm.weight": (_PREFIX + "self_attn.indexer.k_layernorm.weight", "norm"),
    "blk.48.ffn_gate_inp.weight": (_PREFIX + "mlp.gate.weight", "bf16"),
    "blk.48.ffn_gate_inp_shexp.weight": (_PREFIX + "mlp.shared_expert_gate.weight", "vector"),
    "blk.48.ffn_gate_shexp.weight": (_PREFIX + "mlp.shared_expert.gate_proj.weight", "bf16"),
    "blk.48.ffn_up_shexp.weight": (_PREFIX + "mlp.shared_expert.up_proj.weight", "bf16"),
    "blk.48.ffn_down_shexp.weight": (_PREFIX + "mlp.shared_expert.down_proj.weight", "bf16"),
    "blk.48.ffn_gate_exps.weight": (_PREFIX + "mlp.experts.gate_up_proj", "experts:0:640"),
    "blk.48.ffn_up_exps.weight": (_PREFIX + "mlp.experts.gate_up_proj", "experts:640:1280"),
    "blk.48.ffn_down_exps.weight": (_PREFIX + "mlp.experts.down_proj", "experts:0:2560"),
}
for _role in ("attn", "ffn"):
    _source = f"{_PREFIX}{'attn' if _role == 'attn' else 'mlp'}_hyper_connection."
    TENSOR_PLAN[f"blk.48.hc_{_role}_norm.weight"] = (_source + "hc_norm.weight", "norm")
    TENSOR_PLAN[f"blk.48.hc_{_role}_down.weight"] = (_source + "input_mix_weight_down.weight", "bf16")
    TENSOR_PLAN[f"blk.48.hc_{_role}_up.weight"] = (_source + "input_mix_weight_up.weight", "bf16")
    TENSOR_PLAN[f"blk.48.hc_{_role}_inject.weight"] = (_source + "block_inject_weight.weight", "bf16")


def _base_url(drafter: dict) -> str:
    return f"https://huggingface.co/{drafter['repository']}/resolve/{drafter['revision']}/"


def fetch_mtp_source(
    drafter: dict, target: Path, *, opener: Callable = urllib.request.urlopen,
    progress: Callable[[int], None] | None = None,
) -> None:
    """Range-fetch the pinned MTP tensors into one safetensors file, verifying each SHA256.

    Several connections fetch 64 MiB pieces straight into their places in the file; each tensor is
    then hashed from the file, so no tensor is held in memory. ``progress`` (called from those
    connections' threads) receives the byte counts.
    """
    base = _base_url(drafter)
    pinned: dict[str, dict] = drafter["source_tensors"]
    headers: dict[str, tuple[int, dict]] = {}

    def read(url: str, start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with opener(request, timeout=600) as response:
            return response.read()

    def header(shard: str) -> tuple[int, dict]:
        if shard not in headers:
            size = struct.unpack("<Q", read(base + shard, 0, 7))[0]
            if not 0 < size < 64 * 1024 * 1024:
                raise ValueError("invalid safetensors header")
            headers[shard] = (size, json.loads(read(base + shard, 8, 7 + size)))
        return headers[shard]

    names = sorted(pinned)
    meta, offset = {}, 0
    for name in names:
        size, table = header(pinned[name]["shard"])
        info = table[name]
        length = info["data_offsets"][1] - info["data_offsets"][0]
        if info["dtype"] != "BF16" or length != pinned[name]["bytes"]:
            raise ValueError(f"MTP tensor changed upstream: {name}")
        meta[name] = {"dtype": "BF16", "shape": info["shape"], "data_offsets": [offset, offset + length]}
        offset += length
    blob = json.dumps(meta, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    data = 8 + len(blob)
    pieces = []
    for name in names:
        size, table = header(pinned[name]["shard"])
        source = 8 + size + table[name]["data_offsets"][0]
        for piece in range(0, pinned[name]["bytes"], _PIECE):
            length = min(_PIECE, pinned[name]["bytes"] - piece)
            pieces.append((base + pinned[name]["shard"], source + piece, data + meta[name]["data_offsets"][0] + piece,
                           length))
    partial = target.with_name(target.name + ".part")

    def fetch(url: str, start: int, destination: int, length: int) -> None:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        received = 0
        with opener(request, timeout=600) as response, open(partial, "r+b") as output:
            output.seek(destination)
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > length:
                    raise ValueError("MTP tensor download exceeded its range")
                output.write(block)
                if progress is not None:
                    progress(len(block))
        if received != length:
            raise ValueError("MTP tensor download ended early")

    try:
        with open(partial, "wb") as output:
            output.write(struct.pack("<Q", len(blob)) + blob)
            output.truncate(data + offset)
        with ThreadPoolExecutor(_CONNECTIONS, thread_name_prefix="straydog-mtp") as pool:
            futures = [pool.submit(fetch, *piece) for piece in pieces]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        with open(partial, "rb") as output:
            for name in names:
                first, last = meta[name]["data_offsets"]
                output.seek(data + first)
                digest, remaining = hashlib.sha256(), last - first
                while remaining and (block := output.read(min(remaining, 8 * 1024**2))):
                    digest.update(block)
                    remaining -= len(block)
                if remaining or digest.hexdigest() != pinned[name]["sha256"]:
                    raise ValueError(f"MTP tensor failed authentication: {name}")
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _source_tensors(path: Path) -> dict[str, np.ndarray]:
    """Read-only views of the source tensors; the mapping lives as long as the views."""
    file = open(path, "rb")
    try:
        mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    finally:
        file.close()
    size = struct.unpack("<Q", mapped[:8])[0]
    meta = json.loads(mapped[8:8 + size])
    tensors = {}
    for name, info in meta.items():
        start, end = info["data_offsets"]
        count = (end - start) // 2
        tensors[name] = np.frombuffer(mapped, "<u2", count=count, offset=8 + size + start).reshape(info["shape"])
    return tensors


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _quantize_q2_0(bits: np.ndarray, lib: ct.CDLL) -> bytes:
    values = np.ascontiguousarray(bits, dtype="<u2")
    output = np.empty(values.size // 64 * 18, np.uint8)
    if lib.sd_quantize_q2_0(values.ctypes.data, values.size, output.ctypes.data):
        raise RuntimeError("Straydog Q2_0 quantization failed")
    return output.tobytes()


def _string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def convert_mtp(
    source: Path, target: Path, library: Path, *, revision: str,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Write the drafter GGUF from the authenticated BF16 source tensors.

    ``progress(done, total)`` follows the routed-expert values quantized, nearly all of the work.
    """
    tensors = _source_tensors(source)
    lib = load_library(library)
    lib.sd_quantize_q2_0.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p]
    lib.sd_quantize_q2_0.restype = ct.c_int
    total = done = 0
    for origin, transform in TENSOR_PLAN.values():
        if transform.startswith("experts:"):
            first, last = map(int, transform.split(":")[1:])
            experts, _rows, cols = tensors[origin].shape
            total += experts * (last - first) * cols
    entries = []
    for name, (origin, transform) in TENSOR_PLAN.items():
        bits = tensors[origin]
        if transform == "norm":
            entries.append((name, [bits.size], _F32, (_bf16_to_f32(bits) + np.float32(1)).astype("<f4").tobytes()))
        elif transform == "vector":
            entries.append((name, [bits.size], _F32, _bf16_to_f32(bits).astype("<f4").tobytes()))
        elif transform == "bf16":
            entries.append((name, list(reversed(bits.shape)), _BF16, np.ascontiguousarray(bits).tobytes()))
        elif transform.startswith("rows:"):
            first, last = map(int, transform.split(":")[1:])
            rows = np.ascontiguousarray(bits[first:last])
            entries.append((name, list(reversed(rows.shape)), _BF16, rows.tobytes()))
        else:
            first, last = map(int, transform.split(":")[1:])
            # Q2_0 blocks are independent, so a few experts at a time give the same bytes as all at once.
            parts = []
            for expert in range(0, bits.shape[0], _EXPERT_GROUP):
                group = np.ascontiguousarray(bits[expert:expert + _EXPERT_GROUP, first:last, :])
                parts.append(_quantize_q2_0(group, lib))
                done += group.size
                if progress is not None:
                    progress(done, total)
            entries.append((name, [bits.shape[2], last - first, bits.shape[0]], _Q2_0, b"".join(parts)))
    metadata = [("general.architecture", "qwen4exp"), ("straydog.drafter", "mtp"),
                ("straydog.drafter.source_revision", revision)]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(metadata) + 1)
    for key, value in metadata:
        header += _string(key) + struct.pack("<I", 8) + _string(value)
    header += _string("general.alignment") + struct.pack("<II", 4, _ALIGNMENT)
    offset = 0
    for name, dims, kind, data in entries:
        header += _string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        header += struct.pack("<IQ", kind, offset)
        offset += len(data) + (-len(data) % _ALIGNMENT)
    header += b"\0" * (-len(header) % _ALIGNMENT)
    partial = target.with_name(target.name + ".part")
    with open(partial, "wb") as output:
        output.write(header)
        for _name, _dims, _kind, data in entries:
            output.write(data + b"\0" * (-len(data) % _ALIGNMENT))
    os.replace(partial, target)
