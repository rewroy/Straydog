"""Authenticated, demand-paged GGUF weights for the independent Straydog engine.

Rust validates header snapshots from the exact handles Python hashes and maps.
The individually selected gguf package is used only for standard block decoding;
it does not supply a model graph, scheduler, server, or generation implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from straydog import host as straydog_host
from straydog.bundles import QUANTIZATIONS, model_lock
from straydog.devices import GpuGroup
from straydog.digest import HINT_STEP, files_sha256
from straydog.formats import BLOCKS
from straydog.gpu import GpuBackend
from straydog.hybrid import expert_bytes
from straydog.platform.spawn import no_window_kwargs

logger = logging.getLogger(__name__)

_PARALLEL_ROWS, _ROW_THREADS = 256, 16
_CACHED_BATCH, _CACHED_ROWS = 8, 4096   # token embeddings kept decoded: 4096 rows of 2560 floats, 40 MiB


def lock_budget(memory: int, banks: int, pinned_state: int = 0) -> int:
    """Expert bank bytes to page-lock given the host memory (0 if unknown).

    All of them when the memory holds them twice over. With at least 1.25 times their size, half
    the memory less the page-locked sequence state: a prompt reads every layer's experts in turn,
    so banks larger than the page cache are evicted before their next use and reread from the SSD
    on every request; locked layers never are, and copy to the GPU by DMA. With less memory
    nothing is locked: the unlocked layers and the PLE table would then thrash in what is left.
    """
    if memory <= 0 or 4 * memory < 5 * banks:
        return 0
    if memory >= 2 * banks:
        return banks
    return max(0, memory // 2 - pinned_state)


def _byte_counter(
    stage: str, total: int, progress: Callable[[str, int, int], None] | None,
) -> Callable[[int], None] | None:
    """Sum the byte counts of the hashing threads into ``progress(stage, done, total)``."""
    if progress is None:
        return None
    lock, done = threading.Lock(), [0]
    progress(stage, 0, total)

    def count(size: int) -> None:
        with lock:
            done[0] += size
            progress(stage, done[0], total)

    return count


def _open_authenticated(root: Path, assets: list[dict], progress: Callable[[int], None] | None = None) -> list[tuple]:
    """Hash separate files on two CPU cores, retaining the exact verified handles.

    ``progress`` receives the hashed byte counts (from both hashing threads).
    """
    opened, seen = [], set()
    try:
        for asset in assets:
            name = asset["name"]
            if (
                not isinstance(name, str) or name in {"", ".", ".."}
                or Path(name).name != name or name in seen or "\\" in name
            ):
                raise ValueError("invalid Straydog asset name")
            seen.add(name)
            file = (root / name).open("rb")
            opened.append((asset, file, None))
            stat = os.fstat(file.fileno())
            opened[-1] = (asset, file, stat)
            if stat.st_size != asset["bytes"]:
                raise ValueError(f"Straydog asset size mismatch: {name}")
        digests = files_sha256([file for _asset, file, _stat in opened], progress)
        for (asset, _file, _stat), digest in zip(opened, digests, strict=True):
            if digest != asset["sha256"]:
                raise ValueError(f"Straydog asset digest mismatch: {asset['name']}")
        return opened
    except BaseException:
        for _asset, file, _stat in opened:
            file.close()
        raise


def dequantize(data: np.ndarray, kind: int) -> np.ndarray:
    """Decode one bounded tensor slice into float32; type 42 is implemented here."""
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if kind == 0:
        return data.view("<f4")
    if kind == 1:
        return data.view("<f2").astype(np.float32)
    if kind == 30:
        return (data.view("<u2").astype(np.uint32) << 16).view(np.float32)
    if kind == 42:
        if data.size % 18:
            raise ValueError("incomplete Straydog Q2_0 block")
        blocks = data.reshape(-1, 18)
        scales = np.ascontiguousarray(blocks[:, :2]).view("<f2").astype(np.float32)
        codes = (blocks[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
        return ((codes.reshape(-1, 64).astype(np.float32) - 1) * scales).reshape(-1)
    from gguf import GGMLQuantizationType, quants

    return quants.dequantize(data, GGMLQuantizationType(kind)).astype(np.float32, copy=False).reshape(-1)


class StraydogWeights:
    """A fixed, authenticated model bundle with bounded tensor slices."""

    drafter = False   # the pinned MTP draft layer is authenticated and usable

    def __init__(
        self, root: Path, manifest: Path, inspector: Path, gpu: GpuBackend | GpuGroup, *, drafter: Path | None = None,
        quantization: str = "Q2_0", host_memory_bytes: int = 0,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """``progress(stage, done, total)`` follows authentication in bytes ("authenticate", "drafter")."""
        self.root, self.gpu = root.resolve(), gpu
        # Measured by the runtime outside isolation (cgroup limits included); else this process's view.
        self.host_memory_bytes = host_memory_bytes if host_memory_bytes > 0 else straydog_host.host_memory_bytes()
        self._files: list[Any] = []
        self._maps: list[mmap.mmap] = []
        self._signatures: list[tuple[int, int]] = []
        self._tensors: dict[str, tuple[int, dict[str, Any]]] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._projection_cache: dict[tuple[str, int | None], tuple] = {}
        # Decoded token embeddings of recent tokens: speculative decoding looks one or a few up
        # per draft step, and replies reuse their tokens (insertion order evicts the oldest).
        self._embedding_rows: dict[int, np.ndarray] = {}
        self._row_pool: ThreadPoolExecutor | None = None
        self.integer_arrays: dict[str, list[int]] = {}
        self.resources: dict[str, bytes] = {}
        lock = model_lock(quantization, manifest)
        if lock.get("schema") != 1 or lock.get("model") != "Qwen3.8-Flash-Next":
            raise ValueError("unsupported Straydog model manifest")
        if lock.get("quantization") not in QUANTIZATIONS:
            raise ValueError("unsupported Straydog model quantization")
        if not inspector.is_absolute() or not inspector.is_file():
            raise ValueError("Straydog inspector must be an explicit executable file")
        self.quantization = lock["quantization"]
        self.context_limit = 262144
        assets = lock.get("assets", [])
        if len(assets) != 2:
            raise ValueError("Straydog requires both fixed model shards")
        if {item.get("name") for item in lock.get("resources", [])} != {
            "tokenizer.json",
            "chat_template.jinja",
            "config.json",
        }:
            raise ValueError("Straydog requires authenticated tokenizer, template, and config")
        authenticated = []
        try:
            files = assets + lock.get("resources", [])
            total = sum(int(asset.get("bytes", 0)) for asset in files)
            authenticated = _open_authenticated(self.root, files, _byte_counter("authenticate", total, progress))
            for asset, file, stat in authenticated:
                name = asset["name"]
                try:
                    if asset not in assets:
                        if stat.st_size > 32 * 1024 * 1024:
                            raise ValueError("Straydog resource exceeds size limit")
                        file.seek(0)
                        resource = file.read()
                        if hashlib.sha256(resource).hexdigest() != asset["sha256"]:
                            raise ValueError("Straydog resource changed during authentication")
                        self.resources[name] = resource
                        continue
                    file.seek(0)
                    snapshot = file.read(min(stat.st_size, 16 * 1024 * 1024))
                    result = subprocess.run(
                        [str(inspector), "inspect-header", str(stat.st_size)],
                        input=snapshot,
                        capture_output=True,
                        timeout=60,
                        env=_inspector_env(),
                        **no_window_kwargs(),
                    )
                    if result.returncode:
                        raise ValueError("Straydog GGUF preflight: " + result.stderr.decode("utf-8", "replace"))
                    header = json.loads(result.stdout)
                    index = len(self._files)
                    if header["shard_count"] != 2 or header["shard_index"] != index:
                        raise ValueError("Straydog shard identity mismatch")
                    if header["architecture"] not in ({"qwen4exp"} if index == 0 else {"qwen4exp", ""}):
                        raise ValueError("Straydog architecture mismatch")
                    if index == 0 and header["declared_context"] != self.context_limit:
                        raise ValueError("Straydog model context declaration mismatch")
                    for tensor in header["tensors"]:
                        if tensor["name"] in self._tensors:
                            raise ValueError("duplicate tensor across Straydog shards")
                        self._tensors[tensor["name"]] = (index, tensor)
                    self.integer_arrays.update(header["integer_arrays"])
                    if os.fstat(file.fileno()).st_mtime_ns != stat.st_mtime_ns:
                        raise ValueError("Straydog asset changed during authentication")
                    self._files.append(file)
                    self._signatures.append((stat.st_size, stat.st_mtime_ns))
                    self._maps.append(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ))
                    file = None
                finally:
                    if file is not None:
                        file.close()
            if len(self._tensors) != 1224:
                raise ValueError("Straydog model tensor count mismatch")
            if drafter is not None:
                self._load_drafter(Path(drafter).resolve(), lock.get("drafter"), inspector, progress)
            if hasattr(gpu, "balance_layers"):
                gpu.balance_layers(*self._placement_bytes())
        except BaseException:
            self.close()
            raise
        finally:
            for _asset, file, _stat in authenticated:
                if file not in self._files:
                    file.close()

    def _load_drafter(
        self, root: Path, pin: dict | None, inspector: Path, progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Authenticate the pinned MTP drafter GGUF and index its tensors after the main shards."""
        from straydog.mtp import TENSOR_PLAN

        if not isinstance(pin, dict) or pin.get("kind") != "mtp":
            raise ValueError("Straydog drafter is not pinned in the model manifest")
        asset = {key: pin[key] for key in ("name", "bytes", "sha256")}
        ((_asset, file, stat),) = _open_authenticated(root, [asset], _byte_counter("drafter", asset["bytes"], progress))
        try:
            file.seek(0)
            result = subprocess.run(
                [str(inspector), "inspect-header", str(stat.st_size)],
                input=file.read(min(stat.st_size, 16 * 1024 * 1024)), capture_output=True, timeout=60,
                env=_inspector_env(), **no_window_kwargs(),
            )
            if result.returncode:
                raise ValueError("Straydog drafter preflight: " + result.stderr.decode("utf-8", "replace"))
            header = json.loads(result.stdout)
            names = {tensor["name"] for tensor in header["tensors"]}
            if header["shard_count"] != 1 or header["architecture"] != "qwen4exp" or names != set(TENSOR_PLAN):
                raise ValueError("Straydog drafter layout mismatch")
            if names & set(self._tensors):
                raise ValueError("duplicate tensor across Straydog drafter and model")
            index = len(self._files)
            for tensor in header["tensors"]:
                self._tensors[tensor["name"]] = (index, tensor)
            if os.fstat(file.fileno()).st_mtime_ns != stat.st_mtime_ns:
                raise ValueError("Straydog drafter changed during authentication")
            self._files.append(file)
            self._signatures.append((stat.st_size, stat.st_mtime_ns))
            self._maps.append(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ))
            file = None
        finally:
            if file is not None:
                file.close()
        self._drafter_names = names
        self.drafter = True

    def _placement_bytes(self) -> tuple[list[int], int]:
        """GPU bytes of each decoder layer and of the rest placed with the last layer (head, MTP).

        Under expert parallelism every GPU holds half of each layer's routed experts wherever
        the layer boundary falls, so only the rest is balanced.
        """
        layers, tail = [0] * 48, 0
        split = getattr(self.gpu, "expert_parallel", False)
        for name, (_index, tensor) in self._tensors.items():
            if name in {"token_embd.weight", "per_layer_token_embd.weight"}:
                continue
            size = tensor["bytes"]
            if len(tensor["dimensions"]) == 1 or name.endswith("ssm_conv1d.weight"):
                size = max(size, int(np.prod(tensor["dimensions"])) * 4)
            layer = int(name.split(".", 2)[1]) if name.startswith("blk.") else 48
            if split and layer < 48 and name.endswith("_exps.weight"):
                continue
            if layer < 48:
                layers[layer] += size
            else:
                tail += size
        return layers, tail

    def _drop_drafter(self) -> None:
        """Keep the drafter mapped but stop using it (it does not fit this placement)."""
        self.drafter = False

    def layers(self) -> range:
        """Decoder layers to place: the 48 model layers plus the MTP layer when drafting."""
        return range(49 if self.drafter else 48)

    def mtp_prologue(self, embeddings, hidden):
        """MTP layer inputs from token embeddings and backbone (four-stream) hidden rows."""
        operands = [self._vector_operand("mtp.embed_norm.weight"),
                    self._projection("mtp.fc_embedding.weight", None, None)[:4],
                    self._vector_operand("mtp.hidden_norm.weight"),
                    self._projection("mtp.fc_hidden.weight", None, None)[:4]]
        return self.gpu.mtp_prologue(operands, embeddings, hidden)

    def mtp_head(self, values, *, logits=False):
        """Draft tokens from MTP rows through its own mixer and the shared vocabulary head."""
        operands = [self._vector_operand("mtp.hc_out_norm.weight"),
                    *[self._projection(name, None, None)[:4]
                      for name in ("mtp.hc_out_down.weight", "mtp.hc_out_up.weight", "output.weight")]]
        return self.gpu.output_head(operands, values, logits=logits)

    def check_unchanged(self) -> None:
        """Refuse a new generation if an authenticated file has changed."""
        for file, expected in zip(self._files, self._signatures, strict=True):
            stat = os.fstat(file.fileno())
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise RuntimeError("Straydog model changed after authentication")

    def shape(self, name: str) -> tuple[int, ...]:
        """Return GGUF dimensions in on-disk (input-dimension-first) order."""
        return tuple(self._tensors[name][1]["dimensions"])

    def _raw(self, name: str, start: int = 0, size: int | None = None) -> tuple[np.ndarray, int]:
        index, tensor = self._tensors[name]
        if size is None:
            size = tensor["bytes"]
        if start < 0 or size < 0 or start + size > tensor["bytes"]:
            raise ValueError("Straydog tensor slice is out of range")
        return np.frombuffer(self._maps[index], np.uint8, count=size, offset=tensor["offset"] + start), tensor[
            "quant_type"
        ]

    def vector(self, name: str) -> np.ndarray:
        """Read a small norm, scalar, or convolution tensor into independent storage."""
        if name not in self._vectors:
            if np.prod(self.shape(name)) > 65536:
                raise ValueError("Straydog vector request is too large")
            raw, kind = self._raw(name)
            values = dequantize(raw, kind).copy()
            if not np.isfinite(values).all():
                raise ValueError("non-finite Straydog model parameters")
            self._vectors[name] = values.reshape(self.shape(name)[::-1])
        return self._vectors[name]

    def row(self, name: str, index: int) -> np.ndarray:
        """Decode only one token or n-gram embedding row from the mapped model."""
        dims = self.shape(name)
        if len(dims) != 2 or not 0 <= index < dims[1]:
            raise ValueError("Straydog embedding row is out of range")
        tensor = self._tensors[name][1]
        row_bytes = tensor["bytes"] // dims[1]
        raw, kind = self._raw(name, index * row_bytes, row_bytes)
        values = dequantize(raw, kind).copy()
        if values.size != dims[0] or not np.isfinite(values).all():
            raise ValueError("invalid Straydog embedding row")
        return values

    def rows(self, name: str, indices: np.ndarray) -> np.ndarray:
        """Gather a bounded set of packed embedding rows and decode them as one batch."""
        indices = np.asarray(indices)
        if name == "token_embd.weight" and indices.ndim == 1 and 1 <= len(indices) <= _CACHED_BATCH:
            cached = [self._embedding_rows.get(int(index)) for index in indices]
            if all(row is not None for row in cached):
                return np.stack(cached)
            decoded = self._decoded_rows(name, indices)
            for index, row in zip(indices, decoded, strict=True):
                if len(self._embedding_rows) >= _CACHED_ROWS:
                    del self._embedding_rows[next(iter(self._embedding_rows))]
                self._embedding_rows[int(index)] = row.copy()
            return decoded
        return self._decoded_rows(name, indices)

    def _decoded_rows(self, name: str, indices: np.ndarray) -> np.ndarray:
        dims = self.shape(name)
        if (
            len(dims) != 2 or indices.ndim != 1 or not 1 <= len(indices) <= 65536
            or len(indices) * dims[0] > 16 * 1024**2 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= dims[1])
        ):
            raise ValueError("invalid Straydog embedding row batch")
        tensor = self._tensors[name][1]
        raw, kind = self._raw(name)
        table = raw.reshape(dims[1], tensor["bytes"] // dims[1])
        if len(indices) >= _PARALLEL_ROWS:
            # Prompt rows of the 28.8 GB PLE table are random page faults when it is not cached: one thread
            # takes them one at a time (about 5k/s on an NVMe); several keep its queue full.
            if getattr(self, "_row_pool", None) is None:
                self._row_pool = ThreadPoolExecutor(max_workers=_ROW_THREADS, thread_name_prefix="straydog-rows")
            parts = np.array_split(indices, _ROW_THREADS)
            packed = np.concatenate(list(self._row_pool.map(table.__getitem__, parts)))
        else:
            packed = table[indices]
        output = dequantize(packed.reshape(-1), kind).reshape(len(indices), dims[0])
        if not np.isfinite(output).all():
            raise ValueError("non-finite Straydog embedding row batch")
        return output

    def _projection(self, name: str, vector: np.ndarray, expert: int | None):
        cache_key = (name, expert)
        cached = self._projection_cache.get(cache_key)
        if cached is not None:
            return (*cached, vector)
        dims = self.shape(name)
        if len(dims) not in {2, 3} or (len(dims) == 3) != (expert is not None):
            raise ValueError("Straydog matrix rank mismatch")
        if expert is not None and not 0 <= expert < dims[2]:
            raise ValueError("Straydog expert index is out of range")
        cols, rows = dims[:2]

        def matrix() -> tuple[np.ndarray, int]:
            tensor = self._tensors[name][1]
            size = tensor["bytes"] // (dims[2] if expert is not None else 1)
            raw, kind = self._raw(name, (expert or 0) * size, size)
            if kind not in BLOCKS:
                raise ValueError(f"unsupported Straydog weight format {kind}: {name}")
            return raw, kind

        descriptor = (f"{name}:{expert}", matrix, rows, cols)
        self._projection_cache[cache_key] = descriptor
        return (*descriptor, vector)

    def project(self, name: str, vector: np.ndarray, expert: int | None = None) -> np.ndarray:
        """Run a dense or individual expert projection on the owned GPU stream."""
        dims = self.shape(name)
        if len(dims) == 1:
            if expert is not None or vector.shape != dims:
                raise ValueError("Straydog scalar projection shape mismatch")
            return np.array([np.dot(self.vector(name), vector)], np.float32)
        return self.gpu.project(*self._projection(name, vector, expert))

    def project_many(self, requests: list[tuple[str, np.ndarray, int | None]]) -> list[np.ndarray]:
        """Batch independent projections, bounded by native descriptor/output capacity."""
        if not 1 <= len(requests) <= 131072:
            raise ValueError("Straydog projection request count is out of range")
        results = [None] * len(requests)
        projected = []
        for index, (name, vector, expert) in enumerate(requests):
            if len(self.shape(name)) == 1:
                results[index] = self.project(name, vector, expert)
            else:
                projected.append((index, self._projection(name, vector, expert)))
        # Adjacent uses of the same expert reuse its upload across prompt tokens.
        projected.sort(key=lambda item: item[1][0])
        batch, indices, rows = [], [], 0
        for index, request in projected:
            if batch and (
                len(batch) == getattr(self.gpu, "batch_capacity", 32)
                or rows + request[2] > getattr(self.gpu, "output_capacity", 262144)
            ):
                for position, result in zip(indices, self.gpu.project_many(batch), strict=True):
                    results[position] = result
                batch, indices, rows = [], [], 0
            batch.append(request)
            indices.append(index)
            rows += request[2]
        if batch:
            for position, result in zip(indices, self.gpu.project_many(batch), strict=True):
                results[position] = result
        return results

    def close(self) -> None:
        """Release read-only model mappings and handles."""
        if hasattr(self.gpu, "unregister_hosts"):
            self.gpu.unregister_hosts()
        cpu = getattr(self, "_cpu", None)
        if cpu is not None:
            cpu.close()
            self._cpu = None
        if getattr(self, "_row_pool", None) is not None:
            self._row_pool.shutdown()
            self._row_pool = None
        self._vectors.clear()
        self._projection_cache.clear()
        for mapping in self._maps:
            mapping.close()
        self._maps.clear()
        for file in self._files:
            file.close()
        self._files.clear()

    def hyperconnection(self, prefix: str, values: np.ndarray, *, inject: bool):
        """Project authenticated HyperConnection weights with native GPU post-operations."""
        names = [prefix + "_down.weight", prefix + "_up.weight"]
        if inject:
            names.append(prefix + "_inject.weight")
        matrices = [self._projection(name, None, None)[:4] for name in names]
        return self.gpu.hyperconnection(prefix, values, self.vector(prefix + "_norm.weight"), matrices)

    def gated_delta(self, prefix: str, values: np.ndarray) -> np.ndarray:
        """Use only authenticated weights for fused native DeltaNet execution."""
        names = ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight")
        matrices = [self._projection(prefix + name, None, None)[:4] for name in names]
        vectors = [
            (prefix + name, self.vector(prefix + name))
            for name in ("ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight")
        ]
        return self.gpu.gated_delta(prefix, values, matrices, vectors)

    def moe(self, prefix, values, selected, probabilities, shared_mix) -> np.ndarray:
        """Authenticate and map exactly the experts required by the native full-top-10 path."""
        experts = np.unique(selected)
        matrices = [
            self._projection(prefix + "ffn_" + role + "_exps.weight", None, int(expert))[:4]
            for expert in experts for role in ("gate", "up", "down")
        ]
        matrices += [
            self._projection(prefix + "ffn_" + role + "_shexp.weight", None, None)[:4]
            for role in ("gate", "up", "down")
        ]
        return self.gpu.moe(prefix, values, np.searchsorted(experts, selected), probabilities, shared_mix, matrices)

    def sparse_attention(self, prefix, values, *, position, context) -> np.ndarray:
        """Supply authenticated QSA projections and norms to the native causal path."""
        names = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "indexer.q_proj.weight",
                 "indexer.k_proj.weight", "attn_output.weight")
        matrices = [self._projection(prefix + name, None, None)[:4] for name in names]
        vectors = [(prefix + name, self.vector(prefix + name)) for name in (
            "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight",
        )]
        return self.gpu.sparse_attention(prefix, values, matrices, vectors, position=position, context=context)

    def resident_requirements(self, *, drafter: bool) -> dict[str, int]:
        """GPU bytes of every weight full residency pins (with or without the MTP drafter's)."""
        requirements = {}
        excluded = set() if drafter else getattr(self, "_drafter_names", set())
        for name, (_index, tensor) in self._tensors.items():
            if name in {"token_embd.weight", "per_layer_token_embd.weight"} or name in excluded:
                continue
            size = tensor["bytes"]
            if len(tensor["dimensions"]) == 1 or name.endswith("ssm_conv1d.weight"):
                size = max(size, int(np.prod(tensor["dimensions"])) * 4)
            requirements[name] = size
        return requirements

    def resident_fits(self, reserved: list[int] | None, *, drafter: bool) -> bool:
        """Whether full residency fits beside ``reserved`` VRAM per device (a placement's planned state)."""
        if not hasattr(self.gpu, "can_pin_model"):
            return False
        requirements = self.resident_requirements(drafter=drafter and bool(getattr(self, "drafter", False)))
        return self.gpu.can_pin_model(requirements, reserved)

    def prepare_resident_moe(self) -> bool:
        """Reserve full expert residency only after the complete weight footprint fits every GPU."""
        if not hasattr(self.gpu, "can_pin_model"):
            return False
        if self.drafter and not self.gpu.can_pin_model(self.resident_requirements(drafter=True)):
            self._drop_drafter()
        if not self.gpu.can_pin_model(self.resident_requirements(drafter=bool(self.drafter))):
            return False
        for layer in self.layers():
            prefix = f"blk.{layer}."
            banks = []
            for role in ("gate", "up", "down"):
                name = prefix + "ffn_" + role + "_exps.weight"
                cols, rows, experts = self.shape(name)
                banks.append((name + ":bank", lambda name=name: self._raw(name), rows * experts, cols))
            matrices = banks + [self._projection(prefix + "ffn_" + role + "_shexp.weight", None, None)[:4]
                                for role in ("gate", "up", "down")]
            matrices.append(self._projection(prefix + "ffn_gate_inp.weight", None, None)[:4])
            name = prefix + "ffn_gate_inp_shexp.weight"
            matrices.append((name + ":vector", lambda name=name: (self.vector(name), 0), 1, 2560))
            self.gpu.register_resident_moe(prefix, matrices, experts=512)
        return True

    def _dense_requirements(self) -> dict[str, int]:
        requirements = {}
        drafter = getattr(self, "_drafter_names", set())
        for name, (_index, tensor) in self._tensors.items():
            if (name in {"token_embd.weight", "per_layer_token_embd.weight"} or name.endswith("_exps.weight")
                    or name in drafter):
                continue
            size = tensor["bytes"]
            if len(tensor["dimensions"]) == 1 or name.endswith("ssm_conv1d.weight"):
                size = max(size, int(np.prod(tensor["dimensions"])) * 4)
            requirements[name] = size
        return requirements

    def prepare_hybrid_moe(self, *, cpu_threads: int = 0, pin_host: bool | None = None, assist_routes: int = 1,
                           pinned_state_bytes: int = 0) -> bool:
        """Partial residency: pin every dense operand, bind host expert banks for CPU/streaming.

        The CPU engine evaluates absent routed experts in F32; prefill streams them into
        staging slots. ``pin_host`` page-locks the mapped expert banks for direct DMA: all of
        them (True), none (False), or by default whole layers within ``lock_budget`` of the
        host memory (``pinned_state_bytes``: page-locked sequence state already allocated).
        Without the grouped FP16 prefill path (RDNA4 WMMA, CUDA tensor cores) the model keeps
        the bounded expert cache instead.
        """
        from straydog.cpu import CpuExperts

        if not hasattr(self.gpu, "register_hybrid_moe") or not hasattr(self.gpu, "can_pin_model"):
            return False
        if not self.gpu.hybrid_supported():
            return False
        # TODO(STRAYDOG-SPECULATIVE): Run the MTP drafter with partially resident experts.
        self._drop_drafter()
        kinds = {layer: self.expert_kinds(layer) for layer in range(48)}
        reserve = 64 * max(expert_bytes(value) for value in kinds.values()) + 256 * 1024**2
        requirements = self._dense_requirements()
        requirements["@hybrid:staging"] = reserve
        if not self.gpu.can_pin_model(requirements):
            return False
        self._cpu = CpuExperts(self.gpu.library, threads=cpu_threads)
        banks_total = 0
        for layer in range(48):
            prefix = f"blk.{layer}."
            banks = [self._raw(prefix + "ffn_" + role + "_exps.weight")[0] for role in ("gate", "up", "down")]
            banks_total += sum(bank.nbytes for bank in banks)
            matrices = [self._projection(prefix + "ffn_" + role + "_shexp.weight", None, None)[:4]
                        for role in ("gate", "up", "down")]
            matrices.append(self._projection(prefix + "ffn_gate_inp.weight", None, None)[:4])
            name = prefix + "ffn_gate_inp_shexp.weight"
            matrices.append((name + ":vector", lambda name=name: (self.vector(name), 0), 1, 2560))
            self.gpu.register_hybrid_moe(prefix, matrices, banks, experts=512, kinds=kinds[layer])
        # The head and its mixing are not decoder-layer operands; keep them resident too.
        self.gpu.pin_matrices([self._projection(name, None, None)[:4]
                               for name in ("output.weight", "output_hc_down.weight", "output_hc_up.weight")])
        budget = banks_total if pin_host else 0
        if pin_host is None:
            budget = lock_budget(getattr(self, "host_memory_bytes", 0), banks_total, pinned_state_bytes)
        # Whole layers from the last: the authentication pass read those most recently, so they are
        # the likeliest to be cached; the rest of the locked range is read ahead meanwhile.
        names = {layer: [f"blk.{layer}.ffn_{role}_exps.weight" for role in ("gate", "up", "down")]
                 for layer in range(48)}
        layers, locked = [], 0
        for layer in reversed(range(48)):
            size = sum(self._tensors[name][1]["bytes"] for name in names[layer])
            if locked + size > budget:
                break
            layers.append(layer)
            locked += size
        registered = 0
        if layers:
            helper = threading.Thread(target=self._read_ahead_banks, args=([names[layer] for layer in layers],),
                                      name="straydog-read-ahead", daemon=True)
            helper.start()
            try:
                for layer in layers:
                    if not all(self.gpu.register_host(self._raw(name)[0]) for name in names[layer]):
                        break
                    registered += 1
            finally:
                helper.join()
        if registered and assist_routes:
            # The GPU reads a few absent experts per layer over PCIe while the CPU computes the rest.
            # Both read host DRAM, so more than one route per layer mostly adds contention.
            self.gpu.enable_assist(assist_routes)
        self.gpu.attach_cpu(self._cpu.handle)
        return True

    def _read_ahead_banks(self, layers: list[list[str]]) -> None:
        """Request these layers' expert banks from the device in order (hints only; see _read_ahead)."""
        if not hasattr(os, "posix_fadvise"):
            return
        try:
            for names in layers:
                for name in names:
                    index, tensor = self._tensors[name]
                    fd = self._files[index].fileno()
                    for offset in range(tensor["offset"], tensor["offset"] + tensor["bytes"], HINT_STEP):
                        os.posix_fadvise(fd, offset, HINT_STEP, os.POSIX_FADV_WILLNEED)
        except OSError:
            logger.debug("Straydog read-ahead hint refused", exc_info=True)

    def expert_kinds(self, layer: int) -> tuple[int, int, int]:
        """GGML kinds of the routed gate, up and down banks of one layer."""
        return tuple(self._tensors[f"blk.{layer}.ffn_{role}_exps.weight"][1]["quant_type"]
                     for role in ("gate", "up", "down"))

    def finish_hybrid(self) -> int:
        """After every dense operand is pinned, give the remaining budget to expert slots."""
        return self.gpu.create_expert_slots()

    def resident_moe(self, prefix, values) -> np.ndarray:
        """Use the preflighted native plan without constructing CPU expert routes."""
        return self.gpu.resident_moe(prefix, values)

    def prepare_decoder(self, context: int) -> bool:
        """Authenticate and pin each complete decoder layer after full-residency preflight."""
        def vector(name, key=None):
            value = self.vector(name)
            rows, cols = value.shape if value.ndim == 2 else (1, value.size)
            return (key or "@vector:" + name, lambda value=value: (value, 0), rows, cols)

        def hc(prefix):
            return [vector(prefix + "_norm.weight", "@norm:" + prefix)] + [
                self._projection(prefix + suffix, None, None)[:4]
                for suffix in ("_down.weight", "_up.weight", "_inject.weight")
            ]

        for layer in self.layers():
            prefix = f"blk.{layer}."
            matrices = hc(prefix + "hc_attn")
            if layer % 4 != 3 and layer < 48:
                names = ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight")
                norms = ("ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight")
            else:
                names = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "indexer.q_proj.weight",
                         "indexer.k_proj.weight", "attn_output.weight")
                norms = ("attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight")
            matrices += [self._projection(prefix + name, None, None)[:4] for name in names]
            matrices += [vector(prefix + name) for name in norms]
            matrices += hc(prefix + "hc_ffn")
            self.gpu.register_decoder_layer(layer, matrices, context=context)
        return True

    def _vector_operand(self, name, key=None):
        value = self.vector(name)
        rows, cols = value.shape if value.ndim == 2 else (1, value.size)
        return (key or "@vector:" + name, lambda value=value: (value, 0), rows, cols)

    def head_operands(self) -> list[tuple]:
        """Output HyperConnection mix (norm, down, up) and the vocabulary head."""
        names = ("output_hc_down.weight", "output_hc_up.weight", "output.weight")
        projections = [self._projection(name, None, None)[:4] for name in names]
        return [self._vector_operand("output_hc_norm.weight")] + projections

    def output_head(self, values, *, logits=False):
        """First maximum of each row's logits through the resident head (exact one-row arithmetic)."""
        return self.gpu.output_head(self.head_operands(), values, logits=logits)

    def ple_rows(self, values, embeddings):
        """PLE for speculative rows with the one-row arithmetic."""
        prefix = "blk.1."
        matrices = [self._projection(prefix + name, None, None)[:4] for name in ("ple_key.weight", "ple_value.weight")]
        vectors = [(prefix + name, self.vector(prefix + name)) for name in
                   ("ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight")]
        return self.gpu.ple_rows(prefix, values, embeddings, matrices, vectors)

    def verify_span(self, begin, end, values, *, position):
        """Verify up to four speculative tokens through a resident span."""
        return self.gpu.verify_span(begin, end, values, position=position)

    def verify_commit(self, begin, end, kept):
        self.gpu.verify_commit(begin, end, kept)

    def ple(self, values, embeddings):
        """Supply only authenticated PLE projections and normalization operands."""
        prefix = "blk.1."
        matrices = [self._projection(prefix + name, None, None)[:4] for name in ("ple_key.weight", "ple_value.weight")]
        vectors = [(prefix + name, self.vector(prefix + name)) for name in
                   ("ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight")]
        return self.gpu.ple(prefix, values, embeddings, matrices, vectors)

    def prefill_ple(self, embeddings):
        """Apply authenticated PLE operands in place to resident prefill rows."""
        prefix = "blk.1."
        matrices = [self._projection(prefix + name, None, None)[:4] for name in ("ple_key.weight", "ple_value.weight")]
        vectors = [(prefix + name, self.vector(prefix + name)) for name in
                   ("ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight")]
        return self.gpu.prefill_ple(prefix, embeddings, matrices, vectors, layer=1)

    def prefill_layers(self, begin, end, values, *, position, steps, output_rows=0):
        """Advance resident prefill rows through a preflighted span on its owning GPU."""
        return self.gpu.prefill_layers(begin, end, values, position=position, steps=steps, output_rows=output_rows)

    def decode_span(self, begin, end, values, *, position):
        """Execute the preflighted native span using the shared layer placement."""
        return self.gpu.decode_span(begin, end, values, position=position)


def _inspector_env() -> dict[str, str]:
    # A fixed Rust executable needs only Windows runtime/temp paths. No provider
    # credentials or user-configured dynamic-library search paths are inherited.
    return {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "TEMP", "TMP") if key in os.environ}
