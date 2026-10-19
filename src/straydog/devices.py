"""Layer placement across owner-selected GPUs; one engine and one sequence state."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from straydog.gpu import GpuBackend


class GpuGroup:
    """Use each device's bounded cache for a contiguous range of decoder layers.

    Layers execute in sequence on their owner. With two GPUs holding the whole model,
    every layer's routed experts are split between them instead (expert parallelism):
    both evaluate their half of each layer's routes at once.
    """

    def __init__(self, library: Path, *, devices: list[int], budget_bytes: int, split_mode: str = "experts") -> None:
        if split_mode not in ("layer", "experts"):
            raise ValueError("Straydog split mode must be layer or experts")
        if (
            not 1 <= len(devices) <= 4
            or len(set(devices)) != len(devices)
            or any(isinstance(device, bool) or not isinstance(device, int) or device < 0 for device in devices)
        ):
            raise ValueError("Straydog requires one to four distinct GPU ordinals")
        self._backends: list[GpuBackend] = []
        # First layer owned by each device: an even split until balance_layers sees the weights.
        self._starts = [-(-index * 48 // len(devices)) for index in range(len(devices))]
        try:
            for device in devices:
                self._backends.append(GpuBackend(library, device=device, budget_bytes=budget_bytes))
        except BaseException:
            self.close()
            raise
        self._baseline = [backend.memory()[0] for backend in self._backends]
        self._split_mode = split_mode
        self._linked = False

    @property
    def expert_parallel(self) -> bool:
        """Whether a fully resident model splits every layer's routed experts across the two GPUs.

        It needs the grouped FP16 prefill path on both (explicit FP16 GEMM on RDNA4 or CUDA
        tensor cores). Split mode ``layer`` keeps the layer split.
        """
        return (len(self._backends) == 2 and self._split_mode == "experts"
                and all(backend.hybrid_supported() for backend in self._backends))

    def _expert_half(self, index: int, experts: int = 512) -> tuple[int, int]:
        """Routed experts [first, first + count) device ``index`` holds under expert parallelism."""
        half = experts // 2
        return (0, half) if index == 0 else (half, experts - half)

    def balance_layers(self, layer_bytes: list[int], tail_bytes: int) -> None:
        """Split the 48 layers into contiguous ranges with the smallest largest weight share.

        The last device also holds ``tail_bytes`` (output head, MTP layer). Placement
        must be decided before any weight or state is placed.
        """
        count = len(self._backends)
        if len(layer_bytes) != 48 or min(layer_bytes) < 0 or tail_bytes < 0:
            raise ValueError("invalid Straydog layer sizes")
        if any(backend.memory()[0] != self._baseline[index] for index, backend in enumerate(self._backends)):
            raise RuntimeError("Straydog layers must be balanced before placement")
        prefix = np.concatenate([[0], np.cumsum(layer_bytes)])
        best, starts = None, self._starts

        def search(first: int, chosen: list[int]) -> None:
            nonlocal best, starts
            if len(chosen) == count - 1:
                bounds = [0, *chosen, 48]
                loads = [prefix[bounds[i + 1]] - prefix[bounds[i]] for i in range(count)]
                loads[-1] += tail_bytes
                if best is None or max(loads) < best:
                    best, starts = max(loads), [0, *chosen]
                return
            for split in range(first, 48 - (count - 2 - len(chosen))):
                search(split + 1, [*chosen, split])

        search(1, [])
        self._starts = starts

    @property
    def batch_capacity(self) -> int:
        return min(backend.batch_capacity for backend in self._backends)

    @property
    def output_capacity(self) -> int:
        return min(backend.output_capacity for backend in self._backends)

    def configure_gemm(self, **kwargs) -> None:
        """Apply one explicit precision policy to every device."""
        for backend in self._backends:
            backend.configure_gemm(**kwargs)

    def _index(self, name: str) -> int:
        if not self._backends:
            raise RuntimeError("Straydog GPU group is closed")
        if name.startswith("blk."):
            layer = int(name.split(".", 2)[1])
            if not 0 <= layer <= 48:
                raise ValueError("Straydog layer is out of range")
            # Layer 48 is the MTP draft layer; it stays with the head on the last GPU.
            return max(index for index, start in enumerate(self._starts) if layer >= start)
        return len(self._backends) - 1

    @property
    def stats(self) -> dict[str, float] | None:
        enabled = [backend.stats for backend in self._backends if backend.stats is not None]
        if not enabled:
            return None
        return {key: sum(stats.get(key, 0) for stats in enabled) for key in set().union(*enabled)}

    @stats.setter
    def stats(self, value: dict[str, float] | None) -> None:
        for backend in self._backends:
            backend.stats = {} if value is not None else None

    def project(self, key, matrix, rows, cols, vector) -> np.ndarray:
        """Project on the device owning this layer's immutable weights."""
        return self._backends[self._index(key)].project(key, matrix, rows, cols, vector)

    def project_many(self, requests: list[tuple]) -> list[np.ndarray]:
        """Preserve caller order while grouping projections by owner device."""
        groups: dict[int, list[tuple[int, tuple]]] = {}
        for position, request in enumerate(requests):
            groups.setdefault(self._index(request[0]), []).append((position, request))
        result = [None] * len(requests)
        for index, members in groups.items():
            outputs = self._backends[index].project_many([request for _, request in members])
            for (position, _), output in zip(members, outputs, strict=True):
                result[position] = output
        return result

    def delta_sequence(self, state_key, *arrays) -> np.ndarray:
        """Keep recurrence on the same device as its layer's projections."""
        return self._backends[self._index(state_key)].delta_sequence(state_key, *arrays)

    def hyperconnection(self, prefix, *args):
        """Execute fused residual-stream mixing on the layer's owning device."""
        return self._backends[self._index(prefix)].hyperconnection(prefix, *args)

    def gated_delta(self, prefix, *args):
        """Keep all DeltaNet projections and causal state updates on its owning GPU."""
        return self._backends[self._index(prefix)].gated_delta(prefix, *args)

    def moe(self, prefix, *args):
        """Keep routed and shared expert intermediates on their layer's device."""
        return self._backends[self._index(prefix)].moe(prefix, *args)

    def sparse_attention(self, prefix, *args, **kwargs):
        """Keep fused QSA projections, indexer and K/V on the layer's owning device."""
        return self._backends[self._index(prefix)].sparse_attention(prefix, *args, **kwargs)

    def can_pin_model(self, requirements: dict[str, int], reserved: list[int] | None = None) -> bool:
        """Check full-model weight placement independently against each device's remaining budget.

        ``reserved`` lists, per device, VRAM a placement still needs (planned_state_bytes).
        """
        groups = [{} for _ in self._backends]
        parallel = self.expert_parallel
        for key, size in requirements.items():
            layer = int(key.split(".", 2)[1]) if key.startswith("blk.") else 48
            if parallel and layer < 48 and key.endswith("_exps.weight"):
                for index, group in enumerate(groups):
                    group[key] = size * self._expert_half(index)[1] // 512
                continue
            groups[self._index(key)][key] = size
            if parallel and key == "output.weight":
                groups[0][key + ":rows0"] = size // 2   # the lower vocabulary half the first GPU evaluates
        return all(
            backend.can_pin_model(group) if reserved is None else backend.can_pin_model(group, reserved[index])
            for index, (backend, group) in enumerate(zip(self._backends, groups, strict=True))
        )

    def register_resident_moe(self, prefix, matrices, *, experts: int):
        """Pin each layer's expert banks on its owner, or half of them on each GPU (expert parallelism).

        The MTP layer keeps its experts whole on its owner: it drafts one row at a time.
        """
        owner = self._index(prefix)
        if not self.expert_parallel or int(prefix.split(".")[1]) >= 48:
            self._backends[owner].register_resident_moe(prefix, matrices, experts=experts)
            return
        if not self._linked:
            raise RuntimeError("Straydog expert parallelism needs linked GPUs (reserve_model_state)")
        for index, backend in enumerate(self._backends):
            first, count = self._expert_half(index, experts)
            parts = [(f"{key}:{first}+{count}", _bank_slice(factory, first, count, experts),
                      rows * count // experts, cols) for key, factory, rows, cols in matrices[:3]]
            backend.register_resident_moe(prefix, parts + list(matrices[3:]), experts=experts, first=first,
                                          count=count, shared=index == owner, helper=index != owner)

    def resident_moe(self, prefix, values):
        """Execute GPU-only routing and expert combination on the owning device."""
        return self._backends[self._index(prefix)].resident_moe(prefix, values)

    def ple(self, prefix, *arrays):
        """Keep PLE with the layer that consumes its output."""
        return self._backends[self._index(prefix)].ple(prefix, *arrays)

    def register_decoder_layer(self, layer, matrices, **kwargs):
        """Keep a native decoder layer with the same owner as all its operands."""
        self._backends[self._index(f"blk.{layer}.")].register_decoder_layer(layer, matrices, **kwargs)

    def decoder_spans(self, start=1):
        """Partition remaining layers at the existing device boundaries."""
        spans, begin = [], start
        for layer in range(start + 1, 48):
            if self._index(f"blk.{layer}.") != self._index(f"blk.{begin}."):
                spans.append((begin, layer))
                begin = layer
        spans.append((begin, 48))
        return spans

    def decode_span(self, begin, end, values, **kwargs):
        """Transfer only the contiguous span's input and output at its owning GPU."""
        owner = self._index(f"blk.{begin}.")
        if any(self._index(f"blk.{layer}.") != owner for layer in range(begin, end)):
            raise ValueError("Straydog decoder span crosses devices")
        return self._backends[owner].decode_span(begin, end, values, **kwargs)

    def verify_span(self, begin, end, values, **kwargs):
        """Verify speculative tokens through a contiguous span on its owning GPU."""
        owner = self._index(f"blk.{begin}.")
        if any(self._index(f"blk.{layer}.") != owner for layer in range(begin, end)):
            raise ValueError("Straydog decoder span crosses devices")
        return self._backends[owner].verify_span(begin, end, values, **kwargs)

    def verify_commit(self, begin, end, kept):
        self._backends[self._index(f"blk.{begin}.")].verify_commit(begin, end, kept)

    def output_head(self, matrices, values, **kwargs):
        """The vocabulary head and its mixing live on the last GPU.

        In the two-GPU mode the first GPU also evaluates the lower half of the vocabulary rows
        from its own copy, at the same time (tokens and logits are the one-GPU head's).
        """
        last = self._backends[-1]
        if not (self.expert_parallel and self._linked):
            return last.output_head(matrices, values, **kwargs)
        key, factory, rows, cols = matrices[3]
        split = rows // 2
        lower = self._backends[0]._prepare(f"{key}:rows0", _row_slice(factory, 0, split, rows), split, cols,
                                           {f"{key}:rows0"})
        self._backends[0]._resident_weights.add(f"{key}:rows0")
        return last.output_head_split(matrices, values, split=split, peer_head=lower[0], **kwargs)

    def mtp_prologue(self, matrices, embeddings, hidden):
        return self._backends[-1].mtp_prologue(matrices, embeddings, hidden)

    def ple_rows(self, prefix, *arrays):
        return self._backends[self._index(prefix)].ple_rows(prefix, *arrays)

    def copy_state(self, target, source, *, owner):
        self._backends[self._index(owner)].copy_state(target, source)

    @property
    def library(self) -> Path:
        return self._backends[0].library

    def pin_matrices(self, matrices: list[tuple]) -> None:
        """Pin each dense operand on the device that owns its layer."""
        groups: dict[int, list[tuple]] = {}
        for item in matrices:
            groups.setdefault(self._index(item[0]), []).append(item)
        for index, items in groups.items():
            self._backends[index].pin_matrices(items)

    def register_hybrid_moe(self, prefix, matrices, banks, **kwargs):
        """Bind a partially resident layer on its owner."""
        self._backends[self._index(prefix)].register_hybrid_moe(prefix, matrices, banks, **kwargs)

    def create_expert_slots(self, **kwargs) -> int:
        """Size every device's expert cache from its remaining budget."""
        return sum(backend.create_expert_slots(**kwargs) for backend in self._backends)

    def hybrid_supported(self) -> bool:
        return all(backend.hybrid_supported() for backend in self._backends)

    def attach_cpu(self, engine: int) -> None:
        for backend in self._backends:
            backend.attach_cpu(engine)

    def rebalance(self, max_swaps: int = 0) -> None:
        for backend in self._backends:
            backend.rebalance(max_swaps)

    def hybrid_counts(self) -> dict[str, int]:
        counts = [backend.hybrid_counts() for backend in self._backends]
        return {key: sum(item[key] for item in counts) for key in counts[0]}

    def enable_assist(self, routes: int) -> int:
        return sum(backend.enable_assist(routes) for backend in self._backends)

    def register_host(self, array: np.ndarray) -> bool:
        """Page locks are process-wide; one backend performs them."""
        return self._backends[0].register_host(array)

    def unregister_hosts(self) -> None:
        for backend in self._backends:
            backend.unregister_hosts()

    def device_of(self, layer: int) -> int:
        """Index of the GPU that owns ``layer`` in this contiguous placement."""
        return self._index(f"blk.{layer}.")

    def prefill_layers(self, begin, end, values, **kwargs):
        """Advance resident prefill rows on the span's single owning GPU."""
        owner = self._index(f"blk.{begin}.")
        if any(self._index(f"blk.{layer}.") != owner for layer in range(begin, end)):
            raise ValueError("Straydog decoder span crosses devices")
        return self._backends[owner].prefill_layers(begin, end, values, **kwargs)

    def prefill_ple(self, prefix, *arrays, **kwargs):
        """Apply PLE in place on the GPU that owns its layer."""
        return self._backends[self._index(prefix)].prefill_ple(prefix, *arrays, **kwargs)

    def attention_sequence(self, state_key, *arrays, **kwargs) -> np.ndarray:
        """Keep sparse attention K/V on the device owning its layer."""
        return self._backends[self._index(state_key)].attention_sequence(state_key, *arrays, **kwargs)

    def reserve_model_state(
        self, context: int, *, attention: bool, prefill: int = 1024, drafter: bool = False,
        host_attention: bool = False,
    ) -> None:
        """Reserve each layer's state on its owner (and the MTP layer's) before accepting requests.

        Under expert parallelism the two GPUs are also linked, with inboxes for a prefill batch.
        """
        for index, backend in enumerate(self._backends):
            layers = [layer for layer in range(49 if drafter else 48) if self._index(f"blk.{layer}.") == index]
            backend.reserve_model_state(context, attention=attention, layers=layers, prefill=prefill,
                                        **({"host_attention": True} if host_attention else {}))
        if self.expert_parallel and not self._linked:
            self._backends[0].link_peer(self._backends[1], min(context, prefill))
            self._linked = True

    def planned_state_bytes(
        self, context: int, *, attention: bool, prefill: int = 1024, drafter: bool = False,
        host_attention: bool = False,
    ) -> list[int]:
        """Per device, the VRAM a placement would still allocate (its owned layers' state)."""
        # Expert parallelism links the GPUs with inboxes of two prefill batches each (peer VRAM).
        inbox = min(context, prefill) * 2560 * 2 * 4 + 256 if self.expert_parallel else 0
        return [
            backend.planned_state_bytes(
                context, attention=attention, prefill=prefill, host_attention=host_attention,
                # As reserve_model_state: the MTP layer's state goes with its owner.
                layers=[layer for layer in range(49 if drafter else 48) if self._index(f"blk.{layer}.") == index],
            ) + inbox
            for index, backend in enumerate(self._backends)
        ]

    def host_state_supported(self) -> bool:
        return all(backend.host_state_supported() for backend in self._backends)

    @property
    def logical_capacity(self) -> int:
        """Use only the batch capacity every selected device can support."""
        return min(backend.logical_capacity for backend in self._backends)

    def reset_state(self) -> None:
        """Clear sequence state on every owned device."""
        for backend in self._backends:
            backend.reset_state()

    def reserve_checkpoints(self, count: int) -> int:
        """Reserve each device's host copies of the recurrent state it owns; their total bytes."""
        return sum(backend.reserve_checkpoints(count) for backend in self._backends)

    def save_checkpoint(self, slot: int) -> None:
        for backend in self._backends:
            backend.save_checkpoint(slot)

    def load_checkpoint(self, slot: int) -> None:
        for backend in self._backends:
            backend.load_checkpoint(slot)

    def memory(self) -> tuple[int, int]:
        """Sum owned allocation and per-device cache budgets."""
        pairs = [backend.memory() for backend in self._backends]
        return sum(pair[0] for pair in pairs), sum(pair[1] for pair in pairs)

    def info(self) -> dict:
        """Return aggregate telemetry with individual physical device records."""
        if not self._backends:
            raise RuntimeError("Straydog GPU group is closed")
        devices = [backend.info() for backend in self._backends]
        return {
            "name": " + ".join(device["name"] for device in devices),
            "device": devices[0]["device"],
            "devices": devices,
            **{
                key: sum(device[key] for device in devices)
                for key in (
                    "free_bytes",
                    "total_bytes",
                    "allocated_bytes",
                    "budget_bytes",
                    "gemm_calls",
                    "gemm_vectors",
                )
            },
            # Partial-residency counters exist only on backends with an expert slot pool.
            **{
                key: sum(device.get(key, 0) for device in devices)
                for key in ("gpu_experts", "cpu_experts", "streamed_experts", "cached_experts", "wait_ns", "cpu_ns",
                            "swapped_experts", "assisted_experts")
            },
        }

    def close(self) -> None:
        """Release all owned caches, recurrence and BLAS handles."""
        for backend in self._backends:
            backend.close()
        self._backends.clear()

    def __enter__(self) -> GpuGroup:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _row_slice(factory, first: int, count: int, rows: int):
    """The bytes of rows [first, first + count) of a packed row-major matrix."""
    def load():
        data, kind = factory()
        data = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        per_row = data.nbytes // rows
        return data[first * per_row:(first + count) * per_row], kind
    return load


def _bank_slice(factory, first: int, count: int, experts: int):
    """The bytes of experts [first, first + count) of an expert-major packed bank."""
    def load():
        data, kind = factory()
        data = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        per_expert = data.nbytes // experts
        return data[first * per_expert:(first + count) * per_expert], kind
    return load
