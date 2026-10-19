"""Straydog's independent Qwen3.8-Flash-Next decoder with layer-major prefill.

Implements Gated DeltaNet, QSA, gated residual streams, PLE and full MoE.
GPU kernels own projections, recurrence and bounded sparse attention; NumPy
supplies residual/activation arithmetic and the CPU state reference path.
Speculative decoding drafts with the base model's own MTP layer and verifies up to
four tokens per pass with the one-token arithmetic, so greedy output is identical to
token-by-token decode and sampling keeps the model's distribution. Images and
above-native RoPE scaling are deliberately not advertised.

A new prompt resumes from the longest prefix it shares with what the model holds:
the live sequence when the prompt extends it, else the deepest conversation
checkpoint the prompt still matches (the end of the system prompt, and where the
last user-initiated and the last tool-result request began generating).
"""

from __future__ import annotations

# TODO(STRAYDOG-CONTEXT): Validate sparse QSA retrieval and consumer 8GB / 32GB / 64GB configurations.
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from straydog.numeric import (
    causal_conv,
    delta_step,
    ngram_rows,
    ngram_rows_many,
    rms_norm,
    rotary,
    sigmoid,
    silu,
    softmax,
)
from straydog.weights import StraydogWeights

_MAX_DRAFTS = 3   # one verification pass holds the next token and up to three drafts
_END_TOKENS = frozenset({248044, 248046})
_ROW_COST = 0.45  # verifying (and drafting) one more row, relative to a one-token pass
_IM_START, _IM_END = 248045, 248046
# Conversation checkpoints, each a page-locked host copy of the recurrent state (~118 MB):
#   root  - the end of the system prompt, for every chat that shares it;
#   turn  - where the reply to the last user message began (the next user turn diverges there
#           when a client drops that reply's reasoning);
#   prior - the same for the user message before (an edited or retried last message);
#   step  - where the reply to the last tool result began.
# Only a system prompt this long is worth a copy.
_CHECKPOINTS = ("root", "turn", "prior", "step")
_ROOT_MINIMUM = 2048


@dataclass
class _Checkpoint:
    slot: int
    position: int
    tokens: np.ndarray
    final_hidden: np.ndarray | None
    host: list[np.ndarray]


def draft_length(acceptance: float, limit: int) -> int:
    """Drafts that maximize expected tokens per pass cost for a per-draft acceptance rate."""
    best, chosen = 1.0, 0
    for drafts in range(1, limit + 1):
        expected = sum(acceptance**index for index in range(drafts + 1))
        value = expected / (1 + _ROW_COST * drafts)
        if value > best:
            best, chosen = value, drafts
    return chosen


def context_state_bytes(context: int) -> int:
    """Host bytes of one sequence's state before the GPU takes its attention or recurrence part.

    QSA K/V + pooled indexer keys, DeltaNet state/conv, PLE history and the partial
    indexer pooling buffers; the RAM budget must hold this when nothing moves to the GPU.
    """
    state_bytes = 12 * (context * 2 * 2 * 256 + (context // 4 + 1) * 128) * 4
    state_bytes += 36 * (48 * 128 * 128 + 4 * 10240) * 4 + 10 * 10240 * 4
    return state_bytes + 12 * 4 * 128 * 4


class StraydogModel:
    """One resident decoder state and one serialized generation sequence."""

    def __init__(
        self,
        weights: StraydogWeights,
        *,
        context: int = 8192,
        state_budget_bytes: int = 2**30,
        prefill_batch_size: int = 256,
        speculative: int = 0,
    ) -> None:
        if isinstance(context, bool) or not 1 <= context <= weights.context_limit:
            raise ValueError("Straydog context must be within the native model limit")
        if isinstance(prefill_batch_size, bool) or not 1 <= prefill_batch_size <= 4096:
            raise ValueError("Straydog prefill batch must be between 1 and 4096")
        if isinstance(speculative, bool) or not 0 <= speculative <= _MAX_DRAFTS:
            raise ValueError(f"Straydog speculative drafts must be between 0 and {_MAX_DRAFTS}")
        drafting = speculative > 0 and getattr(weights, "drafter", False)
        self.prefill_batch_size = prefill_batch_size
        state_bytes = context_state_bytes(context)
        self._gpu_recurrence = hasattr(getattr(weights, "gpu", None), "delta_sequence")
        gpu = getattr(weights, "gpu", None)
        if prefill_batch_size > getattr(gpu, "logical_capacity", 4096):
            raise ValueError("Straydog prefill exceeds this GPU workspace capacity")
        attention_capable = hasattr(gpu, "attention_sequence") and getattr(gpu, "batch_capacity", 32) == 256
        kv_bytes = 12 * context * 1024 * 4
        fits = attention_capable and kv_bytes <= gpu.memory()[1] // 4
        host_capable = attention_capable and hasattr(gpu, "host_state_supported") and gpu.host_state_supported()
        recurrent_bytes = 36 * 48 * 128 * 128 * 4 if self._gpu_recurrence else 0
        # K/V beyond a quarter of the VRAM budget (a long context on a small card) stays in host RAM,
        # mapped so the GPU kernels read the selected rows over the bus instead of a CPU attention path.
        host, batch = attention_capable and not fits and host_capable, prefill_batch_size
        if host_capable and state_bytes - recurrent_bytes <= state_budget_bytes:
            host, batch = self._resident_placement(
                weights, gpu, context, prefill_batch_size, bool(drafting), vram_kv=fits, default=(host, batch))
        self.prefill_batch_size = batch
        self._host_attention = host
        self._gpu_attention = fits or self._host_attention
        if self._gpu_attention and not self._host_attention:
            state_bytes -= kv_bytes
        state_bytes -= recurrent_bytes
        if state_bytes > state_budget_bytes:
            raise ValueError(f"Straydog context state needs {state_bytes} bytes, over its RAM budget")
        self.weights, self.context, self.state_bytes = weights, context, state_bytes
        self.history: list[int] = []
        self.last_metrics: dict[str, float | int | None] = {}
        self.stats: dict[str, float] | None = None
        self._linear: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._attention: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._ple_history = np.zeros((10, 10240), np.float32)
        self._validate_geometry()
        self._allocate_states()
        if self._gpu_recurrence:
            weights.gpu.reserve_model_state(
                context, attention=self._gpu_attention, prefill=self.prefill_batch_size,
                **({"drafter": True} if drafting else {}),
                **({"host_attention": True} if self._host_attention else {}),
            )
        checkpoint_bytes = (
            weights.gpu.reserve_checkpoints(len(_CHECKPOINTS))
            if self._gpu_recurrence and hasattr(weights.gpu, "reserve_checkpoints") else 0
        )
        self._resident_moe = hasattr(weights, "prepare_resident_moe") and weights.prepare_resident_moe()
        # Partial residency: dense layers stay on the GPU; routed experts use GPU slots,
        # the CPU expert engine (decode) and staging slots (prefill).
        self._hybrid = (
            not self._resident_moe and self._gpu_attention and hasattr(weights, "prepare_hybrid_moe")
            and getattr(getattr(weights, "gpu", None), "batch_capacity", 0) == 256
            and weights.prepare_hybrid_moe(
                pinned_state_bytes=checkpoint_bytes + (12 * context * 1024 * 4 if self._host_attention else 0)
            )
        )
        self._decoder_ready = (
            (self._resident_moe or self._hybrid) and self._gpu_attention and weights.prepare_decoder(context)
        )
        if self._hybrid:
            weights.finish_hybrid()
        # MTP drafting needs the fully resident decoder (exact multi-token verification).
        self.speculative = speculative if (
            drafting and self._decoder_ready and self._resident_moe and getattr(weights, "drafter", False)
            and hasattr(weights.gpu, "verify_span")
        ) else 0
        self._final_hidden: np.ndarray | None = None   # last processed token's four-stream hidden
        self._checkpoints: dict[str, _Checkpoint] = {}

    @staticmethod
    def _resident_placement(weights, gpu, context, prefill, drafting, *, vram_kv, default) -> tuple[bool, int]:
        """(K/V in host memory, prefill batch) that lets every weight live on the GPUs, else ``default``.

        Full residency (and with it MTP speculation) is worth more than K/V in VRAM: with the K/V in
        mapped host memory and, if still needed, a smaller prefill workspace, IQ3_S on two 32 GB cards
        at the native context decodes ~97 instead of ~56 tokens/s. A placement keeping the drafter wins
        over one that drops it; the configured batch and VRAM K/V are tried first.
        """
        if not hasattr(weights, "resident_fits") or not hasattr(gpu, "planned_state_bytes"):
            return default
        batches = [prefill]
        while batches[-1] > 1024:
            batches.append(max(1024, batches[-1] // 2))
        candidates = [(False, prefill)] if vram_kv else []
        candidates += [(True, batch) for batch in batches]
        for keep_drafter in (True, False) if drafting else (False,):
            for host, batch in candidates:
                planned = gpu.planned_state_bytes(context, attention=True, prefill=batch, drafter=drafting,
                                                  host_attention=host)
                if weights.resident_fits(planned, drafter=keep_drafter):
                    return host, batch
        return default

    def _validate_geometry(self) -> None:
        expected = {
            "token_embd.weight": (2560, 248320),
            "output.weight": (2560, 248320),
            "per_layer_token_embd.weight": (160, 320001536),
            "output_hc_norm.weight": (10240,),
            "output_hc_down.weight": (10240, 320),
            "output_hc_up.weight": (320, 10240),
        }
        for layer in range(48):
            prefix = f"blk.{layer}."
            for operation in ("hc_attn", "hc_ffn"):
                expected.update(
                    {
                        prefix + operation + "_norm.weight": (10240,),
                        prefix + operation + "_down.weight": (10240, 320),
                        prefix + operation + "_up.weight": (320, 10240),
                        prefix + operation + "_inject.weight": (10240, 4),
                    }
                )
            expected.update(
                {
                    prefix + "ffn_gate_inp.weight": (2560, 512),
                    prefix + "ffn_gate_inp_shexp.weight": (2560,),
                    prefix + "ffn_gate_exps.weight": (2560, 640, 512),
                    prefix + "ffn_up_exps.weight": (2560, 640, 512),
                    prefix + "ffn_down_exps.weight": (640, 2560, 512),
                }
            )
            if layer % 4 != 3:
                expected.update(
                    {
                        prefix + "attn_qkv.weight": (2560, 10240),
                        prefix + "attn_gate.weight": (2560, 6144),
                        prefix + "ssm_conv1d.weight": (4, 10240),
                        prefix + "ssm_out.weight": (6144, 2560),
                    }
                )
            else:
                expected.update(
                    {
                        prefix + "attn_q.weight": (2560, 12288),
                        prefix + "attn_k.weight": (2560, 512),
                        prefix + "attn_v.weight": (2560, 512),
                        prefix + "attn_output.weight": (6144, 2560),
                    }
                )
        for name, shape in expected.items():
            if self.weights.shape(name) != shape:
                raise ValueError(f"Straydog model geometry mismatch: {name}")
        if self.weights.integer_arrays.get("qwen4exp.ple.layers") != [1]:
            raise ValueError("Straydog model PLE placement mismatch")

    def _allocate_states(self) -> None:
        # Touch the owned buffers rather than declaring capacity from metadata.
        for layer in range(48):
            if layer % 4 != 3:
                self._linear[layer] = (
                    np.zeros((4, 10240), np.float32),
                    None if self._gpu_recurrence else np.zeros((48, 128, 128), np.float32),
                )
            else:
                self._attention[layer] = (
                    None if self._gpu_attention else np.empty((self.context, 2, 256), np.float32),
                    None if self._gpu_attention else np.empty((self.context, 2, 256), np.float32),
                    np.empty((self.context // 4 + 1, 128), np.float32),
                    np.zeros((4, 128), np.float32),
                )
                for value in self._attention[layer]:
                    if value is not None:
                        value.fill(0)

    def reset(self) -> None:
        """Reset sequence-dependent state while retaining the bounded GPU weight cache."""
        self.history.clear()
        self._final_hidden = None
        self._ple_history.fill(0)
        if getattr(self, "_gpu_recurrence", False):
            self.weights.gpu.reset_state()
        for conv, recurrent in self._linear.values():
            conv.fill(0)
            if recurrent is not None:
                recurrent.fill(0)
        # Attention uses only [0:len(history)], so stale tail cells are inaccessible.
        for state in self._attention.values():
            state[3].fill(0)

    def _host_state(self) -> list[np.ndarray]:
        """Recurrent state the model keeps in host arrays (all of it on the CPU reference path)."""
        arrays = [self._ple_history]
        for conv, recurrent in self._linear.values():
            arrays += [conv] if recurrent is None else [conv, recurrent]
        return arrays + [state[3] for state in self._attention.values()]

    def _save_checkpoint(self, kind: str) -> None:
        checkpoints = self._checkpoints
        if kind == "turn" and "turn" in checkpoints:
            checkpoints.pop("prior", None)
            checkpoints["prior"] = checkpoints.pop("turn")
        used = {checkpoint.slot for name, checkpoint in checkpoints.items() if name != kind}
        slot = min(set(range(len(_CHECKPOINTS))) - used)
        if self._gpu_recurrence:
            self.weights.gpu.save_checkpoint(slot)
        checkpoints[kind] = _Checkpoint(
            slot, len(self.history), np.asarray(self.history, np.int32),
            None if self._final_hidden is None else self._final_hidden.copy(),
            [array.copy() for array in self._host_state()],
        )

    def _resume(self, tokens: list[int]) -> tuple[int, str]:
        """Prompt tokens already held, after restoring the state they need; at least one token is left to run.

        Attention K/V below the restored position comes from the live sequence, which matches the
        prompt there; the checkpoint restores the recurrent state at that position.
        """
        held, limit = self.history, len(tokens) - 1
        if len(held) <= limit and held == tokens[: len(held)]:
            return len(held), "live"
        common = 0
        if held:
            span = min(len(held), limit)
            differs = np.flatnonzero(np.asarray(held[:span]) != np.asarray(tokens[:span]))
            common = int(differs[0]) if differs.size else span
        best = None
        for kind, checkpoint in self._checkpoints.items():
            if (checkpoint.position <= common and (best is None or checkpoint.position > best[1].position)
                    and np.array_equal(checkpoint.tokens, tokens[: checkpoint.position])):
                best = (kind, checkpoint)
        if best is None:
            self.reset()
            return 0, "none"
        kind, checkpoint = best
        if self._gpu_recurrence:
            self.weights.gpu.load_checkpoint(checkpoint.slot)
        for target, saved in zip(self._host_state(), checkpoint.host, strict=True):
            np.copyto(target, saved)
        self._final_hidden = None if checkpoint.final_hidden is None else checkpoint.final_hidden.copy()
        self.history[:] = checkpoint.tokens.tolist()
        return checkpoint.position, kind

    def _checkpoint_plan(self, tokens: list[int], reused: int, *, turn: bool, keep: bool) -> dict[int, list[str]]:
        """Positions (multiples of four) this prefill should checkpoint at, with their kinds.

        A request no later one continues (``keep`` false) saves only the system-prompt root.
        """
        plan: dict[int, list[str]] = {}
        if not self._checkpoints.get("root") or not np.array_equal(
            self._checkpoints["root"].tokens, tokens[: self._checkpoints["root"].position]
        ):
            if _IM_END in tokens:
                root = (tokens.index(_IM_END) + 1) // 4 * 4
                if root >= _ROOT_MINIMUM and reused < root < len(tokens):
                    plan.setdefault(root, []).append("root")
        if not keep:
            return plan
        # The reply starts at the prompt's last <|im_start|> (the generation prompt).
        for index in range(len(tokens) - 1, max(-1, len(tokens) - 64), -1):
            if tokens[index] == _IM_START:
                start = index // 4 * 4
                if reused < start:
                    plan.setdefault(start, []).append("turn" if turn else "step")
                break
        return plan

    def _timed(self, name: str, function: Callable, *args, **kwargs):
        stats = getattr(self, "stats", None)
        if stats is None:
            return function(*args, **kwargs)
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            stats[name] = stats.get(name, 0) + perf_counter() - started

    def _project_rows(self, names: tuple[str, ...], values: np.ndarray) -> list[np.ndarray]:
        outputs = self.weights.project_many([(name, row, None) for name in names for row in values])
        count = len(values)
        return [np.stack(outputs[i * count : (i + 1) * count]) for i in range(len(names))]

    def _hc(self, prefix: str, value: np.ndarray, inject: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        weights = self.weights
        single = value.ndim == 1
        values = value.reshape(-1, 10240)
        if hasattr(getattr(weights, "gpu", None), "hyperconnection") and weights.gpu.batch_capacity == 256:
            mixed, injection = weights.hyperconnection(prefix, values, inject=inject)
            return (mixed[0], injection[0] if inject else None) if single else (mixed, injection)
        normalized = rms_norm(values, weights.vector(prefix + "_norm.weight"), 2560)
        names = (prefix + "_down.weight", prefix + "_inject.weight") if inject else (prefix + "_down.weight",)
        projections = self._project_rows(names, normalized)
        low = silu(projections[0] / np.float32(4))
        up = self._project_rows((prefix + "_up.weight",), low)[0]
        gate = sigmoid(up).reshape(-1, 4, 2560)
        mixed = np.mean(gate * normalized.reshape(-1, 4, 2560), axis=1)
        injection = 2 * sigmoid(projections[1] / 4) if inject else None
        return (mixed[0], injection[0] if inject else None) if single else (mixed, injection)

    def _ple_embeddings(self, tokens: list[int]):
        weights = self.weights
        history = self.history[-2:].copy()
        if hasattr(weights, "rows"):
            indices = ngram_rows_many(history, tokens, weights.integer_arrays)
            return weights.rows("per_layer_token_embd.weight", indices.reshape(-1)).reshape(-1, 2560)
        embeddings = []
        for token in tokens:
            rows = ngram_rows(history, token, weights.integer_arrays)
            embeddings.append(np.concatenate([weights.row("per_layer_token_embd.weight", row) for row in rows]))
            history.append(token)
        return embeddings

    def _ple(self, values: np.ndarray, tokens: list[int]) -> np.ndarray:
        weights, prefix = self.weights, "blk.1."
        embeddings = self._ple_embeddings(tokens)
        if hasattr(getattr(weights, "gpu", None), "ple") and weights.gpu.output_capacity >= 256 * 33280:
            return weights.ple(values, np.asarray(embeddings))
        key_projection, value_projection = self._project_rows(
            (prefix + "ple_key.weight", prefix + "ple_value.weight"), np.stack(embeddings)
        )
        key = rms_norm(key_projection, weights.vector(prefix + "ple_norm_key.weight"), 2560).reshape(-1, 4, 2560)
        query = rms_norm(values, weights.vector(prefix + "ple_norm_query.weight"), 2560).reshape(-1, 4, 2560)
        gate = np.sum(key * query, axis=2) / np.sqrt(np.float32(2560))
        gate = np.sign(gate) * np.sqrt(np.maximum(np.abs(gate), np.float32(1e-6)))
        gated = (sigmoid(gate)[:, :, None] * value_projection[:, None, :]).reshape(-1, 10240)
        normalized = rms_norm(gated, weights.vector(prefix + "ple_norm_conv.weight"), 2560)
        conv = causal_conv(normalized, self._ple_history, weights.vector(prefix + "ple_conv1d.weight"), dilation=3)
        return values + gated + silu(conv)

    def _gdn(self, layer: int, values: np.ndarray) -> np.ndarray:
        weights, prefix = self.weights, f"blk.{layer}."
        if hasattr(getattr(weights, "gpu", None), "gated_delta") and weights.gpu.batch_capacity == 256:
            return weights.gated_delta(prefix, values)
        conv, recurrent = self._linear[layer]
        projections = self._project_rows(
            tuple(
                prefix + name for name in ("attn_qkv.weight", "ssm_beta.weight", "ssm_alpha.weight", "attn_gate.weight")
            ),
            values,
        )
        qkv = silu(causal_conv(projections[0], conv, weights.vector(prefix + "ssm_conv1d.weight")))
        queries = qkv[:, :2048].reshape(-1, 16, 128)
        keys = qkv[:, 2048:4096].reshape(-1, 16, 128)
        incoming_values = qkv[:, 4096:].reshape(-1, 48, 128)
        beta = sigmoid(projections[1])
        alpha = projections[2] + weights.vector(prefix + "ssm_dt.bias")
        decay = weights.vector(prefix + "ssm_a") * np.logaddexp(np.float32(0), alpha)
        if self._gpu_recurrence:
            outputs = weights.gpu.delta_sequence(prefix, queries, keys, incoming_values, decay, beta)
        else:
            outputs = np.stack(
                [
                    delta_step(q, k, v, d, b, recurrent)
                    for q, k, v, d, b in zip(queries, keys, incoming_values, decay, beta, strict=True)
                ]
            )
        outputs = rms_norm(outputs, weights.vector(prefix + "ssm_norm.weight"))
        outputs *= sigmoid(projections[3].reshape(-1, 48, 128))
        return self._project_rows((prefix + "ssm_out.weight",), outputs.reshape(-1, 6144))[0]

    def _qsa(self, layer: int, values: np.ndarray) -> np.ndarray:
        prefix = f"blk.{layer}."
        if self._gpu_attention and hasattr(self.weights.gpu, "sparse_attention"):
            return self.weights.sparse_attention(prefix, values, position=len(self.history), context=self.context)
        projections = self._project_rows(
            tuple(
                prefix + name
                for name in (
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "indexer.q_proj.weight",
                    "indexer.k_proj.weight",
                )
            ),
            values,
        )
        # Normalize and rotate each token independently in a single bounded NumPy
        # operation. Indexer pooling/selection below still advances in causal order.
        positions = np.arange(len(self.history), len(self.history) + len(values))[:, None]
        qgate = projections[0].reshape(-1, 24, 512)
        qgate[:, :, :256] = rotary(
            rms_norm(qgate[:, :, :256], self.weights.vector(prefix + "attn_q_norm.weight")), positions
        )
        projections[0] = qgate
        projections[1] = rotary(
            rms_norm(projections[1].reshape(-1, 2, 256), self.weights.vector(prefix + "attn_k_norm.weight")), positions
        )
        projections[3] = rotary(
            rms_norm(projections[3].reshape(-1, 4, 128), self.weights.vector(prefix + "indexer.q_norm.weight")),
            positions,
        )
        rows = [
            self._qsa_state(layer, len(self.history) + i, *row, prepared=True)
            for i, row in enumerate(zip(*projections, strict=True))
        ]
        if self._gpu_attention:
            queries, keys, incoming, indices, gates = zip(*rows, strict=True)
            outputs = self.weights.gpu.attention_sequence(
                prefix,
                np.stack(queries),
                np.stack(keys),
                np.stack(incoming),
                list(indices),
                position=len(self.history),
                context=self.context,
            )
            outputs *= sigmoid(np.stack(gates))
            outputs = outputs.reshape(-1, 6144)
        else:
            outputs = np.stack(rows)
        return self._project_rows((prefix + "attn_output.weight",), outputs)[0]

    def _qsa_state(
        self, layer, position, q_projection, k_projection, v_projection, iq_projection, ik_projection, *, prepared=False
    ):
        weights, prefix = self.weights, f"blk.{layer}."
        keys, values, block_keys, raw_keys = self._attention[layer]
        qgate = q_projection.reshape(24, 512)
        query = qgate[:, :256] if prepared else rotary(
            rms_norm(qgate[:, :256], weights.vector(prefix + "attn_q_norm.weight")), position
        )
        incoming_key = k_projection.reshape(2, 256) if prepared else rotary(
            rms_norm(
                k_projection.reshape(2, 256),
                weights.vector(prefix + "attn_k_norm.weight"),
            ),
            position,
        )
        incoming_value = v_projection.reshape(2, 256)
        index_query = iq_projection.reshape(4, 128) if prepared else rotary(
            rms_norm(
                iq_projection.reshape(4, 128),
                weights.vector(prefix + "indexer.q_norm.weight"),
            ),
            position,
        )
        raw_keys[position % 4] = ik_projection
        complete = (position + 1) // 4
        if (position + 1) % 4 == 0:
            pooled = rms_norm(np.mean(raw_keys, axis=0), weights.vector(prefix + "indexer.k_norm.weight"))
            block_keys[complete - 1] = rotary(pooled, position - 3)
        if complete > 512:
            scores = np.maximum(block_keys[:complete] @ index_query.T, 0).sum(axis=1)
            blocks = np.argpartition(scores, -512)[-512:]
        else:
            blocks = np.arange(complete)
        selected = np.concatenate(
            ((blocks[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(complete * 4, position + 1))
        )
        if self._gpu_attention:
            return query, incoming_key, incoming_value, selected, qgate[:, 256:]
        keys[position], values[position] = incoming_key, incoming_value
        output = np.empty((24, 256), np.float32)
        for head in range(24):
            kv_head = head // 12
            scores = keys[selected, kv_head] @ query[head] / np.float32(16)
            output[head] = softmax(scores) @ values[selected, kv_head]
        output *= sigmoid(qgate[:, 256:])
        return output.reshape(-1)

    def _moe(self, layer: int, values: np.ndarray) -> np.ndarray:
        weights, prefix = self.weights, f"blk.{layer}."
        if self._resident_moe:
            return weights.resident_moe(prefix, values)
        routers, shared_mix = self._project_rows(
            (prefix + "ffn_gate_inp.weight", prefix + "ffn_gate_inp_shexp.weight"), values
        )
        selected = np.argpartition(routers, -10, axis=1)[:, -10:]
        probabilities = softmax(np.take_along_axis(routers, selected, axis=1), axis=1)
        if hasattr(getattr(weights, "gpu", None), "moe") and weights.gpu.batch_capacity == 256:
            return weights.moe(prefix, values, selected, probabilities, sigmoid(shared_mix))
        requests = [
            (prefix + name, value, int(index))
            for value, indices in zip(values, selected, strict=True)
            for index in indices
            for name in ("ffn_gate_exps.weight", "ffn_up_exps.weight")
        ]
        inputs = weights.project_many(requests)
        products = silu(np.stack(inputs[::2])) * np.stack(inputs[1::2])
        outputs = weights.project_many(
            [
                (prefix + "ffn_down_exps.weight", product, int(index))
                for index, product in zip(selected.reshape(-1), products, strict=True)
            ]
        )
        shared_gate, shared_up = self._project_rows(
            (prefix + "ffn_gate_shexp.weight", prefix + "ffn_up_shexp.weight"), values
        )
        shared = self._project_rows((prefix + "ffn_down_shexp.weight",), silu(shared_gate) * shared_up)[0]
        expert_outputs = np.stack(outputs).reshape(len(values), 10, 2560)
        result = np.zeros_like(values)
        for expert in range(10):
            result += probabilities[:, expert, None] * expert_outputs[:, expert]
        return result + sigmoid(shared_mix) * shared

    def _forward_tokens(self, tokens: list[int], *, logits: bool, cancelled=lambda: False) -> np.ndarray | None:
        if len(self.history) + len(tokens) > self.context or any(not 0 <= token < 248320 for token in tokens):
            raise ValueError("Straydog token or context is out of range")
        if hasattr(self.weights, "rows"):
            embeddings = self._timed("embedding", self.weights.rows, "token_embd.weight", np.asarray(tokens))
            hidden = np.tile(embeddings, (1, 4))
        else:
            hidden = np.stack(
                [np.tile(self._timed("embedding", self.weights.row, "token_embd.weight", token), 4) for token in tokens]
            )
        previous = getattr(self, "_final_hidden", None)
        if self._decoder_ready and len(tokens) > 1 and hasattr(self.weights, "prefill_layers"):
            hidden = self._resident_prefill(hidden, tokens, cancelled)
            if hidden is None:
                return None
            if getattr(self, "speculative", 0):
                self._timed("mtp", self._mtp_prompt, tokens, hidden, previous)
        elif self._decoder_ready:
            position = len(self.history)
            hidden = self._timed("decoder", self.weights.decode_span, 0, 1, hidden, position=position)
            if cancelled():
                self.reset()
                return None
            hidden = self._timed("ple", self._ple, hidden, tokens)
            for begin, end in self.weights.gpu.decoder_spans():
                if cancelled():
                    self.reset()
                    return None
                chunks = range(begin, end, 4) if len(tokens) > 1 else (begin,)
                for first in chunks:
                    if cancelled():
                        self.reset()
                        return None
                    last = min(first + 4, end) if len(tokens) > 1 else end
                    hidden = self._timed("decoder", self.weights.decode_span, first, last, hidden, position=position)
        else:
            for layer in range(48):
                if cancelled():
                    self.reset()
                    return None
                if layer == 1:
                    hidden = self._timed("ple", self._ple, hidden, tokens)
                prefix = f"blk.{layer}."
                mixed, injection = self._timed("hc", self._hc, prefix + "hc_attn", hidden)
                output = (
                    self._timed("qsa", self._qsa, layer, mixed)
                    if layer % 4 == 3
                    else self._timed("gdn", self._gdn, layer, mixed)
                )
                hidden += (injection[:, :, None] * output[:, None, :]).reshape(-1, 10240)
                mixed, injection = self._timed("hc", self._hc, prefix + "hc_ffn", hidden)
                hidden += (injection[:, :, None] * self._timed("moe", self._moe, layer, mixed)[:, None, :]).reshape(
                    -1, 10240
                )
                if not np.isfinite(hidden).all():
                    raise RuntimeError(f"Straydog non-finite state in layer {layer}")
        self.history.extend(tokens)
        self._final_hidden = np.array(hidden[-1], np.float32)
        if not logits:
            return None
        mixed, _ = self._hc("output_hc", hidden[-1], inject=False)
        return self._timed("head", self.weights.project, "output.weight", mixed)

    # ---- speculative decoding (MTP drafts, exact verification) ----

    def _mtp_layer(self, rows: np.ndarray, position: int) -> np.ndarray:
        """Advance the MTP layer's K/V over prologue rows starting at ``position``."""
        weights = self.weights
        if len(rows) == 1 or len(rows) <= 4 and not hasattr(weights, "prefill_layers"):
            return weights.verify_span(48, 49, rows, position=position)
        return weights.prefill_layers(48, 49, rows, position=position, steps=len(rows), output_rows=0)

    def _mtp_prompt(self, tokens: list[int], rows: np.ndarray, previous: np.ndarray | None) -> None:
        """Fill the MTP layer's K/V for a prompt batch: pair (hidden_t, token_t+1) sits at position t."""
        start = len(self.history)
        hidden, following, first = list(rows[:-1]), list(tokens[1:]), start
        if start > 0 and previous is not None:
            hidden.insert(0, previous)
            following.insert(0, tokens[0])
            first = start - 1
        for offset in range(0, len(hidden), 512):
            chunk = np.stack(hidden[offset:offset + 512])
            embedded = self.weights.rows("token_embd.weight", np.asarray(following[offset:offset + 512]))
            self._mtp_layer(self.weights.mtp_prologue(embedded, chunk), first + offset)

    def _draft(self, hidden: list[np.ndarray], tokens: list[int], position: int, limit: int) -> list[int]:
        """Catch the MTP layer up on the accepted pairs, then chain ``limit`` greedy drafts."""
        weights = self.weights
        rows = weights.mtp_prologue(weights.rows("token_embd.weight", np.asarray(tokens)), np.stack(hidden))
        output = weights.verify_span(48, 49, rows, position=position)
        if limit == 0:
            return []
        drafts = [int(weights.mtp_head(output[-1:])[0])]
        multi, position = output[-1], position + len(tokens)
        while len(drafts) < limit:
            rows = weights.mtp_prologue(weights.rows("token_embd.weight", np.asarray(drafts[-1:])), multi[None])
            output = weights.verify_span(48, 49, rows, position=position)
            drafts.append(int(weights.mtp_head(output)[0]))
            multi, position = output[0], position + 1
        return drafts

    def _verify(self, tokens: list[int], sampling: bool):
        """Run the next token and its drafts through every layer with the one-token arithmetic."""
        weights, position = self.weights, len(self.history)
        hidden = np.tile(weights.rows("token_embd.weight", np.asarray(tokens)), (1, 4))
        first = weights.verify_span(0, 1, hidden, position=position)
        embeddings = np.asarray(self._ple_embeddings(tokens), np.float32)
        weights.gpu.copy_state("@ple-snapshot:blk.1.", "@ple:blk.1.", owner="blk.1.")
        hidden = weights.ple_rows(first, embeddings)
        for begin, end in weights.gpu.decoder_spans():
            hidden = weights.verify_span(begin, end, hidden, position=position)
        if sampling:
            targets, scores = weights.output_head(hidden, logits=True)
        else:
            targets, scores = weights.output_head(hidden), None
        return first, embeddings, hidden, targets, scores

    def _commit(self, tokens: list[int], kept: int, first: np.ndarray, embeddings: np.ndarray) -> None:
        """Keep the first ``kept`` verified tokens; replay DeltaNet and PLE state for the rest."""
        weights = self.weights
        for begin, end in [(0, 1), *weights.gpu.decoder_spans()]:
            weights.verify_commit(begin, end, kept)
        if kept < len(tokens):
            weights.gpu.copy_state("@ple:blk.1.", "@ple-snapshot:blk.1.", owner="blk.1.")
            weights.ple_rows(first[:kept], embeddings[:kept])
        self.history.extend(tokens[:kept])

    @staticmethod
    def _accept(drafts, targets, scores, temperature: float, rng) -> tuple[int, int]:
        """Accepted draft count and the next token (greedy match, or exact speculative sampling)."""
        if temperature == 0:
            accepted = 0
            while accepted < len(drafts) and drafts[accepted] == int(targets[accepted]):
                accepted += 1
            return accepted, int(targets[accepted])
        for index, draft in enumerate(drafts):
            probabilities = softmax(scores[index] / temperature)
            if rng.random() < probabilities[draft]:
                continue
            # A deterministic draft is a point proposal: resample from the target without it.
            residual = probabilities.copy()
            residual[draft] = 0
            return index, int(rng.choice(residual.size, p=residual / residual.sum()))
        probabilities = softmax(scores[len(drafts)] / temperature)
        return len(drafts), int(rng.choice(probabilities.size, p=probabilities))

    def _generate_speculative(self, logits, max_new, temperature, rng, cancelled, started) -> Iterator[int]:
        if temperature == 0:
            token = int(np.argmax(logits))
        else:
            token = int(rng.choice(logits.size, p=softmax(logits / temperature)))
        if token in _END_TOKENS:
            return
        metrics = self.last_metrics
        metrics.update(ttft_seconds=perf_counter() - started, generated_tokens=1,
                       speculative_passes=0, drafted_tokens=0, accepted_tokens=0)
        yield token
        produced, acceptance, passes = 1, 0.7, 0
        # The MTP layer pairs each hidden state with the token after it, at the hidden's position.
        mtp_hidden, mtp_tokens, mtp_position = [self._final_hidden], [token], len(self.history) - 1
        while produced < max_new:
            if cancelled():
                return
            pass_started = perf_counter()
            limit = min(self.speculative, max_new - produced - 1, self.context - len(self.history) - 1)
            drafting = draft_length(acceptance, limit)
            if not drafting and limit and passes % 8 == 7:
                drafting = 1   # probe: the acceptance estimate must keep tracking the text
            passes += 1
            drafts = self._timed("draft", self._draft, mtp_hidden, mtp_tokens, mtp_position, drafting)
            rows = [token, *drafts]
            if len(self.history) + len(rows) > self.context:
                raise ValueError("Straydog prompt plus output exceeds the allocated context")
            position = len(self.history)
            first, embeddings, hidden, targets, scores = self._timed("verify", self._verify, rows, temperature > 0)
            accepted, following = self._accept(drafts, targets, scores, temperature, rng)
            if drafts:
                attempted = accepted + (accepted < len(drafts))
                acceptance = 0.9 * acceptance + 0.1 * accepted / attempted
            emitted = [*drafts[:accepted], following]
            kept = accepted + 1
            for index, value in enumerate(emitted):
                if value in _END_TOKENS:
                    kept = min(kept, index + 1)
                    emitted = emitted[:index]
                    break
            self._commit(rows, kept, first, embeddings)
            self._final_hidden = np.array(hidden[kept - 1], np.float32)
            metrics["speculative_passes"] += 1
            metrics["drafted_tokens"] += len(drafts)
            metrics["accepted_tokens"] += accepted
            metrics["decode_seconds"] += perf_counter() - pass_started
            for value in emitted:
                produced += 1
                metrics["generated_tokens"] += 1
                metrics["decode_forward_tokens"] += 1
                metrics["decode_tokens_per_second"] = metrics["decode_forward_tokens"] / metrics["decode_seconds"]
                yield value
            if len(emitted) < accepted + 1:
                return
            token = following
            mtp_hidden, mtp_tokens, mtp_position = list(hidden[:accepted + 1]), emitted, position

    def _resident_prefill(self, hidden: np.ndarray, tokens: list[int], cancelled) -> np.ndarray | None:
        """Keep prompt rows on each GPU between spans; return only the last row.

        Control returns to the CPU after at most four layers for cancellation. Rows
        cross the host only at device boundaries; PLE updates the first GPU in place.
        """
        weights, steps, position = self.weights, len(tokens), len(self.history)
        boundaries = weights.gpu.decoder_spans()
        spans = [(0, 1)] + [(first, min(first + 4, end)) for begin, end in boundaries for first in range(begin, end, 4)]
        transfers = {end for _begin, end in boundaries}
        pending, output = hidden, None
        for index, (first, last) in enumerate(spans):
            if cancelled():
                self.reset()
                return None
            if index == 1:
                embeddings = self._timed("ple", self._ple_embeddings, tokens)
                self._timed("ple", weights.prefill_ple, embeddings)
            # Speculation keeps every final row: the MTP layer reads the prompt's hidden states.
            final = steps if getattr(self, "speculative", 0) else 1
            rows = final if last == 48 else steps if last in transfers else 0
            output = self._timed(
                "decoder", weights.prefill_layers, first, last, pending,
                position=position, steps=steps, output_rows=rows,
            )
            pending = output if rows == steps else None
        return output

    def step(self, token: int, *, logits: bool = True) -> np.ndarray | None:
        """Consume one token using the same graph and arithmetic as batched prefill."""
        return self._forward_tokens([token], logits=logits)

    def generate(
        self,
        tokens: list[int],
        *,
        max_new: int,
        temperature: float = 0,
        seed: int = 0,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int], None] = lambda _done, _total: None,
        turn: bool = True,
        checkpoint: bool = True,
    ) -> Iterator[int]:
        """Generate without truncation or hidden retry; resume from the longest prefix already held.

        ``turn`` says the prompt ends with a user message rather than a tool result; ``checkpoint``
        false marks a one-off request whose reply no later request continues.
        """
        started = perf_counter()
        self.last_metrics = {}
        if not tokens or not 1 <= max_new <= self.context or len(tokens) + max_new > self.context:
            raise ValueError("Straydog prompt plus output exceeds the allocated context")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("invalid Straydog temperature")
        self.weights.check_unchanged()
        reused, resumed = self._resume(tokens)
        self.last_metrics = {
            "cached_tokens": reused,
            "resumed_from": resumed,
            "prefill_tokens": len(tokens) - reused,
            "decode_forward_tokens": 0,
            "decode_seconds": 0.0,
            "decode_tokens_per_second": None,
            "generated_tokens": 0,
            "ttft_seconds": None,
        }
        output = None
        prefill_started = perf_counter()
        batch_size = getattr(self, "prefill_batch_size", 1)
        plan = self._checkpoint_plan(tokens, reused, turn=turn, keep=checkpoint)
        position = reused
        for stop in [*sorted(plan), len(tokens)]:
            while position < stop:
                if cancelled():
                    return
                end = min(position + batch_size, stop)
                if batch_size == 1:
                    output = self.step(tokens[position], logits=end == len(tokens))
                else:
                    output = self._forward_tokens(tokens[position:end], logits=end == len(tokens), cancelled=cancelled)
                if cancelled():
                    return
                position = end
                progress(end, len(tokens))
            for kind in plan.get(stop, ()):
                self._save_checkpoint(kind)
        elapsed = perf_counter() - prefill_started
        self.last_metrics.update(prefill_seconds=elapsed, prefill_tokens_per_second=(len(tokens) - reused) / elapsed)
        if getattr(self, "_hybrid", False):
            # Re-pick cached experts from this prompt's routing before decoding.
            started_rebalance = perf_counter()
            self.weights.gpu.rebalance()
            self.last_metrics["rebalance_seconds"] = perf_counter() - started_rebalance
        rng = np.random.default_rng(seed)
        if getattr(self, "speculative", 0) and output is not None:
            yield from self._generate_speculative(output, max_new, temperature, rng, cancelled, started)
            return
        for index in range(max_new):
            if cancelled():
                return
            token = (
                int(np.argmax(output))
                if temperature == 0
                else int(rng.choice(output.size, p=softmax(output / temperature)))
            )
            if token in {248044, 248046}:
                return
            if self.last_metrics["ttft_seconds"] is None:
                self.last_metrics["ttft_seconds"] = perf_counter() - started
            self.last_metrics["generated_tokens"] += 1
            yield token
            if index + 1 < max_new:
                decode_started = perf_counter()
                output = self.step(token)
                self.last_metrics["decode_seconds"] += perf_counter() - decode_started
                self.last_metrics["decode_forward_tokens"] += 1
                self.last_metrics["decode_tokens_per_second"] = (
                    self.last_metrics["decode_forward_tokens"] / self.last_metrics["decode_seconds"]
                )
