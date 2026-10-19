"""Independent float32 reference operations for Qwen3.8-Flash-Next.

Equations and GGUF layout contracts are derived from the official Qwen model
and GGML conversion specification. These operations do not import an inference
engine or execute model-supplied code.
"""

from __future__ import annotations

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Stable sigmoid without overflow at extreme finite activations."""
    # One exponential, with a nonpositive argument, instead of exp(logaddexp()).
    magnitude = np.exp(-np.abs(value))
    denominator = np.float32(1) + magnitude
    return np.where(value >= 0, np.float32(1) / denominator, magnitude / denominator)


def silu(value: np.ndarray) -> np.ndarray:
    """SiLU activation in the input's floating-point precision."""
    return value * sigmoid(value)


def causal_conv(values: np.ndarray, history: np.ndarray, coefficients: np.ndarray, *, dilation: int = 1) -> np.ndarray:
    """Four causal taps over a token batch, updating the rolling history in place."""
    if (
        values.ndim != 2
        or not len(values)
        or dilation not in {1, 3}
        or history.shape != (3 * dilation + 1, values.shape[1])
        or coefficients.shape != (values.shape[1], 4)
    ):
        raise ValueError("invalid Straydog convolution geometry")
    previous = np.concatenate((history[1:], values))
    taps = np.ascontiguousarray(coefficients.T)
    result = previous[: len(values)] * taps[0]
    for tap in range(1, 4):
        offset = tap * dilation
        result += previous[offset : offset + len(values)] * taps[tap]
    history[:] = previous[-len(history) :]
    return result


def rms_norm(value: np.ndarray, weight: np.ndarray, group: int | None = None) -> np.ndarray:
    """RMS normalization with GGUF's already offset-adjusted norm weights."""
    shape = value.shape
    values = value.reshape(-1, group or shape[-1])
    normalized = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + np.float32(1e-6))
    return normalized.reshape(shape) * weight


def rotary(value: np.ndarray, position: int | np.ndarray, dimension: int = 64, base: float = 1e7) -> np.ndarray:
    """Text-only partial RoPE, pairing the two halves of the rotated prefix."""
    positions = np.asarray(position, dtype=np.float32)
    if dimension <= 0 or dimension % 2 or dimension > value.shape[-1] or np.any(positions < 0):
        raise ValueError("invalid Straydog rotary geometry")
    half = dimension // 2
    angles = positions[..., None] / np.power(np.float32(base), np.arange(half, dtype=np.float32) / half)
    cosine, sine = np.cos(angles), np.sin(angles)
    result = value.copy()
    first, second = value[..., :half], value[..., half:dimension]
    result[..., :half] = first * cosine - second * sine
    result[..., half:dimension] = second * cosine + first * sine
    return result


def delta_step(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """One gated DeltaNet recurrence; state is updated in place.

    GGUF stores value heads in tiled order, so Q/K heads repeat by tiling,
    rather than the grouped repeat used before GGUF conversion.
    """
    if query.shape != key.shape or value.shape[0] % key.shape[0]:
        raise ValueError("invalid Straydog DeltaNet head geometry")
    repeat = value.shape[0] // key.shape[0]
    query = np.tile(query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6), (repeat, 1))
    key = np.tile(key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6), (repeat, 1))
    query /= np.float32(np.sqrt(query.shape[-1]))
    state *= np.exp(decay)[:, None, None]
    prediction = np.einsum("hkv,hk->hv", state, key)
    correction = (value - prediction) * beta[:, None]
    state += key[:, :, None] * correction[:, None, :]
    return np.einsum("hkv,hk->hv", state, query)


def softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    """Float32 softmax with a shifted exponent."""
    weights = np.exp(value - np.max(value, axis=axis, keepdims=True))
    return weights / np.sum(weights, axis=axis, keepdims=True)


def ngram_rows(history: list[int], token: int, arrays: dict[str, list[int]]) -> list[int]:
    """Exact integer PLE hashes, including EOS segmentation and head offsets."""
    eos = 248044
    previous = history[-1] if history else eos
    earlier = history[-2] if len(history) > 1 and previous != eos else eos
    multipliers = arrays["qwen4exp.ple.layer_multipliers"]
    offsets = arrays["qwen4exp.ple.head_offsets"]
    sizes = arrays["qwen4exp.ple.head_vocab_sizes"]
    if len(multipliers) != 3 or len(offsets) != 16 or len(sizes) != 16 or min(sizes) <= 0:
        raise ValueError("invalid Straydog PLE constants")
    bigram = token * multipliers[0] ^ previous * multipliers[1]
    trigram = bigram ^ earlier * multipliers[2]
    return [(bigram if head < 8 else trigram) % sizes[head] + offsets[head] for head in range(16)]


def ngram_rows_many(history: list[int], tokens: list[int], arrays: dict[str, list[int]]) -> np.ndarray:
    """Compute the exact fixed-model PLE hashes for a bounded token batch."""
    if not tokens or len(tokens) > 4096:
        raise ValueError("invalid Straydog PLE token batch")
    multipliers = arrays["qwen4exp.ple.layer_multipliers"]
    offsets = arrays["qwen4exp.ple.head_offsets"]
    sizes = arrays["qwen4exp.ple.head_vocab_sizes"]
    if len(multipliers) != 3 or len(offsets) != 16 or len(sizes) != 16 or min(sizes) <= 0:
        raise ValueError("invalid Straydog PLE constants")
    limit = max(tokens + history[-2:] + [248044])
    if min(multipliers + tokens + history[-2:]) < 0 or max(multipliers) * limit > 2**64 - 1:
        result, previous = [], history[-2:].copy()
        for token in tokens:
            result.append(ngram_rows(previous, token, arrays))
            previous.append(token)
        return np.asarray(result, dtype=np.int64)
    current = np.asarray(tokens, dtype=np.uint64)
    previous = np.concatenate((np.array([history[-1] if history else 248044], dtype=np.uint64), current[:-1]))
    prefix = [248044] * (2 - len(history[-2:])) + history[-2:]
    earlier = np.concatenate((np.asarray(prefix, dtype=np.uint64), current))
    earlier = earlier[:len(tokens)]
    earlier = np.where(previous == 248044, np.uint64(248044), earlier)
    first = current * np.uint64(multipliers[0]) ^ previous * np.uint64(multipliers[1])
    second = first ^ earlier * np.uint64(multipliers[2])
    hashes = np.concatenate((np.repeat(first[:, None], 8, axis=1), np.repeat(second[:, None], 8, axis=1)), axis=1)
    return (hashes % np.asarray(sizes, np.uint64) + np.asarray(offsets, np.uint64)).astype(np.int64)
