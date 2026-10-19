from __future__ import annotations

import numpy as np
import pytest

from straydog.numeric import causal_conv, delta_step, ngram_rows, rms_norm, rotary, sigmoid, silu, softmax
from straydog.weights import dequantize


def test_batched_rotary_positions_match_independent_token_calls():
    values = np.random.default_rng(83).normal(size=(9, 4, 128)).astype(np.float32)
    positions = np.arange(2039, 2048)
    expected = np.stack([rotary(value, int(position)) for value, position in zip(values, positions, strict=True)])
    np.testing.assert_array_equal(rotary(values, positions[:, None]), expected)
    with pytest.raises(ValueError, match="geometry"):
        rotary(values, -positions[:, None])


def test_extreme_activations_and_softmax_are_finite() -> None:
    values = np.array([-1e4, 0, 1e4], np.float32)
    np.testing.assert_allclose(sigmoid(values), [0, 0.5, 1])
    np.testing.assert_allclose(silu(values), [0, 0, 1e4])
    np.testing.assert_allclose(softmax(values), [0, 0, 1])


def test_sigmoid_matches_double_precision_across_transition_and_tails() -> None:
    values = np.concatenate((np.linspace(-30, 30, 10001), [-1e4, 1e4])).astype(np.float32)
    with np.errstate(over="raise", invalid="raise"):
        observed = sigmoid(values)
        magnitude = np.exp(-np.abs(values.astype(np.float64)))
        reference = np.where(values >= 0, 1 / (1 + magnitude), magnitude / (1 + magnitude))
    assert observed.dtype == np.float32
    np.testing.assert_allclose(observed, reference, rtol=2e-7, atol=8e-8)


def test_grouped_norm_does_not_mix_residual_streams() -> None:
    values = np.array([3, 4, 30, 40], np.float32)
    result = rms_norm(values, np.ones(4, np.float32), 2)
    np.testing.assert_allclose(result[:2], result[2:], atol=1e-6)
    np.testing.assert_allclose(result[:2], [3 / np.sqrt(12.5), 4 / np.sqrt(12.5)], atol=1e-6)


@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("steps", [1, 2, 17, 256])
def test_batched_causal_conv_matches_scalar_taps_and_continuation(dilation, steps):
    rng = np.random.default_rng(723)
    history = rng.normal(size=(3 * dilation + 1, 19)).astype(np.float32)
    reference = history.copy()
    coefficients = rng.normal(size=(19, 4)).astype(np.float32)
    values = rng.normal(size=(steps + 1, 19)).astype(np.float32)
    expected = []
    for row in values:
        reference[:-1] = reference[1:]
        reference[-1] = row
        expected.append(np.sum(reference[::dilation].T * coefficients, axis=1))
    observed = causal_conv(values[:-1], history, coefficients, dilation=dilation)
    continuation = causal_conv(values[-1:], history, coefficients, dilation=dilation)
    np.testing.assert_array_equal(observed, expected[:-1])
    np.testing.assert_array_equal(continuation[0], expected[-1])
    np.testing.assert_array_equal(history, reference)
    with pytest.raises(ValueError, match="convolution"):
        causal_conv(values, history, coefficients[:, :3], dilation=dilation)


def test_partial_rope_preserves_norm_and_unrotated_tail() -> None:
    values = np.arange(16, dtype=np.float32).reshape(2, 8)
    result = rotary(values, 37, dimension=4)
    np.testing.assert_array_equal(result[:, 4:], values[:, 4:])
    np.testing.assert_allclose(np.sum(result * result, axis=1), np.sum(values * values, axis=1), rtol=1e-6)
    np.testing.assert_array_equal(rotary(values, 0, dimension=4), values)
    with pytest.raises(ValueError):
        rotary(values, 1, dimension=9)


def test_delta_recurrence_updates_and_retains_orthogonal_memory() -> None:
    state = np.zeros((1, 2, 2), np.float32)
    first = delta_step(
        np.array([[1, 0]], np.float32),
        np.array([[1, 0]], np.float32),
        np.array([[2, 4]], np.float32),
        np.zeros(1, np.float32),
        np.array([0.5], np.float32),
        state,
    )
    np.testing.assert_allclose(first, [[1 / np.sqrt(2), 2 / np.sqrt(2)]], atol=3e-6)
    second = delta_step(
        np.array([[0, 1]], np.float32),
        np.array([[0, 1]], np.float32),
        np.array([[6, 8]], np.float32),
        np.array([np.log(0.5)], np.float32),
        np.ones(1, np.float32),
        state,
    )
    np.testing.assert_allclose(second, [[6 / np.sqrt(2), 8 / np.sqrt(2)]], atol=1e-5)
    np.testing.assert_allclose(state[0, 0], [0.5, 1], atol=2e-6)


def test_gguf_delta_heads_are_tiled_not_grouped() -> None:
    keys = np.eye(2, dtype=np.float32)
    state = np.zeros((4, 2, 1), np.float32)
    delta_step(keys, keys, np.ones((4, 1), np.float32), np.zeros(4, np.float32), np.ones(4, np.float32), state)
    assert np.argmax(state[:, :, 0], axis=1).tolist() == [0, 1, 0, 1]


def test_exact_ple_hashes_do_not_round_large_integers_and_reset_at_eos() -> None:
    arrays = {
        "qwen4exp.ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
        "qwen4exp.ple.head_offsets": list(range(0, 16000, 1000)),
        "qwen4exp.ple.head_vocab_sizes": [997] * 16,
    }
    rows = ngram_rows([10, 20], 30, arrays)
    assert len(rows) == 16 and all(i * 1000 <= row < i * 1000 + 997 for i, row in enumerate(rows))
    assert ngram_rows([10, 248044], 30, arrays) == ngram_rows([], 30, arrays)
    assert rows != ngram_rows([11, 20], 30, arrays)
    # The first eight heads are bigrams and ignore the token two positions back.
    assert rows[:8] == ngram_rows([11, 20], 30, arrays)[:8]


def test_q2_block_codes_scale_order_and_truncation() -> None:
    block = np.concatenate((np.array([2], dtype="<f2").view(np.uint8), np.full(16, 0b11100100, np.uint8)))
    decoded = dequantize(block, 42)
    np.testing.assert_array_equal(decoded, np.tile([-2, 0, 2, 4], 16))
    with pytest.raises(ValueError, match="incomplete"):
        dequantize(block[:-1], 42)


def test_float_storage_formats_and_standard_quantization_dependency() -> None:
    values = np.array([1, -2, 0.5], np.float32)
    np.testing.assert_array_equal(dequantize(values.view(np.uint8), 0), values)
    np.testing.assert_array_equal(dequantize(values.astype(np.float16).view(np.uint8), 1), values)
    bf16 = (values.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    np.testing.assert_array_equal(dequantize(bf16, 30), values)
    block = np.concatenate((np.array([0.5], dtype="<f2").view(np.uint8), np.arange(32, dtype=np.int8).view(np.uint8)))
    np.testing.assert_array_equal(dequantize(block, 8), np.arange(32) / 2)


@pytest.mark.parametrize("history", [[], [7], [8, 9], [8, 248044]])
@pytest.mark.parametrize("large", [False, True])
def test_batched_ngram_hashes_match_exact_python_history_and_eos(history, large):
    from straydog.numeric import ngram_rows_many

    arrays = {
        "qwen4exp.ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
        "qwen4exp.ple.head_offsets": [i * 1009 for i in range(16)],
        "qwen4exp.ple.head_vocab_sizes": [1009] * 16,
    }
    if large:
        arrays["qwen4exp.ple.layer_multipliers"][0] = 2**100 + 31
    tokens, previous, expected = [17, 248044, 19, 20, 21], history.copy(), []
    for token in tokens:
        expected.append(ngram_rows(previous, token, arrays))
        previous.append(token)
    np.testing.assert_array_equal(ngram_rows_many(history, tokens, arrays), expected)


def test_batched_embedding_rows_preserve_packed_row_boundaries():
    from straydog.weights import StraydogWeights

    rng = np.random.default_rng(143)
    raw = rng.integers(0, 256, (9, 5, 18), dtype=np.uint8)
    raw[:, :, :2] = np.frombuffer(np.float16(0.02).tobytes(), np.uint8)
    weights = StraydogWeights.__new__(StraydogWeights)
    weights._tensors = {"embedding": (0, {"dimensions": [160, 9], "bytes": raw.nbytes})}
    weights._raw = lambda name: (raw.reshape(-1), 2)
    selected = np.array([8, 3, 8, 0])
    expected = np.stack([dequantize(raw[row].reshape(-1), 2) for row in selected])
    np.testing.assert_array_equal(weights.rows("embedding", selected), expected)
    # Prompt-sized batches are gathered on a thread pool in the same order.
    many = rng.integers(0, 9, 300)
    expected = np.stack([dequantize(raw[row].reshape(-1), 2) for row in many])
    np.testing.assert_array_equal(weights.rows("embedding", many), expected)
    weights._row_pool.shutdown()
    for invalid in (np.array([9]), np.array([-1]), np.array([[1]]), np.array([0.5])):
        with pytest.raises(ValueError, match="row batch"):
            weights.rows("embedding", invalid)
