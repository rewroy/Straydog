"""Golden values of authenticated weights from the Python reference (design phase R3).

Block decoding of the host-decoded formats on seeded random blocks, and, from a real
model bundle: the tensor table, the integer arrays, every vector the engine uploads,
token-embedding rows and PLE n-gram rows. Rust must reproduce them bit for bit.

    uv run python scripts/golden_weights.py --model /path/to/bundle --quantization IQ3_S
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from straydog.formats import BLOCKS, iq_tables
from straydog.numeric import ngram_rows_many
from straydog.weights import StraydogWeights, dequantize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "crates/straydog-engine/tests/golden/weights.json"
HOST_KINDS = (1, 30, 11, 20, 21, 23)   # F16, BF16, Q3_K, IQ4_NL, IQ3_S, IQ4_XS


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def random_blocks(kind: int, count: int, rng: random.Random) -> bytes:
    """Blocks whose FP16 scales are finite (a random scale can be NaN or infinite)."""
    _values, size = BLOCKS[kind]
    out = bytearray()
    while len(out) < count * size:
        block = bytearray(rng.getrandbits(8) for _ in range(size))
        decoded = dequantize(np.frombuffer(bytes(block), np.uint8), kind)
        if np.isfinite(decoded).all():
            out += block
    return bytes(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quantization", required=True)
    args = parser.parse_args()
    rng = random.Random(42)
    blocks = []
    for kind in HOST_KINDS:
        data = random_blocks(kind, 24, rng)
        decoded = dequantize(np.frombuffer(data, np.uint8), kind)
        blocks.append({"kind": kind, "data": data.hex(), "f32": np.asarray(decoded, "<f4").tobytes().hex()})
    grids = hashlib.sha256(b"".join(table.tobytes() for table in iq_tables())).hexdigest()

    class NoGroup:   # weights balance layers only for a GPU group
        pass

    weights = StraydogWeights(args.model, ROOT / "src/straydog/assets.json", ROOT / ".build/bin/straydog-inspect",
                              NoGroup(), quantization=args.quantization)
    try:
        table = [[name, index, tensor["dimensions"], tensor["quant_type"], tensor["offset"], tensor["bytes"]]
                 for name, (index, tensor) in sorted(weights._tensors.items())]
        vectors = {}
        for name, (_index, tensor) in sorted(weights._tensors.items()):
            if len(tensor["dimensions"]) == 1 or name.endswith("conv1d.weight"):
                vectors[name] = digest(weights.vector(name))
        ids = [0, 1, 2, 13, 1000, 151643, 248044, 248045, 248046, 248319]
        rows = {str(token): digest(weights.rows("token_embd.weight", np.asarray([token]))) for token in ids}
        rows["batch"] = digest(weights.rows("token_embd.weight", np.asarray(ids)))
        prng = np.random.default_rng(5)
        sequences = []
        for history_length, count in ((0, 1), (1, 3), (2, 7), (5, 300), (3, 1)):
            history = [int(value) for value in prng.integers(0, 248320, history_length)]
            tokens = [int(value) for value in prng.integers(0, 248320, count)]
            if count == 3:
                tokens[1] = 248044   # an end token resets the trigram
            indices = ngram_rows_many(history, tokens, weights.integer_arrays)
            embedded = weights.rows("per_layer_token_embd.weight", indices.reshape(-1)).reshape(-1, 2560)
            sequences.append({"history": history, "tokens": tokens, "indices": indices.tolist(),
                              "embeddings": digest(embedded)})
        model = {"quantization": args.quantization, "tensors": table, "integer_arrays": weights.integer_arrays,
                 "vectors": vectors, "token_rows": rows, "ple": sequences}
    finally:
        weights.close()
    OUTPUT.write_text(json.dumps({"iq_grids_sha256": grids, "blocks": blocks, "model": model}, indent=1) + "\n")
    print(f"wrote {OUTPUT} ({len(table)} tensors, {len(vectors)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
