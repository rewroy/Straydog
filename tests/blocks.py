"""Well-formed random GGML blocks for the Straydog format tests."""
from __future__ import annotations

import numpy as np


def set_scales(raw: np.ndarray, kind: int, scales) -> np.ndarray:
    """Write each block's FP16 scale: bytes 0-1, or for IQ1_M (29) the top nibbles of its four sub-scale words."""
    bits = np.broadcast_to(np.asarray(scales, np.float16).reshape(-1).view(np.uint16), (len(raw),))
    if kind != 29:
        raw[:, 0:2] = np.ascontiguousarray(bits).view(np.uint8).reshape(-1, 2)
        return raw
    words = np.ascontiguousarray(raw[:, 48:56]).view(np.uint16)
    for k in range(4):
        words[:, k] = (words[:, k] & 0x0FFF) | (((bits >> (4 * k)) & 0xF) << 12)
    raw[:, 48:56] = words.view(np.uint8)
    return raw
