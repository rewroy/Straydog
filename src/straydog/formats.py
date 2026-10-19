"""GGML block geometry of the formats Straydog decodes, and the IQ grid tables.

The IQ1/IQ2/IQ3 lattice grids are format definitions: they come from the selected
gguf package's reference tables and are handed once per process to the native
library, which uploads them to every GPU it opens and uses them on the CPU.
IQ1 grid values (-1, 0, 1) are stored as value + 1.
"""

from __future__ import annotations

import ctypes as ct
from functools import cache
from typing import Any

import numpy as np

# kind -> (values per block, bytes per block)
BLOCKS: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 30: (1, 2),
    2: (32, 18), 6: (32, 22), 8: (32, 34), 20: (32, 18),
    11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 23: (256, 136),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 21: (256, 110), 22: (256, 82), 29: (256, 56),
    42: (64, 18),
}
# Formats the CPU expert engine evaluates (routed experts of a partially resident model).
CPU_EXPERT_KINDS = frozenset({8, 16, 17, 18, 20, 21, 22, 23, 29, 42})


def matrix_bytes(kind: int, rows: int, cols: int) -> int:
    """Packed size of a rows x cols matrix (rows of whole blocks)."""
    values, size = BLOCKS.get(kind, (0, 0))
    if not values or cols % values:
        raise ValueError("unsupported Straydog weight format")
    return rows * cols // values * size


@cache
def iq_tables() -> tuple[np.ndarray, ...]:
    """IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S (also IQ1_M) grids, one byte per value."""
    from gguf import quants

    tables = []
    for kind, shape, offset in ((quants.IQ2_XXS, (256, 8), 0), (quants.IQ2_XS, (512, 8), 0),
                                (quants.IQ2_S, (1024, 8), 0), (quants.IQ3_XXS, (256, 4), 0),
                                (quants.IQ3_S, (512, 4), 0), (quants.IQ1_S, (2048, 8), 1)):
        kind.init_grid()
        grid = np.asarray(kind.grid, dtype=np.float32).reshape(shape) + offset
        if not ((grid >= 0) & (grid <= 255) & (grid == np.round(grid))).all():
            raise ValueError("unexpected IQ grid values")
        tables.append(np.ascontiguousarray(grid.astype(np.uint8)))
    return tuple(tables)


def load_iq_tables(lib: Any) -> None:
    """Give the native library the IQ grids (before it opens any GPU)."""
    lib.sd_iq_tables.argtypes = [ct.c_void_p] * 6
    lib.sd_iq_tables.restype = ct.c_int
    if lib.sd_iq_tables(*(table.ctypes.data for table in iq_tables())):
        lib.sd_cpu_error.restype = ct.c_char_p
        raise RuntimeError("Straydog IQ tables: " + lib.sd_cpu_error().decode("utf-8", "replace"))
