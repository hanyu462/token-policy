"""SONIC v1.1 FSQ token grid: 64 scalars, each one of 32 levels k/16 with k in [-16, 15]
(vector-quantize-pytorch FSQ with even levels: codes -L/2 .. L/2-1 divided by L//2)."""
from __future__ import annotations

import numpy as np

LEVELS = 32
HALF = LEVELS // 2        # 16
K_MIN, K_MAX = -HALF, HALF - 1
TOKEN_DIM = 64


def k_to_z(k) -> np.ndarray:
    return np.asarray(k, float) / HALF


def z_to_k(z) -> np.ndarray:
    return np.clip(np.rint(np.asarray(z, float) * HALF), K_MIN, K_MAX).astype(np.int64)


def snap(z) -> np.ndarray:
    """Nearest grid value."""
    return k_to_z(z_to_k(z))


def on_grid(z, atol: float = 1e-6) -> bool:
    z = np.asarray(z, float)
    return bool(np.all(np.abs(z - snap(z)) < atol) and z.min() >= K_MIN / HALF and z.max() <= K_MAX / HALF)
