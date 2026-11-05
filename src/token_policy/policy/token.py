"""External token policies for SONIC's ``token_hook(z_base, state) -> z``.

    ResidualToken(source, scale):  z = z_base + scale * source(state)   (safe exploration, current stage)
    ReplaceToken(source):          z = source(state)                    (full 64-D token generation, later)

``source(state) -> (64,)`` is anything: zero, a fixed vector, a random draw, a learned policy, a VLA.
"""
from __future__ import annotations

import numpy as np


class ZeroSource:
    def __init__(self, dim: int = 64):
        self.value = np.zeros(dim)

    def __call__(self, state: dict) -> np.ndarray:
        return self.value


class FixedSource:
    """A constant vector for the whole run (e.g. +eps * e_i for a sensitivity sweep)."""

    def __init__(self, value):
        self.value = np.asarray(value, float).copy()

    def __call__(self, state: dict) -> np.ndarray:
        return self.value


class RandomSource(FixedSource):
    """One N(0, I) draw per run, held constant. Pass ``seed`` or an existing ``rng``."""

    def __init__(self, seed: int | None = None, rng: np.random.Generator | None = None, dim: int = 64):
        rng = rng if rng is not None else np.random.default_rng(seed)
        super().__init__(rng.standard_normal(dim))


class ResidualToken:
    def __init__(self, source, scale: float):
        self.source = source
        self.scale = float(scale)

    def reset(self):
        if hasattr(self.source, "reset"):
            self.source.reset()

    def __call__(self, z_base, state):
        return z_base + self.scale * self.source(state)


class ReplaceToken:
    needs_base = False          # the encoder output is not used -> SonicONNX may skip the encoder

    def __init__(self, source):
        self.source = source

    def reset(self):
        if hasattr(self.source, "reset"):
            self.source.reset()

    def __call__(self, z_base, state):
        return self.source(state)


class GridOffsetToken:
    needs_base = False

    """Teacher action on the FSQ grid: ``k = clip(k_base + delta_k, K_MIN, K_MAX)``, ``z = k / 16``.

    ``k_base`` is a fixed integer token (the quantized standing token saved with the snapshot);
    ``delta_source(state) -> int array (64,)`` supplies the offset each tick (bounded, not accumulated).
    """

    def __init__(self, k_base, delta_source):
        from .fsq import K_MAX, K_MIN, k_to_z
        self._k_min, self._k_max, self._k_to_z = K_MIN, K_MAX, k_to_z
        self.k_base = np.asarray(k_base, np.int64).copy()
        self.delta_source = delta_source
        self.k_last = self.k_base.copy()

    def reset(self):
        self.k_last = self.k_base.copy()
        if hasattr(self.delta_source, "reset"):
            self.delta_source.reset()

    def __call__(self, z_base, state):
        dk = np.asarray(self.delta_source(state), np.int64)
        self.k_last = np.clip(self.k_base + dk, self._k_min, self._k_max)
        return self._k_to_z(self.k_last)
