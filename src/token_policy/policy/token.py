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
    def __init__(self, source):
        self.source = source

    def reset(self):
        if hasattr(self.source, "reset"):
            self.source.reset()

    def __call__(self, z_base, state):
        return self.source(state)
