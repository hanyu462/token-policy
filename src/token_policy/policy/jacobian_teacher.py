"""Privileged teacher: closed-loop proportional controller in SONIC token space.

Every ``hold`` ticks:  u = gain * J^+ e,  delta_k = clip(round(u), -R, R),  e = goal error (fwd, lat) in mm
in the task's local frame; J (2 x 64) = foot displacement per grid step of each token dim, measured once
on a nominal standing state (experiments/measure_jacobian.py).  The token offset is held in between.
"""
from __future__ import annotations

import numpy as np


class JacobianTeacher:
    def __init__(self, J: dict, gain: float = 1.0, hold: int = 25, dk_range: int = 2, max_err_mm: float | None = None):
        """``J``: {effector: (2, 64) array, rows (fwd, lat) in mm per grid step}.
        ``max_err_mm``: cap on |e| fed to the controller (rate limit: a larger goal is approached in steps)."""
        self.Jp = {eff: np.asarray(Jm, float).T @ np.linalg.inv(np.asarray(Jm, float) @ np.asarray(Jm, float).T)
                   for eff, Jm in J.items()}                      # (64, 2) min-norm pseudo-inverse per effector
        self.gain, self.hold, self.dk_range = float(gain), int(hold), int(dk_range)
        self.max_err_mm = max_err_mm
        self.reset()

    @classmethod
    def from_file(cls, path: str, **kw) -> "JacobianTeacher":
        f = np.load(path)
        return cls({eff: f[f"J_{eff}"] for eff in ("R", "L") if f"J_{eff}" in f.files}, **kw)

    def reset(self):
        self.t = 0
        self.dk = np.zeros(64, np.int64)

    def __call__(self, effector: str, error_local: np.ndarray) -> np.ndarray:
        """``error_local``: goal - foot in the local frame (m); returns the held delta_k (64,)."""
        if self.t % self.hold == 0:
            e = np.asarray(error_local, float)[:2] * 1e3
            if self.max_err_mm is not None and np.linalg.norm(e) > self.max_err_mm:
                e = e * (self.max_err_mm / np.linalg.norm(e))
            u = self.gain * self.Jp[effector] @ e
            self.dk = np.clip(np.rint(u), -self.dk_range, self.dk_range).astype(np.int64)
        self.t += 1
        return self.dk
