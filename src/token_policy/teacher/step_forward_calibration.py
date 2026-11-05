"""Global d <-> lambda calibration for the STEP_FORWARD token window (grid-snapped tokens, cut 0).

Built from the measured mean realized dx per lambda over the 31 snapshots (experiments/step_forward_sweep.py --snap).
Used only for the teacher's initial guess; the teacher then corrects lambda per snapshot by execution.
"""
from __future__ import annotations

import json

import numpy as np

# measured 2026-11-05: lambda -> mean dx (mm), 31 snapshots, cut 0, snapped tokens
DEFAULT_TABLE = {0.55: 202.0, 0.60: 213.0, 0.65: 238.0, 0.678: 238.0, 0.70: 251.0, 0.75: 286.0,
                 0.802: 313.0, 0.85: 336.0, 0.90: 358.0, 0.926: 374.0, 1.00: 377.0}
SLOPE_M = 0.40        # d(dx)/d(lambda) in the stable range (m per unit lambda), from the unsnapped linear fit
LAMBDA_RANGE = (0.55, 1.0)


class StepForwardCalibration:
    def __init__(self, table: dict | None = None):
        t = DEFAULT_TABLE if table is None else table
        self.lams = np.array(sorted(t), float)
        dx = np.array([t[l] for l in self.lams], float) / 1e3
        self.dx = np.maximum.accumulate(dx) + 1e-9 * np.arange(len(dx))   # snapping plateaus -> strictly increasing

    @classmethod
    def from_sweeps(cls, paths, cut: int = 0) -> "StepForwardCalibration":
        acc = {}
        for p in paths:
            for r in json.load(open(p)):
                if r["cut"] == cut:
                    acc.setdefault(round(r["lam"], 3), []).append(r["dx_mm"])
        return cls({l: float(np.mean(v)) for l, v in acc.items()})

    def lambda_for(self, d: float) -> float:
        """Initial lambda for a commanded forward distance d (m), clipped to the calibrated range."""
        return float(np.clip(np.interp(d, self.dx, self.lams), *LAMBDA_RANGE))

    def d_for(self, lam: float) -> float:
        return float(np.interp(lam, self.lams, self.dx))
