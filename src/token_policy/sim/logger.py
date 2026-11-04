"""Per-tick state log shared by SyncSim.run() and AsyncSim (appended from the sim thread)."""
from __future__ import annotations

import numpy as np

DEFAULT_KEYS = ["time", "base_pos", "base_rpy", "base_lin_vel", "base_ang_vel", "q", "dq", "tau",
                "lfoot", "rfoot", "lfoot_force", "rfoot_force"]


class StateLog:
    """Collects ``state()`` snapshots (selected keys) plus any extra scalars per tick."""

    def __init__(self, keys=None):
        self.keys = list(keys) if keys is not None else list(DEFAULT_KEYS)
        self._rows = {k: [] for k in self.keys}
        self._extra = {}

    def append(self, state: dict, **extra):
        for k in self.keys:
            self._rows[k].append(state[k])
        for k, v in extra.items():
            self._extra.setdefault(k, []).append(v)

    def __len__(self):
        return len(self._rows[self.keys[0]]) if self.keys else 0

    def clear(self):
        self._rows = {k: [] for k in self.keys}
        self._extra = {}

    def arrays(self) -> dict:
        out = {k: np.asarray(v) for k, v in self._rows.items()}
        out.update({k: np.asarray(v) for k, v in self._extra.items()})
        if "time" in out:
            out["t"] = out["time"]
        return out
