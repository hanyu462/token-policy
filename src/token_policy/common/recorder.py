"""Per-tick recorder shared by the sim runners and whatever controls the robot.

One object does both jobs:
  - storage: ``append(state, **extras)`` keeps selected state keys plus any extra values per tick;
    ``arrays()`` / ``save(path)`` / ``Recorder.load(path)``
  - console: if ``console`` is given, a line ``console(state, extras) -> str`` is printed whenever
    sim time crosses a ``period`` grid (no drift, no per-tick spam)

The recorder does not know what the extras mean; a controller can contribute its own through
``controller.log_extras() -> dict`` (see ``controller_extras``).
"""
from __future__ import annotations

import sys

import numpy as np

DEFAULT_KEYS = ["time", "base_pos", "base_rpy", "base_lin_vel", "base_ang_vel", "q", "dq", "tau",
                "lfoot", "rfoot", "lfoot_force", "rfoot_force"]


class Recorder:
    def __init__(self, keys=None, console=None, period: float = 0.5, stream=None):
        self.keys = list(keys) if keys is not None else list(DEFAULT_KEYS)
        self.console = console
        self.period = period
        self.stream = stream or sys.stdout
        self._rows = {k: [] for k in self.keys}
        self._extra = {}
        self._grid = -1

    # ---- storage ----
    def append(self, state: dict, **extras):
        n = len(self)
        for k in self.keys:
            self._rows[k].append(state[k])
        # extras stay aligned with the state rows: a key that appears late is back-filled with NaN,
        # a key missing this tick gets NaN (controllers attach/detach mid-run)
        for k, v in extras.items():
            v = np.asarray(v, dtype=float)
            if k not in self._extra:
                self._extra[k] = [np.full(v.shape, np.nan)] * n
            self._extra[k].append(v)
        for k, rows in self._extra.items():
            if len(rows) == n:
                rows.append(np.full(np.shape(rows[-1]), np.nan))
        if self.console is not None:
            g = int(state["time"] // self.period)
            if g > self._grid:
                self._grid = g
                line = self.console(state, extras)
                if line:
                    self.stream.write(line + "\n")
                    self.stream.flush()

    def __len__(self):
        return len(self._rows[self.keys[0]]) if self.keys else 0

    def clear(self):
        self._rows = {k: [] for k in self.keys}
        self._extra = {}
        self._grid = -1

    def arrays(self) -> dict:
        out = {k: np.asarray(v) for k, v in self._rows.items()}
        out.update({k: np.asarray(v) for k, v in self._extra.items()})
        if "time" in out:
            out["t"] = out["time"]
        return out

    def save(self, path: str, **meta) -> str:
        a = self.arrays(); a.update(meta)
        np.savez_compressed(path, **a)
        return path

    @staticmethod
    def load(path: str) -> dict:
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}


def controller_extras(controller) -> dict:
    """``controller.log_extras() -> dict`` if the controller has it, else {}."""
    f = getattr(controller, "log_extras", None)
    return f() if f is not None else {}
