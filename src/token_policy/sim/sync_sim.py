"""Synchronous simulation: the caller advances time one control tick at a time.

Deterministic and as fast as possible. For training loops (``step``) and batch runs (``run``).

Controller interface: see token_policy.sim.controllers.
"""
from __future__ import annotations

import itertools
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from .g1_params import DEFAULT_ANGLES, KD, KP
from .g1_sim import G1Sim
from .gantry import Gantry
from .logger import StateLog
from .viewer import close_viewer, open_viewer


@dataclass
class SyncConfig:
    q_init: np.ndarray | None = None   # reset pose; default standing pose
    gantry: bool = True                # hold the pelvis with the virtual gantry from t = 0
    t_release: float | None = None     # if set, ramp the gantry off over this many seconds from t = 0
    stop_on_fall: bool = True          # run(): stop at the first fall
    viewer: bool = False               # run(): show a mujoco.viewer window
    realtime: bool = False             # run(): pace to wall clock (with or without the viewer)


class SyncSim:
    def __init__(self, sim: G1Sim, controller=None, cfg: SyncConfig | None = None):
        self.sim = sim
        self.cfg = cfg or SyncConfig()
        self.controller = controller
        self.gantry = Gantry()
        self.tick = 0
        self.reset()

    # ---- same face as AsyncSim ----
    def attach(self, controller):
        self.controller = controller

    def detach(self):
        self.controller = None

    def set_gantry(self, gain: float):
        self.gantry.set_gain(gain)

    def reset(self, q_init=None):
        q = self.cfg.q_init if q_init is None else q_init
        q = DEFAULT_ANGLES if q is None else np.asarray(q, float)
        self.sim.reset(q)
        self.sim.set_command(q, KP, KD)
        if self.cfg.gantry:
            self.gantry.attach(self.sim, 1.0)
        self.tick = 0
        self._state = self.sim.state()
        return self._state

    def state(self) -> dict:
        return self._state

    @property
    def time(self) -> float:
        return self._state["time"]

    # ---- one control tick ----
    def step(self) -> dict:
        """controller(state) -> command -> physics for one control period; returns the new state."""
        if self.cfg.gantry and self.cfg.t_release is not None:
            self.gantry.set_gain(1.0 - self.tick * self.sim.control_dt / max(self.cfg.t_release, 1e-9))
        if self.controller is not None:
            cmd = self.controller(self._state)
            if cmd is not None:
                self.sim.set_command(*cmd)
        self.sim.step_control()
        self.tick += 1
        self._state = self.sim.state()
        return self._state

    # ---- batch run ----
    def run(self, duration: float | None = None, on_tick=None, log: StateLog | None = None, draw=None) -> dict:
        """Reset, then step for ``duration`` seconds (None = until the viewer closes, a fall with
        stop_on_fall, or Ctrl-C). Returns per-tick arrays (StateLog keys + band) and ``fell_at``.

        ``draw(scn, state)``: optional, called each tick with the viewer's user scene (visual only).
        """
        self.reset()
        log = log if log is not None else StateLog()
        fell_at = None
        n = None if duration is None else int(round(duration / self.sim.control_dt))
        viewer = open_viewer(self.sim) if self.cfg.viewer else None
        t_wall0 = time.time()
        try:
            for _ in (range(n) if n is not None else itertools.count()):
                st = self._state
                fallen = self.sim.fallen()
                log.append(st, band=self.gantry.gain)
                if on_tick is not None:
                    on_tick(self, st)
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    if draw is not None:
                        viewer.user_scn.ngeom = 0
                        draw(viewer.user_scn, st)
                    viewer.cam.lookat[:] = st["base_pos"]
                    viewer.sync()
                if self.cfg.realtime:
                    lag = t_wall0 + st["time"] - time.time()
                    if lag > 0:
                        time.sleep(lag)
                if fallen and fell_at is None:
                    fell_at = st["time"]
                    if self.cfg.stop_on_fall:
                        break
                with (viewer.lock() if viewer is not None else nullcontext()):   # GUI thread copies mjData
                    self.step()
        except KeyboardInterrupt:
            pass
        finally:
            if viewer is not None:
                close_viewer(viewer)
        out = log.arrays()
        out["fell_at"] = np.nan if fell_at is None else fell_at
        return out
