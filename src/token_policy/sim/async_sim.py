"""Asynchronous simulation: a background thread advances time at wall-clock rate.

Mirrors a real robot: the sim keeps running on its own; controllers are attached (called from
the sim thread each tick) or write commands from any other thread whenever they like. Not
deterministic. For deployment-style checks (latency, hot-swapping controllers, live viewing).

Controller interface: see token_policy.sim.controllers (same as SyncSim).
"""
from __future__ import annotations

import threading
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
class AsyncConfig:
    q_init: np.ndarray | None = None   # reset pose; default standing pose
    gantry: bool = True                # hold the pelvis with the virtual gantry from t = 0
    realtime_factor: float = 1.0       # 1.0 = real time, 2.0 = twice as fast, 0 = as fast as possible


class AsyncSim:
    """Handle to a simulation stepping in its own thread (one control tick per ``control_dt`` of
    wall clock / realtime_factor, physics sub-steps inside).

      - ``attach(controller)``: the sim thread calls ``controller(state)`` every control tick
      - ``set_command(q_target, kp, kd)``: any thread writes the command buffer; the sim applies the
        latest one at its next tick (later writes overwrite earlier ones)
      - ``state()``: latest snapshot (a copy), never touches the physics
      - ``reset()`` / ``set_gantry()``: queued, applied by the sim thread at its next tick
      - ``log`` (StateLog): if given, every tick is recorded from inside the sim thread, so nothing is
        missed regardless of how often the outside reads ``state()``
    """

    def __init__(self, sim: G1Sim, controller=None, cfg: AsyncConfig | None = None, log: StateLog | None = None):
        self.sim = sim
        self.cfg = cfg or AsyncConfig()
        self.log = log                    # if given, every tick is appended from the sim thread
        self.gantry = Gantry()
        self._lock = threading.Lock()
        self._controller = controller
        self._pending_cmd = None
        self._pending_reset = None
        self._pending_gain = None
        self._stop = threading.Event()
        self._viewer = None               # set while run_viewer() is active: physics must hold viewer.lock()
        self._thread = threading.Thread(target=self._loop, name="AsyncSim", daemon=True)
        self.ticks = 0
        self.overruns = 0                 # ticks where the thread fell behind wall clock by > control_dt
        self._do_reset(self.cfg.q_init)   # thread not started yet: safe to touch the sim directly

    # ---- external interface (any thread) ----
    def attach(self, controller):
        with self._lock:
            self._controller = controller

    def detach(self):
        self.attach(None)

    def set_command(self, q_target, kp=None, kd=None):
        with self._lock:
            self._pending_cmd = (np.asarray(q_target, float).copy(),
                                 None if kp is None else np.asarray(kp, float).copy(),
                                 None if kd is None else np.asarray(kd, float).copy())

    def set_gantry(self, gain: float):
        with self._lock:
            self._pending_gain = float(gain)

    def reset(self, q_init=None):
        with self._lock:
            self._pending_reset = (q_init,)

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    @property
    def time(self) -> float:
        return self._state["time"]

    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        self._thread.join(timeout)

    def running(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    # ---- sim thread ----
    def _do_reset(self, q_init):
        q = self.cfg.q_init if q_init is None else q_init
        q = DEFAULT_ANGLES if q is None else np.asarray(q, float)
        self.sim.reset(q)
        self.sim.set_command(q, KP, KD)
        if self.cfg.gantry:
            self.gantry.attach(self.sim, 1.0)
        self._state = self.sim.state()

    def _apply_pending(self):
        if self._pending_reset is not None:
            (q_init,) = self._pending_reset
            self._pending_reset = None
            self._do_reset(q_init)
        if self._pending_gain is not None:
            self.gantry.set_gain(self._pending_gain)
            self._pending_gain = None
        if self._pending_cmd is not None:
            q, kp, kd = self._pending_cmd
            self._pending_cmd = None
            self.sim.set_command(q, kp, kd)

    def _loop(self):
        dt = self.sim.control_dt
        rtf = self.cfg.realtime_factor
        t_wall0 = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            # The passive viewer's GUI thread copies mjData on its own; stepping under viewer.lock()
            # keeps it from reading mid-step (otherwise mj_copyDataVisual crashes).
            vlock = self._viewer.lock() if self._viewer is not None else nullcontext()
            with self._lock, vlock:
                self._apply_pending()
                if self._controller is not None:
                    cmd = self._controller(self._state)
                    if cmd is not None:
                        self.sim.set_command(*cmd)
                self.sim.step_control()
                self._state = self.sim.state()
                self.ticks += 1
                if self.log is not None:
                    self.log.append(self._state, band=self.gantry.gain)
            k += 1
            if rtf > 0:
                lag = t_wall0 + k * dt / rtf - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                elif lag < -dt:
                    self.overruns += 1
            else:
                time.sleep(1e-4)   # brief yield so other threads can take the lock

    # ---- optional viewer on the calling (main) thread ----
    def run_viewer(self, on_tick=None, hz: float = 50.0, duration: float | None = None):
        """Block and render until the window is closed, ``stop()`` is called, or sim time passes ``duration``."""
        # launch_passive touches mjData (mj_forward) on this thread while the sim thread may be
        # stepping it: open the window with the sim paused, and register the viewer lock before
        # stepping resumes.
        with self._lock:
            viewer = open_viewer(self.sim)
            self._viewer = viewer
        try:
            while viewer.is_running() and self.running() and (duration is None or self.time < duration):
                with self._lock:
                    viewer.cam.lookat[:] = self.sim.base_pos()
                    viewer.sync()
                if on_tick is not None:
                    on_tick(self, self.state())
                time.sleep(1.0 / hz)
        finally:
            self._viewer = None
            close_viewer(viewer)
