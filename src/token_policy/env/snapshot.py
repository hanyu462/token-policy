"""Standing snapshot: the full physics + controller state after the standard start-up
(gantry 1 s -> SONIC holds the standing reference 3 s while the gantry is released), captured
at the start of the first free tick.  Restoring it is a 0-tick reset that continues bit-identically
to a run that went through the whole timeline.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from ..common.rotations import quat_from_angle_axis
from ..sim import DEFAULT_ANGLES, KD, KP, G1Sim, Gantry, SyncConfig, SyncSim

STATE_SPEC = (mujoco.mjtState.mjSTATE_FULLPHYSICS | mujoco.mjtState.mjSTATE_CTRL
              | mujoco.mjtState.mjSTATE_WARMSTART)


@dataclass
class StandingSnapshot:
    physics: np.ndarray          # mj_getState(STATE_SPEC)
    q_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    hist: list                   # StateHistory entries (deep copies)
    heading: tuple               # (init_base_quat, init_ref_root_rot, delta_heading)
    last_action: np.ndarray
    tick: int
    k_stand: np.ndarray | None = None   # quantized standing token (FSQ integer codes) at capture

    @classmethod
    def capture(cls, sim: G1Sim, sonic, tick: int) -> "StandingSnapshot":
        """``sonic``: SonicJointController whose history/heading are the controller's state."""
        n = mujoco.mj_stateSize(sim.model, STATE_SPEC)
        physics = np.empty(n)
        mujoco.mj_getState(sim.model, sim.data, physics, STATE_SPEC)
        c = sonic.ctrl
        from ..policy.fsq import z_to_k
        k_stand = z_to_k(sonic.last["z_base"]) if sonic.last is not None else None
        return cls(physics, sim.q_target.copy(), sim.kp.copy(), sim.kd.copy(),
                   copy.deepcopy(list(c.hist.buf)),
                   (c.heading.init_base_quat.copy(), c.heading.init_ref_root_rot.copy(), c.heading.delta_heading),
                   c.last_action.copy(), tick, k_stand)

    def restore(self, sim: G1Sim, sonic):
        mujoco.mj_setState(sim.model, sim.data, self.physics, STATE_SPEC)
        sim.data.xfrc_applied[:] = 0.0
        sim.wrench_source = None
        mujoco.mj_forward(sim.model, sim.data)
        sim.set_command(self.q_target, self.kp, self.kd)
        c = sonic.ctrl
        c.hist.buf.clear()
        c.hist.buf.extend(copy.deepcopy(self.hist))
        c.heading.init_base_quat = self.heading[0].copy()
        c.heading.init_ref_root_rot = self.heading[1].copy()
        c.heading.delta_heading = self.heading[2]
        c.last_action = self.last_action.copy()
        sonic.reference.reset()
        sonic._need_heading = False
        sonic.last = None

    # ---- file round trip (so worker processes can load it instead of re-running the start-up) ----
    def save(self, path: str):
        from ..sonic.onnx_policy import StateEntry  # noqa: F401
        np.savez(path, physics=self.physics, q_target=self.q_target, kp=self.kp, kd=self.kd,
                 hist_base_quat=np.array([e.base_quat for e in self.hist]),
                 hist_base_ang_vel=np.array([e.base_ang_vel for e in self.hist]),
                 hist_body_q=np.array([e.body_q for e in self.hist]),
                 hist_body_dq=np.array([e.body_dq for e in self.hist]),
                 hist_last_action=np.array([e.last_action for e in self.hist]),
                 heading_init_base_quat=self.heading[0], heading_init_ref_root_rot=self.heading[1],
                 heading_delta=self.heading[2], last_action=self.last_action, tick=self.tick,
                 k_stand=self.k_stand if self.k_stand is not None else np.zeros(0, np.int64))

    @classmethod
    def load(cls, path: str) -> "StandingSnapshot":
        from ..sonic.onnx_policy import StateEntry
        f = np.load(path)
        hist = [StateEntry(f["hist_base_quat"][i], f["hist_base_ang_vel"][i], f["hist_body_q"][i],
                           f["hist_body_dq"][i], f["hist_last_action"][i]) for i in range(len(f["hist_body_q"]))]
        k_stand = f["k_stand"] if "k_stand" in f.files and len(f["k_stand"]) else None
        return cls(f["physics"], f["q_target"], f["kp"], f["kd"], hist,
                   (f["heading_init_base_quat"], f["heading_init_ref_root_rot"], float(f["heading_delta"])),
                   f["last_action"], int(f["tick"]), k_stand)


def build_standing_snapshot(sim: G1Sim, sonic, t_settle: float = 1.0, t_release: float = 1.0,
                            t_hold: float = 3.0, yaw: float = 0.0, q_offset=None, push=None) -> StandingSnapshot:
    """Run the standard start-up with ``sonic`` (standing reference, no token hook) and capture
    the state at the start of tick t_settle + t_hold (same timeline as python -m token_policy).

    Perturbations for diverse initial states (all off by default):
      yaw        initial base heading (rad)
      q_offset   (29,) added to the reset / gantry-phase joint pose (MuJoCo order)
      push       (t_start, duration, fx, fy, fz): world force (N) on the pelvis during the hold phase
    """
    ss = SyncSim(sim, None, SyncConfig(gantry=True))
    q0 = DEFAULT_ANGLES if q_offset is None else DEFAULT_ANGLES + np.asarray(q_offset, float)
    ss.reset(q0)
    if yaw != 0.0:
        sim.data.qpos[3:7] = quat_from_angle_axis(yaw, np.array([0.0, 0.0, 1.0]))
        mujoco.mj_forward(sim.model, sim.data)
        ss.gantry.attach(sim, 1.0)                      # re-anchor at the rotated pose
        ss._state = sim.state()
    sonic.reset()
    dt = sim.control_dt
    k_attach, k_cap = int(round(t_settle / dt)), int(round((t_settle + t_hold) / dt))
    gantry = ss.gantry
    if push is not None:
        t0, dur, *f = push
        f = np.asarray(f, float)
        def wrench(s, _g=gantry):
            w = _g(s)
            if t0 <= s.time < t0 + dur:
                w = w + np.concatenate([f, np.zeros(3)])
            return w
        sim.wrench_source = wrench
    for k in range(k_cap):
        if k < k_attach:
            sonic.observe_only(ss.state())
        elif k == k_attach:
            ss.attach(sonic)
        else:
            ss.set_gantry(1.0 - (k - k_attach) * dt / max(t_release, 1e-9))
        ss.step()
    ss.set_gantry(1.0 - (k_cap - k_attach) * dt / max(t_release, 1e-9))   # gain at the captured tick (0)
    ss.detach()
    return StandingSnapshot.capture(sim, sonic, k_cap)
