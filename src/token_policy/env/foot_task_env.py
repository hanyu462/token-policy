"""T0 teacher environment: privileged state, FSQ-grid token action, foot displacement goal.

    env = FootTaskEnv(TaskConfig(effector="R", commands=[(0, 0.01), (0, -0.01)]))
    obs = env.reset(seed=0)                 # privileged obs (150)
    obs, r, done, info = env.step(delta_k)  # delta_k: int (64,) in [-dk_range, dk_range]
                                            # token z = (k_stand + delta_k) / 16 -> frozen SONIC

Goal: move the effector foot by (dx, dy) in the episode-start heading frame; reward is progress
toward the target plus success, minus fall / stance-foot drift / slip / large delta_k.
Success = within success_dist at low speed for success_hold ticks, then a post_hold (0.8 s) in
which the policy must keep the pose, then the episode ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common.rotations import gravity_in_body_frame
from ..paths import SONIC_MODEL_DIR
from ..policy import GridOffsetToken
from ..policy.fsq import k_to_z
from ..sim import G1Sim, SyncConfig, SyncSim, mujoco_to_isaac
from ..sim.g1_params import DEFAULT_ANGLES
from ..sonic import SonicController, SonicJointController, SonicONNX, load_reference
from ..task import LocalFrame
from .snapshot import StandingSnapshot, build_standing_snapshot


@dataclass
class TaskConfig:
    model_dir: str = SONIC_MODEL_DIR
    onnx_threads: int = 1
    decoder_backend: str = "onnx"             # "onnx" (CPU) | "torch" (GPU, batch-1 here)
    device: str = "cuda"
    snapshot_path: str | None = None
    effector: str = "R"                       # "L" | "R" | "both" (sampled per episode)
    commands: tuple = ((0.0, 0.01), (0.0, -0.01))   # (dx forward, dy lateral) in metres, sampled per episode
    dk_range: int = 2                         # delta_k in [-dk_range, dk_range]
    max_steps: int = 300                      # 6 s to reach the goal
    success_dist: float = 0.005
    success_vel: float = 0.03
    success_hold: int = 5                     # ticks (100 ms)
    post_hold_s: float = 0.8                  # keep running after success (labels for the student chunk)
    w_progress: float = 100.0                 # per metre of distance reduction (1 cm -> +1)
    w_success: float = 10.0                   # once, when success triggers
    w_hold: float = 0.1                       # per tick inside tolerance after success
    w_fall: float = 20.0
    w_stance: float = 20.0                    # per metre of stance-foot drift, per tick
    w_slip: float = 1.0                       # per m/s of stance-foot speed while in contact, per tick
    w_dk: float = 0.0005                      # * ||delta_k||^2 per tick (random policy ~ -0.08/tick)
    contact_force: float = 20.0               # N: a foot is "in contact" above this
    record: bool = False                      # keep per-tick k / state for dataset generation


class _DeltaSource:
    def __init__(self, dim=64):
        self.value = np.zeros(dim, np.int64)

    def __call__(self, state):
        return self.value


class FootTaskEnv:
    OBS_LAYOUT = [("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("gravity", 3),
                  ("lfoot_local", 3), ("rfoot_local", 3), ("lfoot_vel", 3), ("rfoot_vel", 3), ("contact", 2),
                  ("com_local", 3), ("goal_error", 3), ("effector", 2), ("k_stand", 64)]
    obs_dim = sum(d for _, d in OBS_LAYOUT)   # 150

    @classmethod
    def obs_index(cls, name: str) -> slice:
        """Slice of one OBS_LAYOUT block in the observation vector."""
        start = 0
        for n, d in cls.OBS_LAYOUT:
            if n == name:
                return slice(start, start + d)
            start += d
        raise KeyError(name)
    act_dim = 64

    def __init__(self, cfg: TaskConfig | None = None):
        self.cfg = cfg or TaskConfig()
        self.sim = G1Sim()
        self.sync = SyncSim(self.sim, None, SyncConfig(gantry=False))
        self.delta = _DeltaSource()
        self.sonic = SonicJointController(
            SonicController(SonicONNX(self.cfg.model_dir, threads=self.cfg.onnx_threads,
                                      decoder_backend=self.cfg.decoder_backend, device=self.cfg.device)), load_reference("standing"))
        self.snapshot = (StandingSnapshot.load(self.cfg.snapshot_path) if self.cfg.snapshot_path
                         else build_standing_snapshot(self.sim, self.sonic))
        self.set_snapshot(self.snapshot)
        self.rng = np.random.default_rng()
        self.dt = self.sim.control_dt
        self.n_action = 2 * self.cfg.dk_range + 1

    def set_snapshot(self, snapshot: StandingSnapshot):
        """Initial state for the following resets; the token offset is relative to its k_stand."""
        assert snapshot.k_stand is not None, "snapshot has no k_stand (rebuild it)"
        self.snapshot = snapshot
        self.token = GridOffsetToken(snapshot.k_stand, self.delta)
        self.sonic.token_hook = self.token          # GridOffsetToken is itself a (z_base, state) -> z hook
        self.z_stand = k_to_z(snapshot.k_stand)

    # ---- episode ----
    def reset(self, seed: int | None = None, effector: str | None = None, command=None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.snapshot.restore(self.sim, self.sonic)
        self.token.reset()
        self.delta.value[:] = 0
        self.sync.attach(self.sonic)
        self.sync._state = self.sim.state()
        self.sync.tick = 0
        st = self.sync.state()
        eff = effector or self.cfg.effector
        self.effector = self.rng.choice(["L", "R"]) if eff == "both" else eff
        dx, dy = command if command is not None else self.cfg.commands[self.rng.integers(len(self.cfg.commands))]
        self.command = (float(dx), float(dy))
        self.frame = LocalFrame.from_base(st["base_pos"], st["base_quat"])
        self.p_eff0 = st[self._eff_key()].copy()
        self.p_stance0 = st[self._stance_key()].copy()
        self.p_target = self.p_eff0 + dx * self.frame.forward + dy * self.frame.lateral
        self.prev_feet = (st["lfoot"].copy(), st["rfoot"].copy())
        self.d_prev = float(np.linalg.norm(st[self._eff_key()] - self.p_target))
        self.t = 0
        self.success_tick = None
        self.in_tol_count = 0
        self.records = [] if self.cfg.record else None
        return self._obs(st, np.zeros(3), np.zeros(3))

    def step(self, delta_k):
        self.delta.value[:] = np.clip(np.asarray(delta_k, np.int64), -self.cfg.dk_range, self.cfg.dk_range)
        st = self.sync.step()
        self.t += 1
        lf, rf = st["lfoot"], st["rfoot"]
        v_l = (lf - self.prev_feet[0]) / self.dt
        v_r = (rf - self.prev_feet[1]) / self.dt
        self.prev_feet = (lf.copy(), rf.copy())
        p_eff, v_eff = (rf, v_r) if self.effector == "R" else (lf, v_l)
        p_st, v_st, f_st = (lf, v_l, st["lfoot_force"]) if self.effector == "R" else (rf, v_r, st["rfoot_force"])

        d = float(np.linalg.norm(p_eff - self.p_target))
        fallen = self.sim.fallen()
        stance_drift = float(np.linalg.norm(p_st - self.p_stance0))
        slip = float(np.linalg.norm(v_st[:2])) if f_st > self.cfg.contact_force else 0.0
        in_tol = d < self.cfg.success_dist and np.linalg.norm(v_eff) < self.cfg.success_vel
        self.in_tol_count = self.in_tol_count + 1 if in_tol else 0

        c = self.cfg
        r = c.w_progress * (self.d_prev - d) - c.w_stance * stance_drift - c.w_slip * slip - c.w_dk * float(np.sum(self.delta.value ** 2))
        if self.success_tick is None and self.in_tol_count >= c.success_hold:
            self.success_tick = self.t
            r += c.w_success
        if self.success_tick is not None and in_tol:
            r += c.w_hold
        if fallen:
            r -= c.w_fall
        self.d_prev = d

        hold_done = self.success_tick is not None and (self.t - self.success_tick) * self.dt >= c.post_hold_s
        done = bool(fallen or hold_done or self.t >= c.max_steps + (int(c.post_hold_s / self.dt) if self.success_tick else 0))
        obs = self._obs(st, v_l, v_r)
        info = {"t": self.t, "dist": d, "error_local": self.frame.to_local(self.p_target) - self.frame.to_local(p_eff),
                "disp_local": self.frame.to_local(p_eff) - self.frame.to_local(self.p_eff0),
                "stance_drift": stance_drift, "slip": slip, "fallen": fallen, "in_tol": in_tol,
                "success": self.success_tick is not None, "phase": "hold" if self.success_tick else "reach",
                "effector": self.effector, "command": self.command, "k": self.token.k_last.copy()}
        if self.records is not None:
            self.records.append({"k": self.token.k_last.copy(), "obs": obs, "state": st, "info": {k: v for k, v in info.items() if k != "k"}})
        return obs, float(r), done, info

    # ---- helpers ----
    def _eff_key(self):
        return "rfoot" if self.effector == "R" else "lfoot"

    def _stance_key(self):
        return "lfoot" if self.effector == "R" else "rfoot"

    def _obs(self, st, v_l, v_r):
        p_eff = st[self._eff_key()]
        com = self.sim.data.subtree_com[self.sim.pelvis_id]
        parts = [mujoco_to_isaac(st["q"] - DEFAULT_ANGLES), mujoco_to_isaac(st["dq"]), st["base_ang_vel"],
                 gravity_in_body_frame(st["base_quat"]),
                 self.frame.to_local(st["lfoot"]), self.frame.to_local(st["rfoot"]),
                 self.frame.R.T @ v_l, self.frame.R.T @ v_r,
                 np.clip([st["lfoot_force"], st["rfoot_force"]], 0, 500) / 100.0,
                 self.frame.to_local(com),
                 self.frame.R.T @ (self.p_target - p_eff),
                 [1.0, 0.0] if self.effector == "L" else [0.0, 1.0],
                 self.z_stand]
        o = np.concatenate([np.asarray(p, float).ravel() for p in parts])
        assert o.shape == (self.obs_dim,), o.shape
        return o

    def state(self) -> dict:
        return self.sync.state()
