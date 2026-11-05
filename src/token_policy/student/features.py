"""State features for the Student: everything a real G1 can observe, expressed in the primitive-start frame F0.

F0 = heading frame of the base at the primitive start (LocalFrame: columns forward / lateral(left) / up); it never
rotates afterwards.  No world pose enters the features.  Per frame (150-D):

    joint_pos 29 (q - default) | joint_vel 29 | base_lin_vel_f0 3 | base_ang_vel 3 (body) | gravity_body 3 |
    heading_rel 2 (cos, sin of yaw - yaw0) | feet_rel_base_f0 6 | foot_contact 2 (F / 500 N, clipped to [0, 1]) |
    progress_base 3 (R0^T (p_base - p_base0)) | progress_feet 6 | prev_token 64 (k_{t-1} / 16)

Tick alignment (dataset V2 rollouts): state_history[i] is the state AFTER token_k[i] was executed, so the state
"before tick t" is state_start for t = 0 and state_history[t - 1] otherwise; prev_token for tick t is token_k[t - 1]
(k_stand for t = 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.rotations import calc_heading, gravity_in_body_frame
from ..sim.g1_params import DEFAULT_ANGLES
from ..task import LocalFrame

FEATURE_BLOCKS = [("joint_pos", 29), ("joint_vel", 29), ("base_lin_vel_f0", 3), ("base_ang_vel", 3), ("gravity_body", 3),
                  ("heading_rel", 2), ("feet_rel_base_f0", 6), ("foot_contact", 2), ("progress_base", 3),
                  ("progress_feet", 6), ("prev_token", 64)]
FEATURE_DIM = sum(d for _, d in FEATURE_BLOCKS)          # 150
UNNORMALIZED = ("heading_rel", "foot_contact", "prev_token")  # already bounded / on the grid
CONTACT_FORCE_SCALE = 500.0                                # N: ~1.4 x body weight -> single-support load ~0.7
STATE_KEYS = ("q", "dq", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "lfoot", "rfoot", "lfoot_force", "rfoot_force")


def block_slices() -> dict:
    out, i = {}, 0
    for name, d in FEATURE_BLOCKS:
        out[name] = slice(i, i + d)
        i += d
    return out


@dataclass
class StartFrame:
    """Primitive start: F0 and the start positions the progress features are relative to."""
    frame: LocalFrame
    base_pos0: np.ndarray
    lfoot0: np.ndarray
    rfoot0: np.ndarray

    @classmethod
    def from_state(cls, st: dict) -> "StartFrame":
        return cls(LocalFrame.from_base(st["base_pos"], st["base_quat"]), np.asarray(st["base_pos"], float).copy(),
                   np.asarray(st["lfoot"], float).copy(), np.asarray(st["rfoot"], float).copy())


def frame_features(st: dict, start: StartFrame, prev_k: np.ndarray) -> np.ndarray:
    """One 150-D feature vector from a state dict (sim.state() keys) before the token of this tick is applied."""
    R = start.frame.R                                     # columns: forward, lateral, up (world)
    base_pos = np.asarray(st["base_pos"], float)
    parts = [
        np.asarray(st["q"], float) - DEFAULT_ANGLES,
        np.asarray(st["dq"], float),
        R.T @ np.asarray(st["base_lin_vel"], float),
        np.asarray(st["base_ang_vel"], float),
        gravity_in_body_frame(st["base_quat"]),
        _heading_rel(st["base_quat"], start.frame.yaw),
        R.T @ (np.asarray(st["lfoot"], float) - base_pos),
        R.T @ (np.asarray(st["rfoot"], float) - base_pos),
        np.clip([st["lfoot_force"], st["rfoot_force"]], 0.0, CONTACT_FORCE_SCALE) / CONTACT_FORCE_SCALE,
        R.T @ (base_pos - start.base_pos0),
        R.T @ (np.asarray(st["lfoot"], float) - start.lfoot0),
        R.T @ (np.asarray(st["rfoot"], float) - start.rfoot0),
        np.asarray(prev_k, float) / 16.0,
    ]
    x = np.concatenate([np.atleast_1d(np.asarray(p, float)).ravel() for p in parts])
    assert x.shape == (FEATURE_DIM,), x.shape
    return x


def _heading_rel(base_quat, yaw0: float) -> np.ndarray:
    d = calc_heading(base_quat) - yaw0
    return np.array([np.cos(d), np.sin(d)])


def rollout_features(state_start: dict, state_history: dict, token_k: np.ndarray, k_stand: np.ndarray) -> np.ndarray:
    """(T, 150): features of the state before each tick t of a dataset V2 rollout (T = len(token_k))."""
    T = len(token_k)
    start = StartFrame.from_state(state_start)
    out = np.empty((T, FEATURE_DIM))
    for t in range(T):
        st = state_start if t == 0 else {k: state_history[k][t - 1] for k in STATE_KEYS}
        out[t] = frame_features(st, start, k_stand if t == 0 else token_k[t - 1])
    return out


def history_window(feats: np.ndarray, t: int, T_s: int = 10) -> np.ndarray:
    """(T_s, 150): features for ticks t-T_s+1 .. t; ticks before 0 repeat tick 0 (the robot stood still)."""
    idx = np.clip(np.arange(t - T_s + 1, t + 1), 0, None)
    return feats[idx]


class NormStats:
    """Per-dimension mean / std fitted on train features only; bounded blocks are left untouched."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, normalize: np.ndarray, source: str = ""):
        self.mean, self.std, self.normalize, self.source = mean, std, normalize, source

    @classmethod
    def fit(cls, feats: np.ndarray, source: str = "") -> "NormStats":
        feats = feats.reshape(-1, FEATURE_DIM)
        mean, std = feats.mean(0), feats.std(0)
        std = np.where(std < 1e-6, 1.0, std)
        normalize = np.ones(FEATURE_DIM, bool)
        sl = block_slices()
        for name in UNNORMALIZED:
            normalize[sl[name]] = False
        mean[~normalize], std[~normalize] = 0.0, 1.0
        return cls(mean, std, normalize, source)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std, normalize=self.normalize, source=np.array(self.source))

    @classmethod
    def load(cls, path: str) -> "NormStats":
        f = np.load(path)
        return cls(f["mean"], f["std"], f["normalize"], str(f["source"]))
