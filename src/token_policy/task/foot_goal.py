"""Robot-local frame F0 for foot / hand goals.

Fixed at the task start (t = 0) from the pelvis pose:

    forward  = pelvis heading (+x of the pelvis body projected on the floor)
    lateral  = forward rotated +90 deg about +z  (== pelvis-local +y)
    vertical = world +z

The frame never rotates afterwards, even if the base yaw drifts.
Convention (verified by tests/task/test_local_frame.py on the G1 model):

    positive lateral = robot LEFT
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.rotations import calc_heading

LATERAL_CONVENTION = "positive lateral = robot left (pelvis-local +y at task start)"


@dataclass
class LocalFrame:
    origin: np.ndarray      # world position of the pelvis at t = 0
    yaw: float              # pelvis heading at t = 0 (rad)
    R: np.ndarray           # (3,3); columns = forward, lateral, vertical in world coordinates

    @classmethod
    def from_base(cls, base_pos, base_quat) -> "LocalFrame":
        yaw = float(calc_heading(base_quat))
        return cls.from_yaw(base_pos, yaw)

    @classmethod
    def from_yaw(cls, origin, yaw: float) -> "LocalFrame":
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return cls(np.asarray(origin, float).copy(), float(yaw), R)

    @property
    def forward(self): return self.R[:, 0].copy()
    @property
    def lateral(self): return self.R[:, 1].copy()
    @property
    def vertical(self): return self.R[:, 2].copy()

    def to_local(self, p_world):
        return self.R.T @ (np.asarray(p_world, float) - self.origin)

    def to_world(self, p_local):
        return self.origin + self.R @ np.asarray(p_local, float)
