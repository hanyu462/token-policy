"""Virtual gantry: a world-frame spring-damper that holds the pelvis near its reset pose
while the robot is initialised. ``gain`` in [0, 1] scales the wrench (0 = released).
Gains follow the GEAR-SONIC sim2sim elastic band; the anchoring is simplified (no rope length).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.rotations import quat_rotate


@dataclass
class GantryConfig:
    kp_pos: float = 10000.0
    kd_pos: float = 1000.0
    kp_ang: float = 1000.0
    kd_ang: float = 10.0


class Gantry:
    def __init__(self, cfg: GantryConfig | None = None):
        self.cfg = cfg or GantryConfig()
        self.gain = 0.0
        self.anchor = np.zeros(3)

    def attach(self, sim, gain: float = 1.0):
        """Anchor at the current pelvis position and register as the sim's external-wrench source."""
        self.anchor = sim.base_pos()
        self.gain = float(np.clip(gain, 0.0, 1.0))
        sim.wrench_source = self

    def set_gain(self, gain: float):
        self.gain = float(np.clip(gain, 0.0, 1.0))

    def __call__(self, sim) -> np.ndarray:
        """World-frame (force, torque) on the pelvis for the current sim state."""
        if self.gain <= 0.0:
            return np.zeros(6)
        c = self.cfg
        p, v = sim.base_pos(), sim.data.qvel[0:3]
        q = sim.base_quat()
        w_world = quat_rotate(q, sim.data.qvel[3:6])
        rotvec = 2.0 * q[1:4] * np.sign(q[0] if q[0] != 0 else 1.0)   # small-angle rotation vector (identity target)
        f = c.kp_pos * (self.anchor - p) - c.kd_pos * v
        tau = -c.kp_ang * rotvec - c.kd_ang * w_world
        return self.gain * np.concatenate([f, tau])
