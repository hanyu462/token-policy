"""
Observation layout of the SONIC encoder / decoder inputs.

Source: Hugging Face nvidia/GEAR-SONIC, sonic_v1_1/observation_config.yaml

The yaml lists which observations make up each input vector and in what order; this module
turns that into offset tables (encoder 1751, decoder 994) the same way the deploy stack does.
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml

# observation name -> vector size
OBS_DIMS = {
    "token_state": None,  # set from encoder.dimension
    "encoder_mode": 3,
    "encoder_mode_4": 4,
    "motion_joint_positions": 29,
    "motion_joint_velocities": 29,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "motion_joint_positions_wrists_10frame_step1": 60,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_heading": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_anchor_orientation_heading_10frame_step5": 60,
    "motion_root_z_position": 1,
    "motion_root_z_position_10frame_step5": 10,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "smpl_anchor_orientation_heading_10frame_step1": 60,
    "base_angular_velocity": 3,
    "body_joint_positions": 29,
    "body_joint_velocities": 29,
    "last_actions": 29,
    "gravity_dir": 3,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
}


@dataclass
class Layout:
    """Where each observation sits in one input vector: obs[slice(name)] = value."""
    names: list
    offsets: dict
    dims: dict
    total: int

    def slice(self, name: str) -> slice:
        return slice(self.offsets[name], self.offsets[name] + self.dims[name])

    def __contains__(self, name):
        return name in self.offsets


def _build(entries, token_dim):
    names, offsets, dims = [], {}, {}
    off = 0
    for e in entries:
        if not e.get("enabled", True):
            continue
        n = e["name"]
        d = token_dim if n == "token_state" else OBS_DIMS[n]
        if d is None:
            raise KeyError(f"unknown observation dimension for {n}")
        names.append(n)
        offsets[n] = off
        dims[n] = d
        off += d
    return Layout(names, offsets, dims, off)


@dataclass
class SonicObsConfig:
    policy: Layout
    encoder: Layout
    token_dim: int
    encoder_modes: dict  # name -> (mode_id, required observation names)

    @classmethod
    def load(cls, path: str) -> "SonicObsConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        enc = cfg.get("encoder", {})
        token_dim = int(enc.get("dimension", 0))
        policy = _build(cfg["observations"], token_dim)
        encoder = _build(enc.get("encoder_observations", []), token_dim)
        modes = {m["name"]: (int(m["mode_id"]), list(m["required_observations"]))
                 for m in enc.get("encoder_modes", [])}
        return cls(policy, encoder, token_dim, modes)
