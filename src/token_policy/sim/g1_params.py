"""
Unitree G1 29-DOF constants for the SONIC policy.

SONIC: NVIDIA's whole-body motion-tracking controller for the G1 humanoid
(GEAR-SONIC, github.com/NVlabs/GR00T-WholeBodyControl, Apache-2.0).

Per-joint arrays are in MuJoCo/URDF order unless stated otherwise; the policy
works in IsaacLab order and is remapped through the index arrays below.
"""
from __future__ import annotations

import math

import numpy as np

NUM_JOINTS = 29

# MuJoCo / URDF joint order (== DDS motor order == MJCF actuator order)
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ISAACLAB_TO_MUJOCO[i] = IsaacLab index of MuJoCo-order joint i
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], dtype=int)
# MUJOCO_TO_ISAACLAB[i] = MuJoCo index of IsaacLab-order joint i
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], dtype=int)

# ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB[i]] == i  for all i in 0..28
assert (ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB] == np.arange(NUM_JOINTS)).all()


def mujoco_to_isaac(q_mj: np.ndarray) -> np.ndarray:
    """Reorder a MuJoCo-order joint vector into IsaacLab order."""
    return np.asarray(q_mj)[..., MUJOCO_TO_ISAACLAB]


def isaac_to_mujoco(a_isaac: np.ndarray) -> np.ndarray:
    """Reorder an IsaacLab-order joint vector into MuJoCo order."""
    return np.asarray(a_isaac)[..., ISAACLAB_TO_MUJOCO]


# actuator constants
# G1 uses four motor types: "5020", "7520_14", "7520_22", "4010"

# armature: motor inertia seen at the joint (kg * m^2)
#   armature = N^2 * J_rotor    (N = gear ratio)
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}

# effort: torque limit (N * m)
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}

# PD Gains for joint position control
#   J * q'' + kd * q' + kp * q = 0    (J = armature)
#   kp = J * w^2,   kd = 2 * zeta * J * w

# w: 10 Hz natural frequency (rad/s)
_W = 10.0 * 2.0 * math.pi

# zeta: damping ratio
_ZETA = 2.0

_STIFF = {k: v * _W * _W for k, v in _ARMATURE.items()}             # kp per motor type
_DAMP = {k: 2.0 * _ZETA * v * _W for k, v in _ARMATURE.items()}     # kd per motor type

# (motor type, gain multiplier) per MuJoCo joint
_MOTOR = [
    ("7520_22", 1), ("7520_22", 1), ("7520_14", 1), ("7520_22", 1), ("5020", 2), ("5020", 2),   # left leg
    ("7520_22", 1), ("7520_22", 1), ("7520_14", 1), ("7520_22", 1), ("5020", 2), ("5020", 2),   # right leg
    ("7520_14", 1), ("5020", 2), ("5020", 2),                                                   # waist
    ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("4010", 1), ("4010", 1),  # left arm
    ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("4010", 1), ("4010", 1),  # right arm
]

# policy action scale: 0.25 * effort_limit / stiffness   (MuJoCo order)
ACTION_SCALE = np.array([0.25 * _EFFORT[m] / _STIFF[m] for m, _ in _MOTOR])
KP = np.array([g * _STIFF[m] for m, g in _MOTOR])
KD = np.array([g * _DAMP[m] for m, g in _MOTOR])

# default standing pose (rad, MuJoCo order)
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # hip pitch/roll/yaw, knee, ankle pitch/roll
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,                          # waist yaw/roll/pitch
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

# default pelvis height (m)
DEFAULT_ROOT_HEIGHT = 0.788740


def action_to_q_target(action_isaac: np.ndarray) -> np.ndarray:
    """Raw policy output (IsaacLab order) -> joint position target (rad, MuJoCo order)."""
    return DEFAULT_ANGLES + ACTION_SCALE * isaac_to_mujoco(action_isaac)
