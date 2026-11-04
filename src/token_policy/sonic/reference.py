"""Reference providers for the SONIC encoder (g1 mode).

A provider answers one question per control tick: "what are the next
``num_frames`` reference frames, ``step`` ticks apart?"  It returns joint
positions (IsaacLab order), joint velocities (zero-filled while not playing,
as the deploy stack does) and the reference root quaternion per frame.

  * ``StandingReference``  - one pose held forever (never "playing").
  * ``MotionReference``    - an official GEAR-SONIC reference motion exported
                             with gear_sonic_deploy/reference/convert_motions.py
                             (joint_pos.csv / joint_vel.csv / body_quat.csv /
                             body_pos.csv at 50 Hz, joints in IsaacLab order,
                             body 0 = pelvis root).  Held at frame 0 until
                             ``play()``; then advances one frame per 50 Hz tick
                             and holds the last frame at the end.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..paths import MOTION_DIR
from ..sim.g1_params import DEFAULT_ANGLES, DEFAULT_ROOT_HEIGHT, isaac_to_mujoco, mujoco_to_isaac


class ReferenceProvider:
    name: str          # set by subclasses (no default here: StandingReference is a dataclass)
    encoder_mode = 0   # g1

    @property
    def playing(self) -> bool:
        return False

    @property
    def cursor(self) -> int:
        return 0

    @property
    def length(self) -> int:
        return 1

    @property
    def root_quat(self) -> np.ndarray:  # root quaternion of the current frame (wxyz)
        raise NotImplementedError

    def future_frames(self, num_frames: int = 10, step: int = 5):
        """(q (N,29) IsaacLab, dq (N,29), root_quat (N,4), root_pos (N,3)) for the next frames."""
        raise NotImplementedError

    def advance(self):
        pass

    def play(self):
        pass

    def reset(self):
        pass


@dataclass
class StandingReference(ReferenceProvider):
    name: str
    joint_pos_isaac: np.ndarray   # (29,) IsaacLab order, rad
    root_quat_: np.ndarray        # (4,) wxyz
    root_height: float

    @classmethod
    def default(cls) -> "StandingReference":
        return cls("standing", mujoco_to_isaac(DEFAULT_ANGLES).copy(),
                   np.array([1.0, 0.0, 0.0, 0.0]), DEFAULT_ROOT_HEIGHT)

    @property
    def root_quat(self) -> np.ndarray:
        return self.root_quat_

    @property
    def joint_pos_mujoco(self) -> np.ndarray:
        return isaac_to_mujoco(self.joint_pos_isaac)

    def future_frames(self, num_frames: int = 10, step: int = 5):
        q = np.tile(self.joint_pos_isaac, (num_frames, 1))
        dq = np.zeros_like(q)
        roots = np.tile(self.root_quat_, (num_frames, 1))
        pos = np.tile([0.0, 0.0, self.root_height], (num_frames, 1))
        return q, dq, roots, pos


class MotionReference(ReferenceProvider):
    """Official GEAR-SONIC reference-motion directory (CSV export), played at 50 Hz."""

    def __init__(self, motion_dir: str, start_frame: int = 0):
        self.motion_dir = os.path.normpath(motion_dir)
        self.name = os.path.basename(self.motion_dir)
        ld = lambda f: np.atleast_2d(np.loadtxt(os.path.join(self.motion_dir, f), delimiter=",", skiprows=1))  # noqa: E731
        self.joint_pos = ld("joint_pos.csv")[:, :29].astype(float)          # (T,29) IsaacLab order
        self.joint_vel = ld("joint_vel.csv")[:, :29].astype(float)          # (T,29)
        bq = ld("body_quat.csv"); bp = ld("body_pos.csv")
        self.root_quat_all = bq[:, 0:4].astype(float)                        # body 0 = pelvis, wxyz
        self.root_pos_all = bp[:, 0:3].astype(float)
        self.T = self.joint_pos.shape[0]
        assert self.joint_vel.shape[0] == self.T == self.root_quat_all.shape[0] == self.root_pos_all.shape[0]
        self.start_frame = int(np.clip(start_frame, 0, self.T - 1))
        self.reset()

    def reset(self):
        self._cursor = self.start_frame
        self._playing = False
        self.completed = False
        self.needs_heading_reset = False

    def play(self):
        self._playing = True

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def length(self) -> int:
        return self.T

    @property
    def root_quat(self) -> np.ndarray:
        return self.root_quat_all[self._cursor]

    def _target_frame(self, k: int, step: int) -> int:
        f = self._cursor
        if self._playing:
            f = min(f + k * step, self.T - 1)      # clamp: hold the final pose
        return int(np.clip(f, 0, self.T - 1))

    def future_frames(self, num_frames: int = 10, step: int = 5):
        idx = [self._target_frame(k, step) for k in range(num_frames)]
        q = self.joint_pos[idx]
        dq = self.joint_vel[idx] if self._playing else np.zeros_like(q)  # deploy zero-fills while not playing
        return q, dq, self.root_quat_all[idx], self.root_pos_all[idx]

    def advance(self):
        """One 50 Hz control tick (deploy: current_frame_++ while playing; at the end the motion
        stops, the cursor returns to frame 0 (held) and the heading alignment is re-initialised)."""
        if not self._playing:
            return
        self._cursor += 1
        if self._cursor >= self.T:
            self._cursor = 0
            self._playing = False
            self.completed = True
            self.needs_heading_reset = True


def load_reference(spec: str) -> ReferenceProvider:
    """'standing' | 'default' -> StandingReference; a motion name under assets/motions or a directory -> MotionReference."""
    if spec in ("standing", "default"):
        return StandingReference.default()
    if os.path.isdir(spec):
        return MotionReference(spec)
    cand = os.path.join(MOTION_DIR, spec)
    if os.path.isdir(cand):
        return MotionReference(cand)
    raise FileNotFoundError(f"unknown reference '{spec}' (not 'standing', not a motion directory, not under {MOTION_DIR})")


def list_motions() -> list:
    return sorted(d for d in os.listdir(MOTION_DIR) if os.path.isdir(os.path.join(MOTION_DIR, d))) if os.path.isdir(MOTION_DIR) else []
