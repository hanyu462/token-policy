"""SONIC v1.1 ONNX encoder/decoder with a latent-residual injection point.

Pipeline (one 50 Hz tick)::

    standing reference ──► encoder obs (g1 mode, 1751) ──► model_encoder.onnx
        ──► z_stand (64, FSQ-quantized tokens)
        ──► z = z_stand + scale * delta_z            <-- residual injection
        ──► decoder obs = [z | his_* proprio history] (994) ──► model_decoder.onnx
        ──► action (29, IsaacLab order) ──► q* = default + action_scale * action

The encoder graph contains the FSQ quantizer (Round ops), the decoder graph is
a plain MLP, so adding the residual to ``encoded_tokens`` is the same
operation as ``latent_residual_mode="post_quantization"`` in the training code
(UniversalTokenModule.forward: ``all_tokens = all_tokens + residual``).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from ..sim.g1_params import DEFAULT_ANGLES, NUM_JOINTS, mujoco_to_isaac
from .obs_config import SonicObsConfig
from ..common.rotations import (calc_heading_quat, calc_heading_quat_inv, gravity_in_body_frame,
                                quat_conjugate, quat_mul, quat_to_rotation_matrix)

HIST_FRAMES = 10


@dataclass
class StateEntry:
    """One logged robot state (deploy StateLogger::Entry)."""
    base_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))  # wxyz; zero entry = all zeros
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_q: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))   # IsaacLab, minus default
    body_dq: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))  # IsaacLab
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))  # raw policy output


class StateHistory:
    """Ring buffer of the last N states; oldest -> newest with zero padding."""

    def __init__(self, capacity: int = 100):
        self.buf = deque(maxlen=capacity)

    def log(self, base_quat, base_ang_vel, q_mujoco, dq_mujoco, last_action_isaac):
        self.buf.append(StateEntry(
            np.asarray(base_quat, float).copy(), np.asarray(base_ang_vel, float).copy(),
            mujoco_to_isaac(np.asarray(q_mujoco, float) - DEFAULT_ANGLES),
            mujoco_to_isaac(np.asarray(dq_mujoco, float)),
            np.asarray(last_action_isaac, float).copy()))

    def latest(self, n: int):
        have = list(self.buf)[-n:]
        return [StateEntry() for _ in range(n - len(have))] + have

    def __len__(self):
        return len(self.buf)


class HeadingState:
    """Deploy heading alignment: ref heading re-anchored to the robot heading at start."""

    def __init__(self):
        self.init_base_quat = np.array([1.0, 0, 0, 0])
        self.init_ref_root_rot = np.array([1.0, 0, 0, 0])
        self.delta_heading = 0.0

    def reset(self, base_quat, ref_root_quat):
        self.init_base_quat = np.asarray(base_quat, float).copy()
        self.init_ref_root_rot = np.asarray(ref_root_quat, float).copy()
        self.delta_heading = 0.0

    def apply_delta_heading(self):
        return quat_mul(calc_heading_quat(self.init_base_quat),
                        calc_heading_quat_inv(self.init_ref_root_rot))


def anchor_orientation_heading(ref_root_quat, base_quat, heading: HeadingState,
                               apply_delta=None, base_heading_inv=None):
    """6-D (first two rotation-matrix columns, row-major) of base_heading^-1 * ref_root.

    ``apply_delta`` / ``base_heading_inv`` are constant within a tick; pass them precomputed
    when calling for several frames.
    """
    if apply_delta is None:
        apply_delta = heading.apply_delta_heading()
    if base_heading_inv is None:
        base_heading_inv = quat_conjugate(calc_heading_quat(base_quat))
    new_ref = quat_mul(apply_delta, ref_root_quat)
    base_to_ref = quat_mul(base_heading_inv, new_ref)
    m = quat_to_rotation_matrix(base_to_ref)
    return np.array([m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[2, 0], m[2, 1]])


class SonicONNX:
    def __init__(self, model_dir: str, providers=("CPUExecutionProvider",), threads: int = 4):
        """``threads``: ONNX Runtime intra-op threads. The 150 MB decoder is memory-bound on CPU:
        measured p50 encoder+decoder 10 ms at 1 thread, 7 ms at 4 (no gain beyond)."""
        self.cfg = SonicObsConfig.load(os.path.join(model_dir, "observation_config.yaml"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.enc = ort.InferenceSession(os.path.join(model_dir, "model_encoder.onnx"), so, providers=list(providers))
        self.dec = ort.InferenceSession(os.path.join(model_dir, "model_decoder.onnx"), so, providers=list(providers))
        self.enc_in = self.enc.get_inputs()[0].name
        self.dec_in = self.dec.get_inputs()[0].name
        enc_shape = self.enc.get_inputs()[0].shape
        dec_shape = self.dec.get_inputs()[0].shape
        tok_shape = self.enc.get_outputs()[0].shape
        assert enc_shape[-1] == self.cfg.encoder.total, (enc_shape, self.cfg.encoder.total)
        assert dec_shape[-1] == self.cfg.policy.total, (dec_shape, self.cfg.policy.total)
        assert tok_shape[-1] == self.cfg.token_dim == 64, (tok_shape, self.cfg.token_dim)
        self.token_dim = self.cfg.token_dim
        self.g1_mode_id = self.cfg.encoder_modes["g1"][0]

    # ---- encoder -----------------------------------------------------------
    def encoder_obs(self, ref, base_quat, heading: HeadingState) -> np.ndarray:
        """g1-mode encoder observation from a ReferenceProvider (10 future frames, 5 ticks apart).

        Layout (observation_config.yaml order): encoder_mode_4 | motion_joint_positions_10frame_step5
        | motion_joint_velocities_10frame_step5 | motion_anchor_orientation_heading_10frame_step5;
        every other slot stays zero in g1 mode.  Frames are frame-major (frame f -> [f*29:(f+1)*29]).
        """
        L = self.cfg.encoder
        q, dq, roots, _ = ref.future_frames(10, 5)
        assert q.shape == (10, 29) and dq.shape == (10, 29) and roots.shape == (10, 4), (q.shape, dq.shape, roots.shape)
        obs = np.zeros(L.total, dtype=np.float32)
        obs[L.slice("encoder_mode_4")][0] = self.g1_mode_id
        obs[L.slice("motion_joint_positions_10frame_step5")] = q.ravel()
        obs[L.slice("motion_joint_velocities_10frame_step5")] = dq.ravel()
        apply_delta = heading.apply_delta_heading()                       # constant within the tick
        base_heading_inv = quat_conjugate(calc_heading_quat(base_quat))
        anchors = np.concatenate([anchor_orientation_heading(roots[f], base_quat, heading, apply_delta, base_heading_inv)
                                  for f in range(10)])
        obs[L.slice("motion_anchor_orientation_heading_10frame_step5")] = anchors
        return obs

    def encode(self, obs: np.ndarray) -> np.ndarray:
        z = self.enc.run(None, {self.enc_in: obs[None, :].astype(np.float32)})[0][0]
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("encoder produced non-finite tokens")
        return z.astype(np.float64)

    # ---- decoder -----------------------------------------------------------
    def decoder_obs(self, token: np.ndarray, hist: StateHistory) -> np.ndarray:
        L = self.cfg.policy
        obs = np.zeros(L.total, dtype=np.float32)
        assert token.shape == (self.token_dim,), token.shape
        obs[L.slice("token_state")] = token
        entries = hist.latest(HIST_FRAMES)  # oldest -> newest
        obs[L.slice("his_base_angular_velocity_10frame_step1")] = np.concatenate([e.base_ang_vel for e in entries])
        obs[L.slice("his_body_joint_positions_10frame_step1")] = np.concatenate([e.body_q for e in entries])
        obs[L.slice("his_body_joint_velocities_10frame_step1")] = np.concatenate([e.body_dq for e in entries])
        obs[L.slice("his_last_actions_10frame_step1")] = np.concatenate([e.last_action for e in entries])
        obs[L.slice("his_gravity_dir_10frame_step1")] = np.concatenate(
            [gravity_in_body_frame(e.base_quat) for e in entries])
        return obs

    def decode(self, obs: np.ndarray) -> np.ndarray:
        a = self.dec.run(None, {self.dec_in: obs[None, :].astype(np.float32)})[0][0]
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("decoder produced non-finite action")
        return a.astype(np.float64)

    # ---- full tick with residual injection -----------------------------------
    def step(self, ref, base_quat, heading: HeadingState, hist: StateHistory,
             delta_z: np.ndarray | None = None, scale: float = 0.0, inject: bool = True):
        """Returns dict with z_stand, z (decoder input), action, action_base.

        ``action_base`` is the decoder output for the *same* proprio history
        with the un-modified token, so ``q*(z+λδz) - q*(z)`` can be reported.
        When ``inject`` is False the residual path is bypassed entirely.
        """
        z_stand = self.encode(self.encoder_obs(ref, base_quat, heading))
        if inject and delta_z is not None:
            assert delta_z.shape == (self.token_dim,), delta_z.shape
            z = z_stand + scale * delta_z
        else:
            z = z_stand.copy()
        action = self.decode(self.decoder_obs(z, hist))
        if inject and delta_z is not None and scale != 0.0:
            action_base = self.decode(self.decoder_obs(z_stand, hist))
        else:
            action_base = action.copy()
        return {"z_stand": z_stand, "z": z, "action": action, "action_base": action_base}
