"""Layout / dimension / NaN checks for the SONIC v1.1 ONNX pipeline (run: .venv/bin/python -m pytest -q)."""
import numpy as np

from token_policy.paths import SONIC_MODEL_DIR as MODEL_DIR
from token_policy.sim import DEFAULT_ANGLES, isaac_to_mujoco, mujoco_to_isaac
from token_policy.sonic import HeadingState, SonicONNX, StandingReference, StateHistory


def test_joint_remap_roundtrip():
    q = np.arange(29.0)
    assert np.array_equal(isaac_to_mujoco(mujoco_to_isaac(q)), q)
    # IsaacLab index 21/22 are the elbows (MuJoCo 18/25)
    assert mujoco_to_isaac(q)[21] == 18 and mujoco_to_isaac(q)[22] == 25


def test_layout_matches_onnx():
    p = SonicONNX(MODEL_DIR)
    enc, dec = p.cfg.encoder, p.cfg.policy
    assert enc.total == 1751 and dec.total == 994 and p.token_dim == 64
    assert enc.offsets["motion_joint_positions_10frame_step5"] == 4
    assert enc.offsets["motion_joint_velocities_10frame_step5"] == 294
    assert enc.offsets["motion_anchor_orientation_heading_10frame_step5"] == 584
    assert dec.offsets["token_state"] == 0 and dec.offsets["his_gravity_dir_10frame_step1"] == 964


def test_encoder_g1_mode_live_slots_and_fsq_grid():
    """Only the g1-mode blocks change the token; the token lies on the FSQ grid (multiples of 1/16)."""
    p = SonicONNX(MODEL_DIR)
    ref = StandingReference.default()
    obs = p.encoder_obs(ref, np.array([1.0, 0, 0, 0]), HeadingState())
    z = p.encode(obs)
    assert z.shape == (64,) and np.all(np.isfinite(z))
    assert np.allclose(z * 16, np.round(z * 16), atol=1e-5)
    rng = np.random.default_rng(0)
    for name in ["motion_anchor_orientation_heading", "motion_joint_positions_lowerbody_10frame_step5",
                 "vr_3point_local_target", "smpl_joints_10frame_step1"]:
        o = obs.copy()
        o[p.cfg.encoder.slice(name)] += rng.normal(0, 0.5, p.cfg.encoder.dims[name]).astype(np.float32)
        assert np.abs(p.encode(o) - z).max() == 0.0, name
    o = obs.copy()
    o[p.cfg.encoder.slice("motion_joint_positions_10frame_step5")] += 0.3
    assert np.abs(p.encode(o) - z).max() > 0.0


def test_token_hook_reaches_decoder():
    p = SonicONNX(MODEL_DIR)
    ref = StandingReference.default()
    hist = StateHistory()
    for _ in range(10):
        hist.log(np.array([1.0, 0, 0, 0]), np.zeros(3), DEFAULT_ANGLES, np.zeros(29), np.zeros(29))
    dz = np.random.default_rng(0).standard_normal(64)
    a0 = p.step(ref, np.array([1.0, 0, 0, 0]), HeadingState(), hist)
    a1 = p.step(ref, np.array([1.0, 0, 0, 0]), HeadingState(), hist, token_hook=lambda z, st: z + 0.1 * dz)
    a2 = p.step(ref, np.array([1.0, 0, 0, 0]), HeadingState(), hist, token_hook=lambda z, st: a0["z"])
    assert np.allclose(a1["z_base"], a0["z"]) and np.allclose(a1["z"] - a1["z_base"], 0.1 * dz)
    assert np.abs(a1["action"] - a0["action"]).max() > 0.0        # residual changes the action
    assert np.allclose(a2["action"], a0["action"])                 # replacing with the same token does not
    assert np.all(np.isfinite(a1["action"]))
