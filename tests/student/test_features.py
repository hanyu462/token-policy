"""Student state features: F0 / heading-local invariance, contact normalization, tick alignment."""
import numpy as np

from token_policy.common.rotations import quat_from_angle_axis, quat_mul, quat_rotate
from token_policy.sim.g1_params import DEFAULT_ANGLES
from token_policy.student.features import (CONTACT_FORCE_SCALE, FEATURE_DIM, NormStats, StartFrame, block_slices,
                                           frame_features, history_window, rollout_features)

RNG = np.random.default_rng(0)


def _state(base_pos, base_quat, q, dq, v_world, w_body, lfoot, rfoot, fl, fr):
    return {"q": q, "dq": dq, "base_pos": np.asarray(base_pos, float), "base_quat": np.asarray(base_quat, float),
            "base_lin_vel": np.asarray(v_world, float), "base_ang_vel": np.asarray(w_body, float),
            "lfoot": np.asarray(lfoot, float), "rfoot": np.asarray(rfoot, float), "lfoot_force": fl, "rfoot_force": fr}


def _rotate_state(st, yaw, about):
    """The same physical configuration seen in a world frame rotated by yaw about ``about`` (world z)."""
    qz = quat_from_angle_axis(yaw, [0, 0, 1.0])
    rot = lambda p: about + quat_rotate(qz, np.asarray(p, float) - about)   # noqa: E731
    out = dict(st)
    out["base_pos"], out["lfoot"], out["rfoot"] = rot(st["base_pos"]), rot(st["lfoot"]), rot(st["rfoot"])
    out["base_quat"] = quat_mul(qz, st["base_quat"])
    out["base_lin_vel"] = quat_rotate(qz, st["base_lin_vel"])     # world-frame velocity rotates
    return out                                                     # body-frame gyro, q, dq, forces unchanged


def _random_pair(tilt=0.1):
    q0 = quat_from_angle_axis(RNG.uniform(-tilt, tilt), RNG.normal(size=3))
    q0 = quat_mul(quat_from_angle_axis(RNG.uniform(-np.pi, np.pi), [0, 0, 1.0]), q0)
    base = np.array([1.0, -2.0, 0.76])
    start = _state(base, q0, DEFAULT_ANGLES + RNG.normal(0, 0.05, 29), RNG.normal(0, 0.5, 29), RNG.normal(0, 0.1, 3),
                   RNG.normal(0, 0.2, 3), base + [0.05, 0.12, -0.72], base + [0.02, -0.12, -0.72], 300.0, 250.0)
    q1 = quat_mul(quat_from_angle_axis(0.2, [0, 0, 1.0]), quat_mul(quat_from_angle_axis(0.05, [1, 0, 0]), q0))
    later = _state(base + [0.25, 0.03, -0.01], q1, DEFAULT_ANGLES + RNG.normal(0, 0.1, 29), RNG.normal(0, 1.0, 29),
                   RNG.normal(0, 0.5, 3), RNG.normal(0, 0.5, 3), base + [0.30, 0.10, -0.60], base + [0.28, -0.14, -0.72], 20.0, 400.0)
    return start, later


def test_feature_dim_and_blocks():
    assert FEATURE_DIM == 150
    sl = block_slices()
    assert sl["prev_token"] == slice(86, 150) and sl["joint_pos"] == slice(0, 29)


def test_world_yaw_rotation_invariance():
    """Same physical configuration rotated by any world yaw -> identical local features."""
    for _ in range(5):
        start, later = _random_pair()
        prev_k = RNG.integers(-6, 8, 64)
        ref = frame_features(later, StartFrame.from_state(start), prev_k)
        for yaw in (0.3, -1.7, np.pi, 2.9):
            about = RNG.normal(size=3) * 2
            s2, l2 = _rotate_state(start, yaw, about), _rotate_state(later, yaw, about)
            got = frame_features(l2, StartFrame.from_state(s2), prev_k)
            assert np.allclose(got, ref, atol=1e-9), np.abs(got - ref).max()


def test_world_translation_invariance():
    start, later = _random_pair()
    ref = frame_features(later, StartFrame.from_state(start), np.zeros(64))
    shift = np.array([3.0, -7.0, 0.0])
    for st in (start, later):
        for k in ("base_pos", "lfoot", "rfoot"):
            st[k] = st[k] + shift
    assert np.allclose(frame_features(later, StartFrame.from_state(start), np.zeros(64)), ref)


def test_progress_and_feet_are_in_start_frame():
    """A base displacement along the start heading shows up as +x progress, regardless of world yaw."""
    for yaw in (0.0, 1.0, -2.5):
        q0 = quat_from_angle_axis(yaw, [0, 0, 1.0])
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0]); left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        base = np.array([0.5, 0.5, 0.76])
        start = _state(base, q0, DEFAULT_ANGLES, np.zeros(29), np.zeros(3), np.zeros(3), base + 0.12 * left - [0, 0, 0.72], base - 0.12 * left - [0, 0, 0.72], 300, 300)
        later = dict(start); later["base_pos"] = base + 0.3 * fwd + 0.05 * left; later["lfoot"] = start["lfoot"] + 0.4 * fwd
        later["base_lin_vel"] = 0.7 * fwd
        x = frame_features(later, StartFrame.from_state(start), np.zeros(64)); sl = block_slices()
        assert np.allclose(x[sl["progress_base"]], [0.3, 0.05, 0.0], atol=1e-9)
        assert np.allclose(x[sl["progress_feet"]][:3], [0.4, 0.0, 0.0], atol=1e-9)
        assert np.allclose(x[sl["feet_rel_base_f0"]][3:], [-0.3, -0.17, -0.72], atol=1e-9)   # right foot rel. base, F0
        assert np.allclose(x[sl["base_lin_vel_f0"]], [0.7, 0.0, 0.0], atol=1e-9)
        assert np.allclose(x[sl["heading_rel"]], [1.0, 0.0], atol=1e-9)


def test_contact_normalization_and_gravity():
    start, later = _random_pair()
    later["lfoot_force"], later["rfoot_force"] = 1200.0, -5.0
    x = frame_features(later, StartFrame.from_state(start), np.zeros(64)); sl = block_slices()
    assert np.allclose(x[sl["foot_contact"]], [1.0, 0.0])             # clipped to [0, 1], 500 N = 1
    later["lfoot_force"] = 0.5 * CONTACT_FORCE_SCALE
    x = frame_features(later, StartFrame.from_state(start), np.zeros(64))
    assert np.isclose(x[sl["foot_contact"]][0], 0.5)
    assert np.isclose(np.linalg.norm(x[sl["gravity_body"]]), 1.0)


def test_rollout_alignment_and_history():
    T, k_stand = 12, RNG.integers(-3, 4, 64)
    tok = RNG.integers(-6, 8, (T, 64))
    st0 = _random_pair()[0]
    hist = {k: np.array([st0[k] + (i + 1) * 0.001 if np.ndim(st0[k]) else st0[k] for i in range(T)]) for k in st0}
    F = rollout_features(st0, hist, tok, k_stand); sl = block_slices()
    assert F.shape == (T, FEATURE_DIM)
    assert np.allclose(F[0, sl["prev_token"]], k_stand / 16.0)               # tick 0: standing token
    assert np.allclose(F[5, sl["prev_token"]], tok[4] / 16.0)                # tick t: token of tick t-1
    assert np.allclose(F[3, sl["joint_pos"]], hist["q"][2] - DEFAULT_ANGLES)  # tick t: state after tick t-1
    w = history_window(F, 2, T_s=10)
    assert w.shape == (10, FEATURE_DIM) and np.allclose(w[0], F[0]) and np.allclose(w[-1], F[2]) and np.allclose(w[6], F[0])
    assert np.allclose(history_window(F, 11, 10), F[2:12])


def test_norm_stats_leave_bounded_blocks():
    F = RNG.normal(3.0, 2.0, (500, FEATURE_DIM))
    ns = NormStats.fit(F, source="unit"); sl = block_slices(); y = ns.apply(F)
    assert np.allclose(y[:, sl["joint_pos"]].mean(0), 0, atol=1e-6) and np.allclose(y[:, sl["joint_pos"]].std(0), 1, atol=1e-6)
    for name in ("heading_rel", "foot_contact", "prev_token"):
        assert np.allclose(y[:, sl[name]], F[:, sl[name]])
