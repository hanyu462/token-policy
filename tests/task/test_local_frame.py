"""Robot-local frame tests (run: .venv/bin/python -m pytest -q)."""
import numpy as np

from token_policy.common.rotations import quat_from_angle_axis
from token_policy.sim import DEFAULT_ANGLES, G1Sim
from token_policy.task import LocalFrame

QI = np.array([1.0, 0.0, 0.0, 0.0])


def _standing_sim():
    sim = G1Sim()
    sim.reset(DEFAULT_ANGLES)
    return sim


def test_model_positive_lateral_is_robot_left():
    """On the G1 model the body named left_* foot has the larger lateral coordinate -> +lateral = robot left."""
    sim = _standing_sim()
    lf, rf = sim.foot_pos()
    frame = LocalFrame.from_base(sim.base_pos(), sim.base_quat())
    assert frame.to_local(lf)[1] > frame.to_local(rf)[1] + 0.1
    # and the feet are not displaced along forward (symmetric stance)
    assert abs(frame.to_local(lf)[0] - frame.to_local(rf)[0]) < 0.02
    # forward = pelvis heading: toes (foot box geom centre, +0.035 m in the foot frame) point forward
    bid = sim.lfoot_id
    g = [g for g in sim.foot_geoms if sim.model.geom_bodyid[g] == bid][0]
    toe_dir = sim.data.geom_xpos[g] - sim.data.xpos[bid]
    assert np.dot(toe_dir, frame.forward) > 0.02


def test_shift_along_lateral_axis():
    sim = _standing_sim()
    lf, rf = sim.foot_pos()
    frame = LocalFrame.from_base(sim.base_pos(), sim.base_quat())
    d = lambda p: frame.to_local(p)[1] - frame.to_local(rf)[1]   # noqa: E731
    assert np.isclose(d(rf + 0.03 * frame.lateral), +0.03)        # towards robot-left
    assert np.isclose(d(rf - 0.03 * frame.lateral), -0.03)


def test_frame_is_heading_relative_and_fixed():
    """A world +y shift is +lateral only when the robot faces +x; a yawed robot gets its own lateral axis,
    and the frame stored at t=0 does not follow later base yaw changes."""
    origin = np.array([1.0, 2.0, 0.75])
    rf = origin + np.array([0.0, -0.12, -0.7])
    for yaw in [0.0, np.pi / 2, -np.pi / 2, np.pi, 0.7]:
        frame = LocalFrame.from_base(origin, quat_from_angle_axis(yaw, [0, 0, 1]))
        left_world = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        assert np.allclose(frame.lateral, left_world)
        d = lambda p: frame.to_local(p)[1] - frame.to_local(rf)[1]   # noqa: E731
        assert np.isclose(d(rf + 0.03 * left_world), 0.03)
        assert np.isclose(d(rf + 0.03 * np.array([np.cos(yaw), np.sin(yaw), 0])), 0.0)
    # facing +y: a world +y shift is forward, not lateral; world -x is robot-left
    frame = LocalFrame.from_base(origin, quat_from_angle_axis(np.pi / 2, [0, 0, 1]))
    assert np.isclose(frame.to_local(rf + np.array([0, 0.03, 0]))[1] - frame.to_local(rf)[1], 0.0)
    assert np.isclose(frame.to_local(rf + np.array([-0.03, 0, 0]))[1] - frame.to_local(rf)[1], 0.03)
    # round trip
    p = origin + np.array([0.3, -0.2, -0.5])
    assert np.allclose(frame.to_world(frame.to_local(p)), p)
