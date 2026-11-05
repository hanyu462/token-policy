"""Simulation-only checks: no SONIC, no ONNX."""
import subprocess
import sys

import numpy as np

from token_policy.sim import DEFAULT_ANGLES, G1Sim
from token_policy.sim import AsyncConfig, AsyncSim, Recorder, SyncConfig, SyncSim, hold_pose


def test_sim_package_does_not_import_onnxruntime():
    """Dependency direction: importing the sim package must not pull in the policy stack (fresh process)."""
    code = "import sys, token_policy.sim; assert 'onnxruntime' not in sys.modules and 'token_policy.sonic' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_reset_places_feet_on_floor():
    sim = G1Sim()
    sim.reset(DEFAULT_ANGLES)
    lowest = np.inf
    for g in sim.foot_geoms:
        size = sim.model.geom_size[g]
        R = sim.data.geom_xmat[g].reshape(3, 3)
        c = sim.data.geom_xpos[g]
        corners = np.array([[sx, sy, sz] for sx in (-size[0], size[0]) for sy in (-size[1], size[1]) for sz in (-size[2], size[2])])
        lowest = min(lowest, (c + corners @ R.T)[:, 2].min())
    assert abs(lowest - sim.cfg.clearance) < 1e-9
    assert np.allclose(sim.q(), DEFAULT_ANGLES) and np.allclose(sim.dq(), 0.0)


def test_step_control_advances_one_control_period():
    sim = G1Sim()
    sim.reset()
    t0 = sim.time
    sim.step_control()
    assert abs(sim.time - t0 - sim.control_dt) < 1e-12
    assert sim.n_sub == 10 and abs(sim.dt - 0.002) < 1e-12


def test_gantry_hold_is_stable_and_feet_carry_weight():
    sim = G1Sim()
    r = SyncSim(sim, None, SyncConfig(gantry=True)).run(2.0)
    assert not np.isfinite(r["fell_at"])
    rpy = np.degrees(np.abs(r["base_rpy"]))
    assert rpy[:, :2].max() < 2.0                       # roll / pitch stay small
    assert r["base_pos"][-1, 2] > 0.70                   # pelvis stays up (sags ~3 cm on the spring)
    weight = sum(sim.model.body_mass) * 9.81
    feet = r["lfoot_force"][-1] + r["rfoot_force"][-1]
    assert 0.5 * weight < feet < 1.05 * weight           # feet carry most of the weight, gantry the rest


def test_pd_alone_cannot_balance():
    """Joint PD on a fixed pose is not a balance controller: without the gantry the robot tips over."""
    sim = G1Sim()
    r = SyncSim(sim, hold_pose(), SyncConfig(gantry=False)).run(6.0)
    tilt = np.degrees(np.abs(r["base_rpy"][:, :2])).max()
    assert np.isfinite(r["fell_at"]) or tilt > 10.0


def test_run_sync_is_deterministic():
    a = SyncSim(G1Sim(), hold_pose(), SyncConfig(gantry=True, t_release=0.5)).run(1.0)
    b = SyncSim(G1Sim(), hold_pose(), SyncConfig(gantry=True, t_release=0.5)).run(1.0)
    assert np.array_equal(a["q"], b["q"]) and np.array_equal(a["base_pos"], b["base_pos"])


def test_run_async_steps_in_background_and_takes_external_commands():
    import time
    sim = G1Sim()
    h = AsyncSim(sim, None, AsyncConfig(gantry=True, realtime_factor=0.0)).start()   # as fast as possible
    try:
        time.sleep(0.3)
        assert h.ticks > 10 and h.time > 0.2
        st = h.state()
        assert set(["time", "q", "base_pos", "lfoot_force"]) <= set(st.keys())
        q0 = st["q"][3]                                   # left knee, settled under the hold command
        # external writer: bend the knees 0.1 rad more; the sim follows (PD has a gravity sag, so compare deltas)
        q_new = DEFAULT_ANGLES.copy(); q_new[3] += 0.1; q_new[9] += 0.1
        h.set_command(q_new)
        time.sleep(0.3)
        q1 = h.state()["q"][3]
        assert q1 - q0 > 0.05
        # attached controller takes over from inside the sim thread and brings it back
        h.attach(hold_pose())
        time.sleep(0.3)
        q2 = h.state()["q"][3]
        assert abs(q2 - q0) < 0.02
        assert h.state()["base_pos"][2] > 0.7              # still up on the gantry
    finally:
        h.stop()
    assert not h.running()


def test_sync_step_api():
    """RL-style use: reset once, then step() one control tick at a time."""
    ss = SyncSim(G1Sim(), hold_pose(), SyncConfig(gantry=True))
    st0 = ss.reset()
    st1 = ss.step()
    assert abs(st1["time"] - st0["time"] - ss.sim.control_dt) < 1e-12
    for _ in range(49):
        st = ss.step()
    assert abs(st["time"] - 1.0) < 1e-9 and ss.tick == 50


def test_async_log_records_every_tick():
    import time
    log = Recorder()
    h = AsyncSim(G1Sim(), hold_pose(), AsyncConfig(gantry=True, realtime_factor=0.0), log=log).start()
    time.sleep(0.3)
    h.stop()
    a = log.arrays()
    assert len(log) == h.ticks and a["q"].shape == (h.ticks, 29) and "band" in a
    assert np.allclose(np.diff(a["t"]), h.sim.control_dt)     # no gaps, no duplicates
