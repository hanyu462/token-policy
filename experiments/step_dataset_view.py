#!/usr/bin/env python3
"""Replay a dataset rollout's stored tokens (token_z) on its snapshot: live viewer or mp4.

  .venv/bin/python experiments/step_dataset_view.py results/step_dataset/rollouts/standing_snapshot__d25.npz --viewer
  MUJOCO_GL=egl .venv/bin/python experiments/step_dataset_view.py results/step_dataset/rollouts/standing_snapshot__d25.npz --video out.mp4 --view right
"""
import argparse
import os
import time

import numpy as np

from token_policy.env import StandingSnapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken
from token_policy.sim import G1Sim, SyncConfig, SyncSim
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference
from token_policy.viz import colors
from token_policy.viz.draw import Canvas

DT = 0.02


class StoredTokens:
    def __init__(self, z):
        self.z, self.k = z, 0

    def reset(self):
        self.k = 0

    def __call__(self, state):
        k, self.k = self.k, self.k + 1
        return self.z[min(k, len(self.z) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout")
    ap.add_argument("--snapshots", default="results/snapshots")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--view", default="right")
    ap.add_argument("--distance", type=float, default=2.0)
    args = ap.parse_args()
    f = np.load(args.rollout)
    z, n_win, d_cmd = f["token_z"].astype(float), int(f["window_ticks"]), float(f["command_distance_m"])
    sid = os.path.basename(args.rollout).split("__")[0]
    snap_path = "results/standing_snapshot.npz" if sid == "standing_snapshot" else os.path.join(args.snapshots, sid + ".npz")
    snap = StandingSnapshot.load(snap_path)
    source = StoredTokens(z)
    sim = G1Sim()
    ss = SyncSim(sim, None, SyncConfig(gantry=False))
    sonic = SonicJointController(SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch")),
                                 load_reference("standing"), token_hook=ReplaceToken(source))
    snap.restore(sim, sonic); source.reset()
    ss.attach(sonic); ss._state = sim.state(); ss.tick = 0
    st0 = ss.state()
    yaw = st0["base_rpy"][2]; fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    p0, rf0, lf0 = st0["base_pos"].copy(), st0["rfoot"].copy(), st0["lfoot"].copy()
    ctx = {"k": 0}

    def draw(c, st, _):
        floor = np.array([0, 0, 0.012])
        c.marker(rf0, colors.GRAY, 0.012, 0.05, "R start")
        c.marker(rf0 + d_cmd * fwd, colors.GREEN, 0.012, 0.05, f"R +{d_cmd*100:.0f}")   # right foot start + commanded distance
        c.line(rf0 + floor, rf0 + d_cmd * fwd + floor, 3.0, colors.GREEN)
        for d in (0.1, 0.2, 0.3, 0.4):
            c.sphere(rf0 + d * fwd + floor, 0.006, colors.WHITE, f"{int(d*100)}")
        c.sphere(st["rfoot"], 0.03, colors.RED); c.sphere(st["lfoot"], 0.03, colors.BLUE)
        dR = float(np.dot(st["rfoot"] - rf0, fwd)); dL = float(np.dot(st["lfoot"] - lf0, fwd)); dx = float(np.dot(st["base_pos"] - p0, fwd))
        k = ctx["k"]
        c.text(st["base_pos"] + [0, 0, 0.25] + 0.6 * fwd,
               f"cmd {d_cmd*100:.0f} cm  tick {k}/{n_win} {'TOKEN WINDOW' if k < n_win else 'STANDING'}  R foot {dR*100:+.1f}  L foot {dL*100:+.1f}  body {dx*100:+.1f} cm")

    viewer = snap_r = None; frames = []
    if args.viewer:
        from token_policy.sim.viewer import open_viewer
        viewer = open_viewer(sim); viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = args.distance, 90, -15
    if args.video:
        from token_policy.viz.snapshot import Snapshot
        snap_r = Snapshot(sim, 960, 720)
    t0 = time.time()
    try:
        for k in range(len(z)):
            ctx["k"] = k; st = ss.state()
            if viewer is not None:
                if not viewer.is_running():
                    break
                with viewer.lock():
                    viewer.user_scn.ngeom = 0; draw(Canvas(viewer.user_scn), st, ctx); viewer.cam.lookat[:] = st["base_pos"] - [0, 0, 0.2]
                viewer.sync()
                lag = t0 + k * DT - time.time()
                if lag > 0:
                    time.sleep(lag)
            if snap_r is not None:
                frames.append(snap_r.render([draw], ctx, args.view, lookat=st["base_pos"] - [0, 0, 0.25], distance=args.distance))
            if viewer is not None:
                with viewer.lock():
                    ss.step()
            else:
                ss.step()
    finally:
        if viewer is not None:
            from token_policy.sim.viewer import close_viewer
            close_viewer(viewer)
    st = ss.state()
    print(f"{sid} cmd {d_cmd*100:.0f} cm: R foot {np.dot(st['rfoot']-rf0, fwd)*100:+.1f} cm, L foot {np.dot(st['lfoot']-lf0, fwd)*100:+.1f} cm, body {np.dot(st['base_pos']-p0, fwd)*100:+.1f} cm, fell {sim.fallen()}")
    if snap_r is not None and frames:
        from token_policy.viz.snapshot import write_video
        frames.extend([frames[-1]] * 25)
        print("video", write_video(args.video, frames, fps=50))


if __name__ == "__main__":
    main()
