#!/usr/bin/env python3
"""Watch one STEP_FORWARD(lambda, cut) replay on a chosen initial state: live viewer or mp4.

  .venv/bin/python experiments/step_forward_view.py --lam 0.8 --viewer
  MUJOCO_GL=egl .venv/bin/python experiments/step_forward_view.py --lam 0.6 0.8 1.0 --snapshot results/snapshots/s0_03.npz --video results/step_forward/view.mp4
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_forward_sweep import WindowTokens, swings  # noqa: E402

from token_policy.env import StandingSnapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken
from token_policy.policy.fsq import k_to_z
from token_policy.sim import G1Sim, SyncConfig, SyncSim
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference
from token_policy.viz import colors
from token_policy.viz.draw import Canvas

DT = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="walking_quip_360_R_002__A428")
    ap.add_argument("--snapshot", default="results/standing_snapshot.npz")
    ap.add_argument("--nominal", default="results/standing_snapshot.npz")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.8])
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--view", default="right", help="video camera: right | left | front | back | iso | top")
    ap.add_argument("--distance", type=float, default=2.2)
    ap.add_argument("--snap", action="store_true", help="grid-snapped tokens, as in the dataset")
    args = ap.parse_args()
    f = np.load(f"results/step_stats/{args.motion}.npz")
    sw = sorted([(lo, td) for lo, td in swings(f["lforce"] < 20)] + [(lo, td) for lo, td in swings(f["rforce"] < 20)])
    n = sw[0][1] + args.cut
    source = WindowTokens(f["token"], k_to_z(StandingSnapshot.load(args.nominal).k_stand), snap=args.snap)
    sim = G1Sim()
    ss = SyncSim(sim, None, SyncConfig(gantry=False))
    sonic = SonicJointController(SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch")),
                                 load_reference("standing"), token_hook=ReplaceToken(source))
    snap = StandingSnapshot.load(args.snapshot)
    ctx = {}

    def draw(c, st, _ctx):
        p0, fwd = ctx["p0"], ctx["fwd"]
        c.marker(p0 - [0, 0, 0.7], colors.GRAY, 0.015, 0.05, "start")
        for d in (0.1, 0.2, 0.3, 0.4):                                  # distance ticks along the start heading
            c.sphere(p0 - [0, 0, 0.75] + d * fwd, 0.008, colors.WHITE, f"{int(d*100)}")
        c.sphere(st["lfoot"], 0.03, colors.BLUE); c.sphere(st["rfoot"], 0.03, colors.RED)
        dx = float(np.dot(st["base_pos"] - p0, fwd))
        c.text(st["base_pos"] + [0, 0, 0.55] + 0.7 * fwd, f"lambda {ctx['lam']:.2f}  tick {ctx['k']}/{n}  {'TOKEN WINDOW' if ctx['k'] < n else 'STANDING TOKEN'}  dx {dx*100:+.1f} cm")

    viewer = snap_r = None
    frames = []
    if args.viewer:
        from token_policy.sim.viewer import open_viewer
        viewer = open_viewer(sim)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = args.distance, 90, -15
    if args.video:
        from token_policy.viz.snapshot import Snapshot
        snap_r = Snapshot(sim, 960, 720)
    try:
        for lam in args.lam:
            snap.restore(sim, sonic)
            source.set(k_to_z(snap.k_stand), n, lam)
            ss.attach(sonic); ss._state = sim.state(); ss.tick = 0
            st = ss.state()
            yaw = st["base_rpy"][2]
            ctx.update(p0=st["base_pos"].copy(), fwd=np.array([np.cos(yaw), np.sin(yaw), 0.0]), lam=lam, k=0)
            t0 = time.time()
            for k in range(n + int(args.settle / DT)):
                ctx["k"] = k
                st = ss.state()
                if viewer is not None:
                    if not viewer.is_running():
                        raise KeyboardInterrupt
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        draw(Canvas(viewer.user_scn), st, ctx)
                        viewer.cam.lookat[:] = st["base_pos"] - [0, 0, 0.2]
                    viewer.sync()
                    lag = t0 + k * DT - time.time()
                    if lag > 0:
                        time.sleep(lag)
                if snap_r is not None:
                    frames.append(snap_r.render([draw], ctx, args.view, lookat=st["base_pos"] - [0, 0, 0.2], distance=args.distance))
                with (viewer.lock() if viewer is not None else _null()):
                    ss.step()
            st = ss.state()
            print(f"lambda {lam:.2f}: dx {np.dot(st['base_pos'] - ctx['p0'], ctx['fwd'])*100:+.1f} cm  fell {sim.fallen()}")
            if snap_r is not None:
                frames.extend([frames[-1]] * 25)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            from token_policy.sim.viewer import close_viewer
            close_viewer(viewer)
    if snap_r is not None and frames:
        from token_policy.viz.snapshot import write_video
        print("video", write_video(args.video, frames, fps=50))


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
