#!/usr/bin/env python3
"""Chained single-foot steps from changing start poses: segment = (recorded token slice, lambda, hold pose, settle).
Each segment starts from wherever the previous one ended (split stance), feeds the slice scaled about the
standing token, then holds the given split-stance token.  Default chain: R initiation -> L swing -> R swing -> L swing.

  .venv/bin/python experiments/step_chain_probe.py --lam 1.0 [--viewer] [--video out.mp4]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_stance_probe import split_pose  # noqa: E402
from step_stats import swings  # noqa: E402

from token_policy.env import StandingSnapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken, fsq
from token_policy.policy.fsq import k_to_z
from token_policy.sim import G1Sim, SyncConfig, SyncSim
from token_policy.sim.g1_params import mujoco_to_isaac
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference
from token_policy.sonic.reference import StandingReference
from token_policy.viz import colors
from token_policy.viz.draw import Canvas

DT = 0.02


class SegmentSource:
    """Token for the current tick: scaled recorded slice while it lasts, then the hold token (encoded once, live heading)."""

    def __init__(self, tokens, z_nom, ctrl):
        self.tokens, self.z_nom, self.ctrl = tokens, z_nom, ctrl
        self.z_stand, self.slice, self.scale, self.hold, self.k, self.z_hold = None, (0, 0), 1.0, None, 0, None

    def start(self, z_stand, a, b, scale, hold):
        self.z_stand, self.slice, self.scale, self.hold, self.k, self.z_hold = z_stand, (a, b), scale, hold, 0, None

    def reset(self):
        pass

    def __call__(self, state):
        a, b = self.slice
        k, self.k = self.k, self.k + 1
        if k < b - a:
            return fsq.snap(self.z_stand + self.scale * (self.tokens[a + k] - self.z_nom))
        if self.hold is None:
            return self.z_stand
        if self.z_hold is None:
            pol = self.ctrl.policy
            self.z_hold = fsq.snap(pol.encode(pol.encoder_obs(self.hold, state["base_quat"], self.ctrl.heading)))
        return self.z_hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="walking_quip_360_R_002__A428")
    ap.add_argument("--chain", default="R:0:94,L:94:128,R:128:157,L:157:185",
                    help="foot:slice_start:slice_end[:M] per segment (recording ticks; M = take the slice from the mirrored recording)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--split-deg", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--snapshot", default="results/standing_snapshot.npz")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    f = np.load(f"results/step_stats/{args.motion}.npz")
    f_m = np.load(f"results/step_stats/{args.motion}_M.npz")
    nominal = StandingSnapshot.load("results/standing_snapshot.npz")
    snap = StandingSnapshot.load(args.snapshot)
    sim = G1Sim()
    ss = SyncSim(sim, None, SyncConfig(gantry=False))
    ctrl = SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch"))
    source = SegmentSource(f["token"], k_to_z(nominal.k_stand), ctrl)
    tokens = {"": f["token"], "M": f_m["token"]}
    sonic = SonicJointController(ctrl, load_reference("standing"), token_hook=ReplaceToken(source))
    holds = {}
    for foot in ("R", "L"):
        q_mj, h, gap, _ = split_pose(sim, args.split_deg, foot)
        holds[foot] = StandingReference(f"split{foot}", mujoco_to_isaac(q_mj), np.array([1.0, 0, 0, 0]), h)
    snap.restore(sim, sonic)
    ss.attach(sonic); ss._state = sim.state(); ss.tick = 0
    st0 = ss.state(); yaw = st0["base_rpy"][2]; fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    p0, lf0, rf0 = st0["base_pos"].copy(), st0["lfoot"].copy(), st0["rfoot"].copy()
    viewer = snap_r = None; frames = []
    if args.viewer:
        from token_policy.sim.viewer import open_viewer
        viewer = open_viewer(sim); viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 90, -15
    if args.video:
        from token_policy.viz.snapshot import Snapshot
        snap_r = Snapshot(sim, 960, 720)
    ctx = {"label": ""}

    def draw(c, s_, _):
        c.marker(lf0, colors.GRAY, 0.01, 0.04, "L0"); c.marker(rf0, colors.GRAY, 0.01, 0.04, "R0")
        for d in (0.2, 0.4, 0.6, 0.8, 1.0):
            c.sphere(p0 - [0, 0, 0.755] + d * fwd, 0.006, colors.WHITE, f"{int(d*100)}")
        c.sphere(s_["rfoot"], 0.03, colors.RED); c.sphere(s_["lfoot"], 0.03, colors.BLUE)
        c.text(s_["base_pos"] + [0, 0, 0.25] + 0.6 * fwd, f"{ctx['label']}  R {np.dot(s_['rfoot']-rf0, fwd)*100:+.0f}  L {np.dot(s_['lfoot']-lf0, fwd)*100:+.0f}  body {np.dot(s_['base_pos']-p0, fwd)*100:+.0f} cm")

    t_wall = time.time(); tick = 0; fell = False
    segs = [s.split(":") for s in args.chain.split(",")]
    try:
        for i, seg in enumerate(segs):
            foot, a, b = seg[0], int(seg[1]), int(seg[2])
            source.tokens = tokens[seg[3] if len(seg) > 3 else ""]
            source.start(k_to_z(snap.k_stand), a, b, args.lam, holds[foot])
            st_seg = ss.state(); pre = {"L": st_seg["lfoot"].copy(), "R": st_seg["rfoot"].copy()}
            lforce, rforce = [], []
            for t in range(b - a + int(args.settle / DT)):
                ctx["label"] = f"seg {i+1} {foot} [{a}:{b}] {'tokens' if t < b - a else 'hold'}"
                st = ss.state()
                if viewer is not None:
                    if not viewer.is_running():
                        raise KeyboardInterrupt
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0; draw(Canvas(viewer.user_scn), st, None); viewer.cam.lookat[:] = st["base_pos"] - [0, 0, 0.2]
                    viewer.sync()
                    lag = t_wall + tick * DT - time.time()
                    if lag > 0:
                        time.sleep(lag)
                if snap_r is not None:
                    frames.append(snap_r.render([draw], {}, "right", lookat=st["base_pos"] - [0, 0, 0.25], distance=2.2))
                with (viewer.lock() if viewer is not None else _null()):
                    st = ss.step()
                tick += 1
                lforce.append(st["lfoot_force"]); rforce.append(st["rfoot_force"])
                if sim.fallen():
                    fell = True; break
            st = ss.state()
            n_sw = len(swings(np.array(lforce) < 20)) + len(swings(np.array(rforce) < 20))
            key, other = ("rfoot", "lfoot") if foot == "R" else ("lfoot", "rfoot")
            print(f"seg {i+1} {foot} [{a}:{b}] lam {args.lam}: {foot} moved {np.dot(st[key]-pre[foot], fwd)*100:+5.1f} cm, other {np.dot(st[other]-pre['L' if foot=='R' else 'R'], fwd)*100:+5.1f} cm, "
                  f"swings {n_sw}, feet gap ({foot} - other) {np.dot(st[key]-st[other], fwd)*100:+5.1f} cm, body total {np.dot(st['base_pos']-p0, fwd)*100:+5.1f} cm, "
                  f"end speed {np.linalg.norm(st['base_lin_vel']):.3f}{'  FELL' if fell else ''}", flush=True)
            if fell:
                break
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
