#!/usr/bin/env python3
"""STEP without the catch-up: after the token window, hold a SPLIT-STANCE token instead of the standing token.

The hold token is the encoder output for a reference that holds one walking frame (double support right after
the first right-foot touchdown) forever: StandingReference(joint_pos, root_quat, root_height of that frame).
Measures right/left foot and body displacement, swings, stability.

  .venv/bin/python experiments/split_stance_probe.py --frames 96 98 100 --lam 0.699 [--snapshot ...] [--video out.mp4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_stats import swings  # noqa: E402

from token_policy.env import StandingSnapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken, fsq
from token_policy.policy.fsq import k_to_z
from token_policy.sim import G1Sim, SyncConfig, SyncSim
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference
from token_policy.sonic.reference import StandingReference

DT = 0.02


class WindowThenHold:
    """Scaled, snapped recorded tokens for n ticks, then the encoder token of ``hold_ref`` (computed once, live heading)."""

    def __init__(self, tokens, z_nom, ctrl):
        self.tokens, self.z_nom, self.ctrl = tokens, z_nom, ctrl

    def set(self, z_stand, n, scale, hold_ref):
        self.z_stand, self.n, self.scale, self.hold_ref, self.k, self.z_hold = z_stand, n, scale, hold_ref, 0, None

    def reset(self):
        self.k = 0

    def __call__(self, state):
        k, self.k = self.k, self.k + 1
        if k < self.n:
            return fsq.snap(self.z_stand + self.scale * (self.tokens[k] - self.z_nom))
        if self.hold_ref is None:
            return self.z_stand
        if self.z_hold is None:
            pol = self.ctrl.policy
            self.z_hold = fsq.snap(pol.encode(pol.encoder_obs(self.hold_ref, state["base_quat"], self.ctrl.heading)))
        return self.z_hold


def split_pose(sim, deg, foot="R"):
    """Static split stance from the standing pose: the stepping foot's hip pitch -theta (leg forward), the other
    +theta, ankles compensated so both soles stay flat; returns (joint_pos mujoco, pelvis height on the floor,
    fore-aft gap stepping-foot minus other, height difference)."""
    from token_policy.sim.g1_params import DEFAULT_ANGLES
    th = np.radians(deg)
    q = DEFAULT_ANGLES.copy()
    fwd_hip, fwd_ankle, back_hip, back_ankle = (6, 10, 0, 4) if foot == "R" else (0, 4, 6, 10)   # mujoco indices
    q[fwd_hip] -= th; q[fwd_ankle] += th
    q[back_hip] += th; q[back_ankle] -= th
    sim.reset(q)                     # places the lowest sole corner on the floor
    lf, rf = sim.foot_pos()
    a, b = (rf, lf) if foot == "R" else (lf, rf)
    return q, float(sim.base_pos()[2]), float(a[0] - b[0]), float(a[2] - b[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foot", default="R", choices=["R", "L"], help="stepping foot: R uses the walking recording, L its mirror")
    ap.add_argument("--motion", default=None)
    ap.add_argument("--frames", type=int, nargs="*", default=[])
    ap.add_argument("--blend", type=float, nargs="*", default=[], help="hold = blend * walking frame (first of --frames) + (1-blend) * standing pose")
    ap.add_argument("--split-deg", type=float, nargs="*", default=[], help="hold = designed split stance with this hip pitch offset (deg)")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.699])
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--snapshot", default="results/standing_snapshot.npz")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--no-standing", action="store_true", help="skip the standing-hold baseline runs")
    ap.add_argument("--video", default=None)
    ap.add_argument("--viewer", action="store_true", help="live mujoco.viewer window, real time")
    args = ap.parse_args()
    if args.motion is None:
        args.motion = "walking_quip_360_R_002__A428" + ("_M" if args.foot == "L" else "")
    f = np.load(f"results/step_stats/{args.motion}.npz")
    sw = sorted([(lo, td) for lo, td in swings(f["lforce"] < 20)] + [(lo, td) for lo, td in swings(f["rforce"] < 20)])
    n = sw[0][1] + args.cut
    ref = load_reference(args.motion)
    nominal = StandingSnapshot.load("results/standing_snapshot.npz")
    snap = StandingSnapshot.load(args.snapshot)
    sim = G1Sim()
    ss = SyncSim(sim, None, SyncConfig(gantry=False))
    ctrl = SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch"))
    source = WindowThenHold(f["token"], k_to_z(nominal.k_stand), ctrl)
    sonic = SonicJointController(ctrl, load_reference("standing"), token_hook=ReplaceToken(source))
    import time
    from token_policy.viz import colors
    from token_policy.viz.draw import Canvas
    snap_r, frames, viewer = None, [], None
    if args.video:
        from token_policy.viz.snapshot import Snapshot
        snap_r = Snapshot(sim, 960, 720)
    if args.viewer:
        from token_policy.sim.viewer import open_viewer
        viewer = open_viewer(sim)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 90, -15
    from token_policy.sim.g1_params import DEFAULT_ANGLES, mujoco_to_isaac
    q_std = mujoco_to_isaac(DEFAULT_ANGLES)
    holds = [] if args.no_standing else [("standing", None)]
    holds += [(f"frame {fr}", StandingReference(f"split{fr}", ref.joint_pos[fr].copy(), ref.root_quat_all[fr].copy(), float(ref.root_pos_all[fr, 2]))) for fr in args.frames]
    for b in args.blend:
        fr = args.frames[0] if args.frames else 98
        q = b * ref.joint_pos[fr] + (1 - b) * q_std
        holds.append((f"blend {b:.2f} of frame {fr}", StandingReference(f"blend{b}", q, np.array([1.0, 0, 0, 0]), float(b * ref.root_pos_all[fr, 2] + (1 - b) * 0.78874))))
    for deg in args.split_deg:
        q_mj, h, gap, dz = split_pose(sim, deg, args.foot)
        print(f"designed split {deg:.0f} deg: feet gap {gap*100:+.1f} cm, height diff {dz*100:+.1f} cm, pelvis h {h:.3f}")
        holds.append((f"split {deg:.0f} deg", StandingReference(f"split{deg}", mujoco_to_isaac(q_mj), np.array([1.0, 0, 0, 0]), h)))
    for lam in args.lam:
        for label, hold in holds:
            label = f"lam {lam:.2f} {label}"
            snap.restore(sim, sonic)
            source.set(k_to_z(snap.k_stand), n, lam, hold)
            ss.attach(sonic); ss._state = sim.state(); ss.tick = 0
            st0 = ss.state(); yaw = st0["base_rpy"][2]; fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            sk, ok = ("rfoot", "lfoot") if args.foot == "R" else ("lfoot", "rfoot")   # stepping / other foot keys
            rf0, lf0, p0 = st0[sk].copy(), st0[ok].copy(), st0["base_pos"].copy()
            lforce, rforce, speeds, fell = [], [], [], False
            t_wall = time.time()
            for t in range(n + int(args.settle / DT)):
                def draw(c, s_, _):
                    c.marker(rf0, colors.GRAY, 0.012, 0.05, f"{args.foot} start")
                    for d in (0.1, 0.2, 0.3, 0.4):
                        c.sphere(rf0 + d * fwd + [0, 0, 0.012], 0.006, colors.WHITE, f"{int(d*100)}")
                    c.sphere(s_["rfoot"], 0.03, colors.RED); c.sphere(s_["lfoot"], 0.03, colors.BLUE)
                    c.text(s_["base_pos"] + [0, 0, 0.25] + 0.6 * fwd, f"{label}  tick {t}/{n}  "
                           f"{args.foot} {np.dot(s_[sk]-rf0, fwd)*100:+.1f}  other {np.dot(s_[ok]-lf0, fwd)*100:+.1f}  body {np.dot(s_['base_pos']-p0, fwd)*100:+.1f} cm")
                st = ss.state()
                if snap_r is not None:
                    frames.append(snap_r.render([draw], {}, "right", lookat=st["base_pos"] - [0, 0, 0.25], distance=2.0))
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0; draw(Canvas(viewer.user_scn), st, None); viewer.cam.lookat[:] = st["base_pos"] - [0, 0, 0.2]
                    viewer.sync()
                    lag = t_wall + t * DT - time.time()
                    if lag > 0:
                        time.sleep(lag)
                    with viewer.lock():
                        st = ss.step()
                else:
                    st = ss.step()
                lforce.append(st["lfoot_force"]); rforce.append(st["rfoot_force"]); speeds.append(np.linalg.norm(st["base_lin_vel"]))
                if sim.fallen():
                    fell = True; break
            st = ss.state()
            n_sw = len(swings(np.array(lforce) < 20)) + len(swings(np.array(rforce) < 20))
            print(f"{label:34s}: {args.foot} {np.dot(st[sk]-rf0, fwd)*100:+5.1f}  other {np.dot(st[ok]-lf0, fwd)*100:+5.1f}  "
                  f"body {np.dot(st['base_pos']-p0, fwd)*100:+5.1f} cm  feet gap {np.dot(st[sk]-st[ok], fwd)*100:+5.1f} cm  swings {n_sw}  "
                  f"end speed {np.mean(speeds[-25:]):.3f}  h {st['base_pos'][2]:.3f}  {'FELL' if fell else ''}", flush=True)
            if snap_r is not None:
                frames.extend([frames[-1]] * 25)
    if viewer is not None:
        from token_policy.sim.viewer import close_viewer
        close_viewer(viewer)
    if snap_r is not None and frames:
        from token_policy.viz.snapshot import write_video
        print("video", write_video(args.video, frames, fps=50))


if __name__ == "__main__":
    main()
