#!/usr/bin/env python3
"""Single STEP from standing by token replay: feed the tokens SONIC produced while tracking a walking
reference (results/step_stats/<motion>.npz, from play start) up to ``cut`` ticks after the touchdown of swing
--swing, then switch back to the standing token k_stand/16.  Does the robot take exactly one step and stop?

  .venv/bin/python experiments/single_step_probe.py walking_quip_360_R_002__A428 --swing 0 --cut 0 10 25
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from step_stats import heading_frame, swings  # noqa: E402

from token_policy.cli import parse_config, run
from token_policy.env import StandingSnapshot
from token_policy.policy import ReplaceToken
from token_policy.policy.fsq import k_to_z
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference

DT = 0.02


class WindowTokens:
    """source(state): recorded token while k_play <= tick < k_play + n_window, standing token otherwise."""

    def __init__(self, tokens, z_stand, k_play, n_window, scale=1.0):
        self.tokens, self.z_stand, self.k_play, self.n, self.scale = tokens, z_stand, k_play, n_window, scale

    def __call__(self, state):
        k = int(round(state["time"] / DT)) - self.k_play
        if not (0 <= k < self.n):
            return self.z_stand
        return self.z_stand + self.scale * (self.tokens[k] - self.z_stand)   # scale < 1: shrink the deviation from standing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--swing", type=int, default=0, help="index of the sim swing (ordered by liftoff) to end the window at")
    ap.add_argument("--cut", type=int, nargs="+", default=[0, 10, 25], help="ticks after that touchdown to switch to the standing token")
    ap.add_argument("--force", type=float, default=20.0)
    ap.add_argument("--scale", type=float, default=1.0, help="token deviation scale (1 = recorded)")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    f = np.load(f"results/step_stats/{args.motion}.npz")
    tokens, lf, rf = f["token"], f["lfoot"], f["rfoot"]
    sw = sorted([(lo, td, "L") for lo, td in swings(f["lforce"] < args.force)] + [(lo, td, "R") for lo, td in swings(f["rforce"] < args.force)])
    lo, td, foot = sw[args.swing]
    print(f"{args.motion}: swing {args.swing} = {foot} {lo}-{td} (replay ticks from play start); {len(sw)} swings in the recording")
    snap = StandingSnapshot.load("results/standing_snapshot.npz")
    z_stand = k_to_z(snap.k_stand)
    base = parse_config([], controller="sonic", reference="standing", headless=not args.viewer, no_realtime=not args.viewer, print_period=0)
    k_play = int(round((base.t_settle + base.t_hold) / DT))
    out = []
    for cut in args.cut:
        n = td + cut
        sonic = SonicController(SonicONNX(base.model_dir, decoder_backend="torch"))
        ctrl = SonicJointController(sonic, load_reference("standing"), token_hook=ReplaceToken(WindowTokens(tokens, z_stand, k_play, n, args.scale)))
        cfg = parse_config([], controller="sonic", reference="standing", headless=not args.viewer, no_realtime=not args.viewer,
                           print_period=0, duration=(k_play + n) * DT + 3.0)
        a = run(cfg, controller=ctrl)
        sl = slice(k_play, None)
        L, R = a["lfoot"][sl], a["rfoot"][sl]
        yaw0 = a["base_rpy"][k_play, 2]
        Rm = heading_frame(yaw0)
        sw_l = [(x, y, "L") for x, y in swings(a["lfoot_force"][sl] < args.force)]
        sw_r = [(x, y, "R") for x, y in swings(a["rfoot_force"][sl] < args.force)]
        all_sw = sorted(sw_l + sw_r)
        dL, dR = Rm @ (L[-1] - L[0]), Rm @ (R[-1] - R[0])
        v_end = np.linalg.norm(a["base_lin_vel"][-25:], axis=1).mean()
        fell = np.isfinite(a["fell_at"])
        r = {"cut": cut, "scale": args.scale, "window_ticks": n, "swings": [(x, y, s) for x, y, s in all_sw], "n_swings": len(all_sw),
             "dL_fwd_mm": dL[0] * 1e3, "dL_lat_mm": dL[1] * 1e3, "dR_fwd_mm": dR[0] * 1e3, "dR_lat_mm": dR[1] * 1e3,
             "final_feet_gap_fwd_mm": (Rm @ (R[-1] - L[-1]))[0] * 1e3, "base_speed_end": v_end, "fell": bool(fell),
             "height_end": float(a["base_pos"][-1, 2])}
        out.append(r)
        print(f"cut +{cut:2d} (window {n} ticks = {n*DT:.2f} s): swings {[(s, x, y) for x, y, s in all_sw]}  "
              f"L {dL[0]*1e3:+5.0f}/{dL[1]*1e3:+4.0f} mm  R {dR[0]*1e3:+5.0f}/{dR[1]*1e3:+4.0f} mm  feet gap fwd {r['final_feet_gap_fwd_mm']:+5.0f} mm  "
              f"end speed {v_end:.3f} m/s  h {r['height_end']:.3f}  {'FELL' if fell else ''}", flush=True)
    with open(f"results/step_stats/probe_{args.motion}_swing{args.swing}_s{args.scale}.json", "w") as fo:
        json.dump(out, fo, default=float)


if __name__ == "__main__":
    main()
