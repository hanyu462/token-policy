#!/usr/bin/env python3
"""STEP_FORWARD primitive by token-window replay, swept over initial states s_0, deviation scale lambda and cut.

Token window: the tokens SONIC produced while tracking a walking reference from play start up to
(touchdown of swing --swing) + cut ticks, with the deviation from the standing token scaled by lambda:

    z_t = z_stand(s_0) + lambda * (z_rec,t - z_stand,nominal),   then z_stand(s_0) for --settle seconds.

Per (s_0, lambda, cut): base displacement (heading frame of s_0), yaw change, foot displacements, swing count,
max foot clearance, final base speed / height, fall.

  .venv/bin/python experiments/step_forward_sweep.py --lambdas 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --cuts -8 0 10 --out results/step_forward
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_stats import heading_frame, swings  # noqa: E402

from token_policy.env import StandingSnapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken
from token_policy.policy import fsq
from token_policy.policy.fsq import k_to_z
from token_policy.sim import G1Sim, SyncConfig, SyncSim
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference

DT = 0.02


class WindowTokens:
    """source(state): scaled recorded token for the first n calls after reset, the standing token afterwards."""

    def __init__(self, tokens, z_stand_nominal, snap=False):
        """``snap``: round the scaled token onto the FSQ grid (k/16) so the executed token is a valid code."""
        self.tokens, self.z_nom, self.snap = tokens, z_stand_nominal, snap
        self.set(np.zeros(64), 0, 1.0)

    def set(self, z_stand, n, scale):
        self.z_stand, self.n, self.scale, self.k = z_stand, n, scale, 0

    def reset(self):
        self.k = 0

    def __call__(self, state):
        k, self.k = self.k, self.k + 1
        z = self.z_stand + self.scale * (self.tokens[k] - self.z_nom) if k < self.n else self.z_stand
        return fsq.snap(z) if self.snap else z


def episode(sim, ss, sonic, snap, source, n, scale, settle_ticks, force):
    snap.restore(sim, sonic)
    source.set(k_to_z(snap.k_stand), n, scale)
    ss.attach(sonic)
    ss._state = sim.state()
    ss.tick = 0
    st0 = ss.state()
    R = heading_frame(st0["base_rpy"][2])
    L0, R0, p0, yaw0 = st0["lfoot"].copy(), st0["rfoot"].copy(), st0["base_pos"].copy(), st0["base_rpy"][2]
    zl0, zr0 = L0[2], R0[2]
    lforce, rforce, speeds, hmax, fell = [], [], [], 0.0, False
    for t in range(n + settle_ticks):
        st = ss.step()
        lforce.append(st["lfoot_force"]); rforce.append(st["rfoot_force"])
        speeds.append(np.linalg.norm(st["base_lin_vel"]))
        hmax = max(hmax, st["lfoot"][2] - zl0, st["rfoot"][2] - zr0)
        if sim.fallen():
            fell = True
            break
    st = ss.state()
    d_base = R @ (st["base_pos"] - p0)
    dL, dR = R @ (st["lfoot"] - L0), R @ (st["rfoot"] - R0)
    n_sw = len(swings(np.array(lforce) < force)) + len(swings(np.array(rforce) < force))
    return {"dx_mm": d_base[0] * 1e3, "dy_mm": d_base[1] * 1e3,
            "dyaw_deg": float(np.degrees(np.angle(np.exp(1j * (st["base_rpy"][2] - yaw0))))),
            "dL_fwd_mm": dL[0] * 1e3, "dL_lat_mm": dL[1] * 1e3, "dR_fwd_mm": dR[0] * 1e3, "dR_lat_mm": dR[1] * 1e3,
            "feet_gap_fwd_mm": (R @ (st["rfoot"] - st["lfoot"]))[0] * 1e3, "n_swing": n_sw, "hmax_mm": hmax * 1e3,
            "end_speed": float(np.mean(speeds[-25:])) if speeds else np.nan, "end_height": float(st["base_pos"][2]), "fell": fell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="walking_quip_360_R_002__A428")
    ap.add_argument("--swing", type=int, default=0)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--cuts", type=int, nargs="+", default=[-8, 0, 10])
    ap.add_argument("--snapshots", default="results/snapshots")
    ap.add_argument("--nominal", default="results/standing_snapshot.npz")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--force", type=float, default=20.0)
    ap.add_argument("--out", default="results/step_forward")
    ap.add_argument("--snap", action="store_true", help="execute grid-snapped tokens")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    f = np.load(f"results/step_stats/{args.motion}.npz")
    tokens = f["token"]
    sw = sorted([(lo, td) for lo, td in swings(f["lforce"] < args.force)] + [(lo, td) for lo, td in swings(f["rforce"] < args.force)])
    td = sw[args.swing][1]
    nominal = StandingSnapshot.load(args.nominal)
    source = WindowTokens(tokens, k_to_z(nominal.k_stand), snap=args.snap)
    sim = G1Sim()
    ss = SyncSim(sim, None, SyncConfig(gantry=False))
    sonic = SonicJointController(SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch")),
                                 load_reference("standing"), token_hook=ReplaceToken(source))
    paths = [args.nominal] + sorted(glob.glob(os.path.join(args.snapshots, "s0_*.npz")))
    meta = {m["path"]: m for m in json.load(open(os.path.join(args.snapshots, "meta.json")))} if os.path.exists(os.path.join(args.snapshots, "meta.json")) else {}
    settle_ticks = int(round(args.settle / DT))
    rows = []
    for path in paths:
        snap = StandingSnapshot.load(path)
        name = os.path.basename(path).replace(".npz", "")
        for cut in args.cuts:
            for lam in args.lambdas:
                r = episode(sim, ss, sonic, snap, source, td + cut, lam, settle_ticks, args.force)
                r.update(snapshot=name, yaw_deg=meta.get(path, {}).get("yaw_deg", 0.0), cut=cut, lam=lam, window_ticks=td + cut)
                rows.append(r)
        s = [r for r in rows if r["snapshot"] == name]
        print(f"{name}: " + "  ".join(f"l{r['lam']:.1f}/c{r['cut']:+d}:{r['dx_mm']:+4.0f}{'F' if r['fell'] else ''}" for r in s if r["cut"] == 0), flush=True)
    with open(os.path.join(args.out, "sweep.json"), "w") as fo:
        json.dump(rows, fo, default=float)

    print("\n== dx_base (mm) over snapshots: mean +- sd [min..max] | swings (mode) | fall | end speed ==")
    for cut in args.cuts:
        print(f"-- cut {cut:+d} (window {td + cut} ticks = {(td + cut) * DT:.2f} s)")
        for lam in args.lambdas:
            s = [r for r in rows if r["cut"] == cut and r["lam"] == lam]
            dx = np.array([r["dx_mm"] for r in s]); ns = np.array([r["n_swing"] for r in s])
            print(f"   lambda {lam:.2f}: dx {dx.mean():+6.0f} +- {dx.std():4.0f} [{dx.min():+5.0f}..{dx.max():+5.0f}]  dy {np.mean([r['dy_mm'] for r in s]):+5.0f}  "
                  f"dyaw {np.mean([r['dyaw_deg'] for r in s]):+5.1f} deg  swings {np.bincount(ns).argmax()} ({ns.min()}-{ns.max()})  "
                  f"hmax {np.mean([r['hmax_mm'] for r in s]):4.0f} mm  fall {np.mean([r['fell'] for r in s]):.2f}  "
                  f"end speed {np.mean([r['end_speed'] for r in s]):.3f} m/s  gap {np.mean([r['feet_gap_fwd_mm'] for r in s]):+4.0f} mm")
    print("saved", os.path.join(args.out, "sweep.json"))


if __name__ == "__main__":
    main()
