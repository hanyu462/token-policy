#!/usr/bin/env python3
"""Build diverse standing initial states s_0: heading, joint-pose offset and a pelvis push during the hold phase.

  .venv/bin/python experiments/make_snapshots.py --n 30 --out results/snapshots
"""
import argparse
import json
import os

import numpy as np

from token_policy.env import build_standing_snapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.sim import G1Sim
from token_policy.sonic.controller import SonicController, SonicJointController
from token_policy.sonic.onnx_policy import SonicONNX
from token_policy.sonic.reference import load_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yaw", type=float, default=np.pi, help="heading ~ U(-yaw, yaw) rad")
    ap.add_argument("--q-std", type=float, default=0.05, help="joint pose offset std (rad) during the gantry phase")
    ap.add_argument("--push", type=float, default=40.0, help="pelvis push magnitude ~ U(0, push) N for 0.3 s at t in [2, 2.7] s")
    ap.add_argument("--hold", type=float, default=3.5, help="hold time before capture (s); jittered +-0.25")
    ap.add_argument("--out", default="results/snapshots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sim = G1Sim()
    sonic = SonicJointController(SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch")), load_reference("standing"))
    meta = []
    for i in range(args.n):
        yaw = float(rng.uniform(-args.yaw, args.yaw))
        q_off = rng.normal(0.0, args.q_std, 29)
        ang = rng.uniform(0, 2 * np.pi); mag = rng.uniform(0, args.push)
        push = (float(rng.uniform(2.0, 2.7)), 0.3, mag * np.cos(ang), mag * np.sin(ang), 0.0)
        t_hold = float(args.hold + rng.uniform(-0.25, 0.25))
        snap = build_standing_snapshot(sim, sonic, t_hold=t_hold, yaw=yaw, q_offset=q_off, push=push)
        st = sim.state()
        path = os.path.join(args.out, f"s0_{i:02d}.npz")
        snap.save(path)
        m = {"i": i, "path": path, "yaw_deg": np.degrees(yaw), "q_off_rms": float(np.sqrt(np.mean(q_off ** 2))),
             "push_N": mag, "push_dir_deg": np.degrees(ang), "t_hold": t_hold, "height": float(st["base_pos"][2]),
             "rpy_deg": np.degrees(st["base_rpy"]).tolist(), "fallen": bool(sim.fallen()),
             "feet_dy": float(np.linalg.norm(st["lfoot"][:2] - st["rfoot"][:2])), "k_stand": snap.k_stand.tolist()}
        meta.append(m)
        print(f"s0_{i:02d}: yaw {m['yaw_deg']:+6.1f} deg  push {mag:4.1f} N  h {m['height']:.3f}  rpy {np.round(m['rpy_deg'], 1)}  "
              f"feet {m['feet_dy']*1e3:.0f} mm  fallen {m['fallen']}", flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    ks = np.array([m["k_stand"] for m in meta])
    print("k_stand identical across snapshots:", bool((ks == ks[0]).all()), " falls:", sum(m["fallen"] for m in meta))


if __name__ == "__main__":
    main()
