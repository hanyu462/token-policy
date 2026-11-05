#!/usr/bin/env python3
"""Measure the token-space Jacobian on a standing snapshot: hold delta_k on one dim, read the foot displacement.

For each effector (R, L), dim d and magnitude m: delta_k = m e_d held ``ticks`` ticks from the snapshot;
J[eff][:, d] = (disp(+4) - disp(-4)) / 8 in mm per grid step, rows (fwd, lat) in the task's local frame.

  .venv/bin/python experiments/measure_jacobian.py --out results/jacobian.npz [--snapshot results/standing_snapshot.npz]
"""
import argparse

import numpy as np

from token_policy.env import TaskConfig
from token_policy.env.foot_task_env import FootTaskEnv

MAGS = (-8, -4, -2, 2, 4, 8)


def sweep(env, effector, ticks):
    res = np.zeros((64, len(MAGS), 4))   # lat, fwd, stance drift (mm), fallen
    for d in range(64):
        for j, m in enumerate(MAGS):
            dk = np.zeros(64, int); dk[d] = m
            env.reset(seed=0, effector=effector, command=(0.0, 0.01))
            for _ in range(ticks):
                _, _, _, info = env.step(dk)
                if info["fallen"]:
                    break
            res[d, j] = (info["disp_local"][1] * 1e3, info["disp_local"][0] * 1e3, info["stance_drift"] * 1e3, info["fallen"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="results/standing_snapshot.npz")
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--out", default="results/jacobian.npz")
    args = ap.parse_args()
    cfg = TaskConfig(onnx_threads=4, snapshot_path=args.snapshot, dk_range=8, decoder_backend="torch", max_steps=args.ticks + 10)
    env = FootTaskEnv(cfg)
    out = {}
    ip, im = MAGS.index(4), MAGS.index(-4)
    for eff in ("R", "L"):
        res = sweep(env, eff, args.ticks)
        J = np.stack([(res[:, ip, 1] - res[:, im, 1]) / 8.0, (res[:, ip, 0] - res[:, im, 0]) / 8.0])   # (2, 64): fwd, lat
        out[f"J_{eff}"], out[f"sweep_{eff}"] = J, res
        top = np.argsort(-np.abs(J[1]))[:5]
        print(f"{eff}: |lat| per step max {np.abs(J[1]).max():.2f} mm (dims {top.tolist()}), |fwd| max {np.abs(J[0]).max():.2f} mm, "
              f"falls {int(res[:, :, 3].sum())}, drift@+-4 {res[:, [im, ip], 2].mean():.1f} mm", flush=True)
    np.savez(args.out, mags=np.array(MAGS), ticks=args.ticks, **out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
