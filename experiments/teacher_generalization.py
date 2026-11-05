#!/usr/bin/env python3
"""Nominal-Jacobian teacher on diverse initial states: generalization + foot trajectory quality.

For every snapshot in --snapshots and every command (effector R/L x lateral/forward x +-1, +-2 cm) run the
JacobianTeacher (J measured once on the nominal snapshot, never re-measured) and record:
  success, final error, fall, time to success, stance-foot drift,
  swing clearance   = max rise of the moving foot (mm),
  lift fraction     = ticks with the moving foot unloaded (< contact_force) / ticks,
  drag              = horizontal path of the moving foot while loaded (mm; "dragging" if ~= displacement).

  .venv/bin/python experiments/teacher_generalization.py --snapshots results/snapshots --out results/teacher_gen
"""
import argparse
import glob
import json
import os

import numpy as np

from token_policy.env import StandingSnapshot, TaskConfig
from token_policy.env.foot_task_env import FootTaskEnv
from token_policy.policy import JacobianTeacher


def commands(mags_cm=(1.0, 2.0)):
    out = []
    for eff in ("R", "L"):
        for axis in ("lat", "fwd"):
            for m in mags_cm:
                for sgn in (+1, -1):
                    d = sgn * m * 1e-2
                    out.append((eff, axis, sgn * m, (d, 0.0) if axis == "fwd" else (0.0, d)))
    return out


def run_episode(env: FootTaskEnv, teacher: JacobianTeacher, eff, cmd):
    env.reset(seed=0, effector=eff, command=cmd)
    teacher.reset()
    st0 = env.state()
    key = env._eff_key()
    z0, p_prev = st0[key][2], st0[key][:2].copy()
    clearance, drag, unloaded, n = 0.0, 0.0, 0, 0
    info = {"error_local": env.frame.to_local(env.p_target) - env.frame.to_local(env.p_eff0)}
    done = False
    while not done:
        dk = teacher(eff, info["error_local"])
        _, _, done, info = env.step(dk)
        st = env.state()
        p, f = st[key], st[key + "_force"]
        clearance = max(clearance, p[2] - z0)
        loaded = f > env.cfg.contact_force
        if loaded:
            drag += float(np.linalg.norm(p[:2] - p_prev))
        else:
            unloaded += 1
        p_prev, n = p[:2].copy(), n + 1
    return {"success": info["success"], "fallen": info["fallen"], "dist_mm": info["dist"] * 1e3, "t": info["t"],
            "t_success": env.success_tick if env.success_tick is not None else -1,
            "disp_lat_mm": info["disp_local"][1] * 1e3, "disp_fwd_mm": info["disp_local"][0] * 1e3,
            "stance_drift_mm": info["stance_drift"] * 1e3, "clearance_mm": clearance * 1e3,
            "lift_frac": unloaded / max(n, 1), "drag_mm": drag * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="results/snapshots")
    ap.add_argument("--jacobian", default="results/jacobian.npz")
    ap.add_argument("--nominal", default="results/standing_snapshot.npz")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--hold", type=int, default=25)
    ap.add_argument("--out", default="results/teacher_gen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = TaskConfig(onnx_threads=4, snapshot_path=args.nominal, dk_range=2, decoder_backend="torch", max_steps=300)
    env = FootTaskEnv(cfg)
    teacher = JacobianTeacher.from_file(args.jacobian, gain=args.gain, hold=args.hold, dk_range=cfg.dk_range)
    meta = {m["path"]: m for m in json.load(open(os.path.join(args.snapshots, "meta.json")))} if os.path.exists(os.path.join(args.snapshots, "meta.json")) else {}
    paths = [args.nominal] + sorted(glob.glob(os.path.join(args.snapshots, "s0_*.npz")))
    rows = []
    for path in paths:
        if meta.get(path, {}).get("fallen"):
            continue
        env.set_snapshot(StandingSnapshot.load(path))
        name = os.path.basename(path).replace(".npz", "")
        for eff, axis, mag, cmd in commands():
            r = run_episode(env, teacher, eff, cmd)
            r.update(snapshot=name, effector=eff, axis=axis, mag_cm=mag, yaw_deg=meta.get(path, {}).get("yaw_deg", 0.0))
            rows.append(r)
        s = [r for r in rows if r["snapshot"] == name]
        print(f"{name}: success {np.mean([r['success'] for r in s]):.2f}  fall {np.mean([r['fallen'] for r in s]):.2f}  "
              f"err {np.mean([r['dist_mm'] for r in s]):5.1f} mm  drift {np.mean([r['stance_drift_mm'] for r in s]):5.1f}  "
              f"clear {np.mean([r['clearance_mm'] for r in s]):4.1f}  drag {np.mean([r['drag_mm'] for r in s]):5.1f} mm", flush=True)
    with open(os.path.join(args.out, "episodes.json"), "w") as f:
        json.dump(rows, f, indent=0, default=float)

    def summ(sel, label):
        if not sel:
            return
        print(f"{label:<18s} n {len(sel):3d}  success {np.mean([r['success'] for r in sel]):.2f}  fall {np.mean([r['fallen'] for r in sel]):.2f}  "
              f"err {np.mean([r['dist_mm'] for r in sel]):5.1f} mm  t_succ {np.mean([r['t_success'] for r in sel if r['t_success'] > 0] or [np.nan]) * 0.02:4.2f} s  "
              f"drift {np.mean([r['stance_drift_mm'] for r in sel]):5.1f}  clear {np.mean([r['clearance_mm'] for r in sel]):4.1f}  "
              f"lift {np.mean([r['lift_frac'] for r in sel]):.2f}  drag {np.mean([r['drag_mm'] for r in sel]):5.1f} mm")
    print("\n== by command ==")
    for eff in ("R", "L"):
        for axis in ("lat", "fwd"):
            for mag in (1.0, 2.0):
                summ([r for r in rows if r["effector"] == eff and r["axis"] == axis and abs(r["mag_cm"]) == mag], f"{eff} {axis} +-{mag:.0f} cm")
    print("\n== nominal vs perturbed ==")
    summ([r for r in rows if r["snapshot"] == "standing_snapshot"], "nominal")
    summ([r for r in rows if r["snapshot"] != "standing_snapshot"], "perturbed")
    print("\n== by |yaw| ==")
    for lo, hi in ((0, 30), (30, 90), (90, 181)):
        summ([r for r in rows if r["snapshot"] != "standing_snapshot" and lo <= abs(r["yaw_deg"]) < hi], f"|yaw| {lo}-{hi}")


if __name__ == "__main__":
    main()
