#!/usr/bin/env python3
"""First Language -> Motion dataset: STEP_FORWARD(d_cmd) counterfactual bundles on every snapshot.

Teacher = validated inverse lookup: d_cmd -> lambda (piecewise-linear inverse of the measured, grid-snapped
lambda -> dx table at cut 0) -> token window (walking recording from play start to the first touchdown, deviation
from the standing token scaled by lambda, snapped to the FSQ grid) followed by the standing token for --settle s.
Labels: the semantic magnitude is the COMMAND (d_cmd); the realized displacement is stored separately.
Action supervision = the executed token sequence (token_k on the FSQ grid, token_z = k / 16).

  .venv/bin/python experiments/make_step_dataset.py --out results/step_dataset
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_forward_sweep import WindowTokens, swings  # noqa: E402
from step_stats import heading_frame  # noqa: E402

from token_policy.env import StandingSnapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken
from token_policy.policy.fsq import k_to_z, z_to_k
from token_policy.sim import G1Sim, SyncConfig, SyncSim
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference

DT = 0.02
COMMANDS_M = (0.20, 0.25, 0.30, 0.35)
PARAPHRASES = [
    "Move forward by {cm} cm.",
    "Take a {cm} cm step forward.",
    "Advance forward {cm} centimeters.",
    "Walk forward {cm} cm and stop.",
    "Step forward {cm} centimeters.",
    "앞으로 {cm}cm 이동해.",
    "앞으로 {cm}센티미터 전진해.",
    "{cm}cm 앞으로 걸어가.",
]
STATE_KEYS = ["time", "q", "dq", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel",
              "lfoot", "rfoot", "lfoot_force", "rfoot_force"]


def lambda_table(sweep_paths, cut=0):
    """Measured mean dx (mm) per lambda from grid-snapped sweeps -> sorted (lambda, dx) arrays."""
    acc = {}
    for p in sweep_paths:
        for r in json.load(open(p)):
            if r["cut"] == cut:
                acc.setdefault(round(r["lam"], 3), []).append(r["dx_mm"])
    lams = np.array(sorted(acc)); dx = np.array([np.mean(acc[l]) for l in lams])
    return lams, dx


def inverse_lambda(d_mm, lams, dx):
    assert np.all(np.diff(dx) >= -1.0), "lambda -> dx table must be (near) monotonic"
    dx = np.maximum.accumulate(dx) + 1e-6 * np.arange(len(dx))   # ties (grid snapping plateaus) -> strictly increasing
    assert dx[0] - 15 <= d_mm <= dx[-1] + 15, f"{d_mm} mm outside the calibrated range {dx[0]:.0f}..{dx[-1]:.0f}"
    return float(np.interp(d_mm, dx, lams))


def rollout(sim, ss, sonic, snap, source, n, lam, settle_ticks):
    snap.restore(sim, sonic)
    source.set(k_to_z(snap.k_stand), n, lam)
    ss.attach(sonic); ss._state = sim.state(); ss.tick = 0
    st0 = ss.state()
    R = heading_frame(st0["base_rpy"][2])
    hist = {k: [] for k in STATE_KEYS + ["q_target"]}
    tokens = []
    fell = False
    for t in range(n + settle_ticks):
        st = ss.step()
        tokens.append(sonic.last["z"].copy())
        for k in STATE_KEYS:
            hist[k].append(st[k])
        hist["q_target"].append(sim.q_target.copy())
        if sim.fallen():
            fell = True
            break
    st = ss.state()
    d = R @ (st["base_pos"] - st0["base_pos"])
    lf, rf = np.array(hist["lfoot_force"]), np.array(hist["rfoot_force"])
    out = {k: np.array(v) for k, v in hist.items()}
    z = np.array(tokens); k = z_to_k(z)
    assert np.allclose(k_to_z(k), z), "executed token not on the FSQ grid"
    meta = {"realized_dx_m": float(d[0]), "realized_dy_m": float(d[1]),
            "realized_dyaw_rad": float(np.angle(np.exp(1j * (st["base_rpy"][2] - st0["base_rpy"][2])))),
            "n_swing": len(swings(lf < 20)) + len(swings(rf < 20)), "fell": fell, "n_ticks": len(z),
            "end_speed": float(np.mean(np.linalg.norm(out["base_lin_vel"][-25:], axis=1)))}
    return out, k.astype(np.int8), z.astype(np.float32), {k_: st0[k_] for k_ in STATE_KEYS}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="walking_quip_360_R_002__A428")
    ap.add_argument("--snapshots", default="results/snapshots")
    ap.add_argument("--nominal", default="results/standing_snapshot.npz")
    ap.add_argument("--calib-sweeps", nargs="+", default=["results/step_forward_snap_c0/sweep.json", "results/step_forward_snap/sweep.json"])
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--out", default="results/step_dataset")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "rollouts"), exist_ok=True)

    lams, dx = lambda_table(args.calib_sweeps)
    lam_cmd = {d: inverse_lambda(d * 1e3, lams, dx) for d in COMMANDS_M}
    with open(os.path.join(args.out, "calibration.json"), "w") as fo:
        json.dump({"cut": 0, "table_lambda": lams.tolist(), "table_dx_mm": dx.tolist(), "lambda_per_command_m": {str(d): l for d, l in lam_cmd.items()},
                   "sweeps": args.calib_sweeps}, fo, indent=1)
    print("lambda table (snapped, cut 0):", ", ".join(f"{l:.3f}->{x:.0f}" for l, x in zip(lams, dx)))
    print("lambda per command:", {d: round(l, 3) for d, l in lam_cmd.items()})

    f = np.load(f"results/step_stats/{args.motion}.npz")
    sw = sorted([(lo, td) for lo, td in swings(f["lforce"] < 20)] + [(lo, td) for lo, td in swings(f["rforce"] < 20)])
    n_window = sw[0][1]
    nominal = StandingSnapshot.load(args.nominal)
    source = WindowTokens(f["token"], k_to_z(nominal.k_stand), snap=True)
    sim = G1Sim()
    ss = SyncSim(sim, None, SyncConfig(gantry=False))
    sonic = SonicJointController(SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch")),
                                 load_reference("standing"), token_hook=ReplaceToken(source))
    paths = [args.nominal] + sorted(glob.glob(os.path.join(args.snapshots, "s0_*.npz")))
    settle_ticks = int(round(args.settle / DT))
    index = []
    with open(os.path.join(args.out, "index.jsonl"), "w") as idx:
        for path in paths:
            snap = StandingSnapshot.load(path)
            sid = os.path.basename(path).replace(".npz", "")
            for d in COMMANDS_M:
                cm = int(round(d * 100))
                hist, k, z, st0, meta = rollout(sim, ss, sonic, snap, source, n_window, lam_cmd[d], settle_ticks)
                fname = f"{sid}__d{cm}.npz"
                np.savez_compressed(os.path.join(args.out, "rollouts", fname), token_k=k, token_z=z,
                                    **{f"state_history/{kk}": v for kk, v in hist.items()},
                                    **{f"state_start/{kk}": np.asarray(v) for kk, v in st0.items()},
                                    k_stand=snap.k_stand, window_ticks=n_window, lam=lam_cmd[d], cut=0, command_distance_m=d)
                rec = {"file": f"rollouts/{fname}", "text": PARAPHRASES[0].format(cm=cm), "paraphrases": [p.format(cm=cm) for p in PARAPHRASES],
                       "command_distance_m": d, "snapshot_id": sid, "lambda": lam_cmd[d], "cut": 0, "window_ticks": n_window,
                       "motion": args.motion, **meta}
                index.append(rec)
                idx.write(json.dumps(rec, default=float) + "\n")
            s = [r for r in index if r["snapshot_id"] == sid]
            print(f"{sid}: " + "  ".join(f"{r['command_distance_m']*100:.0f}->{r['realized_dx_m']*100:5.1f}" for r in s)
                  + ("  FELL" if any(r["fell"] for r in s) else ""), flush=True)
    print("saved", os.path.join(args.out, "index.jsonl"), len(index), "rollouts")


if __name__ == "__main__":
    main()
