#!/usr/bin/env python3
"""Dataset V2: STEP_FORWARD closed-loop teacher T(s_0, d) on every snapshot x command; only the final rollout is saved.
Teacher internals (lambda, iterations) are metadata only; the Student never sees lambda.

  .venv/bin/python experiments/make_step_dataset_v2.py --out results/step_dataset_v2
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_stats import swings  # noqa: E402

from token_policy.env import StandingSnapshot
from token_policy.teacher import StepForwardCalibration, StepForwardTeacher

COMMANDS_M = (0.20, 0.25, 0.30, 0.35)
PARAPHRASES = ["Move forward by {cm} cm.", "Take a {cm} cm step forward.", "Advance forward {cm} centimeters.",
               "Walk forward {cm} cm and stop.", "Step forward {cm} centimeters.",
               "앞으로 {cm}cm 이동해.", "앞으로 {cm}센티미터 전진해.", "{cm}cm 앞으로 걸어가."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="walking_quip_360_R_002__A428")
    ap.add_argument("--snapshots", default="results/snapshots")
    ap.add_argument("--nominal", default="results/standing_snapshot.npz")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--no-cut-fallback", action="store_true")
    ap.add_argument("--out", default="results/step_dataset_v2")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "rollouts"), exist_ok=True)
    f = np.load(f"results/step_stats/{args.motion}.npz")
    sw = sorted([(lo, td) for lo, td in swings(f["lforce"] < 20)] + [(lo, td) for lo, td in swings(f["rforce"] < 20)])
    n_window = sw[0][1]
    nominal = StandingSnapshot.load(args.nominal)
    teacher = StepForwardTeacher(f["token"], n_window, nominal.k_stand, StepForwardCalibration(), tol=args.tol, max_iters=args.max_iters,
                                 eta=args.eta, cut_fallback=() if args.no_cut_fallback else (-4, 4))
    paths = [args.nominal] + sorted(glob.glob(os.path.join(args.snapshots, "s0_*.npz")))
    index = []
    with open(os.path.join(args.out, "index.jsonl"), "w") as idx:
        for path in paths:
            snap = StandingSnapshot.load(path)
            sid = os.path.basename(path).replace(".npz", "")
            line = []
            for d in COMMANDS_M:
                r = teacher.generate(snap, d)
                cm = int(round(d * 100))
                fname = f"{sid}__d{cm}.npz"
                np.savez_compressed(os.path.join(args.out, "rollouts", fname), token_k=r["token_k"], token_z=r["token_z"],
                                    **{f"state_history/{k}": v for k, v in r["state_history"].items()},
                                    **{f"state_start/{k}": v for k, v in r["state_start"].items()},
                                    k_stand=snap.k_stand, window_ticks=r["window_ticks"], command_distance_m=d)   # no lambda / cut in the rollout file
                rec = {"file": f"rollouts/{fname}", "text": PARAPHRASES[0].format(cm=cm), "paraphrases": [p.format(cm=cm) for p in PARAPHRASES],
                       "command_distance_m": d, "snapshot_id": sid, "motion": args.motion, "window_ticks": r["window_ticks"],
                       "realized_dx_m": r["realized_dx_m"], "realized_dy_m": r["realized_dy_m"], "realized_dyaw_rad": r["realized_dyaw_rad"],
                       "initial_lambda": r["initial_lambda"], "final_lambda": r["final_lambda"], "final_cut": r["final_cut"],
                       "teacher_iterations": r["teacher_iterations"], "search_history": r["search_history"],
                       "teacher_converged": r["teacher_converged"], "n_swing": r["n_swing"], "fell": r["fell"],
                       "n_ticks": int(len(r["token_k"])), "end_speed": r["end_speed"]}
                index.append(rec)
                idx.write(json.dumps(rec, default=float) + "\n")
                line.append(f"{cm}->{r['realized_dx_m']*100:4.1f}({r['teacher_iterations']}it,l{r['final_lambda']:.2f},c{r['final_cut']:+d}{'' if r['teacher_converged'] else '!'})")
            print(f"{sid}: " + "  ".join(line), flush=True)
    err = np.array([abs(r["realized_dx_m"] - r["command_distance_m"]) for r in index]) * 100
    its = np.array([r["teacher_iterations"] for r in index])
    print(f"\n{len(index)} rollouts: |error| mean {err.mean():.2f} cm (median {np.median(err):.2f}, max {err.max():.2f}), converged {np.mean([r['teacher_converged'] for r in index]):.2f}, "
          f"rollouts per sample mean {its.mean():.2f} (1:{(its==1).sum()} 2:{(its==2).sum()} 3:{(its==3).sum()} 4:{(its==4).sum()} 5:{(its==5).sum()}), "
          f"cut used {sum(r['final_cut'] != 0 for r in index)}, falls {sum(r['fell'] for r in index)}")
    print("saved", os.path.join(args.out, "index.jsonl"))


if __name__ == "__main__":
    main()
