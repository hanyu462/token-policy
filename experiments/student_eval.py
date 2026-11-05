#!/usr/bin/env python3
"""Closed-loop evaluation: Student tokens -> frozen SONIC -> MuJoCo, from Dataset V2 snapshots.

  .venv/bin/python experiments/student_eval.py results/student/run_small/best.pt --bucket heldout-seen
  --bucket heldout-unseen30 | text-sanity | ablation ; --snapshots a,b,c ; --distances 0.2,0.35
Each rollout: 244 ticks (same as the teacher); metrics as in the teacher (dx, dy, dyaw, fall, swings, end speed, foot gap).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_eval_offline import load_model  # noqa: E402

from token_policy.env import StandingSnapshot
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken
from token_policy.sim import G1Sim, SyncConfig, SyncSim
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference
from token_policy.student.dataset import Split
from token_policy.student.features import NormStats
from token_policy.student.policy import StudentTokenSource
from token_policy.student.text_encoder import TextCache
from token_policy.student.texts import TRAIN_TEMPLATES, render
from token_policy.teacher.step_forward_teacher import _heading_frame, _swings

DT = 0.02


class Runner:
    def __init__(self, model, norm, cache, dev, snapshots_dir="results/snapshots", nominal="results/standing_snapshot.npz", ticks=244):
        self.sim = G1Sim()
        self.sync = SyncSim(self.sim, None, SyncConfig(gantry=False))
        self.source = StudentTokenSource(model, norm, cache, "", np.zeros(64, int), dev)
        self.sonic = SonicJointController(SonicController(SonicONNX(SONIC_MODEL_DIR, threads=4, decoder_backend="torch")),
                                          load_reference("standing"), token_hook=ReplaceToken(self.source))
        self.snapshots_dir, self.nominal, self.ticks = snapshots_dir, nominal, ticks
        self._snap = {}

    def snapshot(self, sid):
        if sid not in self._snap:
            self._snap[sid] = StandingSnapshot.load(self.nominal if sid == "standing_snapshot" else os.path.join(self.snapshots_dir, sid + ".npz"))
        return self._snap[sid]

    def run(self, sid: str, text: str) -> dict:
        snap = self.snapshot(sid)
        snap.restore(self.sim, self.sonic)
        self.source.k_stand = snap.k_stand.astype(int); self.source.set_text(text); self.source.reset()
        self.sync.attach(self.sonic); self.sync._state = self.sim.state(); self.sync.tick = 0
        st0 = self.sync.state(); R = _heading_frame(st0["base_rpy"][2])
        lf, rf, speeds, fell = [], [], [], False
        for _ in range(self.ticks):
            st = self.sync.step()
            lf.append(st["lfoot_force"]); rf.append(st["rfoot_force"]); speeds.append(np.linalg.norm(st["base_lin_vel"]))
            if self.sim.fallen():
                fell = True; break
        st = self.sync.state()
        d = R @ (st["base_pos"] - st0["base_pos"])
        k = np.array(self.source.k_pred)
        return {"snapshot_id": sid, "text": text, "dx": float(d[0]), "dy": float(d[1]),
                "dyaw_deg": float(np.degrees(np.angle(np.exp(1j * (st["base_rpy"][2] - st0["base_rpy"][2]))))),
                "fell": fell, "n_swing": _swings(np.array(lf) < 20) + _swings(np.array(rf) < 20),
                "end_speed": float(np.mean(speeds[-25:])), "foot_gap": float((R @ (st["rfoot"] - st["lfoot"]))[0]),
                "n_ticks": len(k), "k_pred": k}


def teacher_table(data):
    rows = [json.loads(l) for l in open(os.path.join(data, "index.jsonl"))]
    return {(r["snapshot_id"], round(r["command_distance_m"], 3)): r for r in rows}


def summarize(label, rows):
    if not rows:
        return
    g = lambda k: np.array([r[k] for r in rows], float)   # noqa: E731
    print(f"  {label:36s} n {len(rows):2d}  E_task {g('e_task').mean()*100:5.2f} cm (max {g('e_task').max()*100:4.1f})  E_teacher {g('e_teacher').mean()*100:5.2f}  "
          f"E_imit {g('e_imit').mean()*100:5.2f}  d_student {g('dx').mean()*100:5.1f}  dy {g('dy').mean()*100:+4.1f}  dyaw {g('dyaw_deg').mean():+4.1f}  "
          f"fall {g('fell').mean():.2f}  swings {g('n_swing').mean():.1f}  end v {g('end_speed').mean():.3f}  gap {g('foot_gap').mean()*100:+4.1f} cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--bucket", default="heldout-seen", choices=["heldout-seen", "heldout-unseen30", "train-seen", "text-sanity", "ablation"])
    ap.add_argument("--snapshots", default=None, help="comma list; default from the bucket")
    ap.add_argument("--distances", default=None, help="comma list in m")
    ap.add_argument("--data", default="results/step_dataset_v2")
    ap.add_argument("--work", default="results/student")
    ap.add_argument("--template", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    model, _ = load_model(args.ckpt, dev)
    split = Split.load(os.path.join(args.work, "split.json"))
    norm = NormStats.load(os.path.join(args.work, "norm_stats.npz"))
    cache = TextCache(os.path.join(args.work, "text_cache.pt"))
    runner = Runner(model, norm, cache, dev)
    teacher = teacher_table(args.data)
    snaps = args.snapshots.split(",") if args.snapshots else (split.train if args.bucket == "train-seen" else split.test)
    if args.distances:
        dists = [float(x) for x in args.distances.split(",")]
    else:
        dists = [0.30] if args.bucket == "heldout-unseen30" else ([0.20, 0.35] if args.bucket in ("text-sanity", "ablation") else [0.20, 0.25, 0.35])
    tmpl = TRAIN_TEMPLATES[args.template]
    rows = []
    print(f"bucket {args.bucket}: {len(snaps)} snapshots x {dists} m")
    for sid in snaps:
        for d in dists:
            variants = [("correct", render(tmpl, d))]
            if args.bucket == "ablation":
                variants += [("shuffled", render(tmpl, 0.35 if d < 0.3 else 0.20)), ("blank", "")]
            for variant, text in variants:
                r = runner.run(sid, text)
                t = teacher[(sid, round(d, 3))]
                r.update(variant=variant, d_cmd=d, d_teacher=t["realized_dx_m"], teacher_converged=t["teacher_converged"],
                         e_task=abs(r["dx"] - d), e_teacher=abs(t["realized_dx_m"] - d), e_imit=abs(r["dx"] - t["realized_dx_m"]),
                         k_diff_vs_teacher=float((r.pop("k_pred")[:94] != np.load(os.path.join(args.data, t["file"]))["token_k"][:94]).mean()))
                rows.append(r)
                print(f"  {sid:18s} cmd {d*100:4.1f} {variant:8s}: student {r['dx']*100:5.1f} cm (teacher {t['realized_dx_m']*100:5.1f})  dy {r['dy']*100:+4.1f}  yaw {r['dyaw_deg']:+5.1f}  "
                      f"swings {r['n_swing']}  end v {r['end_speed']:.3f}  gap {r['foot_gap']*100:+4.1f}  k-diff {r['k_diff_vs_teacher']*100:4.1f}%{'  FELL' if r['fell'] else ''}", flush=True)
    print("\n== summary ==")
    for variant in sorted({r["variant"] for r in rows}):
        for d in dists:
            summarize(f"{variant} x {d*100:.0f} cm", [r for r in rows if r["variant"] == variant and r["d_cmd"] == d])
        summarize(f"{variant} x all", [r for r in rows if r["variant"] == variant])
    for conv in (True, False):
        summarize(f"correct | teacher_converged={conv}", [r for r in rows if r["variant"] == "correct" and r["teacher_converged"] == conv])
    if args.bucket in ("text-sanity", "ablation"):
        for sid in snaps:
            a = [r for r in rows if r["snapshot_id"] == sid and r["variant"] == "correct"]
            if len(a) == 2:
                print(f"  {sid}: dx(35) - dx(20) = {(a[1]['dx'] - a[0]['dx'])*100:+.1f} cm")
    out = os.path.join(os.path.dirname(args.ckpt), f"closed_loop_{args.bucket}.json")
    json.dump(rows, open(out, "w"), indent=1, default=float)
    print("saved", out)


if __name__ == "__main__":
    main()
