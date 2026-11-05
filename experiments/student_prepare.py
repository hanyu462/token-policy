#!/usr/bin/env python3
"""Split + normalization for Student S0 from Dataset V2 (30 cm excluded from train/val everywhere).

  .venv/bin/python experiments/student_prepare.py --data results/step_dataset_v2 --out results/student
"""
import argparse
import json
import os

import numpy as np

from token_policy.student.dataset import TEST_DISTANCE, TRAIN_DISTANCES, Split, StepDataset, fit_norm, load_rollouts
from token_policy.student.texts import TRAIN_TEMPLATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/step_dataset_v2")
    ap.add_argument("--out", default="results/student")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.data, "index.jsonl"))]
    snaps = sorted({r["snapshot_id"] for r in rows}, key=lambda s: (s != "standing_snapshot", s))
    split = Split.default(snaps)
    split.save(os.path.join(args.out, "split.json"))
    train = load_rollouts(args.data, split.train, TRAIN_DISTANCES)
    norm = fit_norm(train)
    norm.save(os.path.join(args.out, "norm_stats.npz"))
    src = json.loads(norm.source)
    assert TEST_DISTANCE not in src["distances"] and not set(src["snapshots"]) & set(split.val + split.test)
    print("split: train", len(split.train), "val", len(split.val), "test", len(split.test), "snapshots")
    print("norm stats from", len(train), "rollouts; distances", src["distances"], "; snapshots", len(src["snapshots"]))
    for name, sn, ds in (("train", split.train, TRAIN_DISTANCES), ("val", split.val, TRAIN_DISTANCES),
                         ("test seen-d", split.test, TRAIN_DISTANCES), ("train unseen-d (eval only)", split.train, (TEST_DISTANCE,)),
                         ("test unseen-d (eval only)", split.test, (TEST_DISTANCE,))):
        rs = load_rollouts(args.data, sn, ds)
        m = sum(len(r.motion_ticks()) for r in rs); s = sum(len(r.standing_ticks()) for r in rs)
        print(f"  {name:28s}: {len(rs):3d} rollouts, {m:5d} motion ticks + {s:5d} standing ticks = {m+s:5d} tick samples")
    ds = StepDataset(train, norm, TRAIN_TEMPLATES, n_per_rollout=120, prev_token_dropout=0.5)
    print(f"  epoch (train, 120/rollout): {len(ds)} samples; example text {ds[0]['text']!r}; x {tuple(ds[0]['x'].shape)} y {tuple(ds[0]['y'].shape)}")
    print("saved", os.path.join(args.out, "split.json"), os.path.join(args.out, "norm_stats.npz"))


if __name__ == "__main__":
    main()
