#!/usr/bin/env python3
"""Offline evaluation of a Student checkpoint on Dataset V2 buckets + state-conditioning analysis.

  .venv/bin/python experiments/student_eval_offline.py results/student/run_small/best.pt            # seen distances only
  .venv/bin/python experiments/student_eval_offline.py results/student/run_small/best.pt --unseen   # opens 30 cm (final)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_train import all_tick_dataset, eval_split  # noqa: E402

from token_policy.student.dataset import TEST_DISTANCE, TRAIN_DISTANCES, Split, load_rollouts
from token_policy.student.features import NormStats, history_window
from token_policy.student.model import StudentConfig, StudentS0
from token_policy.student.text_encoder import TextCache
from token_policy.student.texts import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, render


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    model = StudentS0(StudentConfig(**ck["config"])).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


def bucket(model, rollouts, norm, cache, dev, templates, label, subgroups=True, allow_test=False):
    ds = all_tick_dataset(rollouts, norm, templates, every=1, allow_test_distance=allow_test)
    m = eval_split(model, ds, cache, dev)
    line = (f"{label:44s} n_roll {len(rollouts):2d}  CE {m['ce']:.4f} (motion {m['ce_motion']:.4f} / standing {m['ce_standing']:.4f})  "
            f"acc {m['acc']:.4f} (motion {m['acc_motion']:.4f})  |dk| {m['mean_abs_dk']:.4f}  mag err {m['mag_err_cm']:.2f} cm")
    print(line, flush=True)
    out = {label: m}
    if subgroups:
        for conv in (True, False):
            sub = [r for r in rollouts if bool(r.meta.get("teacher_converged", True)) == conv]
            if sub:
                ms = eval_split(model, all_tick_dataset(sub, norm, templates, every=1, allow_test_distance=allow_test), cache, dev)
                print(f"    teacher_converged={conv!s:5s} n_roll {len(sub):2d}  motion CE {ms['ce_motion']:.4f}  acc {ms['acc_motion']:.4f}  mag err {ms['mag_err_cm']:.2f} cm")
                out[f"{label} | converged={conv}"] = ms
    return out


def predict_rollout(model, r, norm, cache, dev, text, ticks):
    """Student k-hat[0] for the given ticks of a rollout (offline: teacher states as input)."""
    X = torch.from_numpy(np.stack([norm.apply(history_window(r.feats, t)) for t in ticks]).astype(np.float32)).to(dev)
    H, M = cache.batch([text] * len(ticks))
    with torch.no_grad():
        k = model.predict_k(H.to(dev), M.to(dev), X)[:, 0].cpu().numpy()      # (n_ticks, 64): the token executed at each tick
    return k


def state_conditioning(model, rollouts, norm, cache, dev, d, text, label):
    """Same command, different snapshots: do teacher and student tokens differ across s_0 in the same way?"""
    rs = [r for r in rollouts if abs(r.d_cmd - d) < 1e-6]
    n = min(r.window_ticks for r in rs)
    ticks = list(range(n))
    T = {r.snapshot_id: r.token_k[:n] for r in rs}
    S = {r.snapshot_id: predict_rollout(model, r, norm, cache, dev, text, ticks) for r in rs}
    ids = sorted(T)
    t_diff, s_diff, agree, t_dist, s_dist = [], [], [], [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            td, sd = T[a] != T[b], S[a] != S[b]
            t_diff.append(td.mean()); s_diff.append(sd.mean()); t_dist.append(td.mean()); s_dist.append(sd.mean())
            if td.any():
                agree.append((sd & td).sum() / td.sum())        # of the cells where the teacher differs, how many does the student also change
    corr = np.corrcoef(t_dist, s_dist)[0, 1] if len(t_dist) > 2 else np.nan
    # per-snapshot delta w.r.t. the nominal rollout: does the student reproduce the teacher's snapshot-specific k?
    base = "standing_snapshot" if "standing_snapshot" in T else ids[0]
    exact_delta = []
    for s in ids:
        if s == base: continue
        cells = T[s] != T[base]
        if cells.any():
            exact_delta.append((S[s][cells] == T[s][cells]).mean())
    print(f"{label:30s} d {d*100:.0f} cm: cross-snapshot differing cells  teacher {np.mean(t_diff)*100:4.1f}%  student {np.mean(s_diff)*100:4.1f}%  | "
          f"student changes where teacher changes {np.mean(agree)*100:4.1f}%  | pairwise distance corr {corr:+.2f}  | "
          f"student reproduces teacher's snapshot-specific k in {np.mean(exact_delta)*100:4.1f}% of those cells", flush=True)
    return {"teacher_diff": float(np.mean(t_diff)), "student_diff": float(np.mean(s_diff)), "agree": float(np.mean(agree)),
            "pairwise_corr": float(corr), "exact_delta": float(np.mean(exact_delta))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="results/step_dataset_v2")
    ap.add_argument("--work", default="results/student")
    ap.add_argument("--unseen", action="store_true", help="evaluate the held-out 30 cm distance (only after the checkpoint is frozen)")
    args = ap.parse_args()
    dev = "cuda"
    model, ck = load_model(args.ckpt, dev)
    print(f"checkpoint {args.ckpt}: epoch {ck.get('epoch')}, val {ck.get('val', {}).get('ce_motion', float('nan')):.4f} motion CE, {model.num_params()/1e6:.2f}M params")
    split = Split.load(os.path.join(args.work, "split.json"))
    norm = NormStats.load(os.path.join(args.work, "norm_stats.npz"))
    cache = TextCache(os.path.join(args.work, "text_cache.pt"))
    res = {}
    print("\n== seen distances {20, 25, 35} cm, train templates ==")
    sets = {"train snapshots": split.train, "val snapshots": split.val, "held-out snapshots": split.test}
    for name, sn in sets.items():
        rs = load_rollouts(args.data, sn, TRAIN_DISTANCES)
        res.update(bucket(model, rs, norm, cache, dev, TRAIN_TEMPLATES, f"{name} x seen d"))
    rs = load_rollouts(args.data, split.test, TRAIN_DISTANCES)
    res.update(bucket(model, rs, norm, cache, dev, HELDOUT_TEMPLATES, "held-out snapshots x seen d x HELD-OUT templates", subgroups=False))
    print("\n== state conditioning (same text, different s_0), seen distances ==")
    for d in TRAIN_DISTANCES:
        rs = load_rollouts(args.data, split.train + split.test, (d,))
        res[f"state_cond_{d}"] = state_conditioning(model, rs, norm, cache, dev, d, render(TRAIN_TEMPLATES[0], d), "train+held-out snapshots")
    if args.unseen:
        print("\n== UNSEEN 30 cm (first look) ==")
        for name, sn in (("train snapshots", split.train), ("held-out snapshots", split.test)):
            rs = load_rollouts(args.data, sn, (TEST_DISTANCE,))
            res.update(bucket(model, rs, norm, cache, dev, TRAIN_TEMPLATES, f"{name} x UNSEEN 30 cm", allow_test=True))
            dh = []
            ds = all_tick_dataset(rs, norm, TRAIN_TEMPLATES, every=10, allow_test_distance=True)
            from token_policy.student.dataset import collate
            with torch.no_grad():
                for i in range(0, len(ds), 512):
                    b = collate([ds[j] for j in range(i, min(i + 512, len(ds)))], cache)
                    dh.append(model(b["H"].to(dev), b["mask"].to(dev), b["x"].to(dev))["d_hat"].cpu() * 0.1)
            dh = torch.cat(dh)
            print(f"    predicted magnitude d_hat for '30 cm' texts: mean {dh.mean()*100:.1f} cm, sd {dh.std()*100:.1f} cm")
        rs = load_rollouts(args.data, split.train + split.test, (TEST_DISTANCE,))
        res["state_cond_0.3"] = state_conditioning(model, rs, norm, cache, dev, TEST_DISTANCE, render(TRAIN_TEMPLATES[0], TEST_DISTANCE), "train+held-out snapshots")
    out = os.path.join(os.path.dirname(args.ckpt), "offline_eval" + ("_unseen" if args.unseen else "") + ".json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
