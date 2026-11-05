#!/usr/bin/env python3
"""Statistics of a STEP_FORWARD dataset (make_step_dataset.py output)."""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/step_dataset")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(args.data, "index.jsonl"))]
    cmds = sorted({r["command_distance_m"] for r in rows}); snaps = sorted({r["snapshot_id"] for r in rows})
    dc = np.array([r["command_distance_m"] for r in rows]); dr = np.array([r["realized_dx_m"] for r in rows])
    print(f"1. rollouts: {len(rows)}  ({len(snaps)} snapshots x {len(cmds)} commands), ticks per rollout {rows[0]['n_ticks']} ({rows[0]['n_ticks']*0.02:.2f} s), "
          f"window {rows[0]['window_ticks']} ticks, paraphrases per rollout {len(rows[0]['paraphrases'])}")
    print("2. realized dx per command (cm):")
    for d in cmds:
        s = [r for r in rows if r["command_distance_m"] == d]
        x = np.array([r["realized_dx_m"] for r in s]) * 100; y = np.array([r["realized_dy_m"] for r in s]) * 100
        yaw = np.degrees([r["realized_dyaw_rad"] for r in s]); ns = [r["n_swing"] for r in s]
        print(f"   cmd {d*100:.0f}: dx {x.mean():5.1f} +- {x.std():4.1f} [{x.min():5.1f}..{x.max():5.1f}]  bias {x.mean()-d*100:+4.1f}  "
              f"dy {y.mean():+4.1f} +- {y.std():3.1f}  dyaw {yaw.mean():+4.1f} +- {yaw.std():3.1f} deg  swings {min(ns)}-{max(ns)}  lambda {s[0]['lambda']:.3f}")
    print(f"3. fall rate: {np.mean([r['fell'] for r in rows]):.3f}   end speed mean {np.mean([r['end_speed'] for r in rows]):.3f} m/s")
    print(f"4. corr(d_cmd, d_real) = {np.corrcoef(dc, dr)[0,1]:.3f};  within-snapshot slope d(real)/d(cmd): "
          f"{np.mean([np.polyfit([r['command_distance_m'] for r in rows if r['snapshot_id']==s], [r['realized_dx_m'] for r in rows if r['snapshot_id']==s], 1)[0] for s in snaps]):.2f}  "
          f"|d_real - d_cmd| mean {np.abs(dr-dc).mean()*100:.1f} cm, max {np.abs(dr-dc).max()*100:.1f} cm; ordering preserved in "
          f"{np.mean([np.all(np.diff([r['realized_dx_m'] for r in sorted([r for r in rows if r['snapshot_id']==s], key=lambda r: r['command_distance_m'])]) > 0) for s in snaps]):.2f} of snapshots")
    print("5. per-snapshot offset (mean of d_real - d_cmd over the 4 commands, cm):")
    off = {s: np.mean([(r["realized_dx_m"] - r["command_distance_m"]) * 100 for r in rows if r["snapshot_id"] == s]) for s in snaps}
    v = np.array(list(off.values()))
    print(f"   offset mean {v.mean():+.1f}, sd {v.std():.1f}, range [{v.min():+.1f}..{v.max():+.1f}]; most negative {min(off, key=off.get)} ({off[min(off, key=off.get)]:+.1f}), "
          f"most positive {max(off, key=off.get)} ({off[max(off, key=off.get)]:+.1f}); within-snapshot sd of the residual across commands {np.mean([np.std([(r['realized_dx_m']-r['command_distance_m'])*100 for r in rows if r['snapshot_id']==s]) for s in snaps]):.1f} cm")

    # ---- token sequences ----
    def load(r):
        f = np.load(os.path.join(args.data, r["file"]))
        return f["token_k"].astype(int), int(f["window_ticks"]), f["k_stand"].astype(int)
    K = {(r["snapshot_id"], r["command_distance_m"]): load(r) for r in rows}
    print("6. token sequence differences between commands (same snapshot, window ticks only):")
    for a, b in [(cmds[i], cmds[j]) for i in range(len(cmds)) for j in range(i + 1, len(cmds))]:
        cells, absdk, l2 = [], [], []
        for s in snaps:
            ka, n, _ = K[(s, a)]; kb, _, _ = K[(s, b)]
            d = ka[:n] - kb[:n]
            cells.append((d != 0).mean()); absdk.append(np.abs(d).mean()); l2.append(np.linalg.norm(d / 16.0, axis=1).mean())
        print(f"   {a*100:.0f} vs {b*100:.0f} cm: differing cells {np.mean(cells)*100:4.1f}%  mean |dk| {np.mean(absdk):.3f} grid steps  mean per-tick ||dz|| {np.mean(l2):.3f}")
    dev, std_frac = [], []
    for s in snaps:
        for d in cmds:
            k, n, ks = K[(s, d)]
            dev.append(np.abs(k[:n] - ks).mean()); std_frac.append((k[n:] == ks).all())
    print(f"   (reference: mean |k - k_stand| inside the window {np.mean(dev):.3f} grid steps; standing phase equals k_stand in {np.mean(std_frac)*100:.0f}% of rollouts)")
    print("7. same command across snapshots (window ticks): differing cells vs the nominal snapshot:")
    for d in cmds:
        k0, n, _ = K[("standing_snapshot", d)]
        c = [(K[(s, d)][0][:n] != k0[:n]).mean() for s in snaps if s != "standing_snapshot"]
        print(f"   cmd {d*100:.0f}: {np.mean(c)*100:4.1f}% (max {np.max(c)*100:4.1f}%)  -> Z varies with s_0 only through k_stand")
    print("8. schema / size:")
    f = np.load(os.path.join(args.data, rows[0]["file"]))
    for k in f.files:
        print(f"   {k:28s} {str(f[k].shape):14s} {f[k].dtype}")
    print("   index.jsonl keys:", list(rows[0].keys()))
    tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.data, "rollouts", "*.npz")))
    print(f"   rollouts: {len(glob.glob(os.path.join(args.data, 'rollouts', '*.npz')))} files, {tot/1e6:.1f} MB total ({tot/len(rows)/1e3:.0f} kB each), "
          f"index.jsonl {os.path.getsize(os.path.join(args.data, 'index.jsonl'))/1e3:.0f} kB")


if __name__ == "__main__":
    main()
