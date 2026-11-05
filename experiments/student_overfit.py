#!/usr/bin/env python3
"""Single-batch overfit test for Student S0: one fixed batch, many steps; L_action must approach 0 and the target
FSQ codes must be reproduced (checks data alignment / architecture before any full training).

  .venv/bin/python experiments/student_overfit.py --batch 32 --steps 600 [--small] [--dropout 0.0]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from token_policy.student.dataset import TRAIN_DISTANCES, Split, StepDataset, collate, load_rollouts
from token_policy.student.features import NormStats
from token_policy.student.losses import LossConfig, compute_losses, offline_metrics
from token_policy.student.model import StudentConfig, StudentS0
from token_policy.student.text_encoder import TextCache
from token_policy.student.texts import TRAIN_TEMPLATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/step_dataset_v2")
    ap.add_argument("--work", default="results/student")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--prev-token-dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    split = Split.load(os.path.join(args.work, "split.json"))
    norm = NormStats.load(os.path.join(args.work, "norm_stats.npz"))
    cache = TextCache(os.path.join(args.work, "text_cache.pt"))
    rollouts = load_rollouts(args.data, split.train[:4], TRAIN_DISTANCES)
    ds = StepDataset(rollouts, norm, TRAIN_TEMPLATES, n_per_rollout=8, motion_frac=0.75, seed=args.seed, prev_token_dropout=args.prev_token_dropout)
    batch = collate([ds[i] for i in range(args.batch)], cache)
    dev = "cuda"
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    print(f"batch: x {tuple(batch['x'].shape)} y {tuple(batch['y'].shape)} H {tuple(batch['H'].shape)} mask {tuple(batch['mask'].shape)} "
          f"motion samples {int(batch['is_motion'].sum())}/{args.batch}; distinct texts {len(set(batch['text']))}; "
          f"y classes used {sorted(set(batch['y'].unique().tolist()))}")
    cfg = StudentConfig.small() if args.small else StudentConfig()
    cfg.dropout = args.dropout
    model = StudentS0(cfg).to(dev)
    print(f"model {'small' if args.small else 'default'}: {model.num_params()/1e6:.2f}M trainable params, dropout {cfg.dropout}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    lcfg = LossConfig()
    curve = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        out = model(batch["H"], batch["mask"], batch["x"])
        losses = compute_losses(out, batch, lcfg)
        opt.zero_grad(); losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                o = model(batch["H"], batch["mask"], batch["x"])
                m = offline_metrics(o["logits"], batch["y"], batch["is_motion"])
                d_err = (o["d_hat"] * lcfg.mag_scale - batch["d_cmd"]).abs().mean().item()
            curve.append({"step": step, "ce_train": losses["action"].item(), **m, "d_err_m": d_err})
            print(f"step {step:4d}: train CE {losses['action'].item():.4f}  eval CE {m['ce']:.4f} (motion {m.get('ce_motion', float('nan')):.4f} / standing {m.get('ce_standing', float('nan')):.4f})  "
                  f"acc {m['acc']:.4f} (motion {m.get('acc_motion', float('nan')):.4f})  mean|dk| {m['mean_abs_dk']:.4f}  |d_hat-d| {d_err*100:.2f} cm  [{time.time()-t0:.0f}s]", flush=True)
    os.makedirs(args.work, exist_ok=True)
    json.dump(curve, open(os.path.join(args.work, f"overfit_{'small' if args.small else 'default'}.json"), "w"), indent=1)
    # does the model actually reproduce the target chunk?
    model.eval()
    with torch.no_grad():
        k_pred = model.predict_k(batch["H"], batch["mask"], batch["x"]); k_gt = batch["y"] - 16
    exact_rows = (k_pred == k_gt).all(-1).float().mean().item()
    print(f"final: exact cells {(k_pred == k_gt).float().mean().item():.4f}, exact (tick, all 64 dims) rows {exact_rows:.4f}, "
          f"max |dk| {(k_pred - k_gt).abs().max().item()}")


if __name__ == "__main__":
    main()
