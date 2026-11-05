#!/usr/bin/env python3
"""Student S0 full training (Dataset V2, seen distances only; 30 cm never read).

  .venv/bin/python experiments/student_train.py --small --out results/student/run_small
Checkpoint selection: validation motion-only CE (val snapshots x seen distances, every tick, no input dropout).
"""
import argparse
import json
import math
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


def eval_split(model, ds, cache, dev, batch_size=512):
    """All samples of ``ds`` (fixed ticks); returns metrics incl. motion / standing CE and magnitude error."""
    model.eval()
    logits, ys, motions, dh, dc = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            b = collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))], cache)
            out = model(b["H"].to(dev), b["mask"].to(dev), b["x"].to(dev))
            logits.append(out["logits"].cpu()); ys.append(b["y"]); motions.append(b["is_motion"])
            dh.append(out["d_hat"].cpu()); dc.append(b["d_cmd"])
    logits, y, mo = torch.cat(logits), torch.cat(ys), torch.cat(motions)
    m = offline_metrics(logits, y, mo)
    m["mag_err_cm"] = ((torch.cat(dh) * 0.1 - torch.cat(dc)).abs().mean() * 100).item()
    return m


def all_tick_dataset(rollouts, norm, templates, every=1, allow_test_distance=False):
    """Deterministic evaluation set: every ``every``-th tick of each rollout, template cycled by tick."""
    ticks = {(r.snapshot_id, r.d_cmd): list(range(0, r.T, every)) for r in rollouts}
    ds = StepDataset(rollouts, norm, templates, fixed_ticks=ticks, prev_token_dropout=0.0, seed=123, allow_test_distance=allow_test_distance)
    ds.rng = np.random.default_rng(123)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/step_dataset_v2")
    ap.add_argument("--work", default="results/student")
    ap.add_argument("--out", default="results/student/run_small")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--prev-token-dropout", type=float, default=0.5)
    ap.add_argument("--n-per-rollout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    split = Split.load(os.path.join(args.work, "split.json"))
    norm = NormStats.load(os.path.join(args.work, "norm_stats.npz"))
    cache = TextCache(os.path.join(args.work, "text_cache.pt"))
    train_r = load_rollouts(args.data, split.train, TRAIN_DISTANCES)      # 30 cm never loaded here
    val_r = load_rollouts(args.data, split.val, TRAIN_DISTANCES)
    train_ds = StepDataset(train_r, norm, TRAIN_TEMPLATES, n_per_rollout=args.n_per_rollout, seed=args.seed,
                           prev_token_dropout=args.prev_token_dropout)
    train_eval_ds = all_tick_dataset(train_r, norm, TRAIN_TEMPLATES, every=4)
    val_ds = all_tick_dataset(val_r, norm, TRAIN_TEMPLATES, every=1)
    cfg = StudentConfig.small() if args.small else StudentConfig()
    model = StudentS0(cfg).to(dev)
    lcfg = LossConfig()
    print(f"train {len(train_r)} rollouts ({len(train_ds)} samples/epoch), val {len(val_r)} rollouts ({len(val_ds)} samples); "
          f"model {'small' if args.small else 'default'} {model.num_params()/1e6:.2f}M; loss = action + {lcfg.w_mag} mag", flush=True)
    json.dump({**vars(args), "model": model.config_dict(), "loss": lcfg.__dict__}, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch)
    total_steps, warm = args.epochs * steps_per_epoch, int(args.warmup * args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total_steps - warm, 1))))
    curve, best, best_epoch, bad = [], float("inf"), -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); train_ds.resample()
        perm = np.random.permutation(len(train_ds))
        tot = {"total": 0.0, "action": 0.0, "mag": 0.0}; n = 0
        for i in range(0, len(perm), args.batch):
            b = collate([train_ds[j] for j in perm[i:i + args.batch]], cache)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["H"], b["mask"], b["x"])
            losses = compute_losses(out, b, lcfg)
            opt.zero_grad(); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k in tot:
                tot[k] += losses[k].item() * len(b["y"])
            n += len(b["y"])
        tr = eval_split(model, train_eval_ds, cache, dev)
        va = eval_split(model, val_ds, cache, dev)
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train_loss": tot["total"] / n, "train_action_loss": tot["action"] / n, "train_mag_loss": tot["mag"] / n,
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}, "time_s": time.time() - t0}
        curve.append(rec)
        improved = va["ce_motion"] < best
        if improved:
            best, best_epoch, bad = va["ce_motion"], epoch, 0
            torch.save({"model": model.state_dict(), "config": model.config_dict(), "epoch": epoch, "val": va, "args": vars(args)}, os.path.join(args.out, "best.pt"))
        else:
            bad += 1
        print(f"ep {epoch:3d}  train CE {tr['ce']:.4f} (m {tr['ce_motion']:.4f} s {tr['ce_standing']:.4f})  val CE {va['ce']:.4f} (m {va['ce_motion']:.4f} s {va['ce_standing']:.4f})  "
              f"val acc {va['acc']:.4f} |dk| {va['mean_abs_dk']:.4f} mag {va['mag_err_cm']:.2f} cm  lr {rec['lr']:.2e}  [{rec['time_s']:.0f}s]{'  *' if improved else ''}", flush=True)
        json.dump(curve, open(os.path.join(args.out, "curve.json"), "w"), indent=1)
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (no val motion-CE improvement for {args.patience} epochs)")
            break
    torch.save({"model": model.state_dict(), "config": model.config_dict(), "epoch": epoch}, os.path.join(args.out, "last.pt"))
    print(f"best val motion CE {best:.4f} at epoch {best_epoch}; saved {args.out}/best.pt")


if __name__ == "__main__":
    main()
