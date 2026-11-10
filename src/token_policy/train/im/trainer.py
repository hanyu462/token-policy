"""Imitation-learning training loop: forward -> action CE -> backward -> grad clip -> optimizer / scheduler step,
then validation, best-checkpoint selection (validation motion-only CE) and early stopping.

Data (datasets, collate, text cache), features and simulator evaluation are supplied by the caller.
Batches are dicts with H (B,N,1024), mask (B,N), x (B,T_s,150), y (B,H,64) class targets, is_motion (B,), key.

Compatibility layer (kept so the STEP_FOOT baseline reproduces bit-exactly): the trainer still requires
train_ds.resample(), the is_motion / key batch fields, and selects the best checkpoint by validation motion CE.
These are recipe decisions and are to move to student/ (then dataset + collate -> DataLoader, verified separately).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Callable

import numpy as np
import torch

from ..checkpoint import save_checkpoint
from ..config import TrainConfig
from ..optim import make_optimizer, make_scheduler
from .losses import action_ce
from .metrics import per_sample_metrics, summarize


@torch.no_grad()
def evaluate(model, ds, collate: Callable, dev, batch: int = 512) -> dict:
    """Every sample of ``ds`` in order -> per-sample arrays: ce, acc, dk, motion flag, rollout key."""
    model.eval()
    ce, acc, dk, mot, keys = [], [], [], [], []
    for i in range(0, len(ds), batch):
        b = collate([ds[j] for j in range(i, min(i + batch, len(ds)))])
        logits = model(b["H"].to(dev), b["mask"].to(dev), b["x"].to(dev))["logits"]
        c, a, d = per_sample_metrics(logits, b["y"].to(dev))
        ce.append(c.cpu()); acc.append(a.cpu()); dk.append(d.cpu())
        mot.append(b["is_motion"]); keys += b["key"]
    return {"ce": torch.cat(ce).numpy(), "acc": torch.cat(acc).numpy(), "dk": torch.cat(dk).numpy(), "motion": torch.cat(mot).numpy(), "key": np.array(keys)}


def fit(model, train_ds, train_eval_ds, val_ds, collate: Callable, cfg: TrainConfig, out_dir: str, dev,
        val_extras: Callable | None = None) -> dict:
    """Train ``model`` in place; writes out_dir/best.pt (lowest val motion CE) and out_dir/curve.json every epoch.

    train_ds must provide resample() (new random samples each epoch).  val_extras(ev_val) -> (record fields, log
    suffix) adds task-specific validation numbers to the curve and the log line.
    """
    os.makedirs(out_dir, exist_ok=True)
    opt = make_optimizer(model.parameters(), cfg)
    sched = make_scheduler(opt, cfg, math.ceil(len(train_ds) / cfg.batch))
    curve, best, best_ep, bad, step, t0 = [], float("inf"), -1, 0, 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train(); train_ds.resample()
        perm = np.random.permutation(len(train_ds)); tot, n = 0.0, 0
        for i in range(0, len(perm), cfg.batch):
            b = collate([train_ds[j] for j in perm[i:i + cfg.batch]])
            loss = action_ce(model(b["H"].to(dev), b["mask"].to(dev), b["x"].to(dev))["logits"], b["y"].to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sched.step()
            tot += loss.item() * len(b["y"]); n += len(b["y"]); step += 1
        ev_tr, ev_va = evaluate(model, train_eval_ds, collate, dev), evaluate(model, val_ds, collate, dev)
        tr, va = summarize(ev_tr), summarize(ev_va)
        extra, suffix = val_extras(ev_va) if val_extras is not None else ({}, "")
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train_loss": tot / n, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}, **extra, "time_s": time.time() - t0}
        curve.append(rec)
        improved = va["ce_motion"] < best
        if improved:
            best, best_ep, bad = va["ce_motion"], epoch, 0
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, epoch, val=va, optimizer=opt, scheduler=sched, global_step=step)
        else:
            bad += 1
        print(f"ep {epoch:3d}  train CE m {tr['ce_motion']:.4f} h {tr['ce_hold']:.4f} acc {tr['acc']:.4f} | val CE m {va['ce_motion']:.4f} h {va['ce_hold']:.4f} "
              f"acc {va['acc']:.4f} (m {va['acc_motion']:.4f}{suffix}) |dk| {va['mean_abs_dk']:.4f}  [{rec['time_s']:.0f}s]{' *' if improved else ''}", flush=True)
        with open(os.path.join(out_dir, "curve.json"), "w") as f:
            json.dump(curve, f, indent=1)
        if bad >= cfg.patience:
            print(f"early stop at epoch {epoch}"); break
    print(f"best val motion CE {best:.4f} at epoch {best_ep}")
    return {"best": best, "best_epoch": best_ep, "curve": curve}
