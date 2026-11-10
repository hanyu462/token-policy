"""Imitation-learning training loop: forward -> action CE -> backward -> grad clip -> optimizer / scheduler step,
then validation, best-checkpoint selection by a caller-supplied criterion, and early stopping.

The trainer knows no task: data (datasets, collate, text cache), the validation summary, the selection criterion
and any task breakdown are supplied by the caller (the recipe in student/).  Batches are dicts with at least
H (B,N,1024), mask (B,N), x (B,T_s,150) and y (B,H,64) class targets.

Still the pre-DataLoader interface (kept so baselines reproduce bit-exactly): train_ds must provide resample(),
and datasets are indexed + collated here; to become DataLoaders (verified separately).
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
from .metrics import per_sample_metrics, summarize as generic_summarize


@torch.no_grad()
def evaluate(model, ds, collate: Callable, dev, batch: int = 512) -> dict:
    """Every sample of ``ds`` in index order -> per-sample arrays: ce (cell mean), acc (exact cell), dk (mean |dk|)."""
    model.eval()
    ce, acc, dk = [], [], []
    for i in range(0, len(ds), batch):
        b = collate([ds[j] for j in range(i, min(i + batch, len(ds)))])
        logits = model(b["H"].to(dev), b["mask"].to(dev), b["x"].to(dev))["logits"]
        c, a, d = per_sample_metrics(logits, b["y"].to(dev))
        ce.append(c.cpu()); acc.append(a.cpu()); dk.append(d.cpu())
    return {"ce": torch.cat(ce).numpy(), "acc": torch.cat(acc).numpy(), "dk": torch.cat(dk).numpy()}


def fit(model, train_ds, train_eval_ds, val_ds, collate: Callable, cfg: TrainConfig, out_dir: str, dev, *,
        summarize: Callable | None = None, selection_metric: Callable | None = None, selection_mode: str = "min",
        val_extras: Callable | None = None) -> dict:
    """Train ``model`` in place; writes out_dir/best.pt (best selection score) and out_dir/curve.json every epoch.

    summarize(ev, ds) -> dict of scalars for an evaluate() result of dataset ds (default: overall ce / acc / |dk|).
    selection_metric(val_summary) -> scalar used for best-checkpoint selection and early stopping (default: val ce);
    selection_mode "min" or "max".  val_extras(ev_val, val_ds) -> (record fields, log suffix) adds task-specific
    validation numbers to the curve and the log line.
    """
    if selection_mode not in ("min", "max"):
        raise ValueError(f"selection_mode must be 'min' or 'max', got {selection_mode!r}")
    summarize = summarize or (lambda ev, ds: generic_summarize(ev))
    selection_metric = selection_metric or (lambda va: va["ce"])
    better = (lambda a, b: a < b) if selection_mode == "min" else (lambda a, b: a > b)
    os.makedirs(out_dir, exist_ok=True)
    opt = make_optimizer(model.parameters(), cfg)
    sched = make_scheduler(opt, cfg, math.ceil(len(train_ds) / cfg.batch))
    best = float("inf") if selection_mode == "min" else -float("inf")
    curve, best_ep, bad, step, t0 = [], -1, 0, 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train(); train_ds.resample()
        perm = np.random.permutation(len(train_ds)); tot, n = 0.0, 0
        for i in range(0, len(perm), cfg.batch):
            b = collate([train_ds[j] for j in perm[i:i + cfg.batch]])
            loss = action_ce(model(b["H"].to(dev), b["mask"].to(dev), b["x"].to(dev))["logits"], b["y"].to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sched.step()
            tot += loss.item() * len(b["y"]); n += len(b["y"]); step += 1
        ev_tr, ev_va = evaluate(model, train_eval_ds, collate, dev), evaluate(model, val_ds, collate, dev)
        tr, va = summarize(ev_tr, train_eval_ds), summarize(ev_va, val_ds)
        extra, suffix = val_extras(ev_va, val_ds) if val_extras is not None else ({}, "")
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train_loss": tot / n, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}, **extra, "time_s": time.time() - t0}
        curve.append(rec)
        score = float(selection_metric(va))
        improved = better(score, best)
        if improved:
            best, best_ep, bad = score, epoch, 0
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, epoch, val=va, optimizer=opt, scheduler=sched, global_step=step)
        else:
            bad += 1
        print(f"ep {epoch:3d}  train loss {tot / n:.4f} | " + " ".join(f"{k} {v:.4f}" for k, v in va.items())
              + f"{suffix} | select {score:.4f}  [{rec['time_s']:.0f}s]{' *' if improved else ''}", flush=True)
        with open(os.path.join(out_dir, "curve.json"), "w") as f:
            json.dump(curve, f, indent=1)
        if bad >= cfg.patience:
            print(f"early stop at epoch {epoch}"); break
    print(f"best selection score {best:.4f} at epoch {best_ep}")
    return {"best": best, "best_epoch": best_ep, "curve": curve}
