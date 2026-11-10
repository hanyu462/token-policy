"""STEP_FOOT point-to-point dataset -> Student samples (text, state history) -> future token chunk.

Inputs the Student sees: a raw command text rendered from (foot, dx, dy) and the 150-D state features
(features.py, start frame F0) of the 10 ticks before tick t.  Never as inputs: foot / dx / dy fields, a*, lambda,
CEM metadata.  Target: token_k[t : t+40] of the executed teacher rollout; past the end the last token (the split-stance
hold token, held by the teacher) is repeated.  Split by snapshot (Split.default); rollouts with drift_flag are removed
from every split into a separate hard-case set.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from ..features.state import NormStats, block_slices, history_window, rollout_features
from .common import HORIZON, MOTION_TAIL, T_S, Split
from .texts import STEPFOOT_TRAIN_TEMPLATES, render_stepfoot

STATE_KEYS = ["q", "dq", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "lfoot", "rfoot", "lfoot_force", "rfoot_force"]
N_WINDOW = 94


EXTRA_SNAPSHOT_DIRS = {"s1": "results/snapshots_extra"}   # id prefix -> directory of additional snapshot batches


def snapshot_path(sid: str, snapshots_dir="results/snapshots", nominal="results/standing_snapshot.npz") -> str:
    if sid == "standing_snapshot":
        return nominal
    return os.path.join(EXTRA_SNAPSHOT_DIRS.get(sid.split("_")[0], snapshots_dir), sid + ".npz")


@dataclass
class FootRollout:
    snapshot_id: str
    foot: str            # used ONLY to render text and for evaluation bookkeeping
    dx: float
    dy: float
    token_k: np.ndarray  # (T, 64)
    k_stand: np.ndarray  # (64,) standing code of the snapshot (prev_token at t = 0)
    feats: np.ndarray    # (T, 150)
    meta: dict = field(default_factory=dict)

    @property
    def T(self):
        return len(self.token_k)

    def target(self, t):
        y = self.token_k[t:t + HORIZON]
        if len(y) < HORIZON:
            y = np.concatenate([y, np.tile(self.token_k[-1], (HORIZON - len(y), 1))])   # hold token stays
        return y

    def motion_ticks(self):
        return np.arange(0, min(N_WINDOW + MOTION_TAIL, self.T))

    def standing_ticks(self):
        return np.arange(min(N_WINDOW + MOTION_TAIL, self.T), self.T)


def load_index(data_dir="results/stepfoot_dataset") -> list:
    return [json.loads(l) for l in open(os.path.join(data_dir, "index.jsonl"))]


def load_foot_rollouts(data_dir, rows) -> list:
    k_stand_cache, out = {}, []
    for r in rows:
        if not r["teacher_success"]:
            continue
        sid = r["snapshot_id"]
        if sid not in k_stand_cache:
            k_stand_cache[sid] = np.load(snapshot_path(sid))["k_stand"].astype(np.int64)
        f = np.load(os.path.join(data_dir, r["file"]))
        tok = f["token_k"].astype(np.int64)
        st0 = {k: f[f"state_start/{k}"] for k in STATE_KEYS}
        hist = {k: f[f"state_history/{k}"] for k in STATE_KEYS}
        out.append(FootRollout(sid, r["foot"], float(r["command_dx"]), float(r["command_dy"]), tok, k_stand_cache[sid],
                               rollout_features(st0, hist, tok, k_stand_cache[sid]), r))
    return out


def make_splits(rows, split: Split) -> dict:
    """{'train','val','test','hard'} -> index rows; hard = drift-flagged rollouts from any split."""
    out = {"train": [], "val": [], "test": [], "hard": []}
    for r in rows:
        if not r["teacher_success"]:
            continue
        if r["drift_flag"]:
            out["hard"].append(r)
            continue
        for name in ("train", "val", "test"):
            if r["snapshot_id"] in getattr(split, name):
                out[name].append(r)
    return out


def fit_foot_norm(train_rollouts) -> NormStats:
    feats = np.concatenate([r.feats for r in train_rollouts])
    return NormStats.fit(feats, source=json.dumps({"dataset": "stepfoot", "snapshots": sorted({r.snapshot_id for r in train_rollouts}),
                                                   "n_rollouts": len(train_rollouts)}))


def is_motion_tick(t: int) -> bool:
    """Motion sample = token window + stop transition; later ticks are the hold."""
    return bool(t < N_WINDOW + MOTION_TAIL)


class StepFootDataset(torch.utils.data.Dataset):
    """Same sampling as StepDataset: per epoch n_per_rollout ticks per rollout, 70 % motion / 30 % hold.

    per_rollout_n: optional {rollout key: ticks per epoch} overriding n_per_rollout (e.g. command-balanced sampling).
    """

    def __init__(self, rollouts, norm: NormStats, templates=STEPFOOT_TRAIN_TEMPLATES, n_per_rollout=120, motion_frac=0.7,
                 seed=0, prev_token_dropout=0.0, fixed_ticks: dict | None = None, per_rollout_n: dict | None = None):
        self.rollouts, self.norm, self.templates = rollouts, norm, list(templates)
        self.n_per_rollout, self.motion_frac, self.prev_token_dropout = n_per_rollout, motion_frac, prev_token_dropout
        self.per_rollout_n = per_rollout_n
        self.rng = np.random.default_rng(seed)
        self.prev_sl = block_slices()["prev_token"]
        self.fixed = fixed_ticks
        self.resample()

    def resample(self):
        items = []
        for i, r in enumerate(self.rollouts):
            if self.fixed is not None:
                items += [(i, t) for t in self.fixed.get(r.meta["key"], range(r.T))]
                continue
            n = self.n_per_rollout if self.per_rollout_n is None else self.per_rollout_n[r.meta["key"]]
            n_m = int(round(n * self.motion_frac))
            m, s = r.motion_ticks(), r.standing_ticks()
            items += [(i, int(t)) for t in self.rng.choice(m, n_m)]
            if len(s):
                items += [(i, int(t)) for t in self.rng.choice(s, n - n_m)]
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t = self.items[idx]
        r = self.rollouts[i]
        x = self.norm.apply(history_window(r.feats, t, T_S)).astype(np.float32)
        if self.prev_token_dropout > 0 and self.rng.random() < self.prev_token_dropout:
            x[:, self.prev_sl] = 0.0
        text = render_stepfoot(self.templates[self.rng.integers(len(self.templates))], r.foot, r.dx, r.dy)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(r.target(t) + 16).long(), "text": text,
                "tick": t, "is_motion": is_motion_tick(t), "key": r.meta["key"]}

    def sample_info(self) -> dict:
        """Per item, in index order and without using the sampling RNG: is_motion (bool) and rollout key."""
        return {"motion": np.array([is_motion_tick(t) for _, t in self.items], bool),
                "key": np.array([self.rollouts[i].meta["key"] for i, _ in self.items])}


def command_balanced_counts(base_rollouts, all_rollouts, n_per_rollout: int) -> dict:
    """Ticks per epoch for each rollout of ``all_rollouts`` so that every command keeps the per-epoch budget it has
    in ``base_rollouts`` (n_per_rollout x its rollout count there), spread evenly over all its rollouts.  Extra
    rollouts thus add state diversity without changing the command frequencies (or the epoch size)."""
    cmd = lambda r: (r.foot, round(r.dx, 2), round(r.dy, 2))
    budget = {}
    for r in base_rollouts:
        budget[cmd(r)] = budget.get(cmd(r), 0) + n_per_rollout
    groups = {}
    for r in all_rollouts:
        groups.setdefault(cmd(r), []).append(r)
    out = {}
    for c, rs in groups.items():
        base, rem = divmod(budget[c], len(rs))
        for j, r in enumerate(sorted(rs, key=lambda r: r.meta["key"])):
            out[r.meta["key"]] = base + (1 if j < rem else 0)
    return out


def collate_foot(batch: list, cache) -> dict:
    H, M = cache.batch([b["text"] for b in batch])
    return {"x": torch.stack([b["x"] for b in batch]), "y": torch.stack([b["y"] for b in batch]), "H": H, "mask": M,
            "is_motion": torch.tensor([b["is_motion"] for b in batch]), "text": [b["text"] for b in batch],
            "key": [b["key"] for b in batch], "tick": torch.tensor([b["tick"] for b in batch])}


def all_ticks(rollouts, every=1) -> dict:
    return {r.meta["key"]: list(range(0, r.T, every)) for r in rollouts}
