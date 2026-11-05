"""Dataset V2 -> Student samples.

A sample at tick t of a rollout: text (a train template rendered with that rollout's d_cmd), the 10-frame state
history before tick t (features.py), the future chunk token_k[t : t + 40] (padded with the rollout's k_stand past
the end) and d_cmd.  Splits are by snapshot; 30 cm rollouts are excluded from everything except the final test.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .features import FEATURE_DIM, NormStats, block_slices, history_window, rollout_features
from .texts import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, render

HORIZON = 40
T_S = 10
TRAIN_DISTANCES = (0.20, 0.25, 0.35)
TEST_DISTANCE = 0.30
MOTION_TAIL = 20          # ticks after the token window still counted as "motion" (stop transition)


@dataclass
class Split:
    train: list
    val: list
    test: list

    @classmethod
    def default(cls, snapshots: list) -> "Split":
        """Held-out 7 spread over the yaw-sorted list; 3 validation; the rest train."""
        test = ["s0_02", "s0_07", "s0_12", "s0_16", "s0_21", "s0_25", "s0_29"]
        val = ["s0_05", "s0_14", "s0_24"]
        train = [s for s in snapshots if s not in test and s not in val]
        assert not set(train) & set(test) and not set(train) & set(val) and not set(val) & set(test)
        return cls(train, val, test)

    def save(self, path: str):
        json.dump(self.__dict__, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "Split":
        return cls(**json.load(open(path)))


@dataclass
class Rollout:
    snapshot_id: str
    d_cmd: float
    token_k: np.ndarray        # (T, 64) int
    k_stand: np.ndarray        # (64,)
    feats: np.ndarray          # (T, 150) raw features of the state before each tick
    window_ticks: int
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.token_k)

    def target(self, t: int) -> np.ndarray:
        """(40, 64) token_k[t : t+40], padded with k_stand (the standing token is held after the rollout ends)."""
        y = self.token_k[t:t + HORIZON]
        if len(y) < HORIZON:
            y = np.concatenate([y, np.tile(self.k_stand, (HORIZON - len(y), 1))])
        return y

    def motion_ticks(self) -> np.ndarray:
        return np.arange(0, min(self.window_ticks + MOTION_TAIL, self.T))

    def standing_ticks(self) -> np.ndarray:
        return np.arange(min(self.window_ticks + MOTION_TAIL, self.T), self.T)


def load_rollouts(data_dir: str, snapshots=None, distances=None) -> list:
    rows = [json.loads(l) for l in open(os.path.join(data_dir, "index.jsonl"))]
    out = []
    for r in rows:
        if snapshots is not None and r["snapshot_id"] not in snapshots:
            continue
        if distances is not None and not any(abs(r["command_distance_m"] - d) < 1e-6 for d in distances):
            continue
        f = np.load(os.path.join(data_dir, r["file"]))
        keys = ["q", "dq", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "lfoot", "rfoot", "lfoot_force", "rfoot_force"]
        st0 = {k: f[f"state_start/{k}"] for k in keys}
        hist = {k: f[f"state_history/{k}"] for k in keys}
        tok = f["token_k"].astype(np.int64)
        out.append(Rollout(r["snapshot_id"], float(r["command_distance_m"]), tok, f["k_stand"].astype(np.int64),
                           rollout_features(st0, hist, tok, f["k_stand"]), int(f["window_ticks"]), r))
    return out


class StepDataset(torch.utils.data.Dataset):
    """Epoch = n_per_rollout samples per rollout; ticks drawn 70 % from the motion range, 30 % from standing
    (early standing weighted 2:1).  Texts are rendered from ``templates`` with the rollout's d_cmd."""

    def __init__(self, rollouts: list, norm: NormStats, templates=TRAIN_TEMPLATES, n_per_rollout: int = 120,
                 motion_frac: float = 0.7, seed: int = 0, prev_token_dropout: float = 0.0, state_noise: float = 0.0,
                 fixed_ticks: dict | None = None, allow_test_distance: bool = False):
        """``allow_test_distance``: only for the final unseen-magnitude evaluation; training/validation must leave it False."""
        self.rollouts, self.norm, self.templates = rollouts, norm, list(templates)
        for r in rollouts:
            assert allow_test_distance or abs(r.d_cmd - TEST_DISTANCE) > 1e-6, f"held-out distance {TEST_DISTANCE} m in a training/validation dataset"
        self.n_per_rollout, self.motion_frac = n_per_rollout, motion_frac
        self.prev_token_dropout, self.state_noise = prev_token_dropout, state_noise
        self.rng = np.random.default_rng(seed)
        self.prev_sl = block_slices()["prev_token"]
        self.fixed = fixed_ticks
        self.resample()

    def resample(self):
        """Draw this epoch's (rollout, tick) pairs."""
        items = []
        for i, r in enumerate(self.rollouts):
            if self.fixed is not None:
                items += [(i, t) for t in self.fixed[(r.snapshot_id, r.d_cmd)]]
                continue
            n_m = int(round(self.n_per_rollout * self.motion_frac))
            m, s = r.motion_ticks(), r.standing_ticks()
            w = np.where(s - s[0] < 50, 2.0, 1.0) if len(s) else None
            items += [(i, t) for t in self.rng.choice(m, n_m)]
            if len(s):
                items += [(i, t) for t in self.rng.choice(s, self.n_per_rollout - n_m, p=w / w.sum())]
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t = self.items[idx]
        r = self.rollouts[i]
        x = self.norm.apply(history_window(r.feats, t, T_S)).astype(np.float32)
        if self.state_noise > 0:
            x = x + self.rng.normal(0, self.state_noise, x.shape).astype(np.float32) * self.norm.normalize
        if self.prev_token_dropout > 0 and self.rng.random() < self.prev_token_dropout:
            x[:, self.prev_sl] = 0.0                                      # neutral: unknown previous token
        text = render(self.templates[self.rng.integers(len(self.templates))], r.d_cmd)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(r.target(t) + 16).long(), "text": text,
                "d_cmd": torch.tensor(r.d_cmd, dtype=torch.float32), "tick": t, "is_motion": bool(t < r.window_ticks + MOTION_TAIL),
                "snapshot_id": r.snapshot_id}


def collate(batch: list, cache) -> dict:
    H, M = cache.batch([b["text"] for b in batch])
    return {"x": torch.stack([b["x"] for b in batch]), "y": torch.stack([b["y"] for b in batch]), "H": H, "mask": M,
            "d_cmd": torch.stack([b["d_cmd"] for b in batch]), "is_motion": torch.tensor([b["is_motion"] for b in batch]),
            "text": [b["text"] for b in batch]}


def fit_norm(rollouts: list) -> NormStats:
    """Normalization from train rollouts only (asserted: no held-out distance)."""
    for r in rollouts:
        assert abs(r.d_cmd - TEST_DISTANCE) > 1e-6
    feats = np.concatenate([r.feats for r in rollouts])
    return NormStats.fit(feats, source=json.dumps({"snapshots": sorted({r.snapshot_id for r in rollouts}),
                                                   "distances": sorted({r.d_cmd for r in rollouts})}))
