"""train/ refactor checks: imports, dependency direction, one training step, schedule, validation, checkpoints."""
import ast
import importlib
import math
import os
import pathlib

import numpy as np
import pytest
import torch

from token_policy.models.config import ModelConfig
from token_policy.models.model import TokenPolicyModel
from token_policy.train.checkpoint import load_model, save_checkpoint
from token_policy.train.config import TrainConfig
from token_policy.train.im.losses import action_ce
from token_policy.train.im.metrics import per_sample_metrics, summarize
from token_policy.train.im.trainer import evaluate, fit
from token_policy.train.optim import make_optimizer, make_scheduler

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "token_policy"
CFG = ModelConfig(d_model=32, num_heads=2, ffn_dim=64, num_skill_queries=4, num_skill_layers=1, num_state_layers=1,
                  num_motion_layers=1, horizon=4, dropout=0.0)


class TinyDS:
    """Random batch items in the trainer's format; resample() draws new ones."""

    def __init__(self, n=12, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.resample()

    def resample(self):
        self.items = [{"H": torch.randn(5, CFG.lang_dim), "x": torch.randn(CFG.history, CFG.feature_dim),
                       "y": torch.from_numpy(self.rng.integers(0, 32, (CFG.horizon, 64))), "is_motion": bool(i % 2), "key": f"r{i}"}
                      for i in range(self.n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.items[i]


def collate(items):
    return {"H": torch.stack([b["H"] for b in items]), "mask": torch.ones(len(items), 5, dtype=torch.bool),
            "x": torch.stack([b["x"] for b in items]), "y": torch.stack([b["y"] for b in items]),
            "is_motion": torch.tensor([b["is_motion"] for b in items]), "key": [b["key"] for b in items]}


def test_imports_have_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for m in ["token_policy.train", "token_policy.train.config", "token_policy.train.optim", "token_policy.train.checkpoint",
              "token_policy.train.im", "token_policy.train.im.losses", "token_policy.train.im.metrics", "token_policy.train.im.trainer",
              "token_policy.train.rl", "token_policy.train.rl.buffer", "token_policy.train.rl.losses", "token_policy.train.rl.metrics",
              "token_policy.train.rl.ppo"]:
        importlib.reload(importlib.import_module(m))
    assert list(tmp_path.iterdir()) == []


def test_lower_layers_never_import_train():
    """student/ (recipes) may use train/; everything below it must not."""
    files = [f for pkg in ("models", "common", "features", "data", "policy", "sonic", "sim", "env", "teacher", "task")
             for f in (SRC / pkg).rglob("*.py")]
    for f in files:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom):
                mod = ("." * node.level) + (node.module or "")
                assert "train" not in mod.split("."), (f, mod)


def test_one_step_updates_parameters_and_schedule():
    torch.manual_seed(0)
    model = TokenPolicyModel(CFG)
    cfg = TrainConfig(epochs=2, batch=4, lr=1e-3, warmup=0.25)
    opt = make_optimizer(model.parameters(), cfg)
    sched = make_scheduler(opt, cfg, steps_per_epoch=4)                 # 8 steps, warm = 2
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 1 / 2)
    b = collate(TinyDS(4).items)
    before = [p.detach().clone() for p in model.parameters()]
    loss = action_ce(model(b["H"], b["mask"], b["x"])["logits"], b["y"])
    opt.zero_grad(); loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sched.step()
    assert any(not torch.equal(a, p) for a, p in zip(before, model.parameters()))
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(6):
        opt.step(); sched.step(); lrs.append(opt.param_groups[0]["lr"])
    expect = [1e-3 * (2 / 2)] + [1e-3 * 0.5 * (1 + math.cos(math.pi * (s - 2) / 6)) for s in range(2, 8)]
    assert lrs == pytest.approx(expect[:7])


def test_evaluate_and_fit(tmp_path):
    torch.manual_seed(0); np.random.seed(0)
    model = TokenPolicyModel(CFG)
    ds, val = TinyDS(12, 0), TinyDS(6, 1)
    ev = evaluate(model, val, collate, "cpu", batch=4)
    assert ev["ce"].shape == (6,) and ev["motion"].sum() == 3 and list(ev["key"]) == [f"r{i}" for i in range(6)]
    logits = model.eval()(*[collate(val.items)[k] for k in ("H", "mask", "x")])["logits"]
    ce, acc, dk = per_sample_metrics(logits, collate(val.items)["y"])
    assert np.allclose(ce.numpy(), ev["ce"], atol=1e-5) and set(summarize(ev)) >= {"ce_motion", "acc_hold", "mean_abs_dk"}
    out = fit(model, ds, ds, val, collate, TrainConfig(epochs=2, batch=4, patience=5), str(tmp_path), "cpu",
              lambda e: ({"val_extra": 1.0}, ", extra"))
    assert len(out["curve"]) == 2 and out["curve"][0]["val_extra"] == 1.0
    ck = torch.load(tmp_path / "best.pt")
    assert {"model", "config", "epoch", "val", "optimizer", "scheduler", "global_step"} <= set(ck)
    m2, _ = load_model(str(tmp_path / "best.pt"))
    assert all(torch.equal(a, b) for a, b in zip(m2.state_dict().values(), [ck["model"][k] for k in m2.state_dict()]))


def test_save_load_roundtrip(tmp_path):
    model = TokenPolicyModel(CFG)
    save_checkpoint(str(tmp_path / "a.pt"), model, epoch=3, val={"ce": 1.0}, note="x")
    m2, ck = load_model(str(tmp_path / "a.pt"))
    assert ck["epoch"] == 3 and ck["note"] == "x" and "optimizer" not in ck
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), m2.state_dict().values()))


@pytest.mark.skipif(not os.path.exists("results/student_stepfoot/run_small/best.pt"), reason="no STEP_FOOT checkpoint")
def test_existing_checkpoint_loads():
    model, ck = load_model("results/student_stepfoot/run_small/best.pt")
    assert isinstance(model, TokenPolicyModel) and ck["epoch"] == 8


def test_models_depend_only_on_common():
    """models/ is the architecture: no imports from data / features / train / student / policy."""
    for f in (SRC / "models").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level >= 2:
                assert (node.module or "").split(".")[0] in ("common",), (f, node.module)
