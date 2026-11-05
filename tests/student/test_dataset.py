"""Split / leakage tests for the Student dataset layer (synthetic rollouts; no SONIC, no files)."""
import json

import numpy as np
import pytest

from token_policy.student.dataset import HORIZON, TEST_DISTANCE, TRAIN_DISTANCES, Rollout, Split, StepDataset, fit_norm
from token_policy.student.features import FEATURE_DIM, block_slices
from token_policy.student.texts import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, all_texts, render

RNG = np.random.default_rng(1)


def _rollout(sid, d, T=244, window=94):
    tok = RNG.integers(-6, 8, (T, 64)); k_stand = RNG.integers(-3, 4, 64); tok[window:] = k_stand
    feats = RNG.normal(d * 10, 1.0, (T, FEATURE_DIM))      # mean depends on d so leakage is detectable
    return Rollout(sid, d, tok, k_stand, feats, window)


def test_split_is_disjoint_and_complete():
    snaps = ["standing_snapshot"] + [f"s0_{i:02d}" for i in range(30)]
    sp = Split.default(snaps)
    assert len(sp.train) == 21 and len(sp.val) == 3 and len(sp.test) == 7
    assert set(sp.train) | set(sp.val) | set(sp.test) == set(snaps)


def test_train_loader_never_renders_the_held_out_distance():
    rs = [_rollout("a", d) for d in TRAIN_DISTANCES]
    ds = StepDataset(rs, fit_norm(rs), n_per_rollout=50)
    texts = {ds[i]["text"] for i in range(len(ds))}
    assert texts and all("30" not in t for t in texts)
    assert all(any(str(int(d * 100)) in t for d in TRAIN_DISTANCES) for t in texts)
    with pytest.raises(AssertionError):
        StepDataset(rs + [_rollout("b", TEST_DISTANCE)], fit_norm(rs))


def test_norm_stats_do_not_depend_on_30cm():
    rs = [_rollout(f"s{i}", d) for i in range(3) for d in TRAIN_DISTANCES]
    ns = fit_norm(rs)
    with pytest.raises(AssertionError):
        fit_norm(rs + [_rollout("x", TEST_DISTANCE)])
    src = json.loads(ns.source)
    assert TEST_DISTANCE not in src["distances"] and sorted(src["distances"]) == sorted(TRAIN_DISTANCES)
    rs2 = rs + [_rollout("x", 0.35)]        # adding an allowed rollout changes the stats; so a 30 cm one would too
    assert not np.allclose(fit_norm(rs2).mean, ns.mean)


def test_target_padding_and_alignment():
    r = _rollout("a", 0.2, T=100, window=60)
    y = r.target(80)
    assert y.shape == (HORIZON, 64) and np.array_equal(y[:20], r.token_k[80:100]) and np.array_equal(y[20:], np.tile(r.k_stand, (20, 1)))
    assert np.array_equal(r.target(0), r.token_k[:HORIZON])


def test_sampler_mix_and_prev_token_dropout():
    rs = [_rollout("a", 0.25)]
    ds = StepDataset(rs, fit_norm(rs), n_per_rollout=200, motion_frac=0.7, prev_token_dropout=1.0)
    ticks = np.array([t for _, t in ds.items])
    assert np.isclose((ticks < 94 + 20).mean(), 0.7, atol=0.01)
    sl = block_slices()["prev_token"]
    assert all(np.all(ds[i]["x"][:, sl].numpy() == 0) for i in range(5))
    ds2 = StepDataset(rs, fit_norm(rs), n_per_rollout=10, prev_token_dropout=0.0)
    assert any(np.any(ds2[i]["x"][:, sl].numpy() != 0) for i in range(10))
    assert ds[0]["y"].min() >= 0 and ds[0]["y"].max() <= 31 and ds[0]["x"].shape == (10, FEATURE_DIM)


def test_templates_split_and_rendering():
    assert not set(TRAIN_TEMPLATES) & set(HELDOUT_TEMPLATES)
    assert render(TRAIN_TEMPLATES[0], 0.25) == "Move forward by 25 cm."
    assert len(all_texts((0.2, 0.25, 0.3, 0.35))) == 4 * 8
