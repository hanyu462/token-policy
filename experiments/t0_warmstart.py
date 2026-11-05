#!/usr/bin/env python3
"""Warm-start the T0 gaussian policy from the token-space Jacobian controller (behaviour cloning), then hand to PPO.

Controller: every decision u = g * J^+ e, e = goal error (fwd, lat) in mm, J (2 x 64) = per-dim foot response slopes
from results/dim_sweep.npy (mm per grid step); delta_k = clip(round(u), -R, R). The policy mean is regressed onto u.

  .venv/bin/python experiments/t0_warmstart.py --episodes 200 --out results/t0_bc
  .venv/bin/python experiments/t0_train.py --action gaussian --action-repeat 25 --init-from results/t0_bc/ppo_bc.zip ...
"""
import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from token_policy.env import TaskConfig
from token_policy.env.gym_wrapper import FootTaskGym

SNAP = "results/standing_snapshot.npz"


def jacobian_pinv(path="results/dim_sweep.npy"):
    res = np.load(path)                                   # (64, 6, [lat, fwd, drift, fell]) for dk in -8,-4,-2,2,4,8
    J = np.stack([(res[:, 4, 1] - res[:, 1, 1]) / 8.0,    # fwd  mm / grid step
                  (res[:, 4, 0] - res[:, 1, 0]) / 8.0])   # lat  mm / grid step
    return J.T @ np.linalg.inv(J @ J.T)                   # (64, 2) min-norm pseudo-inverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--action-repeat", type=int, default=25)
    ap.add_argument("--dk-range", type=int, default=2)
    ap.add_argument("--log-std-init", type=float, default=-1.2, help="initial exploration std after BC (grid steps)")
    ap.add_argument("--bc-epochs", type=int, default=300)
    ap.add_argument("--out", default="results/t0_bc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = TaskConfig(onnx_threads=4, snapshot_path=SNAP, effector="R", commands=((0.0, 0.01), (0.0, -0.01)),
                     dk_range=args.dk_range, decoder_backend="torch")
    env = FootTaskGym(cfg, action_repeat=args.action_repeat, action_mode="gaussian")
    Jp = jacobian_pinv()
    gi = env.env.obs_index("goal_error")                  # slice of the goal error (fwd, lat, up) in the obs

    # ---- collect (obs, u) with the Jacobian controller ----
    X, U, stats = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed * 1000 + ep, options={"command": cfg.commands[ep % 2]})
        done = False
        while not done:
            u = np.clip(args.gain * Jp @ (obs[gi][:2] * 1e3), -args.dk_range, args.dk_range)
            X.append(obs.copy()); U.append(u.astype(np.float32))
            obs, r, te, tr, info = env.step(u)
            done = te or tr
        stats.append((info["success"], info["dist"] * 1e3, info["stance_drift"] * 1e3, info["t"]))
    s = np.array(stats, float)
    print(f"controller: {len(X)} decisions, success {s[:,0].mean():.2f}, final dist {s[:,1].mean():.1f} mm, "
          f"drift {s[:,2].mean():.1f} mm, len {s[:,3].mean():.0f} ticks")
    X = torch.as_tensor(np.array(X)); U = torch.as_tensor(np.array(U))

    # ---- behaviour cloning of the policy mean ----
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=512, n_epochs=5, learning_rate=3e-4, gamma=0.98,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=args.seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=args.log_std_init))
    pol = model.policy
    opt = torch.optim.Adam([p for n, p in pol.named_parameters() if not n.startswith("value_net") and n != "log_std"], lr=1e-3)
    for epoch in range(args.bc_epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 1024):
            idx = perm[i:i + 1024]
            feat = pol.extract_features(X[idx])
            lat_pi = pol.mlp_extractor.forward_actor(feat)
            mean = pol.action_net(lat_pi)
            loss = torch.mean((mean - U[idx]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if epoch % 50 == 0 or epoch == args.bc_epochs - 1:
            print(f"bc epoch {epoch:4d}: mse {tot / len(X):.4f} (u var {float(U.var()):.4f})")
    model.save(os.path.join(args.out, "ppo_bc"))

    # ---- check: deterministic BC policy in the env ----
    out = []
    for ep in range(10):
        obs, _ = env.reset(seed=10_000 + ep, options={"command": cfg.commands[ep % 2]})
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(a); done = te or tr
        out.append((info["success"], info["dist"] * 1e3, info["stance_drift"] * 1e3, info["t"], info["disp_local"][1] * 1e3))
    o = np.array(out, float)
    print(f"BC policy (deterministic, 10 ep): success {o[:,0].mean():.2f}, final dist {o[:,1].mean():.1f} mm, drift {o[:,2].mean():.1f} mm, "
          f"len {o[:,3].mean():.0f}, lat(+10) {o[0::2,4].mean():+.1f} lat(-10) {o[1::2,4].mean():+.1f}")
    print("saved", os.path.join(args.out, "ppo_bc.zip"))


if __name__ == "__main__":
    main()
