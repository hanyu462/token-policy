#!/usr/bin/env python3
"""T0 teacher: PPO on FootTaskEnv (right foot lateral +-1 cm), FSQ-grid offset action.

  .venv/bin/python experiments/t0_train.py --steps 2000000 --n-envs 4 --out results/t0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from token_policy.env import TaskConfig
from token_policy.env.gym_wrapper import FootTaskGym

SNAP = "results/standing_snapshot.npz"


def make_env(seed, cfg, action_repeat=1, action_mode="discrete"):
    def _f():
        env = FootTaskGym(cfg, action_repeat=action_repeat, action_mode=action_mode)
        env.reset(seed=seed)
        return env
    return _f


class EpisodeStats(BaseCallback):
    """Success rate / final distance / falls per rollout, printed and logged to tensorboard."""

    def __init__(self, out_dir):
        super().__init__()
        self.buf = []
        self.path = os.path.join(out_dir, "episodes.jsonl")
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                rec = {"step": self.num_timesteps, "success": bool(info["success"]), "dist_mm": info["dist"] * 1e3,
                       "disp_mm": [float(x) * 1e3 for x in info["disp_local"][:2]], "cmd_mm": [c * 1e3 for c in info["command"]],
                       "fallen": bool(info["fallen"]), "len": info["t"], "stance_mm": info["stance_drift"] * 1e3}
                self.buf.append(rec)
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return True

    def _on_rollout_end(self) -> None:
        if not self.buf:
            return
        b = self.buf; n = len(b)
        s = {"ep": n, "success": np.mean([r["success"] for r in b]), "fall": np.mean([r["fallen"] for r in b]),
             "dist_mm": np.mean([r["dist_mm"] for r in b]), "len": np.mean([r["len"] for r in b]),
             "stance_mm": np.mean([r["stance_mm"] for r in b])}
        for k, v in s.items():
            self.logger.record(f"task/{k}", float(v))
        print(f"[{self.num_timesteps:>8d} steps, {(time.time()-self.t0)/60:5.1f} min] episodes {n:3d}  success {s['success']:.2f}  "
              f"fall {s['fall']:.2f}  final dist {s['dist_mm']:.1f} mm  len {s['len']:.0f}  stance {s['stance_mm']:.1f} mm", flush=True)
        self.buf = []


def bias_toward_zero(model, n_action, bias=2.0):
    """Initial policy prefers delta_k = 0 (index n_action // 2 in every head)."""
    with torch.no_grad():
        b = model.policy.action_net.bias
        b.view(64, n_action)[:, n_action // 2] += bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--dk-range", type=int, default=2)
    ap.add_argument("--out", default="results/t0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="PPO network device (small MLP: cpu is fine)")
    ap.add_argument("--backend", default="torch", choices=["onnx", "torch"], help="SONIC decoder backend inside each env")
    ap.add_argument("--action-repeat", type=int, default=10, help="hold each delta_k for this many ticks (10 = 0.2 s)")
    ap.add_argument("--w-dk", type=float, default=0.0, help="||delta_k||^2 penalty weight")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--zero-bias", type=float, default=1.0, help="initial logit bias toward delta_k = 0")
    ap.add_argument("--action", default="discrete", choices=["discrete", "gaussian"],
                    help="gaussian: continuous u (grid steps), delta_k = clip(round(u))")
    ap.add_argument("--log-std-init", type=float, default=-0.35, help="gaussian: initial log std (grid steps); -0.35 -> 0.7")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=16384, help="save results/<out>/ckpt_<steps>_steps.zip every n decisions (0 = off)")
    ap.add_argument("--resume", default=None, help="PPO zip (ckpt or final) to continue from: keeps optimizer state and step count")
    ap.add_argument("--init-from", default=None, help="PPO zip to start from (e.g. t0_warmstart.py output); hyperparameters below still apply")
    ap.add_argument("--sde", action="store_true", help="gaussian: gSDE (state-dependent, temporally consistent exploration noise)")
    ap.add_argument("--sde-sample-freq", type=int, default=-1, help="gSDE: resample the noise matrix every n decisions (-1 = per rollout)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = TaskConfig(onnx_threads=1, snapshot_path=SNAP, effector="R", commands=((0.0, 0.01), (0.0, -0.01)),
                     dk_range=args.dk_range, decoder_backend=args.backend, w_dk=args.w_dk)
    venv = SubprocVecEnv([make_env(args.seed * 100 + i, cfg, args.action_repeat, args.action) for i in range(args.n_envs)], start_method="spawn")
    policy_kwargs = dict(net_arch=[256, 256])
    if args.action == "gaussian":
        policy_kwargs["log_std_init"] = args.log_std_init
        if args.sde:
            policy_kwargs["use_expln"] = True
    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device, tensorboard_log=os.path.join(args.out, "tb"),
                         custom_objects=dict(learning_rate=args.lr, clip_range=args.clip, n_epochs=args.n_epochs, ent_coef=args.ent_coef))
        print("resumed", args.resume, "at", model.num_timesteps, "steps")
    else:
        model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, n_epochs=args.n_epochs, learning_rate=args.lr, gamma=0.98,
                    gae_lambda=0.95, clip_range=args.clip, ent_coef=args.ent_coef, policy_kwargs=policy_kwargs,
                    use_sde=args.action == "gaussian" and args.sde, sde_sample_freq=args.sde_sample_freq,
                    seed=args.seed, device=args.device, verbose=0, tensorboard_log=os.path.join(args.out, "tb"))
    if args.init_from and not args.resume:
        init = PPO.load(args.init_from, device=args.device)
        model.policy.load_state_dict(init.policy.state_dict())
        print("initialised policy from", args.init_from)
    elif args.action == "discrete" and not args.resume:
        bias_toward_zero(model, 2 * args.dk_range + 1, args.zero_bias)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "task": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}}, f, indent=1)
    callbacks = [EpisodeStats(args.out)]
    if args.ckpt_every > 0:
        callbacks.append(CheckpointCallback(save_freq=max(args.ckpt_every // args.n_envs, 1), save_path=args.out, name_prefix="ckpt"))
    model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=not args.resume)
    model.save(os.path.join(args.out, "ppo_t0"))
    print("saved", os.path.join(args.out, "ppo_t0.zip"))


if __name__ == "__main__":
    main()
