"""Gymnasium view of FootTaskEnv for Stable-Baselines3.

Action modes: "discrete" = MultiDiscrete (64 heads, index -> delta_k); "gaussian" = Box in grid-step units,
delta_k = clip(round(u), -dk, dk) (continuous policy mean, rounding handles the grid)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .foot_task_env import FootTaskEnv, TaskConfig


class FootTaskGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: TaskConfig | None = None, action_repeat: int = 1, action_mode: str = "discrete"):
        """``action_repeat``: hold each delta_k decision for this many control ticks (reward summed).
        Temporally consistent token offsets are what move the feet; per-tick random offsets cancel out."""
        super().__init__()
        self.env = FootTaskEnv(cfg)
        self.dk = self.env.cfg.dk_range
        self.action_repeat = int(action_repeat)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.env.obs_dim,), np.float32)
        assert action_mode in ("discrete", "gaussian"), action_mode
        self.action_mode = action_mode
        if action_mode == "discrete":
            self.action_space = gym.spaces.MultiDiscrete([self.env.n_action] * self.env.act_dim)
        else:
            self.action_space = gym.spaces.Box(-float(self.dk), float(self.dk), (self.env.act_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(seed=seed, **(options or {}))
        return obs.astype(np.float32), {}

    def step(self, action):
        if self.action_mode == "discrete":
            dk = np.asarray(action, np.int64) - self.dk
        else:
            dk = np.clip(np.rint(np.asarray(action, np.float64)), -self.dk, self.dk).astype(np.int64)
        r_sum = 0.0
        for _ in range(self.action_repeat):
            obs, r, done, info = self.env.step(dk)
            r_sum += r
            if done:
                break
        r = r_sum
        terminated = bool(info["fallen"] or (info["success"] and done))
        truncated = bool(done and not terminated)
        return obs.astype(np.float32), r, terminated, truncated, info
