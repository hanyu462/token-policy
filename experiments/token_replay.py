#!/usr/bin/env python3
"""Run A: SONIC tracks the official walking reference, tokens recorded per 50 Hz tick.
Run B: same initial state, the recorded tokens are fed straight into the decoder (ReplaceToken);
the reference/encoder output is ignored, residual 0.  Then compare q_target, q, base, feet.

  .venv/bin/python experiments/token_replay.py [--duration 15] [--headless] [--no-realtime]
"""
import os
import sys

import numpy as np

from token_policy.cli import parse_config, run
from token_policy.paths import SONIC_MODEL_DIR
from token_policy.policy import ReplaceToken
from token_policy.sonic import SonicController, SonicJointController, SonicONNX, load_reference
from token_policy.viz import colors

REF = "walking_quip_360_R_002__A428"
os.makedirs("results", exist_ok=True)


class RecordedTokens:
    """source(state) -> the token recorded at the same control tick of Run A."""

    def __init__(self, tokens, dt):
        self.tokens, self.dt = tokens, dt

    def __call__(self, state):
        k = int(round(state["time"] / self.dt))
        z = self.tokens[min(k, len(self.tokens) - 1)]
        assert np.all(np.isfinite(z)), f"no recorded token at tick {k}"
        return z


def draw(c, st, ctx):
    c.marker(st["rfoot"], colors.RED, label="R")
    c.text(st["base_pos"] + [0, -0.45, 0.7], f"{ctx.get('run', '')}  t={st['time']:.2f}s")


argv = sys.argv[1:]
base = parse_config(argv, controller="sonic", reference=REF, print_period=0)

# ---- Run A: reference -> encoder -> token -> decoder (tokens recorded) ----
cfg_a = parse_config(argv, controller="sonic", reference=REF, print_period=0, log="results/token_replay_A.npz")
a = run(cfg_a, draw_fns=[draw])

# ---- Run B: recorded token -> decoder; the reference is irrelevant (standing), encoder output discarded ----
sonic = SonicController(SonicONNX(base.model_dir))
ctrl_b = SonicJointController(sonic, load_reference("standing"),
                              token_hook=ReplaceToken(RecordedTokens(a["token"], 0.02)))
cfg_b = parse_config(argv, controller="sonic", reference="standing", print_period=0, log="results/token_replay_B.npz")
b = run(cfg_b, controller=ctrl_b, draw_fns=[draw])

# ---- compare ----
n = min(len(a["t"]), len(b["t"]))
m = np.isfinite(a["token"][:n, 0])                           # ticks where SONIC was on (both runs)
print("\n=== Run A (encoder) vs Run B (recorded token -> decoder), ticks with SONIC on:", int(m.sum()))
for key, unit in [("token", ""), ("q_target", "rad"), ("q", "rad"), ("base_pos", "m"), ("base_rpy", "rad"), ("lfoot", "m"), ("rfoot", "m")]:
    d = np.abs(a[key][:n][m] - b[key][:n][m])
    print(f"  {key:9s} max|A-B| = {d.max():.2e} {unit}   mean {d.mean():.2e}")
print(f"  fell: A {'yes' if np.isfinite(a['fell_at']) else 'no'}, B {'yes' if np.isfinite(b['fell_at']) else 'no'}; "
      f"travel A {np.linalg.norm(a['base_pos'][-1,:2]-a['base_pos'][0,:2]):.2f} m, B {np.linalg.norm(b['base_pos'][-1,:2]-b['base_pos'][0,:2]):.2f} m")
print(f"  (B's own encoder output, discarded, differs from the fed token by max {np.nanmax(np.abs(b['token_base'][:n][m] - b['token'][:n][m])):.2f} -> the decoder really used the recorded tokens)")
