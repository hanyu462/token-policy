#!/usr/bin/env python3
"""Learning curve from results/<run>/episodes.jsonl: success rate, final distance, lateral displacement vs command.

  .venv/bin/python experiments/t0_curve.py results/t0 [--window 100]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "results/t0"
win = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 100
rows = [json.loads(l) for l in open(os.path.join(run, "episodes.jsonl"))]
step = np.array([r["step"] for r in rows]); succ = np.array([r["success"] for r in rows], float)
fall = np.array([r["fallen"] for r in rows], float); dist = np.array([r["dist_mm"] for r in rows])
disp = np.array([r["disp_mm"][1] for r in rows]); cmd = np.array([r["cmd_mm"][1] for r in rows])
length = np.array([r["len"] for r in rows]); stance = np.array([r["stance_mm"] for r in rows])
ma = lambda x: np.convolve(x, np.ones(win) / win, mode="valid")  # noqa: E731
s = step[win - 1:]
fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
axs[0].plot(s, ma(succ), label="success rate"); axs[0].plot(s, ma(fall), label="fall rate"); axs[0].set_ylim(-0.02, 1.02); axs[0].legend(); axs[0].set_title(f"{run}: {len(rows)} episodes, moving average {win}")
axs[1].plot(s, ma(dist)); axs[1].axhline(5, ls="--", color="g"); axs[1].set_ylabel("final dist to target (mm)")
for c, col in ((10, "tab:red"), (-10, "tab:blue")):
    m = cmd == c
    if m.sum() > win:
        axs[2].plot(step[m][win - 1:], ma(disp[m]), color=col, label=f"cmd {c:+d} mm")
        axs[2].axhline(c, color=col, ls=":")
axs[2].set_ylabel("final lateral disp (mm)"); axs[2].legend()
axs[3].plot(s, ma(length), label="episode length (ticks)"); axs[3].plot(s, ma(stance) * 10, label="stance drift (mm x10)"); axs[3].legend(); axs[3].set_xlabel("env steps")
for a in axs:
    a.grid(alpha=0.3)
fig.tight_layout(); out = os.path.join(run, "curve.png"); fig.savefig(out, dpi=120); print("saved", out)
print(f"last {win} episodes: success {succ[-win:].mean():.2f}, fall {fall[-win:].mean():.2f}, dist {dist[-win:].mean():.1f} mm, "
      f"disp(+10) {disp[-win:][cmd[-win:] == 10].mean():+.1f} mm, disp(-10) {disp[-win:][cmd[-win:] == -10].mean():+.1f} mm")
