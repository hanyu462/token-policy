#!/usr/bin/env python3
"""SONIC walks the official reference; the experiment decides what to draw.

CLI options still work on top (e.g. --log, --mode async, --duration 10):
  .venv/bin/python experiments/sonic_walk_tracking.py --log
"""
import sys

from token_policy.cli import parse_config, run
from token_policy.viz import colors


def draw(c, st, ctx):
    """Called every tick with a Canvas, the sim state and the run context."""
    c.sphere(st["lfoot"], 0.04, colors.RED)                       # red sphere on the left foot
    c.text(st["lfoot"] + [0, 0, 0.1], "L")
    c.sphere([0.4, 0.3, 0.5], 0.03, colors.BLUE, "fixed point")   # anything else you like
    ref = ctx["reference"]
    c.text(st["base_pos"] + [0, -0.45, 0.7], f"t={st['time']:.2f}s  ref {ref.cursor}/{ref.length} {'PLAY' if ref.playing else 'HOLD'}")


cfg = parse_config(sys.argv[1:], controller="sonic", reference="walking_quip_360_R_002__A428")
run(cfg, draw_fns=[draw])
