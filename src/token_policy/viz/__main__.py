"""python -m token_policy.viz [--mode sync|async] [--duration S]

Drawing demo: the sim holds the standing pose on the gantry while a few test shapes are drawn
every tick: a red sphere at a fixed world point, a marker following the right foot, a world frame.
"""
import argparse

import numpy as np

from ..sim import AsyncConfig, AsyncSim, G1Sim, SyncConfig, SyncSim
from . import colors
from .draw import Canvas


def draw(scn, state):
    c = Canvas(scn)
    c.sphere([0.4, 0.3, 0.5], 0.03, colors.RED, "red sphere @ (0.4, 0.3, 0.5)")     # fixed world point
    c.box([0.4, -0.3, 0.3], [0.03, 0.03, 0.03], colors.BLUE, label="blue box")
    c.marker(state["rfoot"], colors.YELLOW, label="right foot")                       # follows the foot
    c.line(state["lfoot"], state["rfoot"], 3.0, colors.CYAN)                           # between the feet
    c.frame([0.0, 0.0, 0.0], length=0.3)                                        # world axes at origin
    c.text(state["base_pos"] + np.array([0, -0.4, 0.6]), f"t={state['time']:.2f}s  h={state['base_pos'][2]:.3f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()
    sim = G1Sim()
    if args.mode == "sync":
        SyncSim(sim, None, SyncConfig(gantry=True, viewer=True, realtime=True)).run(args.duration, draw=draw)
    else:
        h = AsyncSim(sim, None, AsyncConfig(gantry=True)).start()
        try:
            h.run_viewer(duration=args.duration, draw=draw)
        finally:
            h.stop()


if __name__ == "__main__":
    main()
