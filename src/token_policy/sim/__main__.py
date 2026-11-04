"""python -m token_policy.sim [--mode sync|async] [--headless] [--no-realtime] [--duration S] [--no-gantry] [--release S]

Options mean the same in both modes. Defaults: viewer on, real time, gantry on, unbounded
(runs until the window is closed / Ctrl-C, or a fall in sync mode)."""
import argparse
import time

import numpy as np

from .async_sim import AsyncConfig, AsyncSim
from .g1_sim import G1Sim
from .sync_sim import SyncConfig, SyncSim


def _fmt(st, band):
    return (f"t={st['time']:5.2f}s  h={st['base_pos'][2]:.3f} m  rpy={np.degrees(st['base_rpy']).round(1)} deg  "
            f"feet N=({st['lfoot_force']:.0f},{st['rfoot_force']:.0f})  band={band:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Run the G1 sim with no controller (PD hold of the standing pose).")
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--duration", type=float, default=None, help="simulated seconds to run (default: unbounded)")
    ap.add_argument("--no-gantry", action="store_true", help="start on the floor without the virtual gantry")
    ap.add_argument("--release", type=float, default=None, help="ramp the gantry off over this many seconds from t=0")
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--no-realtime", action="store_true", help="run as fast as possible instead of wall-clock pace")
    args = ap.parse_args()
    sim = G1Sim()
    last = [-1.0]          # last printed 0.5 s grid index; prints at t = 0.0, 0.5, 1.0, ... without drift

    def due(t):
        g = int(t // 0.5)
        if g > last[0]:
            last[0] = g
            return True
        return False

    if args.mode == "sync":
        ss = SyncSim(sim, None, SyncConfig(gantry=not args.no_gantry, t_release=args.release,
                                           viewer=not args.headless, realtime=not args.no_realtime))

        def on_tick(s, st):
            if due(st["time"]):
                print(_fmt(st, s.gantry.gain))
        r = ss.run(args.duration, on_tick=on_tick)
        print(f"done [sync]: {len(r['t'])} ticks, sim time {r['t'][-1]:.2f}s, final height {r['base_pos'][-1, 2]:.3f} m, "
              f"fell {'at %.2fs' % r['fell_at'] if np.isfinite(r['fell_at']) else 'no'}")
        return

    h = AsyncSim(sim, None, AsyncConfig(gantry=not args.no_gantry,
                                        realtime_factor=0.0 if args.no_realtime else 1.0)).start()
    t_wall0 = time.time()

    def on_tick(handle, st):
        if args.release is not None:                      # external "operator" ramps the gantry off
            handle.set_gantry(1.0 - st["time"] / args.release)
        if due(st["time"]):
            print(_fmt(st, handle.gantry.gain))
    try:
        if not args.headless:
            h.run_viewer(on_tick, duration=args.duration)
        else:
            while h.running() and (args.duration is None or h.time < args.duration):
                on_tick(h, h.state())
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        h.stop()
    st = h.state()
    print(f"done [async]: {h.ticks} ticks, sim time {h.time:.2f}s, final height {st['base_pos'][2]:.3f} m, "
          f"fell {'yes' if h.sim.fallen() else 'no'}  |  wall {time.time() - t_wall0:.2f}s, overruns {h.overruns}")


if __name__ == "__main__":
    main()
