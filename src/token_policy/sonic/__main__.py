"""python -m token_policy.sonic [--mode sync|async] [--reference standing|<motion>] [--headless] [--no-realtime]
                               [--duration S] [--t-hold S] [--scale L --seed N]

SONIC driving the G1 in the sim. Timeline: 1 s PD hold on the gantry (SONIC only observes),
then SONIC takes over while the gantry is released over 1 s, then (motions) play after --t-hold.
"""
import argparse
import time

import numpy as np

from ..paths import SONIC_MODEL_DIR
from ..sim import AsyncConfig, AsyncSim, G1Sim, SyncConfig, SyncSim
from .controller import SonicController, SonicJointController
from .onnx_policy import SonicONNX
from .reference import list_motions, load_reference

T_SETTLE, T_RELEASE = 1.0, 1.0


class RandomFixedResidual:
    def __init__(self, seed, dim=64):
        self.value = np.random.default_rng(seed).standard_normal(dim)

    def reset(self):
        pass

    def __call__(self, state):
        return self.value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--model-dir", default=SONIC_MODEL_DIR)
    ap.add_argument("--reference", default="standing")
    ap.add_argument("--list-motions", action="store_true")
    ap.add_argument("--duration", type=float, default=None, help="simulated seconds (default: unbounded)")
    ap.add_argument("--t-hold", type=float, default=3.0, help="seconds after takeover before a motion plays")
    ap.add_argument("--scale", type=float, default=0.0, help="residual scale lambda (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-realtime", action="store_true")
    args = ap.parse_args()
    if args.list_motions:
        print("\n".join(list_motions())); return

    sim = G1Sim()
    ref = load_reference(args.reference)
    residual = RandomFixedResidual(args.seed) if args.scale != 0 else None
    sonic = SonicJointController(SonicController(SonicONNX(args.model_dir)), ref, residual, args.scale)
    t_takeover, t_play = T_SETTLE, T_SETTLE + args.t_hold
    print(f"reference={ref.name} (frames={ref.length})  residual scale={args.scale}  mode={args.mode}")

    phase = {"attached": False, "played": False, "grid": -1}

    def schedule(handle, st):
        """Shared timeline: observe -> attach + release -> play. Works on SyncSim and AsyncSim alike."""
        t = st["time"]
        if not phase["attached"]:
            sonic.observe_only(st)                       # warm the proprio history during the PD hold
            if t >= t_takeover:
                handle.attach(sonic); phase["attached"] = True
        else:
            handle.set_gantry(1.0 - (t - t_takeover) / T_RELEASE)
            if not phase["played"] and t >= t_play:
                sonic.play(); phase["played"] = True
        g = int(t // 0.5)
        if g > phase["grid"]:
            phase["grid"] = g
            z = sonic.last["z"] if sonic.last is not None else None
            cur = f"{ref.cursor}/{ref.length}{'P' if ref.playing else 'H'}" if ref.length > 1 else "held"
            print(f"t={t:5.2f}s  sonic={'on ' if phase['attached'] else 'off'}  ref={cur:10s}  "
                  f"|z|={'  n/a' if z is None else f'{np.linalg.norm(z):5.3f}'}  h={st['base_pos'][2]:.3f} m  "
                  f"rpy={np.degrees(st['base_rpy']).round(1)} deg  band={handle.gantry.gain:.2f}")

    if args.mode == "sync":
        ss = SyncSim(sim, None, SyncConfig(gantry=True, viewer=not args.headless, realtime=not args.no_realtime))
        r = ss.run(args.duration, on_tick=schedule)
        print(f"done [sync]: {len(r['t'])} ticks, sim time {r['t'][-1]:.2f}s, final height {r['base_pos'][-1, 2]:.3f} m, "
              f"fell {'at %.2fs' % r['fell_at'] if np.isfinite(r['fell_at']) else 'no'}")
        return

    h = AsyncSim(sim, None, AsyncConfig(gantry=True, realtime_factor=0.0 if args.no_realtime else 1.0)).start()
    t_wall0 = time.time()
    try:
        if not args.headless:
            h.run_viewer(schedule, duration=args.duration)
        else:
            while h.running() and (args.duration is None or h.time < args.duration):
                schedule(h, h.state())
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        h.stop()
    st = h.state()
    print(f"done [async]: {h.ticks} ticks, sim time {h.time:.2f}s, final height {st['base_pos'][2]:.3f} m, "
          f"fell {'yes' if h.sim.fallen() else 'no'}  |  wall {time.time() - t_wall0:.2f}s, overruns {h.overruns}")


if __name__ == "__main__":
    main()
