"""The one entry point: ``python -m token_policy [options]``.

Runs the G1 sim with a chosen controller (PD hold or SONIC), in the synchronous or the
asynchronous runner, with a per-tick record; experiments draw in the viewer through run(draw_fns=...).

Timeline (same in both modes):
  [0, t_settle)          PD hold of the standing pose on the gantry; the controller only observes
  t_settle               controller attached; gantry released linearly over t_release
  t_settle + t_hold      reference starts playing (motions); residual applied from here
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, fields

import numpy as np

from .common.recorder import Recorder
from .paths import SONIC_MODEL_DIR, default_log_path
from .sim import AsyncConfig, AsyncSim, G1Sim, SyncConfig, SyncSim, hold_pose


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m token_policy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("run")
    g.add_argument("--mode", choices=["sync", "async"], default="sync", help="who advances time: the loop (deterministic) or a thread (wall clock)")
    g.add_argument("--duration", type=float, default=None, help="simulated seconds (default: unbounded)")
    g.add_argument("--headless", action="store_true", help="no viewer window")
    g.add_argument("--no-realtime", action="store_true", help="as fast as possible instead of wall-clock pace")
    g.add_argument("--log", nargs="?", const="", default=None, metavar="PATH", help="save the per-tick record; no PATH -> results/<name>_<date>_<time>.npz")
    g.add_argument("--print-period", type=float, default=0.5, help="terminal status line period (s); 0 = off")
    g = ap.add_argument_group("controller")
    g.add_argument("--controller", choices=["hold", "sonic"], default="sonic", help="hold = PD on the standing pose (falls once released)")
    g.add_argument("--reference", default="standing", help="SONIC reference: 'standing' or a motion name under assets/motions")
    g.add_argument("--model-dir", default=SONIC_MODEL_DIR)
    g.add_argument("--scale", type=float, default=0.0, help="latent residual scale lambda (0 = none)")
    g.add_argument("--seed", type=int, default=0, help="seed of the random fixed residual")
    g.add_argument("--list-motions", action="store_true")
    g = ap.add_argument_group("timeline")
    g.add_argument("--t-settle", type=float, default=1.0)
    g.add_argument("--t-release", type=float, default=1.0)
    g.add_argument("--t-hold", type=float, default=3.0)
    g.add_argument("--no-gantry", action="store_true", help="start on the floor without the gantry")
    return ap


@dataclass
class RunConfig:
    """Everything the CLI can set; experiments build one of these (parse_config) and call run()."""
    mode: str = "sync"
    duration: float | None = None
    headless: bool = False
    no_realtime: bool = False
    log: str | None = None
    print_period: float = 0.5
    controller: str = "sonic"
    reference: str = "standing"
    model_dir: str = SONIC_MODEL_DIR
    scale: float = 0.0
    seed: int = 0
    list_motions: bool = False
    t_settle: float = 1.0
    t_release: float = 1.0
    t_hold: float = 3.0
    no_gantry: bool = False


def parse_config(argv=None, **defaults) -> RunConfig:
    """CLI arguments on top of code defaults: ``parse_config(sys.argv[1:], controller="sonic")``."""
    ap = build_parser()
    ap.set_defaults(**defaults)
    args = ap.parse_args(argv)
    return RunConfig(**{f.name: getattr(args, f.name) for f in fields(RunConfig)})


def main(argv=None) -> dict:
    return run(parse_config(argv))


def run(args: RunConfig, draw_fns=(), controller=None, on_tick=None) -> dict:
    """Run one episode.

    ``draw_fns``: callables ``fn(canvas, state, ctx)`` drawn every tick in the viewer (the
    experiment decides what to show; nothing is drawn by default). ``controller``: a ready ``controller(state)`` object to use instead of the
    one built from ``args.controller`` (hold / sonic). ``on_tick(handle, state)``: extra per-tick hook.
    """
    if args.list_motions:
        from .sonic import list_motions
        print("\n".join(list_motions()))
        return {}

    sim = G1Sim()
    ctx = {"attached": False, "played": False, "band": 1.0 if not args.no_gantry else 0.0, "fallen": False,
           "controller_name": args.controller, "reference": None, "sonic": None}

    # ---- controller ----
    ref, name, hook = None, args.controller, None
    if controller is not None:                      # caller-supplied object
        ref = getattr(controller, "reference", None)
        ctx["controller_name"] = type(controller).__name__
        name = ctx["controller_name"]
    elif args.controller == "sonic":
        from .policy import RandomSource, ResidualToken
        from .sonic import SonicController, SonicJointController, SonicONNX, load_reference
        ref = load_reference(args.reference)
        hook = ResidualToken(RandomSource(args.seed), args.scale) if args.scale != 0 else None
        controller = SonicJointController(SonicController(SonicONNX(args.model_dir)), ref)   # hook armed at t_play
        name = f"sonic_{ref.name}" + (f"_s{args.scale}_seed{args.seed}" if args.scale else "")
    else:
        controller, hook = hold_pose(), None
    if ref is not None:
        ctx.update(reference=ref, sonic=controller)
    if hasattr(controller, "reset"):
        controller.reset()                          # fresh history / heading / reference for every run
    dt = sim.control_dt
    k_attach = int(round(args.t_settle / dt))
    k_play = k_attach + int(round(args.t_hold / dt))

    def schedule(handle, st):
        """Shared timeline; works on SyncSim and AsyncSim through the same four calls.
        Uses the control-tick index (sim time is an accumulated float) so events land on exact ticks."""
        k = int(round(st["time"] / dt))
        if not ctx["attached"]:
            if k >= k_attach:                        # from this tick the controller observes itself
                handle.attach(controller); ctx["attached"] = True
            elif hasattr(controller, "observe_only"):
                controller.observe_only(st)
        elif not args.no_gantry:
            handle.set_gantry(1.0 - (k - k_attach) * dt / max(args.t_release, 1e-9))
        if hasattr(controller, "play") and not ctx["played"] and k >= k_play:
            controller.play(); ctx["played"] = True
            if hook is not None:
                controller.token_hook = hook          # external token enters from here on
        ctx["band"] = handle.gantry.gain
        ctx["fallen"] = handle.sim.fallen()
        if on_tick is not None:
            on_tick(handle, st)

    def console(st, ex):
        z = ex.get("token")
        zs = "  n/a" if z is None or not np.all(np.isfinite(z)) else f"{np.linalg.norm(z):5.3f}"
        cur = (f"{ref.cursor}/{ref.length}{'P' if ref.playing else 'H'}" if ref is not None and ref.length > 1
               else ("held" if ref is not None else "-"))
        return (f"t={st['time']:5.2f}s  {args.controller}={'on ' if ctx['attached'] else 'off'}  ref={cur:10s}  |z|={zs}  "
                f"h={st['base_pos'][2]:.3f} m  rpy={np.degrees(st['base_rpy']).round(1)} deg  band={ex['band']:.2f}")
    rec = Recorder(console=console if args.print_period > 0 else None, period=args.print_period)
    draw = None
    if not args.headless and draw_fns:
        from .viz.draw import Canvas

        def draw(scn, st):
            c = Canvas(scn)
            for fn in draw_fns:
                fn(c, st, ctx)
    print(f"controller={args.controller}" + (f" reference={ref.name} (frames={ref.length}) scale={args.scale}" if ref else "")
          + f"  mode={args.mode}")

    # ---- run ----
    t_wall0 = time.time()
    if args.mode == "sync":
        ss = SyncSim(sim, None, SyncConfig(gantry=not args.no_gantry, viewer=not args.headless, realtime=not args.no_realtime))
        r = ss.run(args.duration, on_tick=schedule, log=rec, draw=draw)
        ticks, t_end, fell = len(r["t"]), r["t"][-1], np.isfinite(r["fell_at"])
        fell_at, extra = r["fell_at"], ""
    else:
        h = AsyncSim(sim, None, AsyncConfig(gantry=not args.no_gantry, realtime_factor=0.0 if args.no_realtime else 1.0), log=rec).start()
        try:
            if not args.headless:
                h.run_viewer(schedule, duration=args.duration, draw=draw)
            else:
                while h.running() and (args.duration is None or h.time < args.duration):
                    schedule(h, h.state()); time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            h.stop()
        ticks, t_end, fell = h.ticks, h.time, h.sim.fallen()
        fell_at = h.time if fell else np.nan
        extra = f"  |  wall {time.time() - t_wall0:.2f}s, overruns {h.overruns}"
    a = rec.arrays()
    a["fell_at"] = fell_at
    print(f"done [{args.mode}]: {ticks} ticks, sim time {t_end:.2f}s, final height {a['base_pos'][-1, 2]:.3f} m, "
          f"fell {'yes' if fell else 'no'}{extra}")
    if args.log is not None:
        path = rec.save(args.log or default_log_path(name), controller=args.controller,
                        reference=ref.name if ref else "", fell_at=fell_at)
        print("saved", path, summarize(a))
    return a


def summarize(a: dict) -> str:
    """Stability over the ticks where the controller ran alone (gantry released)."""
    m = a["band"] == 0.0
    if "token" in a:
        m &= np.isfinite(a["token"][:, 0])
    if not m.any():
        return "(controller never ran alone: gantry still attached)"
    rpy = np.degrees(a["base_rpy"][m]); h = a["base_pos"][m][:, 2]
    return (f"| alone {m.sum()} ticks: roll {rpy[:, 0].mean():+.2f} (max |{np.abs(rpy[:, 0]).max():.2f}|) deg, "
            f"pitch {rpy[:, 1].mean():+.2f} (max |{np.abs(rpy[:, 1]).max():.2f}|) deg, "
            f"height {h.mean():.4f} [{h.min():.4f}, {h.max():.4f}] m, |q_target - q| max {np.abs(a['q_target'][m] - a['q'][m]).max():.3f} rad")


if __name__ == "__main__":
    main(sys.argv[1:])
