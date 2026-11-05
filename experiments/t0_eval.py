#!/usr/bin/env python3
"""Roll out a trained T0 teacher: per-command success / displacement; optional live viewer or video.

  .venv/bin/python experiments/t0_eval.py results/t0_ft/ppo_t0.zip --action gaussian --action-repeat 25 --deterministic
  ... --viewer                 # live MuJoCo window (real time), camera on the right foot
  ... --video results/t0_ft/eval.mp4 [--view right|front|iso]   # offscreen EGL video, 50 fps = sim rate
"""
import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from token_policy.env import TaskConfig
from token_policy.env.gym_wrapper import FootTaskGym
from token_policy.viz import colors


def make_draw(task):
    """Markers: start foot (gray), goal (green), effector foot (red) + arrow to the goal, status text."""
    def draw(c, st, ctx):
        p = st[task._eff_key()]
        c.marker(task.p_eff0, colors.GRAY, 0.008, 0.06, "start")
        c.marker(task.p_target, colors.GREEN, 0.008, 0.06, "goal")
        c.sphere(p, 0.012, colors.RED)
        c.arrow(p, task.p_target, 0.003, colors.YELLOW)
        info = ctx.get("info")
        if info is not None:
            c.text(p + [0, 0, 0.25], f"cmd {ctx['cmd_mm']:+.0f} mm  t={st['time']:.2f}s  dist {info['dist']*1e3:.1f} mm  "
                                     f"{info['phase']}{'  SUCCESS' if info['success'] else ''}")
    return draw


def rollout(env: FootTaskGym, model, cmd, seed, deterministic, on_tick=None):
    """Policy decision every ``action_repeat`` ticks; ``on_tick(state, info)`` after every control tick."""
    task = env.env
    obs, _ = env.reset(seed=seed, options={"command": cmd})
    ret, done, info = 0.0, False, None
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        dk = (np.clip(np.rint(np.asarray(a, np.float64)), -env.dk, env.dk) if env.action_mode == "gaussian"
              else np.asarray(a, np.int64) - env.dk).astype(np.int64)
        for _ in range(env.action_repeat):
            obs, r, done, info = task.step(dk)
            ret += r
            if on_tick is not None:
                on_tick(task.state(), info)
            if done:
                break
        obs = obs.astype(np.float32)
    return info, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--dk-range", type=int, default=2)
    ap.add_argument("--action-repeat", type=int, default=10)
    ap.add_argument("--action", default="discrete", choices=["discrete", "gaussian"])
    ap.add_argument("--viewer", action="store_true", help="live mujoco.viewer window, real time")
    ap.add_argument("--video", default=None, help="write an mp4 of the episodes (offscreen EGL)")
    ap.add_argument("--view", default="right", help="video camera: right | front | iso | left | back | top")
    ap.add_argument("--distance", type=float, default=1.1, help="camera distance (m) for viewer / video")
    args = ap.parse_args()
    cfg = TaskConfig(onnx_threads=4, snapshot_path="results/standing_snapshot.npz", effector="R",
                     commands=((0.0, 0.01), (0.0, -0.01)), dk_range=args.dk_range, record=True, decoder_backend="torch")
    env = FootTaskGym(cfg, action_repeat=args.action_repeat, action_mode=args.action)
    task, sim = env.env, env.env.sim
    model = PPO.load(args.model, device="cpu")
    draw, ctx = make_draw(task), {}

    viewer = snap = None
    frames = []
    if args.viewer:
        from token_policy.sim.viewer import open_viewer
        from token_policy.viz.draw import Canvas
        viewer = open_viewer(sim)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = args.distance, 90, -15
    if args.video:
        from token_policy.viz.snapshot import Snapshot
        snap = Snapshot(sim, 960, 720)

    t_wall = [time.time()]

    def on_tick(st, info):
        ctx["info"] = info
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw(Canvas(viewer.user_scn), st, ctx)
                viewer.cam.lookat[:] = st[task._eff_key()] + [0, 0, 0.1]
            viewer.sync()
            lag = t_wall[0] + st["time"] - time.time()
            if lag > 0:
                time.sleep(lag)
        if snap is not None:
            frames.append(snap.render([draw], ctx, args.view, lookat=st[task._eff_key()] + [0, 0, 0.1], distance=args.distance))

    rows = []
    try:
        for ep in range(args.episodes):
            cmd = cfg.commands[ep % len(cfg.commands)]
            ctx["cmd_mm"] = cmd[1] * 1e3
            t_wall[0] = time.time()
            info, ret = rollout(env, model, cmd, ep, args.deterministic, on_tick)
            rows.append((cmd[1] * 1e3, info["disp_local"][1] * 1e3, info["dist"] * 1e3, info["success"], info["fallen"], info["t"], ret))
            print(f"ep {ep}: cmd lateral {cmd[1]*1e3:+.0f} mm -> disp {info['disp_local'][1]*1e3:+.1f} mm (fwd {info['disp_local'][0]*1e3:+.1f}), "
                  f"dist {info['dist']*1e3:.1f} mm, success {info['success']}, fallen {info['fallen']}, len {info['t']}, return {ret:+.1f}")
            if snap is not None:   # hold the last frame for 0.5 s between episodes
                frames.extend([frames[-1]] * 25)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if viewer is not None:
            from token_policy.sim.viewer import close_viewer
            close_viewer(viewer)
    if rows:
        r = np.array(rows, dtype=float)
        print(f"\nsuccess {r[:,3].mean():.2f}  fall {r[:,4].mean():.2f}  mean final dist {r[:,2].mean():.1f} mm  mean return {r[:,6].mean():+.1f}")
    if snap is not None and frames:
        from token_policy.viz.snapshot import write_video
        print("video", write_video(args.video, frames, fps=50))


if __name__ == "__main__":
    main()
