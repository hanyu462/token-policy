"""Visualization helpers: task markers in an mjvScene (viewer or offscreen), terminal diagnostics.

Everything here only *reads* the simulation state and writes into a
visualization scene, so headless runs and viewer runs share identical dynamics.
"""
from __future__ import annotations

import os
import sys
import time

import mujoco
import numpy as np

RGBA_INIT = (0.1, 0.4, 1.0, 1.0)      # blue   : initial right foot p_R0
RGBA_TARGET = (0.1, 0.9, 0.1, 1.0)    # green  : target p_R*
RGBA_NOW = (1.0, 0.1, 0.1, 1.0)       # red    : current right foot
RGBA_TRACE = (1.0, 0.5, 0.0, 0.9)     # orange : recent right-foot trajectory
RGBA_FWD = (1.0, 0.0, 1.0, 1.0)       # magenta: forward axis
RGBA_LAT = (0.0, 0.8, 0.8, 1.0)       # cyan   : lateral (+) axis
RGBA_RULER = (0.6, 0.6, 0.6, 0.7)


def _add_geom(scn, gtype, size, pos, rgba, label=""):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    g.label = label
    scn.ngeom += 1


def _add_connector(scn, gtype, width, p0, p1, rgba, label=""):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, gtype, width, np.asarray(p0, float), np.asarray(p1, float))
    g.label = label
    scn.ngeom += 1


def _label_at(scn, pos, label, rgba=(0, 0, 0, 0)):
    """Invisible tiny sphere that only carries a text label (lets labels be spread out)."""
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], pos, rgba, label)


def add_task_geoms(scn: mujoco.MjvScene, task, p_R_now, trace=None, max_trace=60, labels=True, p_L_now=None):
    """Draw p_R0 (blue), p_R* (green pole), current right foot (red), trace, and the F0 axes."""
    if task is None:
        return
    sph, line, arrow = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_LINE, mujoco.mjtGeom.mjGEOM_ARROW
    up = np.array([0.0, 0.0, 1.0])
    lat = task.frame.lateral
    # +-3 cm lateral ruler on the floor through p_R0 with 1 cm ticks (the goal range)
    floor = np.array([task.p_R0[0], task.p_R0[1], 0.003])
    _add_connector(scn, line, 3.0, floor - 0.03 * lat, floor + 0.03 * lat, RGBA_RULER)
    for k in range(-3, 4):
        _add_connector(scn, line, 2.0, floor + 0.01 * k * lat, floor + 0.01 * k * lat + 0.015 * task.frame.forward, RGBA_RULER)
    # marker "flags" sit in front of the toes (the foot points themselves are inside the foot/shin
    # geometry); a line joins each flag to its true point.  Heights differ so they never occlude.
    fwd = task.frame.forward
    off_init = 0.17 * fwd + 0.015 * up
    off_tgt = 0.17 * fwd + 0.045 * up
    off_now = 0.17 * fwd + 0.075 * up
    _add_geom(scn, sph, [0.013, 0, 0], task.p_R0 + off_init, RGBA_INIT)
    _add_connector(scn, line, 3.0, task.p_R0, task.p_R0 + off_init, RGBA_INIT)
    _add_geom(scn, sph, [0.013, 0, 0], task.p_target + off_tgt, RGBA_TARGET)
    _add_connector(scn, line, 4.0, [task.p_target[0], task.p_target[1], 0.0], task.p_target + 0.06 * up, RGBA_TARGET)
    _add_connector(scn, line, 3.0, task.p_target, task.p_target + off_tgt, RGBA_TARGET)
    _add_geom(scn, sph, [0.011, 0, 0], p_R_now + off_now, RGBA_NOW)
    _add_connector(scn, line, 3.0, p_R_now, p_R_now + off_now, RGBA_NOW)
    if p_L_now is not None:
        _add_geom(scn, sph, [0.008, 0, 0], p_L_now + off_init, (1, 1, 1, 0.8))
        _add_connector(scn, line, 2.0, p_L_now, p_L_now + off_init, (1, 1, 1, 0.6))
    if trace is not None and len(trace) > 1:
        pts = trace[-max_trace:]
        for a, b in zip(pts[:-1], pts[1:]):
            _add_connector(scn, line, 5.0, a, b, RGBA_TRACE)
    # local frame F0 drawn 0.45 m in front of the pelvis so it does not overlap the feet
    o = np.array([task.frame.origin[0], task.frame.origin[1], 0.02]) + 0.40 * task.frame.forward - 0.25 * lat
    _add_connector(scn, arrow, 0.02, o, o + 0.22 * task.frame.forward, RGBA_FWD)
    _add_connector(scn, arrow, 0.02, o, o + 0.22 * lat, RGBA_LAT)
    _add_connector(scn, arrow, 0.012, o, o + 0.12 * up, (1, 1, 1, 0.8))
    if labels:
        _label_at(scn, task.p_R0 + off_init + 0.03 * fwd, "init R")
        _label_at(scn, task.p_target + off_tgt + 0.03 * fwd, f"target d={task.d:+.3f}")
        _label_at(scn, p_R_now + off_now + 0.03 * fwd, "R now")
        if p_L_now is not None:
            _label_at(scn, p_L_now + off_init + 0.03 * fwd, "L foot")
        _label_at(scn, o + 0.27 * task.frame.forward, "forward")
        _label_at(scn, o + 0.27 * lat + 0.04 * up, "lateral(+) = robot left")


class TerminalDiagnostics:
    """Low-rate terminal print of the task quantities (default 5 Hz)."""

    def __init__(self, period_s: float = 0.2, stream=sys.stdout):
        self.period = period_s
        self.stream = stream
        self.next_t = -np.inf

    def __call__(self, sim, info):
        if info["task"] is None or info["t_run"] < self.next_t:
            return
        self.next_t = info["t_run"] + self.period - 1e-9
        ev, rpy, st = info["ev"], info["rpy"], info["ev"]
        self.stream.write(
            f"t={info['t_run']:5.2f}s  goal: {info['d']:+.3f} m  right foot: {ev['lateral_disp']:+.3f} m  "
            f"error: {ev['error']:+.3f} m  target reward: {ev['reward']:.3f}  left foot drift: {ev['left_drift']:.3f} m  "
            f"roll/pitch: {np.degrees(rpy[0]):+.1f} / {np.degrees(rpy[1]):+.1f} deg  "
            f"hold: {st['hold_s']:.2f}s{'  SUCCESS' if st['success'] else ''}{'  FALL' if info['fallen'] else ''}\n")
        self.stream.flush()


class ViewerSession:
    """Passive MuJoCo viewer; call ``tick`` once per control step from the runner's on_tick hook."""

    def __init__(self, sim, realtime: bool = True):
        self.sim = sim
        self.realtime = realtime
        self.viewer = None
        self.t_wall0 = None

    def __enter__(self):
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        self.viewer.opt.sitegroup[:] = 0  # hide the scene's fixed com_marker site
        self.viewer.cam.lookat[:] = self.sim.base_pos()
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 150
        self.viewer.cam.elevation = -15
        self.t_wall0 = time.time() - self.sim.time
        return self

    def __exit__(self, *exc):
        if self.viewer is not None:
            from ..sim.viewer import close_viewer
            close_viewer(self.viewer)

    def running(self):
        return self.viewer is not None and self.viewer.is_running()

    def tick(self, sim, info):
        if not self.running():
            return
        scn = self.viewer.user_scn
        scn.ngeom = 0
        add_task_geoms(scn, info["task"], info["rf"], info["trace"], p_L_now=info["lf"])
        self.viewer.sync()
        if self.realtime:
            lag = self.t_wall0 + sim.time - time.time()
            if lag > 0:
                time.sleep(lag)


class Snapshotter:
    """Offscreen (EGL) renderer producing labelled PNG snapshots with the task markers."""

    def __init__(self, sim, width=800, height=600):
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.sim = sim
        # offscreen framebuffer size is a visual-only model setting (no effect on dynamics)
        sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
        sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(sim.model, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0  # hide the scene's fixed red com_marker site

    def render(self, task, p_R_now, trace, view="front", p_L_now=None):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        base = self.sim.base_pos()
        yaw = np.degrees(task.frame.yaw) if task is not None else 0.0
        fwd = task.frame.forward if task is not None else np.array([1.0, 0, 0])
        feet = 0.5 * (task.p_R0 + task.p_L0) if task is not None else base
        cam.lookat[:] = np.array([feet[0], feet[1], 0.06]) + 0.12 * fwd
        # MuJoCo free camera: azimuth == heading puts the camera BEHIND the robot (looking along forward)
        if view == "front":          # camera in front of the robot, looking back at it (robot-left = image right)
            cam.distance, cam.azimuth, cam.elevation = 0.8, yaw + 180, -14
        elif view == "top":          # steep view from the front so the feet are not hidden by the torso
            cam.distance, cam.azimuth, cam.elevation = 0.85, yaw + 180, -65
        elif view == "back":         # from behind (robot-left = image left)
            cam.distance, cam.azimuth, cam.elevation = 1.1, yaw, -14
        else:                        # iso, front-left
            cam.distance, cam.azimuth, cam.elevation = 1.3, yaw + 140, -28
        self.renderer.update_scene(self.sim.data, cam, self.opt)
        add_task_geoms(self.renderer.scene, task, p_R_now, trace, p_L_now=p_L_now)
        return self.renderer.render().copy()

    def save(self, path, task, p_R_now, trace, views=("front", "top"), title=None, p_L_now=None):
        import imageio.v2 as imageio
        imgs = [self.render(task, p_R_now, trace, v, p_L_now=p_L_now) for v in views]
        img = np.concatenate(imgs, axis=1)
        if title:
            try:
                from PIL import Image, ImageDraw, ImageFont
                im = Image.fromarray(img)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 22)
                except Exception:  # noqa: BLE001
                    font = ImageFont.load_default()
                ImageDraw.Draw(im).text((12, 10), title, fill=(255, 255, 0), font=font)
                img = np.asarray(im)
            except Exception:  # noqa: BLE001
                pass
        imageio.imwrite(path, img)
        return path


# ---- generic status overlay / generic draw hooks (used by experiments/sonic_viewer.py) ----------

def add_status_geoms(scn, anchor, lines, line_gap=0.07):
    """Stack text lines (label-only geoms) above ``anchor`` (world xyz), first line on top."""
    for i, text in enumerate(lines):
        _label_at(scn, np.asarray(anchor, float) + np.array([0, 0, (len(lines) - 1 - i) * line_gap]), text)


def add_foot_geoms(scn, lf, rf):
    up = np.array([0, 0, 1.0])
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.01, 0, 0], lf + 0.09 * up, (1, 1, 1, 0.9))
    _add_connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2.0, lf, lf + 0.09 * up, (1, 1, 1, 0.6))
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.01, 0, 0], rf + 0.09 * up, RGBA_NOW)
    _add_connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2.0, rf, rf + 0.09 * up, RGBA_NOW)


class GenericViewerSession(ViewerSession):
    """Passive viewer whose per-tick drawing is a user callback ``draw(scn, sim, info)``."""

    def __init__(self, sim, draw, realtime=True, follow=True):
        super().__init__(sim, realtime)
        self.draw = draw
        self.follow = follow

    def tick(self, sim, info):
        if not self.running():
            return
        scn = self.viewer.user_scn
        scn.ngeom = 0
        self.draw(scn, sim, info)
        if self.follow:
            self.viewer.cam.lookat[:] = sim.base_pos()
        self.viewer.sync()
        if self.realtime:
            lag = self.t_wall0 + sim.time - time.time()
            if lag > 0:
                time.sleep(lag)


class GenericSnapshotter(Snapshotter):
    """Offscreen renderer with a user draw callback; camera follows the pelvis."""

    def render_with(self, draw, info, view="iso", distance=2.6):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        base = self.sim.base_pos()
        cam.lookat[:] = [base[0], base[1], 0.62]
        az = {"front": 180, "back": 0, "side": -90, "iso": 135}[view]
        el = {"front": -10, "back": -10, "side": -8, "iso": -18}[view]
        cam.distance, cam.azimuth, cam.elevation = distance, az, el
        self.renderer.update_scene(self.sim.data, cam, self.opt)
        draw(self.renderer.scene, self.sim, info)
        return self.renderer.render().copy()

    def save_with(self, path, draw, info, views=("front", "iso"), title=None):
        import imageio.v2 as imageio
        img = np.concatenate([self.render_with(draw, info, v) for v in views], axis=1)
        if title:
            try:
                from PIL import Image, ImageDraw, ImageFont
                im = Image.fromarray(img)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 22)
                except Exception:  # noqa: BLE001
                    font = ImageFont.load_default()
                ImageDraw.Draw(im).text((12, 10), title, fill=(255, 255, 0), font=font)
                img = np.asarray(im)
            except Exception:  # noqa: BLE001
                pass
        imageio.imwrite(path, img)
        return path
