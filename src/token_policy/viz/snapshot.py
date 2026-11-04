"""Offscreen rendering (EGL) of the sim with the same ``draw(canvas, state, ctx)`` callbacks the
viewer uses: PNG snapshots and frame capture for videos.  Set MUJOCO_GL=egl for headless use."""
from __future__ import annotations

import os

import mujoco
import numpy as np

from .draw import Canvas

VIEWS = {   # azimuth is relative to the robot heading (0 = camera behind the robot)
    "front": (180, -12, 1.6), "back": (0, -12, 1.6), "left": (90, -10, 1.6), "right": (-90, -10, 1.6),
    "iso": (140, -25, 2.0), "top": (180, -70, 1.6),
}


class Snapshot:
    def __init__(self, sim, width: int = 800, height: int = 600):
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.sim = sim
        sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
        sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(sim.model, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, draw_fns=(), ctx=None, view: str = "iso", lookat=None, distance=None) -> np.ndarray:
        """Render the current sim state; ``draw_fns`` are ``fn(canvas, state, ctx)`` like in the viewer."""
        from ..common.rotations import calc_heading
        st = self.sim.state()
        az, el, dist = VIEWS[view]
        yaw = np.degrees(calc_heading(st["base_quat"]))
        self.cam.lookat[:] = st["base_pos"] - np.array([0, 0, 0.25]) if lookat is None else np.asarray(lookat, float)
        self.cam.azimuth, self.cam.elevation, self.cam.distance = yaw + az, el, dist if distance is None else distance
        self.renderer.update_scene(self.sim.data, self.cam, self.opt)
        c = Canvas(self.renderer.scene)
        for fn in draw_fns:
            fn(c, st, ctx if ctx is not None else {})
        return self.renderer.render().copy()

    def save(self, path: str, draw_fns=(), ctx=None, views=("front", "iso"), title: str | None = None, **kw) -> str:
        import imageio.v2 as imageio
        img = np.concatenate([self.render(draw_fns, ctx, v, **kw) for v in views], axis=1)
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


def write_video(path: str, frames, fps: int = 25) -> str:
    import imageio.v2 as imageio
    imageio.mimwrite(path, list(frames), fps=fps, quality=7)
    return path
