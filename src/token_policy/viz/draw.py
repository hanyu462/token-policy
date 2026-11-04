"""Primitive drawing on a MuJoCo scene (viewer.user_scn or renderer.scene).

All positions are world coordinates in metres. Geoms are visual only: nothing here touches
the physics. Clear and redraw every tick to animate.

    c = Canvas(viewer.user_scn)
    c.clear()
    c.sphere([0.3, 0.0, 0.5], 0.03, RED, "here")
"""
from __future__ import annotations

import mujoco
import numpy as np

# (r, g, b, a)
RED = (1.0, 0.1, 0.1, 1.0)
GREEN = (0.1, 0.9, 0.1, 1.0)
BLUE = (0.1, 0.4, 1.0, 1.0)
YELLOW = (1.0, 0.9, 0.1, 1.0)
ORANGE = (1.0, 0.5, 0.0, 1.0)
CYAN = (0.0, 0.8, 0.8, 1.0)
MAGENTA = (1.0, 0.0, 1.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
GRAY = (0.6, 0.6, 0.6, 0.8)
BLACK = (0.0, 0.0, 0.0, 1.0)

_UP = np.array([0.0, 0.0, 1.0])


class Canvas:
    def __init__(self, scn: mujoco.MjvScene):
        self.scn = scn

    def clear(self):
        self.scn.ngeom = 0

    # ---- primitives --------------------------------------------------------
    def _geom(self, gtype, size, pos, rot, color, label):
        scn = self.scn
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        rot = np.eye(3) if rot is None else np.asarray(rot, float)
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                            rot.ravel(), np.asarray(color, np.float32))
        g.label = label
        scn.ngeom += 1
        return g

    def _connector(self, gtype, width, p0, p1, color, label):
        scn = self.scn
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color, np.float32))
        mujoco.mjv_connector(g, gtype, width, np.asarray(p0, float), np.asarray(p1, float))
        g.label = label
        scn.ngeom += 1
        return g

    def sphere(self, pos, radius=0.01, color=RED, label=""):
        return self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], pos, None, color, label)

    def box(self, pos, half_size, color=BLUE, rot=None, label=""):
        """``half_size``: (x, y, z) half extents; ``rot``: 3x3 rotation matrix (world)."""
        return self._geom(mujoco.mjtGeom.mjGEOM_BOX, half_size, pos, rot, color, label)

    def cylinder(self, pos, radius, half_height, color=GREEN, rot=None, label=""):
        return self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [radius, half_height, 0], pos, rot, color, label)

    def line(self, p0, p1, width=2.0, color=WHITE, label=""):
        """``width`` in pixels."""
        return self._connector(mujoco.mjtGeom.mjGEOM_LINE, width, p0, p1, color, label)

    def arrow(self, p0, p1, width=0.01, color=WHITE, label=""):
        """``width`` in metres (shaft radius)."""
        return self._connector(mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1, color, label)

    def text(self, pos, s: str):
        """Text only (an invisible point carrying a label)."""
        return self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], pos, None, (0, 0, 0, 0), s)

    # ---- composites ----------------------------------------------------------
    def frame(self, origin, R=None, length=0.2, width=0.01, labels=("x", "y", "z")):
        """Right-handed axes at ``origin`` with columns of ``R`` (world); x red, y green, z blue."""
        R = np.eye(3) if R is None else np.asarray(R, float)
        o = np.asarray(origin, float)
        for i, (col, lab) in enumerate(zip((RED, GREEN, BLUE), labels)):
            self.arrow(o, o + length * R[:, i], width, col, lab)

    def trace(self, points, color=ORANGE, width=4.0, max_points=60):
        pts = list(points)[-max_points:]
        for a, b in zip(pts[:-1], pts[1:]):
            self.line(a, b, width, color)

    def marker(self, pos, color=RED, radius=0.012, lift=0.08, label=""):
        """Sphere floating ``lift`` above ``pos`` with a line down to it (points inside geometry stay visible)."""
        p = np.asarray(pos, float)
        self.line(p, p + lift * _UP, 2.0, color)
        return self.sphere(p + lift * _UP, radius, color, label)
