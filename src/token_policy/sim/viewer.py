"""Plain mujoco.viewer window for the standalone sim runs (no markers; see token_policy.viz for those)."""
from __future__ import annotations

import threading
import time

from .g1_sim import G1Sim


def open_viewer(sim: G1Sim):
    import mujoco.viewer
    v = mujoco.viewer.launch_passive(sim.model, sim.data)
    v.opt.sitegroup[:] = 0   # hide the scene's fixed com_marker site
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.5, 135, -15
    return v


def close_viewer(v, timeout: float = 5.0):
    """Close the window and wait for the viewer's GUI thread to finish.

    ``Handle.close()`` only requests the exit; the GUI thread tears the window down afterwards.
    Returning to the interpreter shutdown before that finishes segfaults, so join that thread.
    """
    import mujoco.viewer
    v.close()
    for t in threading.enumerate():
        if getattr(t, "_target", None) is mujoco.viewer._launch_internal:
            t.join(timeout)
    time.sleep(0.05)
