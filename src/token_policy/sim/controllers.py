"""Controller interface shared by SyncSim and AsyncSim, plus trivial controllers.

    controller(state: dict) -> (q_target, kp, kd) | None      # None = keep the last command

``state`` is the dict returned by ``G1Sim.state()``; q_target / kp / kd are 29-vectors in MuJoCo order.
"""
from __future__ import annotations

import numpy as np

from .g1_params import DEFAULT_ANGLES, KD, KP


def hold_pose(q: np.ndarray | None = None):
    """Controller that commands a fixed pose with the default PD gains."""
    q = DEFAULT_ANGLES if q is None else np.asarray(q, float)
    return lambda state: (q, KP, KD)
