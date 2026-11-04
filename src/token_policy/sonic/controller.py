"""SONIC controller = ONNX encoder/decoder + the state it carries between ticks
(proprioception history, heading alignment, last action)."""
from __future__ import annotations

import numpy as np

from ..sim.g1_params import NUM_JOINTS
from .onnx_policy import HeadingState, SonicONNX, StateHistory


class SonicController:
    def __init__(self, policy: SonicONNX):
        self.policy = policy
        self.token_dim = policy.token_dim
        self.reset()

    def reset(self):
        self.hist = StateHistory()
        self.heading = HeadingState()
        self.last_action = np.zeros(NUM_JOINTS)

    def observe(self, state: dict):
        """Log the measured robot state for the his_* observations (deploy: gather_robot_state)."""
        self.hist.log(state["base_quat"], state["base_ang_vel"], state["q"], state["dq"], self.last_action)

    def reset_heading(self, base_quat, ref_root_quat):
        self.heading.reset(base_quat, ref_root_quat)

    def act(self, reference, base_quat, delta_z, scale: float, inject: bool) -> dict:
        """One tick: reference -> encoder -> (token + scale*delta_z) -> decoder -> action (IsaacLab order)."""
        out = self.policy.step(reference, base_quat, self.heading, self.hist, delta_z, scale, inject=inject)
        self.last_action = out["action"]
        return out


class SonicJointController:
    """SONIC as a sim controller: ``controller(state) -> (q_target, kp, kd)`` for SyncSim / AsyncSim.

    Wraps SonicController + a ReferenceProvider (+ optional residual source) into the one-call
    interface the sim runners use. Per tick: log the state, run encoder -> token (+ scale * delta_z)
    -> decoder, advance the reference, return joint targets. Call ``play()`` to start a motion.
    """

    def __init__(self, ctrl: SonicController, reference, residual_source=None, scale: float = 0.0):
        from ..sim.g1_params import KD, KP, action_to_q_target
        self._to_q, self._kp, self._kd = action_to_q_target, KP, KD
        self.ctrl = ctrl
        self.reference = reference
        self.residual = residual_source
        self.scale = scale
        self.last = None            # last policy output dict (z_stand, z, action, action_base)
        self._zero = np.zeros(ctrl.token_dim)
        self.reset()

    def reset(self):
        self.ctrl.reset()
        self.reference.reset()
        if self.residual is not None:
            self.residual.reset()
        self._need_heading = True
        self.last = None

    def observe_only(self, state: dict):
        """Feed the state without acting (e.g. while a PD hold runs before SONIC takes over)."""
        self.ctrl.observe(state)

    def play(self):
        self.reference.play()

    def log_extras(self) -> dict:
        """Per-tick values for the sim's Recorder (sim just stores them): token and raw action."""
        if self.last is None:
            nan = np.full(self.ctrl.token_dim, np.nan)
            return {"token": nan, "action": np.full(29, np.nan), "ref_cursor": self.reference.cursor}
        return {"token": self.last["z"].copy(), "action": self.last["action"].copy(), "ref_cursor": self.reference.cursor}

    def __call__(self, state: dict):
        ref = self.ctrl and self.reference
        self.ctrl.observe(state)
        if self._need_heading or getattr(ref, "needs_heading_reset", False):
            self.ctrl.reset_heading(state["base_quat"], ref.root_quat)   # deploy: reset at start / (re)load
            self._need_heading = False
            if hasattr(ref, "needs_heading_reset"):
                ref.needs_heading_reset = False
        dz = self.residual(state) if self.residual is not None else self._zero
        out = self.ctrl.act(ref, state["base_quat"], dz, self.scale, inject=self.residual is not None)
        self.last = out
        ref.advance()
        return self._to_q(out["action"]), self._kp, self._kd
