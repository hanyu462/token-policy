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

    def act(self, reference, base_quat, token_hook=None, state=None) -> dict:
        """One tick: reference -> encoder -> token_hook -> decoder -> action (IsaacLab order)."""
        out = self.policy.step(reference, base_quat, self.heading, self.hist, token_hook, state)
        self.last_action = out["action"]
        return out


class SonicJointController:
    """SONIC as a sim controller: ``controller(state) -> (q_target, kp, kd)`` for SyncSim / AsyncSim.

    Wraps SonicController + a ReferenceProvider (+ an optional external ``token_hook``) into the
    one-call interface the sim runners use. Per tick: log the state, encoder -> token_hook ->
    decoder, advance the reference, return joint targets. Call ``play()`` to start a motion.

    ``token_hook(z_base, state) -> z``: the external token entry (see token_policy.policy.token);
    None = plain SONIC.
    """

    def __init__(self, ctrl: SonicController, reference, token_hook=None):
        from ..sim.g1_params import KD, KP, action_to_q_target
        self._to_q, self._kp, self._kd = action_to_q_target, KP, KD
        self.ctrl = ctrl
        self.reference = reference
        self.token_hook = token_hook
        self.last = None            # last policy output dict (z_base, z, action)
        self.reset()

    def reset(self):
        self.ctrl.reset()
        self.reference.reset()
        if self.token_hook is not None and hasattr(self.token_hook, "reset"):
            self.token_hook.reset()
        self._need_heading = True
        self.last = None

    def observe_only(self, state: dict):
        """Feed the state without acting (e.g. while a PD hold runs before SONIC takes over)."""
        self.ctrl.observe(state)

    def play(self):
        self.reference.play()

    def log_extras(self) -> dict:
        """Per-tick values for the sim's Recorder (sim just stores them): token, raw action, the
        reference frame index and its joint targets q_ref (MuJoCo order), so logs can separate
        q - q_ref (tracking error) from q - q_target (PD error)."""
        from ..sim.g1_params import isaac_to_mujoco
        q_ref = isaac_to_mujoco(self.reference.future_frames(1, 1)[0][0])
        if self.last is None:
            nan = np.full(self.ctrl.token_dim, np.nan)
            return {"token": nan, "token_base": nan, "action": np.full(29, np.nan),
                    "ref_cursor": self.reference.cursor, "q_ref": q_ref}
        return {"token": self.last["z"].copy(), "token_base": self.last["z_base"].copy(),
                "action": self.last["action"].copy(), "ref_cursor": self.reference.cursor, "q_ref": q_ref}

    def __call__(self, state: dict):
        ref = self.ctrl and self.reference
        self.ctrl.observe(state)
        if self._need_heading or getattr(ref, "needs_heading_reset", False):
            self.ctrl.reset_heading(state["base_quat"], ref.root_quat)   # deploy: reset at start / (re)load
            self._need_heading = False
            if hasattr(ref, "needs_heading_reset"):
                ref.needs_heading_reset = False
        out = self.ctrl.act(ref, state["base_quat"], self.token_hook, state)
        self.last = out
        ref.advance()
        return self._to_q(out["action"]), self._kp, self._kd
