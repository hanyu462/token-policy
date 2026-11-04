"""
MuJoCo simulation of the Unitree G1 with PD control.

Takes a joint command (q_target, kp, kd) at 50Hz and runs the physics at 500Hz.

recomputing tau = kp * (q_target - q) - kd * dq every step.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..common.rotations import quat_to_rpy
from ..paths import G1_SCENE_XML
from .g1_params import DEFAULT_ANGLES, KD, KP, MUJOCO_JOINT_NAMES, NUM_JOINTS


@dataclass
class SimConfig:
    scene_xml: str | None = None     # default: assets/g1/scene_43dof.xml
    control_hz: float = 50.0
    clearance: float = 0.002         # foot clearance above the floor at reset (m)
    contact_threshold: float = 1.0   # normal force (N) above which a foot counts as in contact


class G1Sim:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.contact_threshold = self.cfg.contact_threshold

        # ---- model: load the MJCF scene, fix the physics / control step ratio ----
        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene_xml or G1_SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)           # physics step (0.002 s in the MJCF)
        self.control_dt = 1.0 / self.cfg.control_hz        # one step_control() = n_sub physics steps
        self.n_sub = int(round(self.control_dt / self.dt))
        assert abs(self.n_sub * self.dt - self.control_dt) < 1e-9, (self.dt, self.control_dt)

        # ---- joint indices: the 29 body joints in MuJoCo order ----
        # The 14 Dex3 finger joints sit between the arms in the MJCF, so the body joints
        # are not contiguous in qpos / qvel / ctrl; look them up by name once.
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in MUJOCO_JOINT_NAMES]
        assert min(jid) >= 0, "missing body joint in model"
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in jid])   # index in data.qpos
        self.dof_adr = np.array([self.model.jnt_dofadr[j] for j in jid])     # index in data.qvel
        self.act_id = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n[:-len("_joint")])
                                for n in MUJOCO_JOINT_NAMES])                 # index in data.ctrl
        assert self.act_id.min() >= 0, "missing body actuator in model"
        self.ctrl_limit = self.model.actuator_ctrlrange[self.act_id, 1].copy()   # torque limit from the MJCF

        # ---- body indices: pelvis (base) and feet, for state readout and reset ----
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.rfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        # foot sole boxes: reset() uses their lowest corner to place the robot on the floor
        self.foot_geoms = [g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] in (self.lfoot_id, self.rfoot_id)
                           and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]

        # ---- runtime state: current PD command, external wrench hook, PD-loop buffers ----
        self.q_target = DEFAULT_ANGLES.copy()
        self.kp = KP.copy()
        self.kd = KD.copy()
        # external wrench on the pelvis (e.g. the initialisation gantry): callable(sim) -> (6,) or None
        self.wrench_source = None
        # preallocated for the 500 Hz loop (fancy indexing copies, so take(..., out=) into these)
        self._q = np.empty(NUM_JOINTS)
        self._dq = np.empty(NUM_JOINTS)
        self._tau = np.empty(NUM_JOINTS)
        self._tmp = np.empty(NUM_JOINTS)

    # ---- state ---------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.data.time)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def dq(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    def base_quat(self) -> np.ndarray:  # wxyz, world
        return self.data.qpos[3:7].copy()

    def base_ang_vel(self) -> np.ndarray:  # body frame (free-joint qvel convention), like the pelvis IMU gyro
        return self.data.qvel[3:6].copy()

    def base_lin_vel(self) -> np.ndarray:  # world frame
        return self.data.qvel[0:3].copy()

    def base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    def base_rpy(self) -> np.ndarray:
        return quat_to_rpy(self.base_quat())

    def foot_pos(self):
        return self.data.xpos[self.lfoot_id].copy(), self.data.xpos[self.rfoot_id].copy()

    def tau(self) -> np.ndarray:
        """Torque actually applied by the 29 body actuators in the last physics step (N m, MuJoCo order)."""
        return self.data.actuator_force[self.act_id].copy()

    def foot_contact_force(self):
        """Sum of contact normal forces on each foot body (N): (left, right)."""
        f = np.zeros(2)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            for side, bid in enumerate((self.lfoot_id, self.rfoot_id)):
                if bid in (b1, b2):
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    f[side] += abs(wrench[0])   # normal component in the contact frame
        return float(f[0]), float(f[1])

    def state(self) -> dict:
        """One snapshot of everything a controller / task needs (copies)."""
        lf, rf = self.foot_pos()
        lforce, rforce = self.foot_contact_force()
        return {"time": self.time,
                "base_pos": self.base_pos(), "base_quat": self.base_quat(), "base_rpy": self.base_rpy(),
                "base_lin_vel": self.base_lin_vel(), "base_ang_vel": self.base_ang_vel(),
                "q": self.q(), "dq": self.dq(), "tau": self.tau(),
                "lfoot": lf, "rfoot": rf,
                "lfoot_force": lforce, "rfoot_force": rforce,
                "lfoot_contact": lforce > self.contact_threshold, "rfoot_contact": rforce > self.contact_threshold}

    def fallen(self, min_height=0.45, max_tilt=1.0) -> bool:
        rpy = self.base_rpy()
        return bool(self.base_pos()[2] < min_height or abs(rpy[0]) > max_tilt or abs(rpy[1]) > max_tilt)

    # ---- control -------------------------------------------------------------
    def reset(self, q_mujoco: np.ndarray | None = None, clearance: float | None = None):
        """Place the robot at the given joint pose with the feet just touching the floor."""
        clearance = self.cfg.clearance if clearance is None else clearance
        mujoco.mj_resetData(self.model, self.data)
        q_mujoco = DEFAULT_ANGLES if q_mujoco is None else np.asarray(q_mujoco)
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0  # identity quaternion
        self.data.qpos[self.qpos_adr] = q_mujoco
        self.data.qpos[2] = 1.0
        mujoco.mj_forward(self.model, self.data)
        lowest = np.inf
        for g in self.foot_geoms:
            size = self.model.geom_size[g]
            R = self.data.geom_xmat[g].reshape(3, 3)
            c = self.data.geom_xpos[g]
            corners = np.array([[sx, sy, sz] for sx in (-size[0], size[0])
                                for sy in (-size[1], size[1]) for sz in (-size[2], size[2])])
            lowest = min(lowest, (c + corners @ R.T)[:, 2].min())
        self.data.qpos[2] -= lowest - clearance
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.q_target = q_mujoco.copy()
        self.wrench_source = None

    def set_command(self, q_target: np.ndarray, kp: np.ndarray | None = None, kd: np.ndarray | None = None):
        assert q_target.shape == (NUM_JOINTS,)
        self.q_target = np.asarray(q_target, float).copy()
        if kp is not None:
            self.kp = np.asarray(kp, float).copy()
        if kd is not None:
            self.kd = np.asarray(kd, float).copy()

    def step_control(self):
        """Advance one control period (n_sub physics steps) with the current command."""
        qpos, qvel, ctrl = self.data.qpos, self.data.qvel, self.data.ctrl
        for _ in range(self.n_sub):
            np.take(qpos, self.qpos_adr, out=self._q)
            np.take(qvel, self.dof_adr, out=self._dq)
            np.subtract(self.q_target, self._q, out=self._tau)
            self._tau *= self.kp                      # kp * (q_target - q)
            np.multiply(self.kd, self._dq, out=self._tmp)
            self._tau -= self._tmp                    # + kd * (0 - dq)
            np.clip(self._tau, -self.ctrl_limit, self.ctrl_limit, out=self._tau)
            ctrl[:] = 0.0
            ctrl[self.act_id] = self._tau
            if self.wrench_source is not None:
                self.data.xfrc_applied[self.pelvis_id] = self.wrench_source(self)
            mujoco.mj_step(self.model, self.data)
