"""STEP_FORWARD closed-loop teacher:  T(s_0, d) -> Z*  (executed, grid-snapped SONIC token trajectory).

Internally a one-parameter motion family: the tokens SONIC produced while tracking a walking reference, from play
start to the first touchdown (window), with the deviation from the standing token scaled by lambda and snapped to
the FSQ grid, then the standing token of s_0 for ``settle`` seconds.  lambda is found per snapshot by execution:

    lambda_0 = f_global^-1(d);   lambda_{i+1} = clip(lambda_i + eta * (d - d_i) / 0.4, 0.55, 1.0)

with deterministic snapshot resets, until |d_i - d| < tol or max_iters.  If that fails (FSQ snapping makes the
lambda -> d map a staircase with ~5 cm gaps on some snapshots), the window length is tried once per value in
``cut_fallback`` at the best lambda and the smallest-error rollout is kept.  lambda / cut never leave the metadata.
"""
from __future__ import annotations

import numpy as np

from ..env import StandingSnapshot
from ..paths import SONIC_MODEL_DIR
from ..policy import ReplaceToken, fsq
from ..policy.fsq import k_to_z, z_to_k
from ..sim import G1Sim, SyncConfig, SyncSim
from ..sonic import SonicController, SonicJointController, SonicONNX, load_reference
from .step_forward_calibration import LAMBDA_RANGE, SLOPE_M, StepForwardCalibration

DT = 0.02
STATE_KEYS = ["time", "q", "dq", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel",
              "lfoot", "rfoot", "lfoot_force", "rfoot_force"]


def _heading_frame(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # rows: fwd, lat, up


def _swings(unloaded, min_len=3):
    out, start = 0, None
    for t, u in enumerate(list(unloaded) + [False]):
        if u and start is None:
            start = t
        elif not u and start is not None:
            out += (t - start) >= min_len
            start = None
    return out


class _WindowSource:
    """Token source for SONIC: precomputed window sequence, then the standing token of s_0."""

    def __init__(self):
        self.seq, self.z_stand, self.k = None, None, 0

    def set(self, seq, z_stand):
        self.seq, self.z_stand, self.k = seq, z_stand, 0

    def reset(self):
        self.k = 0

    def __call__(self, state):
        k, self.k = self.k, self.k + 1
        return self.seq[k] if k < len(self.seq) else self.z_stand


class StepForwardTeacher:
    def __init__(self, recorded_tokens: np.ndarray, n_window: int, nominal_k_stand: np.ndarray, calibration: StepForwardCalibration | None = None,
                 settle: float = 3.0, tol: float = 0.01, max_iters: int = 3, eta: float = 0.5, cut_fallback=(-4, 4),
                 contact_force: float = 20.0, onnx_threads: int = 4, decoder_backend: str = "torch"):
        """``recorded_tokens``: (T, 64) tokens recorded from play start on the nominal snapshot; ``n_window``: ticks up to
        the first touchdown (the base window; cut adds to it)."""
        self.tokens, self.n_window, self.z_nom = np.asarray(recorded_tokens, float), int(n_window), k_to_z(nominal_k_stand)
        self.calib = calibration or StepForwardCalibration()
        self.settle_ticks, self.tol, self.max_iters, self.eta, self.contact_force = int(round(settle / DT)), tol, max_iters, eta, contact_force
        self.cut_fallback = tuple(cut_fallback)
        self.sim = G1Sim()
        self.sync = SyncSim(self.sim, None, SyncConfig(gantry=False))
        self.source = _WindowSource()
        self.sonic = SonicJointController(SonicController(SonicONNX(SONIC_MODEL_DIR, threads=onnx_threads, decoder_backend=decoder_backend)),
                                          load_reference("standing"), token_hook=ReplaceToken(self.source))

    # ---- motion family ----
    def window_sequence(self, lam: float, k_stand: np.ndarray, cut: int = 0) -> np.ndarray:
        """Grid-snapped token window (n_window + cut ticks) for lambda on a snapshot with standing code k_stand."""
        return fsq.snap(k_to_z(k_stand) + lam * (self.tokens[:self.n_window + cut] - self.z_nom))

    # ---- one deterministic rollout ----
    def rollout(self, snapshot: StandingSnapshot, lam: float, cut: int = 0) -> dict:
        seq = self.window_sequence(lam, snapshot.k_stand, cut)
        snapshot.restore(self.sim, self.sonic)
        self.source.set(seq, k_to_z(snapshot.k_stand))
        self.sync.attach(self.sonic); self.sync._state = self.sim.state(); self.sync.tick = 0
        st0 = self.sync.state()
        R = _heading_frame(st0["base_rpy"][2])
        hist = {k: [] for k in STATE_KEYS + ["q_target"]}
        tokens, fell = [], False
        for _ in range(len(seq) + self.settle_ticks):
            st = self.sync.step()
            tokens.append(self.sonic.last["z"].copy())
            for k in STATE_KEYS:
                hist[k].append(st[k])
            hist["q_target"].append(self.sim.q_target.copy())
            if self.sim.fallen():
                fell = True
                break
        st = self.sync.state()
        d = R @ (st["base_pos"] - st0["base_pos"])
        z = np.array(tokens); k = z_to_k(z)
        assert np.allclose(k_to_z(k), z), "executed token not on the FSQ grid"
        lf, rf = np.array(hist["lfoot_force"]), np.array(hist["rfoot_force"])
        return {"lambda": float(lam), "cut": int(cut), "token_k": k.astype(np.int8), "token_z": z.astype(np.float32), "window_ticks": len(seq),
                "state_start": {kk: np.asarray(st0[kk]) for kk in STATE_KEYS},
                "state_history": {kk: np.array(v) for kk, v in hist.items()},
                "realized_dx_m": float(d[0]), "realized_dy_m": float(d[1]),
                "realized_dyaw_rad": float(np.angle(np.exp(1j * (st["base_rpy"][2] - st0["base_rpy"][2])))),
                "n_swing": _swings(lf < self.contact_force) + _swings(rf < self.contact_force), "fell": fell,
                "end_speed": float(np.mean(np.linalg.norm(hist["base_lin_vel"][-25:], axis=1)) if False else np.mean(np.linalg.norm(np.array(hist["base_lin_vel"])[-25:], axis=1)))}

    # ---- T(s_0, d) ----
    def generate(self, snapshot: StandingSnapshot, d: float) -> dict:
        """Closed-loop search over lambda (cut 0), then cut fallback; returns the best rollout plus teacher metadata."""
        lam = self.calib.lambda_for(d)
        lam0, tried, history, best = lam, {}, [], None

        def consider(r):
            nonlocal best
            if best is None or (not r["fell"] and abs(d - r["realized_dx_m"]) < abs(d - best["realized_dx_m"])):
                best = r
            return abs(d - r["realized_dx_m"]) < self.tol and not r["fell"]

        for _ in range(self.max_iters):
            # snapping plateaus: nudge lambda until the window sequence differs from every sequence already executed
            key = self.window_sequence(lam, snapshot.k_stand).tobytes()
            direction = 0.0 if not history else np.sign(d - history[-1][2])
            while key in tried and direction != 0.0 and LAMBDA_RANGE[0] < lam < LAMBDA_RANGE[1]:
                lam = float(np.clip(lam + 0.01 * direction, *LAMBDA_RANGE))
                key = self.window_sequence(lam, snapshot.k_stand).tobytes()
            if key in tried:
                break                                      # nothing new to execute (range bound or plateau)
            r = self.rollout(snapshot, lam)
            tried[key] = r
            history.append((lam, 0, r["realized_dx_m"]))
            if consider(r):
                break
            lam = float(np.clip(lam + self.eta * (d - r["realized_dx_m"]) / SLOPE_M, *LAMBDA_RANGE))
        converged = abs(d - best["realized_dx_m"]) < self.tol and not best["fell"]
        if not converged:                                  # staircase gap / range bound: try the window length once per cut
            lam_best = best["lambda"]
            for cut in self.cut_fallback:
                r = self.rollout(snapshot, lam_best, cut)
                history.append((lam_best, cut, r["realized_dx_m"]))
                if consider(r):
                    break
            converged = abs(d - best["realized_dx_m"]) < self.tol and not best["fell"]
        best = dict(best)
        best.update(command_distance_m=float(d), initial_lambda=float(lam0), final_lambda=best["lambda"], final_cut=best["cut"],
                    teacher_iterations=len(history), search_history=[(float(l), int(c), float(x)) for l, c, x in history],
                    teacher_converged=bool(converged))
        return best
