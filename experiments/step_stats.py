#!/usr/bin/env python3
"""Natural STEP statistics from official SONIC references: per swing phase, the foot displacement
p_touchdown - p_liftoff (heading frame at liftoff), duration, clearance, and the token sequence
z[liftoff:touchdown] that SONIC produced while tracking the motion in sim.

Two views per motion: the reference kinematics (MuJoCo FK of joint_pos + root pose; swing = foot
height > --ref-clear above its minimum) and the sim replay (swing = contact force < --force N).

  .venv/bin/python experiments/step_stats.py walking_quip_360_R_002__A428 [more motions] --out results/step_stats
"""
import argparse
import json
import os

import mujoco
import numpy as np

from token_policy.cli import parse_config, run
from token_policy.common.rotations import calc_heading, quat_rotate
from token_policy.sim import G1Sim
from token_policy.sim.g1_params import isaac_to_mujoco
from token_policy.sonic.reference import load_reference

DT = 0.02


def heading_frame(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # rows: fwd, lat(left), up


def swings(unloaded: np.ndarray, min_len: int = 3):
    """[(liftoff, touchdown)] = maximal runs of ``unloaded`` of length >= min_len (touchdown = first loaded tick)."""
    out, start = [], None
    for t, u in enumerate(unloaded):
        if u and start is None:
            start = t
        elif not u and start is not None:
            if t - start >= min_len:
                out.append((start, t))
            start = None
    return out


def ref_foot_tracks(sim: G1Sim, ref):
    """Foot world positions / base heading of each reference frame by forward kinematics."""
    T = ref.length
    lf, rf, yaw = np.zeros((T, 3)), np.zeros((T, 3)), np.zeros(T)
    for t in range(T):
        sim.data.qpos[:] = 0.0
        sim.data.qpos[0:3] = ref.root_pos_all[t]
        sim.data.qpos[3:7] = ref.root_quat_all[t]
        sim.data.qpos[sim.qpos_adr] = isaac_to_mujoco(ref.joint_pos[t])
        mujoco.mj_kinematics(sim.model, sim.data)
        lf[t], rf[t] = sim.data.xpos[sim.lfoot_id], sim.data.xpos[sim.rfoot_id]
        yaw[t] = calc_heading(ref.root_quat_all[t])
    return lf, rf, yaw


def analyse(name, lf, rf, yaw, unloaded_l, unloaded_r, tokens=None, tag=""):
    rows = []
    for foot, p, other, unl in (("L", lf, rf, unloaded_l), ("R", rf, lf, unloaded_r)):
        for lo, td in swings(unl):
            R = heading_frame(yaw[lo])
            dp = R @ (p[td] - p[lo])                      # stride: same foot, liftoff -> touchdown
            step = R @ (p[td] - other[td])                # step: swing foot relative to the stance foot at touchdown
            z0 = min(p[lo, 2], p[td, 2])
            rows.append({"motion": name, "view": tag, "foot": foot, "liftoff": lo, "touchdown": td, "dur_s": (td - lo) * DT,
                         "fwd_mm": dp[0] * 1e3, "lat_mm": dp[1] * 1e3, "step_fwd_mm": step[0] * 1e3, "step_lat_mm": step[1] * 1e3,
                         "clear_mm": (p[lo:td, 2].max() - z0) * 1e3,
                         "yaw_change_deg": np.degrees(np.angle(np.exp(1j * (yaw[td] - yaw[lo])))),
                         "tokens": None if tokens is None else tokens[lo:td].tolist()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motions", nargs="+")
    ap.add_argument("--force", type=float, default=20.0, help="sim: foot unloaded below this contact force (N)")
    ap.add_argument("--ref-clear", type=float, default=0.015, help="reference: swing when foot height > min + this (m)")
    ap.add_argument("--out", default="results/step_stats")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sim_fk = G1Sim()
    all_rows = []
    for name in args.motions:
        ref = load_reference(name)
        # ---- reference kinematics ----
        lf, rf, yaw = ref_foot_tracks(sim_fk, ref)
        rows_ref = analyse(name, lf, rf, yaw, lf[:, 2] > lf[:, 2].min() + args.ref_clear, rf[:, 2] > rf[:, 2].min() + args.ref_clear, tag="ref")
        # ---- sim replay with SONIC (standard timeline: play starts at t_settle + t_hold = 4 s) ----
        cfg = parse_config([], controller="sonic", reference=name, headless=True, no_realtime=True,
                           duration=4.0 + ref.length * DT + 1.0, print_period=0)
        a = run(cfg)
        k_play = int(round((cfg.t_settle + cfg.t_hold) / DT))
        sl = slice(k_play, k_play + ref.length)
        yaw_sim = a["base_rpy"][sl, 2]
        rows_sim = analyse(name, a["lfoot"][sl], a["rfoot"][sl], yaw_sim, a["lfoot_force"][sl] < args.force,
                           a["rfoot_force"][sl] < args.force, tokens=a["token"][sl], tag="sim")
        fell = np.isfinite(a["fell_at"])
        np.savez(os.path.join(args.out, f"{name}.npz"), token=a["token"][sl], lfoot=a["lfoot"][sl], rfoot=a["rfoot"][sl],
                 lforce=a["lfoot_force"][sl], rforce=a["rfoot_force"][sl], base_pos=a["base_pos"][sl], base_rpy=a["base_rpy"][sl],
                 q=a["q"][sl], ref_lfoot=lf, ref_rfoot=rf, ref_yaw=yaw, fell=fell)
        for tag, rows in (("ref", rows_ref), ("sim", rows_sim)):
            print(f"\n== {name} [{tag}]  swings {len(rows)}  {'FELL' if fell and tag == 'sim' else ''}")
            for r in rows:
                print(f"  {r['foot']} {r['liftoff']:4d}-{r['touchdown']:4d} ({r['dur_s']:.2f} s)  stride fwd {r['fwd_mm']:+7.1f} lat {r['lat_mm']:+7.1f}  "
                      f"step fwd {r['step_fwd_mm']:+6.1f} lat {r['step_lat_mm']:+6.1f}  clear {r['clear_mm']:5.1f} mm  yaw {r['yaw_change_deg']:+6.1f} deg")
        all_rows += rows_ref + rows_sim
    with open(os.path.join(args.out, "swings.json"), "w") as f:
        json.dump(all_rows, f, default=float)
    for tag in ("ref", "sim"):
        sel = [r for r in all_rows if r["view"] == tag]
        if sel:
            fw, la, sf, sl_, du, cl = (np.array([r[k] for r in sel]) for k in ("fwd_mm", "lat_mm", "step_fwd_mm", "step_lat_mm", "dur_s", "clear_mm"))
            print(f"\n[{tag}] {len(sel)} swings: stride fwd {fw.mean():+.0f} mm (sd {fw.std():.0f}, range {fw.min():+.0f}..{fw.max():+.0f}) lat {la.mean():+.0f} (sd {la.std():.0f}); "
                  f"step fwd {sf.mean():+.0f} mm (sd {sf.std():.0f}, range {sf.min():+.0f}..{sf.max():+.0f}) |lat| {np.abs(sl_).mean():.0f}; "
                  f"dur {du.mean():.2f} s (sd {du.std():.2f}); clearance {cl.mean():.0f} mm (max {cl.max():.0f})")
    print("saved", os.path.join(args.out, "swings.json"))


if __name__ == "__main__":
    main()
