#!/usr/bin/env python3
"""Convert a GEAR-SONIC retargeted motion pickle (robot_filtered: dof (T,29) MuJoCo order, root_trans_offset,
root_rot xyzw, fps) into the CSV reference layout MotionReference reads (50 Hz, joints in IsaacLab order,
body 0 = pelvis root).  Only the columns the g1 encoder needs are written.

  .venv/bin/python experiments/convert_robot_pkl.py assets/hf/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl
"""
import argparse
import os

import joblib
import numpy as np

from token_policy.paths import MOTION_DIR
from token_policy.sim.g1_params import mujoco_to_isaac

OUT_HZ = 50.0


def resample(x, fps):
    t_src = np.arange(len(x)) / fps
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / OUT_HZ)
    return np.stack([np.interp(t_dst, t_src, x[:, j]) for j in range(x.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="+")
    ap.add_argument("--out-dir", default=MOTION_DIR)
    args = ap.parse_args()
    for path in args.pkl:
        for name, m in joblib.load(path).items():
            fps = float(m["fps"])
            q_mj = np.asarray(m["dof"], float)
            quat = np.asarray(m["root_rot"], float)[:, [3, 0, 1, 2]]          # xyzw -> wxyz
            quat[np.sum(quat * quat[:1], 1) < 0] *= -1                          # consistent sign for interpolation
            q = resample(np.stack([mujoco_to_isaac(r) for r in q_mj]), fps)
            pos = resample(np.asarray(m["root_trans_offset"], float), fps)
            qt = resample(quat, fps); qt /= np.linalg.norm(qt, axis=1, keepdims=True)
            dq = np.gradient(q, 1.0 / OUT_HZ, axis=0)
            out = os.path.join(args.out_dir, name)
            os.makedirs(out, exist_ok=True)
            np.savetxt(os.path.join(out, "joint_pos.csv"), q, delimiter=",", header=",".join(f"j{i}" for i in range(29)), comments="")
            np.savetxt(os.path.join(out, "joint_vel.csv"), dq, delimiter=",", header=",".join(f"j{i}" for i in range(29)), comments="")
            np.savetxt(os.path.join(out, "body_pos.csv"), pos, delimiter=",", header="body_0_x,body_0_y,body_0_z", comments="")
            np.savetxt(os.path.join(out, "body_quat.csv"), qt, delimiter=",", header="body_0_w,body_0_x,body_0_y,body_0_z", comments="")
            with open(os.path.join(out, "info.txt"), "w") as f:
                f.write(f"converted from {path} ({len(q_mj)} frames @ {fps:.0f} fps) by experiments/convert_robot_pkl.py: "
                        f"{len(q)} frames @ {OUT_HZ:.0f} Hz, joints IsaacLab order, body 0 = pelvis root (pos, quat wxyz)\n")
            print(f"{name}: {len(q_mj)} @ {fps:.0f} fps -> {len(q)} @ 50 Hz, travel {np.linalg.norm(pos[-1, :2] - pos[0, :2]):.2f} m -> {out}")


if __name__ == "__main__":
    main()
