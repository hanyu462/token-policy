#!/usr/bin/env bash
# Populate assets/: SONIC v1.1 ONNX models + observation config from Hugging Face and the
# G1 MuJoCo model (MJCF + meshes) from a local GR00T-WholeBodyControl checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_WBC="${GR00T_WBC:-$HOME/GR00T-WholeBodyControl}"

mkdir -p "$ROOT/assets/hf" "$ROOT/assets/g1" "$ROOT/assets/motions"
hf download nvidia/GEAR-SONIC \
    --include "sonic_v1_1/model_encoder.onnx" "sonic_v1_1/model_decoder.onnx" "sonic_v1_1/observation_config.yaml" \
    --local-dir "$ROOT/assets/hf"

G1_SRC="$GR00T_WBC/gear_sonic/data/robot_model/model_data/g1"
cp "$G1_SRC/g1_29dof_with_hand.xml" "$G1_SRC/scene_43dof.xml" "$ROOT/assets/g1/"
cp -r "$G1_SRC/meshes" "$ROOT/assets/g1/"
cp -r "$GR00T_WBC/gear_sonic_deploy/reference/example/squat_001__A359" "$ROOT/assets/motions/"
echo "assets ready under $ROOT/assets"
