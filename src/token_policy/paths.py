"""Asset locations. Override the root with the TOKEN_POLICY_ASSETS environment variable."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(os.environ.get("TOKEN_POLICY_ASSETS", REPO_ROOT / "assets"))


def asset(*parts: str) -> str:
    return str(ASSETS_DIR.joinpath(*parts))


G1_SCENE_XML = asset("g1", "scene_43dof.xml")
SONIC_MODEL_DIR = asset("hf", "sonic_v1_1")
MOTION_DIR = asset("motions")
