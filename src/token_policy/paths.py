"""Asset locations. Override the root with the TOKEN_POLICY_ASSETS environment variable."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(os.environ.get("TOKEN_POLICY_ASSETS", REPO_ROOT / "assets"))


def asset(*parts: str) -> str:
    return str(ASSETS_DIR.joinpath(*parts))


G1_SCENE_XML = asset("g1", "scene_43dof.xml")
SONIC_MODEL_DIR = asset("hf", "sonic_v1_1")
MOTION_DIR = asset("motions")

RESULTS_DIR = Path(os.environ.get("TOKEN_POLICY_RESULTS", "results"))


def default_log_path(prefix: str) -> str:
    """<results>/<prefix>_YYYYMMDD_HHMMSS.npz (directory created if missing)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
