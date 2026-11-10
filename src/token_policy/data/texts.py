"""Command texts: templates x distances.  The physics rollouts are indexed by (snapshot, d_cmd); texts are attached
in the dataset layer, so templates can be extended without new rollouts.  Template sets are split for language
generalization tests; the held-out magnitude (30 cm) is never rendered by the train loader (see dataset.py)."""
from __future__ import annotations

TRAIN_TEMPLATES = [
    "Move forward by {cm} cm.",
    "Take a {cm} cm step forward.",
    "Advance forward {cm} centimeters.",
    "Step forward {cm} centimeters.",
    "앞으로 {cm}cm 이동해.",
    "앞으로 {cm}센티미터 전진해.",
]
HELDOUT_TEMPLATES = [
    "Walk forward {cm} cm and stop.",
    "{cm}cm 앞으로 걸어가.",
]
ALL_TEMPLATES = TRAIN_TEMPLATES + HELDOUT_TEMPLATES


def render(template: str, d_m: float) -> str:
    return template.format(cm=int(round(d_m * 100)))


def all_texts(distances_m, templates=ALL_TEMPLATES) -> list:
    """Every (template, distance) string, deterministic order (used to build the text cache)."""
    return [render(t, d) for d in distances_m for t in templates]


# ---- STEP_FOOT(foot, dx, dy): raw text only; foot / dx / dy never reach the Student except through these strings ----
# convention: dy > 0 = robot left, dy < 0 = robot right (task.LocalFrame)
_FOOT_EN = {"L": "left", "R": "right"}
_FOOT_KO = {"L": "왼발", "R": "오른발"}

STEPFOOT_TRAIN_TEMPLATES = [
    ("en", "Step your {foot} foot {dx} cm forward{lat_en}."),
    ("en", "Move the {foot} foot forward by {dx} cm{lat_en}."),
    ("en", "Place your {foot} foot {dx} centimeters ahead{lat_en}."),
    ("en", "Take a step with the {foot} foot, {dx} cm forward{lat_en}."),
    ("ko", "{foot_ko}을 앞으로 {dx}cm{lat_ko} 내디뎌."),
    ("ko", "{foot_ko}을 {dx}센티미터 앞으로{lat_ko} 옮겨."),
]
STEPFOOT_HELDOUT_TEMPLATES = [
    ("en", "Put the {foot} foot down {dx} cm in front{lat_en}."),
    ("ko", "{foot_ko}로 {dx}cm 앞에{lat_ko} 디뎌."),
]


def render_stepfoot(template, foot: str, dx_m: float, dy_m: float) -> str:
    lang, t = template
    dx, dy = int(round(dx_m * 100)), int(round(dy_m * 100))
    side_en, side_ko = ("left", "왼쪽") if dy > 0 else ("right", "오른쪽")
    lat_en = "" if dy == 0 else f" and {abs(dy)} cm to the {side_en}"
    lat_ko = "" if dy == 0 else f", {side_ko}으로 {abs(dy)}cm"
    return t.format(foot=_FOOT_EN[foot], foot_ko=_FOOT_KO[foot], dx=dx, lat_en=lat_en, lat_ko=lat_ko)


def all_stepfoot_texts(commands, templates=STEPFOOT_TRAIN_TEMPLATES + STEPFOOT_HELDOUT_TEMPLATES) -> list:
    return [render_stepfoot(t, f, dx, dy) for (f, dx, dy) in commands for t in templates]
