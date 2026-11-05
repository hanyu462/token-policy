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
