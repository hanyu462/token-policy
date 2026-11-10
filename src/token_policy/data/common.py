"""Shared sample layout and the snapshot split (used by every Student dataset)."""
from __future__ import annotations

import json
from dataclasses import dataclass

HORIZON = 40
T_S = 10
MOTION_TAIL = 20          # ticks after the token window still counted as "motion" (stop transition)


@dataclass
class Split:
    train: list
    val: list
    test: list

    @classmethod
    def default(cls, snapshots: list) -> "Split":
        """Held-out 7 spread over the yaw-sorted list; 3 validation; the rest train."""
        test = ["s0_02", "s0_07", "s0_12", "s0_16", "s0_21", "s0_25", "s0_29"]
        val = ["s0_05", "s0_14", "s0_24"]
        train = [s for s in snapshots if s not in test and s not in val]
        assert not set(train) & set(test) and not set(train) & set(val) and not set(val) & set(test)
        return cls(train, val, test)

    def save(self, path: str):
        json.dump(self.__dict__, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "Split":
        return cls(**json.load(open(path)))
