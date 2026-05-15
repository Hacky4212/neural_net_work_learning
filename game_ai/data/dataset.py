from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


STATE_SIGNAL_KEYS = (
    "hp",
    "mp",
    "position",
    "target",
    "ui_state",
    "player_dead",
    "killed_target",
    "task_done",
)


class JsonlDataset:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[dict]:
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def is_informative_sample(row: dict) -> bool:
    observation = row.get("observation") or {}
    if observation.get("window_found") is False:
        return False
    if observation.get("screenshot_path"):
        return True

    state = row.get("state") or {}
    if state.get("in_combat"):
        return True

    return any(state.get(key) is not None for key in STATE_SIGNAL_KEYS)


def reward_to_sample_weight(reward: object, min_weight: float = 0.1, max_weight: float = 5.0) -> float:
    try:
        value = 1.0 + float(reward)
    except (TypeError, ValueError):
        value = 1.0
    return min(max(value, min_weight), max_weight)
