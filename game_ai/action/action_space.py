from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from game_ai.action.action_schema import Action, ClickAction, KeyAction, NoopAction, WaitAction
from game_ai.learning.action_codec import action_to_label


@dataclass(frozen=True)
class ActionSpec:
    label: str
    action: Action


class DiscreteActionSpace:
    def __init__(self, specs: list[ActionSpec]) -> None:
        if not specs:
            raise ValueError("action space is empty")
        labels = [spec.label for spec in specs]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"duplicate action labels: {duplicates}")

        self.specs = specs
        self.label_to_index = {spec.label: index for index, spec in enumerate(specs)}

    @classmethod
    def load(cls, path: str) -> "DiscreteActionSpace":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise ValueError(f"missing action list in {path}")

        specs = [
            ActionSpec(
                label=row["label"],
                action=_dict_to_action(row["action"]),
            )
            for row in actions
        ]
        return cls(specs)

    def __len__(self) -> int:
        return len(self.specs)

    def action(self, index: int) -> Action:
        if index < 0 or index >= len(self.specs):
            raise IndexError(f"action index out of range: {index}")
        return self.specs[index].action

    def label(self, index: int) -> str:
        if index < 0 or index >= len(self.specs):
            raise IndexError(f"action index out of range: {index}")
        return self.specs[index].label

    def index_for_action_dict(self, action: dict) -> int | None:
        label = action_to_label(action)
        if label in self.label_to_index:
            return self.label_to_index[label]

        # Allow semantic labels such as skill:1 for key:1.
        action_type = action.get("type")
        key = action.get("key")
        if action_type == "key":
            for prefix in ("move", "skill", "potion"):
                label = f"{prefix}:{key}"
                if label in self.label_to_index:
                    return self.label_to_index[label]

        return None


def _dict_to_action(action: dict) -> Action:
    action_type = action.get("type")
    if action_type == "key":
        return KeyAction(str(action["key"]))
    if action_type == "click":
        return ClickAction(int(action["x"]), int(action["y"]))
    if action_type == "wait":
        return WaitAction(int(action.get("ms", 300)))
    return NoopAction()
