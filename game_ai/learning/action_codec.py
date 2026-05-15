from __future__ import annotations

from game_ai.action.action_schema import Action, ClickAction, KeyAction, NoopAction, WaitAction


def action_to_label(action: dict) -> str:
    action_type = action.get("type")
    if action_type == "key":
        return f"key:{action.get('key')}"
    if action_type == "click":
        return f"click:{int(action.get('x', 0))}:{int(action.get('y', 0))}"
    if action_type == "wait":
        return f"wait:{int(action.get('ms', 300))}"
    return "noop"


def action_object_to_label(action: Action) -> str:
    return action_to_label(action.to_dict())


def label_to_action(label: str) -> Action:
    parts = label.split(":")

    if parts[0] == "key" and len(parts) == 2:
        return KeyAction(parts[1])

    if parts[0] == "click" and len(parts) == 3:
        return ClickAction(int(parts[1]), int(parts[2]))

    if parts[0] == "wait" and len(parts) == 2:
        return WaitAction(int(parts[1]))

    return NoopAction()
