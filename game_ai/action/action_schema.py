from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ActionType = Literal["click", "key", "wait", "noop"]


@dataclass(frozen=True)
class Action:
    type: ActionType

    def to_dict(self) -> dict:
        return {"type": self.type}


@dataclass(frozen=True)
class ClickAction(Action):
    x: int
    y: int

    def __init__(self, x: int, y: int) -> None:
        object.__setattr__(self, "type", "click")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def to_dict(self) -> dict:
        return {"type": self.type, "x": self.x, "y": self.y}


@dataclass(frozen=True)
class KeyAction(Action):
    key: str

    def __init__(self, key: str) -> None:
        object.__setattr__(self, "type", "key")
        object.__setattr__(self, "key", key)

    def to_dict(self) -> dict:
        return {"type": self.type, "key": self.key}


@dataclass(frozen=True)
class WaitAction(Action):
    ms: int

    def __init__(self, ms: int) -> None:
        object.__setattr__(self, "type", "wait")
        object.__setattr__(self, "ms", ms)

    def to_dict(self) -> dict:
        return {"type": self.type, "ms": self.ms}


@dataclass(frozen=True)
class NoopAction(Action):
    def __init__(self) -> None:
        object.__setattr__(self, "type", "noop")


@dataclass(frozen=True)
class Decision:
    reason: str
    action: Action
