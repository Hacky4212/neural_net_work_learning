from __future__ import annotations

from game_ai.perception.perceptor import GameState


FEATURE_SIZE = 13


def state_to_vector(state: GameState) -> list[float]:
    return [
        _num(state.hp),
        _known(state.hp),
        _num(state.mp),
        _known(state.mp),
        _pos(state.position, 0),
        _pos(state.position, 1),
        _pos(state.position, 2),
        1.0 if state.target else 0.0,
        1.0 if state.in_combat else 0.0,
        _bool(state.player_dead),
        _bool(state.killed_target),
        _bool(state.task_done),
        1.0 if state.ui_state else 0.0,
    ]


def dict_state_to_vector(state: dict) -> list[float]:
    position = state.get("position")
    return [
        _num(state.get("hp")),
        _known(state.get("hp")),
        _num(state.get("mp")),
        _known(state.get("mp")),
        _pos(position, 0),
        _pos(position, 1),
        _pos(position, 2),
        1.0 if state.get("target") else 0.0,
        1.0 if state.get("in_combat") else 0.0,
        _bool(state.get("player_dead")),
        _bool(state.get("killed_target")),
        _bool(state.get("task_done")),
        1.0 if state.get("ui_state") else 0.0,
    ]


def _num(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _known(value: object | None) -> float:
    return 0.0 if value is None else 1.0


def _pos(position: object, index: int) -> float:
    if not position or len(position) <= index:
        return 0.0
    return float(position[index])


def _bool(value: bool | None) -> float:
    if value is None:
        return 0.0
    return 1.0 if value else -1.0
