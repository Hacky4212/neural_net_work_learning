from __future__ import annotations

from dataclasses import dataclass

from game_ai.observation.observer import RawObservation


@dataclass
class GameState:
    hp: float | None
    mp: float | None
    position: tuple[float, float, float] | None
    target: str | None
    ui_state: str | None
    in_combat: bool
    player_dead: bool | None = None
    killed_target: bool | None = None
    task_done: bool | None = None


class Perceptor:
    def parse(self, observation: RawObservation) -> GameState:
        return GameState(
            hp=observation.hp,
            mp=observation.mp,
            position=observation.position,
            target=observation.target,
            ui_state=observation.ui_state,
            in_combat=observation.target is not None,
            player_dead=observation.hp == 0 if observation.hp is not None else None,
            killed_target=None,
            task_done=None,
        )
