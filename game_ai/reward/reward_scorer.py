from __future__ import annotations

from dataclasses import dataclass, field

from game_ai.config import RewardConfig
from game_ai.perception.perceptor import GameState


@dataclass(frozen=True)
class RewardResult:
    total: float
    breakdown: dict[str, float] = field(default_factory=dict)


class RewardScorer:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def score(self, previous: GameState | None, current: GameState) -> RewardResult:
        parts: dict[str, float] = {}

        if current.player_dead:
            parts["death"] = self.config.death_penalty
        else:
            parts["alive"] = self.config.alive_reward

        if current.in_combat:
            parts["combat"] = self.config.combat_reward
        else:
            parts["idle"] = self.config.idle_penalty

        if current.killed_target:
            parts["kill"] = self.config.kill_reward

        if current.task_done:
            parts["task_done"] = self.config.task_done_reward

        if previous is not None:
            self._score_hp(previous, current, parts)
            self._score_mp(previous, current, parts)
            self._score_target(previous, current, parts)

        return RewardResult(total=sum(parts.values()), breakdown=parts)

    def _score_hp(self, previous: GameState, current: GameState, parts: dict[str, float]) -> None:
        if previous.hp is None or current.hp is None:
            return

        delta = current.hp - previous.hp
        if delta > 0:
            parts["hp_gain"] = delta * self.config.hp_gain_scale
        elif delta < 0:
            parts["hp_loss"] = abs(delta) * self.config.hp_loss_scale

    def _score_mp(self, previous: GameState, current: GameState, parts: dict[str, float]) -> None:
        if previous.mp is None or current.mp is None:
            return

        delta = current.mp - previous.mp
        if delta > 0:
            parts["mp_gain"] = delta * self.config.mp_gain_scale
        elif delta < 0:
            parts["mp_loss"] = abs(delta) * self.config.mp_loss_scale

    def _score_target(self, previous: GameState, current: GameState, parts: dict[str, float]) -> None:
        if previous.target is None and current.target is not None:
            parts["target_acquired"] = self.config.target_acquired_reward
        elif previous.target is not None and current.target is None:
            parts["target_cleared"] = self.config.target_cleared_reward
