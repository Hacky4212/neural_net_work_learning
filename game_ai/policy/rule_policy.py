from __future__ import annotations

from game_ai.action.action_schema import Decision, KeyAction, WaitAction
from game_ai.memory.game_memory import GameMemory
from game_ai.perception.perceptor import GameState


class RulePolicy:
    def __init__(self, hp_low_threshold: float, mp_low_threshold: float) -> None:
        self.hp_low_threshold = hp_low_threshold
        self.mp_low_threshold = mp_low_threshold

    def decide(self, state: GameState, memory: GameMemory) -> Decision:
        if state.hp is not None and state.hp <= self.hp_low_threshold:
            return Decision(reason="hp_low", action=KeyAction("F1"))

        if state.mp is not None and state.mp <= self.mp_low_threshold:
            return Decision(reason="mp_low", action=KeyAction("F2"))

        if state.in_combat:
            return Decision(reason="attack_target", action=KeyAction("1"))

        return Decision(reason="idle_wait", action=WaitAction(300))
