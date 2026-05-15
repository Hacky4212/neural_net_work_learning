from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from game_ai.observation.observer import RawObservation
from game_ai.perception.perceptor import GameState


@dataclass
class MemoryItem:
    observation: RawObservation
    state: GameState
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class GameMemory:
    max_size: int = 30
    history: Deque[MemoryItem] = field(default_factory=deque)

    total_reward: float = 0.0

    def remember(
        self,
        observation: RawObservation,
        state: GameState,
        reward: float = 0.0,
        reward_breakdown: dict[str, float] | None = None,
    ) -> None:
        self.total_reward += reward
        self.history.append(
            MemoryItem(
                observation=observation,
                state=state,
                reward=reward,
                reward_breakdown=reward_breakdown or {},
            )
        )
        while len(self.history) > self.max_size:
            self.history.popleft()

    @property
    def last_state(self) -> GameState | None:
        if not self.history:
            return None
        return self.history[-1].state
