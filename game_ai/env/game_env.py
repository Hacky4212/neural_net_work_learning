from __future__ import annotations

import time
from dataclasses import dataclass

from game_ai.action.action_schema import Action, Decision
from game_ai.data.recorder import Recorder
from game_ai.executor.executor import Executor
from game_ai.memory.game_memory import GameMemory
from game_ai.observation.observer import Observer, RawObservation
from game_ai.perception.perceptor import GameState, Perceptor
from game_ai.reward.reward_scorer import RewardResult, RewardScorer


@dataclass(frozen=True)
class EnvStep:
    observation: RawObservation
    state: GameState
    reward: RewardResult
    done: bool
    info: dict


class GameEnv:
    def __init__(
        self,
        observer: Observer,
        perceptor: Perceptor,
        reward_scorer: RewardScorer,
        executor: Executor,
        recorder: Recorder | None = None,
        tick_seconds: float = 0.3,
        max_steps: int | None = None,
        record_missing_window: bool = False,
    ) -> None:
        self.observer = observer
        self.perceptor = perceptor
        self.reward_scorer = reward_scorer
        self.executor = executor
        self.recorder = recorder
        self.tick_seconds = tick_seconds
        self.max_steps = max_steps
        self.record_missing_window = record_missing_window
        self.memory = GameMemory()
        self.step_count = 0

    def reset(self) -> EnvStep:
        self.memory = GameMemory()
        self.step_count = 0
        return self._observe_only()

    def step(self, action: Action) -> EnvStep:
        self.executor.execute(action)
        time.sleep(self.tick_seconds)
        self.step_count += 1
        result = self._observe_only(action)
        return result

    def _observe_only(self, action: Action | None = None) -> EnvStep:
        raw = self.observer.observe()
        state = self.perceptor.parse(raw)
        reward = self.reward_scorer.score(self.memory.last_state, state)
        self.memory.remember(raw, state, reward.total, reward.breakdown)
        if self.recorder is not None and action is not None and self._should_record(raw):
            self.recorder.record(
                raw,
                state,
                Decision(reason="env_step", action=action),
                reward,
                self.memory.total_reward,
            )
        done = self._is_done(state)
        info = {
            "step": self.step_count,
            "total_reward": self.memory.total_reward,
            "window_found": raw.window_found,
            "window_title": raw.window_title,
        }
        return EnvStep(
            observation=raw,
            state=state,
            reward=reward,
            done=done,
            info=info,
        )

    def _is_done(self, state: GameState) -> bool:
        if state.player_dead or state.task_done:
            return True
        if self.max_steps is not None and self.step_count >= self.max_steps:
            return True
        return False

    def _should_record(self, observation: RawObservation) -> bool:
        return self.record_missing_window or observation.window_found
