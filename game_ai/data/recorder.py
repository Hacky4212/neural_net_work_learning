from __future__ import annotations

import json
import time
from pathlib import Path

from game_ai.action.action_schema import Decision
from game_ai.observation.observer import RawObservation
from game_ai.perception.perceptor import GameState
from game_ai.reward.reward_scorer import RewardResult


class Recorder:
    def __init__(
        self,
        path: str = "data/actions.jsonl",
        save_screenshots: bool = True,
        screenshot_dir: str = "data/screenshots",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_screenshots = save_screenshots
        self.screenshot_dir = Path(screenshot_dir)
        if self.save_screenshots:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def record(
        self,
        observation: RawObservation,
        state: GameState,
        decision: Decision,
        reward: RewardResult,
        total_reward: float,
    ) -> None:
        screenshot_path = self._save_screenshot(observation)
        row = {
            "observation": {
                "timestamp": observation.timestamp,
                "window_found": observation.window_found,
                "window_title": observation.window_title,
                "window_rect": observation.window_rect,
                "screenshot_path": screenshot_path,
            },
            "state": {
                "hp": state.hp,
                "mp": state.mp,
                "position": state.position,
                "target": state.target,
                "ui_state": state.ui_state,
                "in_combat": state.in_combat,
                "player_dead": state.player_dead,
                "killed_target": state.killed_target,
                "task_done": state.task_done,
            },
            "decision": {
                "reason": decision.reason,
                "action": decision.action.to_dict(),
            },
            "reward": {
                "step": reward.total,
                "total": total_reward,
                "breakdown": reward.breakdown,
            },
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_screenshot(self, observation: RawObservation) -> str | None:
        if not self.save_screenshots or observation.screenshot is None:
            return None

        name = f"{int(time.time() * 1000)}_{self.index:06d}.png"
        self.index += 1
        path = self.screenshot_dir / name
        try:
            observation.screenshot.save(path)
        except OSError:
            return None
        return str(path)
