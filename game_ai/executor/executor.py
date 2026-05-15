from __future__ import annotations

import subprocess
import time

from game_ai.action.action_schema import Action, ClickAction, KeyAction, WaitAction
from game_ai.config import ExecutorConfig
from game_ai.window.window_utils import focus_resolved_window, resolve_window, window_to_screen


class Executor:
    def __init__(self, config: ExecutorConfig) -> None:
        self.config = config

    def execute(self, action: Action) -> None:
        if self.config.dry_run:
            print(f"[dry-run] {action.to_dict()}")
            return

        if isinstance(action, ClickAction):
            if not self._ensure_target_window(focus_allowed=self.config.click_backend != "window_message"):
                return
            self._execute_click(action)
            return

        if isinstance(action, KeyAction):
            if not self._ensure_target_window():
                return
            self._run([self.config.key_exe, action.key])
            return

        if isinstance(action, WaitAction):
            time.sleep(action.ms / 1000)
            return

    @staticmethod
    def _run(command: list[str]) -> None:
        subprocess.run(command, check=True)

    def _resolve_click_position(self, action: ClickAction) -> tuple[int, int]:
        if self.config.click_coordinates != "window":
            return action.x, action.y

        screen_position = window_to_screen(
            self.config.window_title,
            action.x,
            action.y,
            self.config.fallback_window_keywords,
        )
        if screen_position is None:
            return action.x, action.y

        return screen_position

    def _execute_click(self, action: ClickAction) -> None:
        if self.config.click_backend in {"window_go", "window_message"}:
            window_title = self._resolved_window_title()
            focus = self.config.focus_before_action and self.config.click_backend != "window_message"
            mode = "message" if self.config.click_backend == "window_message" else "cursor"
            self._run(self._window_click_command(window_title, action, mode, focus))
            return

        x, y = self._resolve_click_position(action)
        self._run([self.config.click_exe, str(x), str(y)])

    def _window_click_command(
        self,
        window_title: str,
        action: ClickAction,
        mode: str,
        focus: bool,
    ) -> list[str]:
        return [
            self.config.click_exe,
            "-title",
            window_title,
            "-x",
            str(action.x),
            "-y",
            str(action.y),
            "-refw",
            str(self.config.click_reference_width),
            "-refh",
            str(self.config.click_reference_height),
            "-area",
            self.config.click_area,
            f"-scale={str(self.config.click_scale_to_window).lower()}",
            f"-focus={str(focus).lower()}",
            "-mode",
            mode,
        ]

    def _resolved_window_title(self) -> str:
        window = resolve_window(self.config.window_title, self.config.fallback_window_keywords)
        if window is None:
            return self.config.window_title
        return window.title

    def _ensure_target_window(self, focus_allowed: bool = True) -> bool:
        if not self.config.require_window:
            return True

        if self.config.focus_before_action and focus_allowed:
            focused = focus_resolved_window(
                self.config.window_title,
                self.config.fallback_window_keywords,
            )
            if focused:
                return True

        window = resolve_window(self.config.window_title, self.config.fallback_window_keywords)
        if window is not None:
            return True

        print(f"[skip] target window not found: {self.config.window_title}")
        return False
