from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from game_ai.config import ObservationConfig
from game_ai.window.window_utils import get_window_rect, resolve_window, restore_window


@dataclass
class RawObservation:
    timestamp: float
    screenshot: Any | None = None
    hp: float | None = None
    mp: float | None = None
    position: tuple[float, float, float] | None = None
    target: str | None = None
    ui_state: str | None = None
    window_found: bool = False
    window_title: str | None = None
    window_rect: tuple[int, int, int, int] | None = None


class Observer:
    def __init__(self, config: ObservationConfig) -> None:
        self.config = config

    def observe(self) -> RawObservation:
        screenshot, window_found, window_title, window_rect = self._capture_screen()
        return RawObservation(
            timestamp=time.time(),
            screenshot=screenshot,
            hp=None,
            mp=None,
            position=None,
            target=None,
            ui_state=None,
            window_found=window_found,
            window_title=window_title,
            window_rect=window_rect,
        )

    def _capture_screen(self) -> tuple[Any | None, bool, str | None, tuple[int, int, int, int] | None]:
        try:
            from PIL import ImageGrab
        except ImportError:
            return None, False, None, None

        window = resolve_window(self.config.window_title, self.config.fallback_window_keywords)
        if window is None:
            return None, False, None, None

        if self.config.restore_before_capture:
            restore_window(window.hwnd)
        rect = get_window_rect(window.hwnd) or window.rect
        return ImageGrab.grab(bbox=rect), True, window.title, rect
