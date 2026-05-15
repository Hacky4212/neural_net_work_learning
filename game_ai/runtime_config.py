from __future__ import annotations

import re

from game_ai.config import ExecutorConfig


def prompt_for_window_resolution(config: ExecutorConfig) -> None:
    if config.dry_run or not config.prompt_window_resolution:
        return

    current = f"{config.click_reference_width}x{config.click_reference_height}"
    value = input(f"Input game window resolution for click coordinates [{current}]: ").strip()
    if not value:
        return

    width, height = parse_resolution(value)
    config.click_reference_width = width
    config.click_reference_height = height
    print(f"click reference resolution={width}x{height}")


def parse_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX,* ]\s*(\d+)\s*", value)
    if not match:
        raise ValueError("resolution format must be like 1280x720")

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("resolution width and height must be positive")
    return width, height
