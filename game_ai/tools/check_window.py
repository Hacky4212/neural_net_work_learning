from __future__ import annotations

from game_ai.config import config
from game_ai.window.window_utils import list_windows, resolve_window


def main() -> None:
    target = resolve_window(
        config.observation.window_title,
        config.observation.fallback_window_keywords,
    )
    print(f"target={target}")
    print("windows:")
    for window in list_windows():
        print(f"- hwnd={window.hwnd} title={window.title!r} rect={window.rect}")


if __name__ == "__main__":
    main()
