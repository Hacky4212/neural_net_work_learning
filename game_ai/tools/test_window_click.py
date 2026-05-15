from __future__ import annotations

from game_ai.action.action_schema import ClickAction
from game_ai.config import config
from game_ai.executor.executor import Executor
from game_ai.runtime_config import prompt_for_window_resolution


def main() -> None:
    prompt_for_window_resolution(config.executor)
    executor = Executor(config.executor)

    print("Input window coordinates as x,y. Example: 640,360")
    print("Input q to quit.")
    while True:
        value = input("click> ").strip()
        if value.lower() in {"q", "quit", "exit"}:
            return

        try:
            x_text, y_text = value.replace(" ", "").split(",", 1)
            action = ClickAction(int(x_text), int(y_text))
        except ValueError:
            print("format must be x,y")
            continue

        executor.execute(action)


if __name__ == "__main__":
    main()
