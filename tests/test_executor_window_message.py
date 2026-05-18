import unittest

from game_ai.action.action_schema import ClickAction
from game_ai.config import ExecutorConfig
from game_ai.executor.executor import Executor


class RecordingExecutor(Executor):
    def __init__(self, config: ExecutorConfig) -> None:
        super().__init__(config, admin_check=lambda: True)
        self.commands: list[list[str]] = []

    def _run(self, command: list[str]) -> None:
        self.commands.append(command)

    def _resolved_window_title(self) -> str:
        return "MIRMG(1)"


class ExecutorWindowMessageTests(unittest.TestCase):
    def test_real_actions_require_admin_by_default(self) -> None:
        config = ExecutorConfig(
            dry_run=False,
            require_window=False,
        )

        with self.assertRaisesRegex(RuntimeError, "administrator"):
            Executor(config, admin_check=lambda: False)

    def test_dry_run_does_not_require_admin(self) -> None:
        config = ExecutorConfig(dry_run=True)

        Executor(config, admin_check=lambda: False)

    def test_admin_check_can_be_disabled_for_real_actions(self) -> None:
        config = ExecutorConfig(
            dry_run=False,
            require_window=False,
            require_admin_for_real_actions=False,
        )

        Executor(config, admin_check=lambda: False)

    def test_window_message_click_uses_message_mode_without_focus(self) -> None:
        config = ExecutorConfig(
            dry_run=False,
            require_window=False,
            click_backend="window_message",
            focus_before_action=True,
        )
        executor = RecordingExecutor(config)

        executor.execute(ClickAction(640, 360))

        self.assertEqual(len(executor.commands), 1)
        command = executor.commands[0]
        self.assertIn("-mode", command)
        self.assertIn("message", command)
        self.assertIn("-focus=false", command)

    def test_window_sendinput_click_uses_sendinput_mode_with_focus(self) -> None:
        config = ExecutorConfig(
            dry_run=False,
            require_window=False,
            click_backend="window_sendinput",
            focus_before_action=True,
        )
        executor = RecordingExecutor(config)

        executor.execute(ClickAction(640, 360))

        self.assertEqual(len(executor.commands), 1)
        command = executor.commands[0]
        self.assertIn("-mode", command)
        self.assertIn("sendinput", command)
        self.assertIn("-focus=true", command)


if __name__ == "__main__":
    unittest.main()
