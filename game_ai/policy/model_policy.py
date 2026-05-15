from __future__ import annotations

from pathlib import Path

from game_ai.action.action_schema import Decision, WaitAction
from game_ai.config import config
from game_ai.learning.action_codec import label_to_action
from game_ai.learning.features import state_to_vector
from game_ai.learning.simple_mlp import SimpleMLP
from game_ai.memory.game_memory import GameMemory
from game_ai.perception.perceptor import GameState


class ModelPolicy:
    def __init__(self, model_path: str = config.model.model_path) -> None:
        self.model_path = model_path
        self.model: SimpleMLP | None = None
        self.labels: list[str] = []
        self._load()

    def decide(self, state: GameState, memory: GameMemory) -> Decision:
        if self.model is None or not self.labels:
            return Decision(reason="model_not_loaded", action=WaitAction(300))

        prediction = self.model.predict(state_to_vector(state))
        if prediction.label < 0 or prediction.label >= len(self.labels):
            return Decision(reason="model_label_out_of_range", action=WaitAction(300))

        label = self.labels[prediction.label]
        action = label_to_action(label)
        return Decision(
            reason=f"model:{label}:confidence={prediction.confidence:.3f}",
            action=action,
        )

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            return

        self.model, self.labels = SimpleMLP.load(self.model_path)
