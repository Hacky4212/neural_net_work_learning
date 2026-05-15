from __future__ import annotations

from pathlib import Path

from game_ai.action.action_schema import Decision, WaitAction
from game_ai.action.action_space import DiscreteActionSpace
from game_ai.learning.features import state_to_vector
from game_ai.memory.game_memory import GameMemory
from game_ai.nn.image_utils import image_to_tensor, zero_image_tensor
from game_ai.nn.models import ActorCriticNet, torch
from game_ai.perception.perceptor import GameState


class TorchPolicy:
    def __init__(
        self,
        model_path: str,
        action_space_path: str,
        vector_size: int,
        image_size: int,
        use_images: bool,
    ) -> None:
        self.model_path = model_path
        self.action_space = DiscreteActionSpace.load(action_space_path)
        self.image_size = image_size
        self.use_images = use_images
        self.model_loaded = False
        self.model = ActorCriticNet(
            vector_size=vector_size,
            action_count=len(self.action_space),
            image_size=image_size,
            use_images=use_images,
        )
        self._load()

    def decide(self, state: GameState, memory: GameMemory) -> Decision:
        if not self.model_loaded:
            return Decision(reason="torch_model_not_found", action=WaitAction(300))

        vector = torch.tensor([state_to_vector(state)], dtype=torch.float32)
        image = self._last_image(memory)
        with torch.no_grad():
            logits, _value = self.model(vector, image)
            probs = torch.softmax(logits, dim=1)
            action_index = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, action_index])

        label = self.action_space.label(action_index)
        return Decision(
            reason=f"torch:{label}:confidence={confidence:.3f}",
            action=self.action_space.action(action_index),
        )

    def _load(self) -> None:
        path = Path(self.model_path)
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
            self.model_loaded = True
        self.model.eval()

    def _last_image(self, memory: GameMemory):
        if not self.use_images or not memory.history:
            return zero_image_tensor(self.image_size, torch, batched=True)

        image = memory.history[-1].observation.screenshot
        return image_to_tensor(image, self.image_size, torch, batched=True)
