from __future__ import annotations

from pathlib import Path

from game_ai.action.action_space import DiscreteActionSpace
from game_ai.data.dataset import JsonlDataset, is_informative_sample, reward_to_sample_weight
from game_ai.learning.features import dict_state_to_vector
from game_ai.nn.image_utils import image_path_to_tensor, zero_image_tensor
from game_ai.nn.torch_utils import require_torch


torch, _, _ = require_torch()


class BehaviorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        action_space: DiscreteActionSpace,
        image_size: int = 84,
        use_images: bool = True,
    ) -> None:
        self.path = Path(path)
        self.action_space = action_space
        self.image_size = image_size
        self.use_images = use_images
        self.samples = self._load()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        vector = torch.tensor(sample["vector"], dtype=torch.float32)
        action = torch.tensor(sample["action"], dtype=torch.long)
        weight = torch.tensor(sample["weight"], dtype=torch.float32)
        image = self._load_image(sample.get("screenshot"))
        return {
            "vector": vector,
            "image": image,
            "action": action,
            "weight": weight,
        }

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []

        samples: list[dict] = []
        for row in JsonlDataset(str(self.path)):
            if not is_informative_sample(row):
                continue

            action = row.get("decision", {}).get("action", {})
            action_index = self.action_space.index_for_action_dict(action)
            if action_index is None:
                continue

            reward = row.get("reward", {}).get("step", 0.0)
            samples.append(
                {
                    "vector": dict_state_to_vector(row.get("state", {})),
                    "screenshot": row.get("observation", {}).get("screenshot_path"),
                    "action": action_index,
                    "weight": reward_to_sample_weight(reward),
                }
            )

        return samples

    def _load_image(self, path: str | None):
        if not self.use_images:
            return zero_image_tensor(self.image_size, torch)
        return image_path_to_tensor(path, self.image_size, torch)
