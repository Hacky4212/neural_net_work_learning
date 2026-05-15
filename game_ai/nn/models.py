from __future__ import annotations

from game_ai.nn.torch_utils import require_torch


torch, nn, F = require_torch()


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        vector_size: int,
        action_count: int,
        image_size: int = 84,
        use_images: bool = True,
    ) -> None:
        super().__init__()
        self.use_images = use_images
        self.image_size = image_size

        if use_images:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                cnn_size = self.cnn(dummy).shape[1]
        else:
            self.cnn = None
            cnn_size = 0

        self.vector_net = nn.Sequential(
            nn.Linear(vector_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        fused_size = cnn_size + 64
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_count)
        self.critic = nn.Linear(128, 1)

    def forward(self, vector, image=None):
        parts = [self.vector_net(vector)]
        if self.use_images:
            if image is None:
                image = torch.zeros(
                    vector.shape[0],
                    3,
                    self.image_size,
                    self.image_size,
                    device=vector.device,
                )
            parts.append(self.cnn(image))

        features = self.fusion(torch.cat(parts, dim=1))
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def act(self, vector, image=None):
        logits, value = self.forward(vector, image)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()
