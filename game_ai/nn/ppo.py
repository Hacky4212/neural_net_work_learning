from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

from game_ai.action.action_space import DiscreteActionSpace
from game_ai.env.game_env import EnvStep, GameEnv
from game_ai.learning.features import state_to_vector
from game_ai.nn.image_utils import image_to_tensor
from game_ai.nn.models import ActorCriticNet, torch, F


@dataclass
class RolloutBatch:
    vectors: list = field(default_factory=list)
    images: list = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list = field(default_factory=list)


class PPOTrainer:
    def __init__(
        self,
        env: GameEnv,
        action_space: DiscreteActionSpace,
        model: ActorCriticNet,
        optimizer,
        gamma: float,
        gae_lambda: float,
        clip: float,
        value_coef: float,
        entropy_coef: float,
        rollout_steps: int,
        ppo_epochs: int,
        image_size: int,
        use_images: bool,
    ) -> None:
        self.env = env
        self.action_space = action_space
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.image_size = image_size
        self.use_images = use_images

    def train(self, updates: int, save_path: str) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        step = self.env.reset()
        for update in range(updates):
            batch, step = self._collect(step)
            metrics = self._update(batch)
            torch.save(self.model.state_dict(), save_path)
            print(
                f"update={update + 1} "
                f"reward={metrics['mean_reward']:.4f} "
                f"done_rate={metrics['done_rate']:.3f} "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f} "
                f"saved={save_path}"
            )

    def _collect(self, current_step: EnvStep) -> tuple[RolloutBatch, EnvStep]:
        batch = RolloutBatch()
        step = current_step
        for _ in range(self.rollout_steps):
            vector = torch.tensor([state_to_vector(step.state)], dtype=torch.float32)
            source_image = step.observation.screenshot if self.use_images else None
            image = image_to_tensor(source_image, self.image_size, torch, batched=True)
            with torch.no_grad():
                action_id, log_prob, value, _entropy = self.model.act(vector, image)

            action_index = int(action_id.item())
            next_step = self.env.step(self.action_space.action(action_index))

            batch.vectors.append(vector.squeeze(0))
            batch.images.append(image.squeeze(0))
            batch.actions.append(action_index)
            batch.log_probs.append(log_prob.squeeze(0))
            batch.values.append(value.squeeze(0))
            batch.rewards.append(next_step.reward.total)
            batch.dones.append(next_step.done)
            step = self.env.reset() if next_step.done else next_step

        return batch, step

    def _update(self, batch: RolloutBatch) -> dict[str, float]:
        vectors = torch.stack(batch.vectors)
        images = torch.stack(batch.images)
        actions = torch.tensor(batch.actions, dtype=torch.long)
        old_log_probs = torch.stack(batch.log_probs).detach()
        values = torch.stack(batch.values).detach()

        returns, advantages = _gae(
            batch.rewards,
            batch.dones,
            values,
            self.gamma,
            self.gae_lambda,
        )

        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        for _ in range(self.ppo_epochs):
            logits, new_values = self.model(vectors, images)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantages
            policy_loss = -torch.min(ratio * advantages, clipped).mean()
            value_loss = F.mse_loss(new_values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_loss_value = float(policy_loss.item())
            value_loss_value = float(value_loss.item())
            entropy_value = float(entropy.item())

        mean_reward = sum(batch.rewards) / len(batch.rewards) if batch.rewards else 0.0
        done_rate = sum(1 for done in batch.dones if done) / len(batch.dones) if batch.dones else 0.0
        return {
            "mean_reward": float(mean_reward),
            "done_rate": float(done_rate),
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
        }


def _gae(rewards: list[float], dones: list[bool], values, gamma: float, gae_lambda: float):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * mask - float(values[index])
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = float(values[index])

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns.detach(), advantages.detach()
