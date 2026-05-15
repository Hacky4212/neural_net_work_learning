from __future__ import annotations

from pathlib import Path

from game_ai.action.action_space import DiscreteActionSpace
from game_ai.config import config
from game_ai.data.recorder import Recorder
from game_ai.env.game_env import GameEnv
from game_ai.executor.executor import Executor
from game_ai.nn.torch_utils import require_torch
from game_ai.observation.observer import Observer
from game_ai.perception.perceptor import Perceptor
from game_ai.reward.reward_scorer import RewardScorer
from game_ai.runtime_config import prompt_for_window_resolution


def main() -> None:
    prompt_for_window_resolution(config.executor)

    try:
        torch, _nn, _F = require_torch()
    except RuntimeError as exc:
        print(exc)
        return

    from game_ai.nn.models import ActorCriticNet
    from game_ai.nn.ppo import PPOTrainer

    action_space = DiscreteActionSpace.load(config.model.action_space_path)
    model = ActorCriticNet(
        vector_size=config.model.vector_size,
        action_count=len(action_space),
        image_size=config.model.image_size,
        use_images=config.model.use_images,
    )

    model_path = Path(config.model.torch_model_path)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    recorder = (
        Recorder(
            config.data.record_path,
            save_screenshots=config.data.save_screenshots,
            screenshot_dir=config.data.screenshot_dir,
        )
        if config.data.enable_recorder
        else None
    )
    env = GameEnv(
        observer=Observer(config.observation),
        perceptor=Perceptor(),
        reward_scorer=RewardScorer(config.reward),
        executor=Executor(config.executor),
        recorder=recorder,
        tick_seconds=config.agent.tick_seconds,
        max_steps=config.agent.max_steps,
        record_missing_window=config.data.record_missing_window,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    trainer = PPOTrainer(
        env=env,
        action_space=action_space,
        model=model,
        optimizer=optimizer,
        gamma=config.model.gamma,
        gae_lambda=config.model.gae_lambda,
        clip=config.model.ppo_clip,
        value_coef=config.model.value_coef,
        entropy_coef=config.model.entropy_coef,
        rollout_steps=config.model.rollout_steps,
        ppo_epochs=config.model.ppo_epochs,
        image_size=config.model.image_size,
        use_images=config.model.use_images,
    )
    trainer.train(updates=config.model.epochs, save_path=config.model.torch_model_path)


if __name__ == "__main__":
    main()
