from __future__ import annotations

import time

from game_ai.config import config
from game_ai.data.recorder import Recorder
from game_ai.executor.executor import Executor, ensure_runtime_permissions
from game_ai.memory.game_memory import GameMemory
from game_ai.observation.observer import Observer, RawObservation
from game_ai.perception.perceptor import Perceptor
from game_ai.policy.rule_policy import RulePolicy
from game_ai.reward.reward_scorer import RewardScorer
from game_ai.runtime_config import prompt_for_window_resolution


def run() -> None:
    try:
        ensure_runtime_permissions(config.executor)
    except RuntimeError as exc:
        print(exc)
        return

    prompt_for_window_resolution(config.executor)

    observer = Observer(config.observation)
    perceptor = Perceptor()
    memory = GameMemory()
    reward_scorer = RewardScorer(config.reward)
    policy = _build_policy()
    executor = Executor(config.executor)
    recorder = (
        Recorder(
            config.data.record_path,
            save_screenshots=config.data.save_screenshots,
            screenshot_dir=config.data.screenshot_dir,
        )
        if config.data.enable_recorder
        else None
    )

    for step in range(config.agent.max_steps):
        raw_observation = observer.observe()
        game_state = perceptor.parse(raw_observation)
        reward = reward_scorer.score(memory.last_state, game_state)
        memory.remember(raw_observation, game_state, reward.total, reward.breakdown)

        decision = policy.decide(game_state, memory)
        if recorder is not None and _should_record(raw_observation):
            recorder.record(raw_observation, game_state, decision, reward, memory.total_reward)

        print(
            f"[step={step}] "
            f"window_found={raw_observation.window_found} "
            f"window_title={raw_observation.window_title!r} "
            f"reward={reward.total:.3f} "
            f"total_reward={memory.total_reward:.3f} "
            f"reason={decision.reason} "
            f"action={decision.action.to_dict()}"
        )

        executor.execute(decision.action)
        time.sleep(config.agent.tick_seconds)


def _build_policy():
    if config.agent.policy_type == "model":
        from game_ai.policy.model_policy import ModelPolicy

        return ModelPolicy(config.model.model_path)

    if config.agent.policy_type == "torch":
        from game_ai.policy.torch_policy import TorchPolicy

        return TorchPolicy(
            model_path=config.model.torch_model_path,
            action_space_path=config.model.action_space_path,
            vector_size=config.model.vector_size,
            image_size=config.model.image_size,
            use_images=config.model.use_images,
        )

    return RulePolicy(
        hp_low_threshold=config.agent.hp_low_threshold,
        mp_low_threshold=config.agent.mp_low_threshold,
    )


def _should_record(observation: RawObservation) -> bool:
    return config.data.record_missing_window or observation.window_found


if __name__ == "__main__":
    run()
