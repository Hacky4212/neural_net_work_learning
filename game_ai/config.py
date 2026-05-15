from dataclasses import dataclass, field


@dataclass
class ExecutorConfig:
    click_exe: str = "tools/window_click.exe"
    key_exe: str = "key.exe"
    dry_run: bool = True
    window_title: str = "MIRMG(1)"
    fallback_window_keywords: tuple[str, ...] = ()
    focus_before_action: bool = True
    require_window: bool = True
    click_coordinates: str = "window"
    click_backend: str = "window_message"
    click_area: str = "client"
    click_reference_width: int = 1280
    click_reference_height: int = 720
    click_scale_to_window: bool = True
    prompt_window_resolution: bool = True


@dataclass
class AgentConfig:
    tick_seconds: float = 0.3
    max_steps: int = 10
    policy_type: str = "rule"
    hp_low_threshold: float = 0.35
    mp_low_threshold: float = 0.25


@dataclass
class RewardConfig:
    alive_reward: float = 0.02
    combat_reward: float = 0.1
    idle_penalty: float = -0.02
    hp_gain_scale: float = 5.0
    hp_loss_scale: float = -10.0
    mp_gain_scale: float = 1.0
    mp_loss_scale: float = -1.0
    target_acquired_reward: float = 2.0
    target_cleared_reward: float = 5.0
    death_penalty: float = -100.0
    kill_reward: float = 20.0
    task_done_reward: float = 50.0


@dataclass
class DataConfig:
    enable_recorder: bool = True
    record_path: str = "data/actions.jsonl"
    record_missing_window: bool = False
    save_screenshots: bool = True
    screenshot_dir: str = "data/screenshots"


@dataclass
class ModelConfig:
    model_path: str = "models/policy_mlp.json"
    torch_model_path: str = "models/policy_actor_critic.pt"
    action_space_path: str = "configs/action_space.json"
    hidden_size: int = 32
    epochs: int = 200
    learning_rate: float = 0.03
    batch_size: int = 32
    validation_ratio: float = 0.2
    validation_seed: int = 42
    early_stopping_patience: int = 20
    image_size: int = 84
    use_images: bool = True
    vector_size: int = 13
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    rollout_steps: int = 128
    ppo_epochs: int = 4


@dataclass
class ObservationConfig:
    window_title: str = "MIRMG(1)"
    fallback_window_keywords: tuple[str, ...] = ()
    restore_before_capture: bool = False


@dataclass
class AppConfig:
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)


config = AppConfig()
