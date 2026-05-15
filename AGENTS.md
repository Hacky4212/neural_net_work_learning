# AGENTS.md

Guidance for coding agents working in this repository.

## Project Scope

This repository is a reusable Python framework for learning and running a 3D game AI agent.

Main runtime flow:

```text
Observation -> Perception -> Memory -> Policy -> Action -> Executor
```

Keep the dev-machine and game-machine boundary clear:

- This machine can be used for development only.
- The physical game machine runs the game and real input actions.
- Do not assume the development machine has the target game window.

## Key Defaults

Important defaults live in `game_ai/config.py`:

- `ExecutorConfig.dry_run = True`
- `ExecutorConfig.window_title = "MIRMG(1)"`
- `ExecutorConfig.click_backend = "window_message"`
- `ObservationConfig.restore_before_capture = False`
- `DataConfig.record_missing_window = False`
- `AgentConfig.policy_type = "rule"`

Preserve these safety defaults unless the user explicitly asks to change runtime behavior.

## Safety Rules

- Reward scoring must stay passive.
- Passive reward code must not click, press keys, move focus, or restore windows.
- Window focus changes belong only in real execution paths.
- Dry-run mode should print actions and avoid calling click/key helpers.
- `window_message` click mode must not move the real mouse or focus the game window.
- Real actions should target the configured game window only.
- If the target window is missing, real click and key actions should be skipped.

## Important Paths

- `game_ai/main.py`: runtime entry point.
- `game_ai/config.py`: runtime, reward, data, model, and observation config.
- `configs/action_space.json`: discrete action table.
- `game_ai/perception/perceptor.py`: game-state parsing.
- `game_ai/reward/reward_scorer.py`: reward rules.
- `game_ai/executor/executor.py`: click/key execution.
- `game_ai/window/window_utils.py`: window lookup and focus helpers.
- `game_ai/train/`: training entry points.
- `game_ai/nn/`: PyTorch model and PPO code.
- `data/actions.jsonl`: recorded training data.
- `models/`: trained model outputs.
- `docs/architecture.md`: architecture notes.

## Commands

Run the agent:

```powershell
python -m game_ai.main
```

Compile-check the package:

```powershell
python -m compileall game_ai
```

Check ML dependencies:

```powershell
python -m game_ai.tools.check_dependencies
```

Train the state-only MLP policy:

```powershell
python -m game_ai.train.train_imitation
```

Train the PyTorch behavior-cloning policy:

```powershell
python -m game_ai.train.train_behavior_cloning_torch
```

Continue online training with PPO:

```powershell
python -m game_ai.train.train_rl
```

Test one real window click on the game machine:

```powershell
python -m game_ai.tools.test_window_click
```

## Dependency Notes

Install training dependencies with:

```powershell
pip install -r requirements.txt
```

`Pillow`, `torch`, and `torchvision` are needed for the full image-based learning path.

## Change Guidelines

- Prefer existing package boundaries over adding new top-level structure.
- Keep game-specific changes in the action table, state parser, reward scorer, and config.
- Keep learning-core changes under `game_ai/learning`, `game_ai/nn`, or `game_ai/train`.
- Avoid committing generated caches such as `__pycache__`.
- Do not record missing-window dry runs as training data unless the user asks for synthetic/no-window data.
- Before claiming a runtime change is safe, run at least `python -m compileall game_ai`.
