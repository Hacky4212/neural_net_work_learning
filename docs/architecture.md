# Architecture

Goal:

```text
Reuse one 3D game AI learning framework across multiple game projects.
```

## Layers

```text
Game Adapter
  Observe the game
  Parse game state
  Execute actions
  Score rewards

Learning Core
  Action space
  Data recording
  Behavior cloning
  PPO reinforcement learning
  Neural network models

Runtime
  Rule policy
  Simple MLP policy
  PyTorch policy
  Executor
```

## Game-Specific Files

Action table:

```text
configs/action_space.json
```

State parser:

```text
game_ai/perception/perceptor.py
```

Reward rules:

```text
game_ai/reward/reward_scorer.py
```

Window and script config:

```text
game_ai/config.py
```

Window click helper:

```text
tools/window_click.go
tools/window_click.exe
```

Default real-click backend:

```text
window_sendinput
```

It accepts window/client coordinates.
It converts them to screen coordinates before clicking.

## Training Data Quality

Only target-window steps are recorded by default.

This prevents dev-machine dry-runs from producing empty no-window samples.

To record synthetic or no-window data, set:

```python
record_missing_window = True
```

## Learning Path

Step 1:

```text
Run the rule policy.
```

Step 2:

```text
Record screenshots, states, actions, and rewards.
```

Step 3:

```text
Train behavior cloning.
```

Behavior cloning:

```text
Learn from human actions or rule-policy actions first.
```

Step 4:

```text
Continue with PPO online reinforcement learning.
```

PPO:

```text
Improve actions while playing, based on rewards.
```

## Training Entry Points

State-only MLP:

```powershell
python -m game_ai.train.train_imitation
```

Image + state PyTorch model:

```powershell
python -m game_ai.train.train_behavior_cloning_torch
```

PPO reinforcement learning:

```powershell
python -m game_ai.train.train_rl
```

## Runtime Entry Point

```powershell
python -m game_ai.main
```

When `dry_run = False`, the runtime asks for the click reference resolution.
It also requires administrator PowerShell before real actions.
The runtime does not auto-open a UAC prompt.

Example:

```text
1280x720
```

Manual click test:

```powershell
python -m game_ai.tools.test_window_click
```

Policy options:

```python
policy_type = "rule"
policy_type = "model"
policy_type = "torch"
```
