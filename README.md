# 3D Game AI Framework

This is a minimal framework for controlling a 3D game with scripts.

Main flow:

```text
Observation -> Perception -> Memory -> Policy -> Action -> Executor
```

## Dev Machine

This machine can be used for coding only.

Keep this setting in `game_ai/config.py`:

```python
dry_run = True
```

The program will print actions.
It will not call `click.exe` or `key.exe`.

## Game Machine

Copy this folder to the physical machine that runs the game.

Mouse clicks use the root click tool:

```text
click.exe
```

The default click backend is `screen_click`.
Python converts window/client coordinates to screen coordinates first.
Then it calls `click.exe x y`.

Full-screen game mode is the most accurate path for this backend.

The Go window clicker is still available as an optional backend:

```text
tools/window_click.exe
```

Its config field is:

```python
window_click_exe = "tools/window_click.exe"
```

Available optional backends in `game_ai/config.py`:

```python
click_backend = "window_message"
click_backend = "window_sendinput"
```

`window_message` does not move the real mouse cursor.
Some games ignore it.

`window_sendinput` moves the real cursor and sends a real left click.

If it is missing, build it:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_window_click.ps1
```

Key actions still use:

```text
key.exe
```

The default game window title is:

```text
MIRMG(1)
```

To execute real actions, change:

```python
dry_run = False
```

Run the project from administrator PowerShell for real actions.
The runtime checks this before clicking.
It does not auto-open a UAC prompt.

The default click reference resolution is:

```text
1920x1080
```

The runtime does not ask for it at startup.
Change these values in `game_ai/config.py` when needed:

```python
click_reference_width = 1920
click_reference_height = 1080
```

Click actions use window/client coordinates.

Meaning:

```text
0,0 = game client top-left
```

Test one click point:

```powershell
python -m game_ai.tools.test_window_click
```

## Safety

The executor only targets `MIRMG(1)` by default.

If the window is not found, real click and key actions are skipped.

## Reward

Runtime reward is enabled.
Reward scoring is passive.
It does not click, press keys, or change focus.

Each step prints:

```text
reward
total_reward
```

Training data is written to:

```text
data/actions.jsonl
```

By default, missing-window steps are not recorded.
This prevents dry-run data from teaching the model to only wait.

To record synthetic or no-window runs, change:

```python
record_missing_window = True
```

Current reward signals:

```text
alive
combat
idle
hp_gain
hp_loss
mp_gain
mp_loss
target_acquired
target_cleared
death
kill
task_done
```

## Neural Learning

Install training dependencies on the game or training machine:

```powershell
pip install -r requirements.txt
```

Train a neural policy from recorded data:

```powershell
python -m game_ai.train.train_imitation
```

Train the PyTorch vision policy:

```powershell
python -m game_ai.train.train_behavior_cloning_torch
```

Continue training online with PPO:

```powershell
python -m game_ai.train.train_rl
```

The supervised trainers now print train/validation metrics and keep the best checkpoint.
PPO now prints rollout reward and update losses.

The model is saved to:

```text
models/policy_mlp.json
```

To run with the neural policy, change:

```python
policy_type = "model"
```

in `game_ai/config.py`.

Engineering notes:

```text
docs/architecture.md
```

## Run

```powershell
python -m game_ai.main
```
