# CodingGame Gym - Python Wrapper for CodingGame Games

A Python wrapper that provides an OpenAI Gym-like interface for CodingGame multiplayer games. Train reinforcement learning agents against CodingGame game engines with support for parallel environments and visual debugging.

## Features

- **Gym-like API** - Familiar `reset()` and `step()` interface for RL training
- **Parallel Environments** - Run multiple games simultaneously with `MultiGameEnv` for fast training
- **Visual Debugging** - Debug mode serves game replays on localhost for visual inspection
- **GameRunner API** - Run batch games with CLI-like interface for evaluation
- **Currently Supports** - `backtrack` game (`cellularena` and `seabed` are not compatible with the current version)

## Installation

Clone the repo and install in development mode:

```bash
git clone <repo-url>
cd codinggame_gym
pip install -e .
```

**Requirements:** See `requirements.txt` for dependencies (JPype1, NumPy, etc.)

## Quick Start

### Single Environment

```python
import codingame_gym.cggym as gym

# List available games
print(gym.list_envs())  # ['backtrack']

# Create environment (num_envs=1 by default)
env = gym.make('backtrack')

# Reset with league level and seed
env.reset(league_level=3, random_seeds=12345)

# Access observations via property (numpy array: [num_players, obs_size])
# State encoding is done in Java (StateEncoder.java)
observations = env.observations

done = False
while not done:
    # Provide actions for all players: List[List[str]]
    # Each player can have multiple actions (e.g., multiple units)
    actions = [
        ['AUTOPLACE 0 0 29 19'],  # Player 0
        ['AUTOPLACE 5 5 25 15']   # Player 1
    ]
    
    done = env.step(actions)
    observations = env.observations  # Access updated observations
    scores = env.get_scores()  # [score_p0, score_p1]

    # Reward calculation can be done here
    # or on the Java side and written in the observations
    reward = calculate_reward(observations, scores)

env.shutdown()
```

### Multi-Environment (Parallel)

For faster training, run multiple games in parallel:

```python
import codingame_gym.cggym as gym
import numpy as np

# Create 100 parallel environments
num_envs = 100
env = gym.make('backtrack', num_envs=num_envs)

# Reset all environments with different seeds
random_seeds = np.random.randint(0, 1000000, num_envs)
env.reset(league_level=5, random_seeds=random_seeds)

# Access observations via property (numpy array: [num_envs, num_players, obs_size])
# State encoding is done in Java (StateEncoder.java) for performance
observations = env.observations

all_done = False
while not all_done:
    # Actions: List[List[List[str]]] - [env][player][actions]
    actions = compute_actions(observations)  # Your policy here
    
    dones = env.step(actions)
    observations = env.observations  # Access updated observations
    scores = env.get_scores()  # List[List[int]] - [env][player_score]

    # Reward calculation can be done here
    # or on the Java side and written in the observations
    rewards = calculate_rewards(observations, scores)
    
    all_done = all(dones)

env.shutdown()
```

### GameRunner - Batch Evaluation

Run multiple games with command-line style interface:

```python
from codingame_gym.game_runner import GameRunner

runner = GameRunner('backtrack')

# Run 10 games for evaluation
scores = runner.run_games(
    num_games=10,
    league_level=3,
    agent_1_cmd="python my_agent.py",
    agent_2_cmd="python bots/boss.py",
    seeds=list(range(100, 110)),
    debug=False
)


## Strategic Bot Baselines

This repo includes a series of hand-crafted Python bots (`strategic_bot_v*.py`) plus helper eval scripts.

Run V16 vs V15 (league 5):

```bash
python eval_v16_vs_v15.py --games 50 --league 5 --seed0 7000
```
# scores is List[List[int]] - [[p0_score, p1_score], ...]
wins = sum(1 for s in scores if s[0] > s[1])
print(f"Agent 1 won {wins}/{len(scores)} games")
```

### Debug Mode - Visual Replay

Run a single game with visual replay server:

```python
runner = GameRunner('backtrack')

# Set debug=True to open replay server
result = runner.run_games(
    num_games=1,
    league_level=3,
    agent_1_cmd="python my_agent.py",
    agent_2_cmd="python bots/boss.py",
    seeds=[12345],
    debug=True  # Serves replay on localhost
)

# Check terminal output for localhost URL
# Open in browser to watch the game
# Press Ctrl+C when done
```

## Rollout & Dataset Tools

These helper scripts live under `scripts/` and speed up experimentation:

- `smoke_test.py` – quick sanity check that the JVM bridge, observation buffers, and JPype wiring all work on your machine.
- `heuristic_autoplace.py` – runs a simple scripted policy that repeatedly issues `AUTOPLACE` for a desired town pair; useful for generating baseline matches or JSONL datasets via `--record`.
- `rl_scaffold.py` – vectorized rollout driver capable of running many env copies in parallel, writing step-by-step JSONL logs (`--record`), and keeping an in-memory replay buffer (`--buffer-capacity` / `--buffer-dump-path`). Swap out the built-in `AutoplaceVectorPolicy` with your model inference function when you are ready to train.
- `dataset_stats.py` – post-process one or more JSONL files (from the recorder or replay buffer dump) to report aggregate episode counts, reward statistics, and the most common action strings:

    ```bash
    python scripts/dataset_stats.py data/buffer.jsonl
    python scripts/dataset_stats.py data/run_*.jsonl --top-actions 10
    ```

- `export_npz.py` – converts JSONL rollouts into compressed NPZ chunks (`observations`, `next_observations`, `rewards`, etc.) that can be consumed by PyTorch/NumPy training loops. Example:

    ```bash
    python scripts/export_npz.py data/run.jsonl --output-prefix data/run_npz/chunk --chunk-size 4096
    ```

Typical workflow: run `rl_scaffold.py` with recording enabled, inspect the resulting dataset with `dataset_stats.py`, and then feed the cleaned JSONL (or the replay buffer dump) into your RL/BC training loop.

## Key Differences from OpenAI Gym

This wrapper adapts CodingGame's multiplayer architecture to a gym-like interface:

### Observations
- **Format:** NumPy arrays (float32) accessed via `.observations` property
- **Shape:** Single env: `[num_players, obs_size]`, Multi env: `[num_envs, num_players, obs_size]`
- **Encoding:** State encoding done in Java (`StateEncoder.java`) for performance
- **Access:** Zero-copy via shared ByteBuffer for efficient memory transfer
- **Debug Mode:** String observations available via `get_frame_strings()` when `debug=True`

### Actions
- **Single env:** `List[List[str]]` - `[player][actions]`
- **Multi env:** `List[List[List[str]]]` - `[env][player][actions]`
- **Format:** Action strings like `"AUTOPLACE 0 0 29 19"` or `"WAIT"`

### Rewards
- **Not provided automatically** - Calculate rewards based on `get_scores()` or state changes
- CodingGame games typically use final scores rather than step rewards

### Step Returns
- **Single env:** `done` (bool) - No observations or reward returned
- **Multi env:** `dones` (list of bool) - No observations or reward returned
- **Access observations:** Via `.observations` property after calling `step()`

## API Reference

### `gym.make(env_name, num_envs=1)`
Create a single or multi-environment.

### `env.reset(league_level, random_seeds)`
Reset environment(s). Returns `None`. Access observations via `.observations` property.
- `random_seeds` can be a single int (for single env) or list of ints (for multi env; must match `num_envs`)
- League level controls game complexity (typically 1-5)

### `env.step(actions)`
Execute actions. Returns `done` (single) or `dones` (multi). Access observations via `.observations` property.

### `env.get_scores()`
Get current scores for all players.

### `env.shutdown()`
Clean up JVM resources (call when done).



## Architecture Notes

- **JPype Bridge:** Python communicates with Java game engines via JPype
- **Single JVM:** Only one JVM per Python process (JPype limitation)
- **Thread Safety:** MultiGameEnv uses Java thread pool for parallelism
- **Memory:** ByteBuffer shared memory for efficient observation transfer

## V25 config overrides + self-play tuning

V25 supports runtime hyperparameter overrides via `run_v25.py`.

```powershell
# Run V25 with a one-off override
./.venv/Scripts/python.exe ./run_v25.py --set SUPPORT_MAX_PLACEMENTS=3

# Or load overrides from a JSON file
./.venv/Scripts/python.exe ./run_v25.py --config ./configs/v25_best.json
```

`eval_v25_vs.py` can evaluate V25 using a config file:

```powershell
./.venv/Scripts/python.exe ./eval_v25_vs.py --opponent v24 --games 10 --league 5 --seed0 11000 --safe --timeout 60 --v25-config ./configs/v25_best.json
```

Self-play tuning (promotion-based: “new config must beat previous config”):

```powershell
./.venv/Scripts/python.exe ./tune_v25.py --iters 60 --games 6 --league 5 --seed0 11000 --seed-bands 15000,18919,22000,26000 --margin 0.02 --patience 25
```

Optuna-based tuning (lighter ML / Bayesian optimization + promotion epochs):

```powershell
./.venv/Scripts/python.exe ./tune_v25_optuna.py --epochs 3 --trials 60 --games 6 --league 5 --seed0 11000 --seed-bands 15000,18919,22000,26000 --margin 0.02
```

Promoted configs are written to `configs/history/` and the latest to `configs/v25_best.json`.
