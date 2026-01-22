# CodingGame Gym - Python Wrapper for CodingGame Games

## Architecture Overview

This package provides an **OpenAI Gym-like interface** for CodingGame multiplayer games using JPype to bridge Python and Java. It enables training reinforcement learning agents against CodingGame game engines.

### Core Components

- **`cggym.py`** - Main API: `make()`, `make_multi()`, `list_envs()`
- **`game_runner.py`** - `GameRunner` class for running multiple games with CLI-like interface
- **`catalogue.py`** - Registry of available games and their JAR paths
- **`utils.py`** - JPype utilities (JVM startup, Java collection conversions)
- **`codingame_gym/games/*.jar`** - Compiled game engines (from separate Java projects)

## Key Workflows

### Installing the Package

```bash
pip install -e .
```

This makes imports work via `import codingame_gym.cggym as gym` from anywhere.

### Available Games

```python
import codingame_gym.cggym as gym
gym.list_envs()  # ['seabed', 'cellularena', 'backtrack']
```

Games are registered in `catalogue.py`. Each points to a JAR in `codingame_gym/games/`.

### Single Environment Training

```python
env = gym.make('cellularena')
obs = env.reset(league_level=3, random_seed=12345)

done = False
while not done:
    actions = [['WAIT'], ['WAIT']]  # List[List[str]]
    obs, done = env.step(actions)
    scores = env.get_scores()
```

### Multi-Environment Training (Parallel)

```python
NUM_ENVS = 5
env = gym.make_multi('backtrack', NUM_ENVS)

random_seeds = np.random.randint(0, 10000, NUM_ENVS)
observations = env.reset(league_level=5, random_seeds=random_seeds)

while not all_done:
    # Actions: List[List[List[str]]] - [env][player][actions]
    actions = [compute_action(obs) for obs in observations]
    observations, dones = env.step(actions)
    scores = env.get_scores()  # List[List[int]] - [env][player_score]
```

### GameRunner for Batch Evaluation

```python
from codingame_gym.game_runner import GameRunner

runner = GameRunner('cellularena')
scores = runner.run_games(
    num_games=10,
    league_level=3,
    agent_1_cmd="python bots/my_agent.py",
    agent_2_cmd="python bots/boss.py",
    seeds=[1,2,3,4,5,6,7,8,9,10],
    debug=False  # True serves replay on localhost, but only 1 game
)
```

## Critical Patterns

### JPype Integration

**JVM Lifecycle:**
- JVM started lazily on first `make()` or `make_multi()` call
- Only one JVM per Python process (JPype limitation)
- JVM started with system property: `-Dgame.mode=multi`
- Cannot restart JVM without restarting Python process

**Java Collection Conversions:**
```python
# Python → Java
convert_to_java_list_of_lists(py_list)          # List[List[str]]
convert_to_java_list_of_lists_of_lists(py_list)  # List[List[List[str]]]

# Java → Python (automatic via JPype)
java_array = resetResult.observations  # float[]
numpy_obs = np.array(java_array, dtype=np.float32)
```

### Action Format Differences

**Single Gym (`GameEnv`):**
- Input: Python `List[List[str]]` (player → actions)
- Converted internally to string: `"action1\naction2,action1\naction2"` (players separated by `,`, actions by `\n`)

**Multi Gym (`MultiGameEnv`):**
- Input: Python `List[List[List[str]]]` (env → player → actions)
- Converted to Java `ArrayList<ArrayList<ArrayList<String>>>`

### Observation Handling

**String Observations (raw):**
- Returned by Java as `List<String>` per player
- Accessible via `env.global_info` after `reset()`
- Format depends on game; typically line-separated entity data

**Numeric Observations (for ML):**
- Java encodes game state as `float[]` with shape metadata
- Python reshapes: `np.array(obs).reshape(shape)`
- Shape typically: `[num_players, width, height, channels]` or similar

### League Levels

Games have progressive difficulty (typically 1-5):
- Level 1: Simplest rules, fastest for testing
- Level 5: Full game complexity
- Passed to both `reset()` and `GameRunner.run_games()`

## Testing & Debugging

**Test Files:**
- `test_gym_environment.py` - Single `GameEnv` functionality
- `test_multigym_environment.py` - `MultiGameEnv` with parallel envs
- `test_game_runner.py` - `GameRunner` batch execution

**Debugging Games:**
```python
runner = GameRunner('cellularena')
runner.run_games(
    num_games=1,
    league_level=3,
    agent_1_cmd="python my_agent.py",
    agent_2_cmd="python boss.py",
    seeds=[12345],
    debug=True  # Opens localhost server with replay
)
```

**Common Issues:**
- **"JVM already started"** - Cannot change classpath after first JVM start; restart Python
- **Mismatched num_envs and seeds** - `len(random_seeds)` must equal `num_envs`
- **Java output spam** - Use `RedirectJavaOutput()` context manager to suppress

## Project-Specific Conventions

### Game Registration

To add a new game:
1. Build the game as JAR (from separate Java project)
2. Copy JAR to `codingame_gym/games/<gamename>.jar`
3. Add entry to `envs_catalog` in `catalogue.py`:
```python
'mygame': {
    'jar_path': games_path / 'mygame.jar',
}
```

### State Compilation Pattern

Games return raw string observations. Projects typically define a `State` class to parse and structure data:

```python
# See train_ppo.py pattern
def compile_states(observations):
    states = []
    for game_obs in observations:
        for player_obs in game_obs:
            state = State.from_strings(player_obs)
            states.append(state)
    return states
```

### Bot Command Formats

Commands passed to `GameRunner` or Java `MultiplayerGameRunner`:
- Python: `"python bots/my_bot.py"`
- C++: Compile first, then `"bash -c /tmp/compiled_bot"`
- Java: `"java -cp . MyBot"`

Game engine spawns these as subprocesses for each player.

### Performance Considerations

- **MultiGymEnv** is significantly faster than running single envs in sequence
- Parallelism is on Java side (efficient thread pool)
- Typical throughput: 500-1000 episodes/second on modern hardware (game-dependent)
- Minimize Python ↔ Java data transfers; batch operations when possible

## Dependencies

**Core:** JPype1, NumPy
**Optional (for training examples):** PyTorch, pandas

The package uses `pkg_resources` for locating JAR files, ensuring correct paths when installed via `pip install -e .`.

## Known Limitations

- **No reward calculation** - Gym `step()` returns only observations and done flag; compute rewards in Python based on `get_scores()` or state changes
- **No rendering** - Use `GameRunner` with `debug=True` or Java `MultiplayerGameRunner` for visual replays
- **Single JVM constraint** - Can only work with one game at a time per Python process (can't mix games)
- **Seed arrays** - `MultiGymEnv` requires numpy array or list of seeds, not single int

## Integration with Java Game Projects

This package expects Java game engines to expose:
- `GymEnv.make()` - Creates single environment
- `MultiGymEnv.make(int numEnvs)` - Creates parallel environments
- Both must implement `.reset()`, `.step()`, `.get_scores()`, `.global_info()` methods
- See CodingGame engine implementation in `cellularena` project for reference architecture
