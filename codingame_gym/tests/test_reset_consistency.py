"""
Test that resetting the Python gym with the same seed produces consistent scores.
"""

import pytest
import numpy as np
import jpype
import json
import subprocess
import random
import time
import sys
from pathlib import Path
import codingame_gym.cggym as gym

# Configuration
LEAGUE_LEVEL = 5
CODINGGAME_GYM_DIR = Path("/Users/fzinno/dev/git/ateam/ML-Development/codinggame_gym")
BOT_PATH = CODINGGAME_GYM_DIR / "bots" / "julien.py"
JAR_PATH = CODINGGAME_GYM_DIR / "codingame_gym" / "games" / "backtrack.jar"


def run_cli_game(seed):
    """Run a game via CLI and capture JSON output."""
    cmd = [
        "java", "-jar", str(JAR_PATH),
        "-s", str(seed),
        "-l", str(LEAGUE_LEVEL),
        "-n", "1",
        "-t",
        "-p", f"python3 {BOT_PATH}", f"python3 {BOT_PATH}"
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(CODINGGAME_GYM_DIR),
        capture_output=True,
        text=True
    )
    
    lines = [line for line in result.stdout.split('\n') if not line.startswith('WARNING')]
    json_output = '\n'.join(lines)
    
    try:
        data = json.loads(json_output)
        game_result = data['results'][0]
        # Calculate steps from actions
        num_steps = min(len(game_result['outputs']['0']), len(game_result['outputs']['1']))
        return {
            'scores': [game_result['scores']['0'], game_result['scores']['1']],
            'actions': game_result['outputs'],
            'steps': num_steps
        }
    except Exception as e:
        return {'error': str(e)}


def convert_cli_actions_to_gym(cli_actions_list):
    """
    Convert list of CLI action dicts to gym format List[List[List[str]]].
    Returns max_turns and the formatted actions array.
    """
    # Find max turns across all games
    max_turns = 0
    for actions in cli_actions_list:
        p0_len = len(actions['0'])
        p1_len = len(actions['1'])
        max_turns = max(max_turns, min(p0_len, p1_len))
    
    gym_actions_per_turn = []
    
    for turn in range(max_turns):
        turn_actions = []
        for game_idx, actions in enumerate(cli_actions_list):
            # Get actions for this turn if available, else WAIT
            p0_acts = ['WAIT']
            p1_acts = ['WAIT']
            
            if turn < len(actions['0']):
                lines = [l for l in actions['0'][turn].split('\n') if l.strip()]
                if lines: p0_acts = lines
                
            if turn < len(actions['1']):
                lines = [l for l in actions['1'][turn].split('\n') if l.strip()]
                if lines: p1_acts = lines
                
            turn_actions.append([p0_acts, p1_acts])
        gym_actions_per_turn.append(turn_actions)
        
    return max_turns, gym_actions_per_turn


def record_games_via_cli(num_games=10):
    """Record N games via CLI to get real bot actions."""
    print(f"Phase 1: Recording {num_games} games via CLI...")
    print("-" * 80)
    
    # Generate test seeds
    np.random.seed(42)
    test_seeds = np.random.randint(0, 1000000, num_games)
    
    recorded_games = []
    
    for i, seed in enumerate(test_seeds):
        if (i + 1) % 5 == 0:
            print(f"  Recording game {i+1}/{num_games}...")
            
        result = run_cli_game(seed)
        
        if 'error' in result:
            raise Exception(f"CLI failed for seed {seed}: {result['error']}")
            
        recorded_games.append({
            'seed': seed,
            'actions': result['actions'],
            'scores': result['scores'],
            'steps': result['steps']
        })
        
    print(f"✓ Successfully recorded {num_games} games\n")
    return test_seeds, recorded_games


@pytest.mark.integration
def test_bulk_reset_consistency():
    """Test that bulk reset (all envs) produces consistent scores."""
    
    # Skip if JVM already running
    if jpype.isJVMStarted():
        pytest.skip("JVM already started by another test - JPype limitation prevents restart")
    
    num_envs = 10
    num_resets = 3
    
    # Phase 1: Record games
    test_seeds, recorded_games = record_games_via_cli(num_envs)
    
    # Prepare actions for replay
    cli_actions_list = [g['actions'] for g in recorded_games]
    max_turns, gym_actions = convert_cli_actions_to_gym(cli_actions_list)
    
    print(f"Phase 2: Testing bulk reset consistency ({num_resets} resets)...")
    print("-" * 80)
    
    # Create gym
    env = gym.make('backtrack', num_envs=num_envs)
    
    all_run_scores = []
    all_run_steps = []
    
    for reset_num in range(num_resets):
        print(f"  Reset #{reset_num + 1}...")
        
        # Reset all envs
        env.reset(league_level=LEAGUE_LEVEL, random_seeds=test_seeds)
        
        # Replay actions and track steps per env
        env_steps = [0] * num_envs
        env_finished = [False] * num_envs
        for turn_actions in gym_actions:
            dones = env.step(turn_actions)
            for i in range(num_envs):
                if not env_finished[i]:
                    env_steps[i] += 1
                    if dones[i]:
                        env_finished[i] = True
            if all(dones):
                break
                
        scores = [[int(s) for s in env_score] for env_score in env.get_scores()]
        all_run_scores.append(scores)
        all_run_steps.append(env_steps)
        
    # Verify consistency
    print("\nChecking consistency...")
    baseline = all_run_scores[0]
    baseline_steps = all_run_steps[0]
    all_consistent = True
    
    # First check: gym scores and steps match CLI
    print("Comparing gym vs CLI scores and steps...")
    for env_idx in range(num_envs):
        gym_score = baseline[env_idx]
        cli_score = recorded_games[env_idx]['scores']
        gym_steps = baseline_steps[env_idx]
        cli_steps = recorded_games[env_idx]['steps']
        
        score_match = gym_score[0] == cli_score[0] and gym_score[1] == cli_score[1]
        steps_match = gym_steps == cli_steps
        
        if not score_match or not steps_match:
            all_consistent = False
            print(f"❌ Env {env_idx} mismatch: gym={gym_score} in {gym_steps} steps, CLI={cli_score} in {cli_steps} steps")
        else:
            print(f"✓ Env {env_idx} matches CLI: {gym_score} in {gym_steps} steps")
    
    # Second check: all gym resets produce same scores
    print("\nComparing gym resets...")
    for env_idx in range(num_envs):
        env_consistent = True
        baseline_score = baseline[env_idx]
        
        for reset_num in range(1, num_resets):
            current_score = all_run_scores[reset_num][env_idx]
            if current_score[0] != baseline_score[0] or current_score[1] != baseline_score[1]:
                env_consistent = False
                all_consistent = False
                print(f"❌ Env {env_idx} mismatch: Run 1={baseline_score}, Run {reset_num+1}={current_score}")
        
        if env_consistent:
            print(f"✓ Env {env_idx} consistent across {num_resets} resets: {baseline_score}")
            
    if not all_consistent:
        pytest.fail("Bulk reset consistency check failed")
        
    print("\n✅ Bulk reset consistency verified!")
    env.shutdown()


@pytest.mark.integration
def test_selective_reset_single_env_error():
    """Test that selective reset raises error for single environment mode."""
    
    # Skip if JVM already running
    if jpype.isJVMStarted():
        pytest.skip("JVM already started by another test - JPype limitation prevents restart")
    
    env = gym.make('backtrack', num_envs=1)
    env.reset(league_level=LEAGUE_LEVEL, random_seeds=12345)
    
    # Should raise error when trying to use selective reset with single env
    with pytest.raises(Exception):  # Could be GymError or ValueError depending on implementation
        env.reset_subset(indices=[0], league_level=LEAGUE_LEVEL, random_seeds=[100])
    
    env.shutdown()


@pytest.mark.integration
def test_selective_reset_consistency():
    """Test that selective reset (reset_subset) produces consistent scores."""
    
    # Skip if JVM already running
    if jpype.isJVMStarted():
        pytest.skip("JVM already started by another test - JPype limitation prevents restart")
        
    num_envs = 10
    num_resets = 3
    
    # Phase 1: Record games
    test_seeds, recorded_games = record_games_via_cli(num_envs)
    
    # Prepare actions
    cli_actions_list = [g['actions'] for g in recorded_games]
    max_turns, gym_actions = convert_cli_actions_to_gym(cli_actions_list)
    cli_steps = [g['steps'] for g in recorded_games]

    print(f"Phase 3: Testing selective reset consistency ({num_resets} completions per env)...")
    print("-" * 80)
    
    env = gym.make('backtrack', num_envs=num_envs)
    
    # Initialize tracking
    env_completions = [0] * num_envs  # How many times each env has completed
    env_scores = {i: [] for i in range(num_envs)}  # Scores for each completion
    env_steps = {i: [] for i in range(num_envs)}  # Steps for each completion
    env_turns = [0] * num_envs  # Per-environment turn counters (also tracks steps)
    
    # Reset all envs to start
    print("  Starting all environments...")
    env.reset(league_level=LEAGUE_LEVEL, random_seeds=test_seeds)
    
    # Main stepping loop
    total_steps = 0
    max_total_steps = max_turns * num_resets * 2  # Safety limit
    
    while any(count < num_resets for count in env_completions) and total_steps < max_total_steps:
        # Build actions for each env from its own turn counter
        actions = []
        for i in range(num_envs):
            turn = env_turns[i]
            if turn < recorded_games[i]['steps']:
                actions.append(gym_actions[turn][i])
            else:
                raise Exception(f"Env {i} exceeded its recorded {recorded_games[i]['steps']} steps at turn {turn}")
        
        # Step all environments
        dones = env.step(actions)
        
        # Update turn counters and check for completions
        for i in range(num_envs):
            env_turns[i] += 1
            
            if dones[i]:
                # Capture score and steps before reset
                scores = [[int(s) for s in env_score] for env_score in env.get_scores()]
                env_scores[i].append(scores[i])
                env_steps[i].append(env_turns[i])  # env_turns tracks the step count
                env_completions[i] += 1
                
                # Reset this env if it needs more completions
                # if env_completions[i] < num_resets:
                env.reset_subset(indices=[i], league_level=LEAGUE_LEVEL, random_seeds=[test_seeds[i]])
                env_turns[i] = 0  # Reset turn counter (which also resets step count)
        
        total_steps += 1
            
        # Progress reporting
        if total_steps % 100 == 0:
            completed = sum(1 for c in env_completions if c >= num_resets)
            print(f"  Step {total_steps}: {completed}/{num_envs} envs completed {num_resets} times")
    
    if total_steps >= max_total_steps:
        print(f"⚠ Warning: Hit safety limit of {max_total_steps} steps")
    
    print(f"\n  Completed in {total_steps} total steps")
    print(f"  Env completions: {env_completions}")
    
    # Verify consistency
    print("\nChecking consistency...")
    all_consistent = True
    
    # First check: first gym completion matches CLI scores and steps
    print("Comparing first gym completion vs CLI scores and steps...")
    for env_idx in range(num_envs):
        scores = env_scores[env_idx]
        steps = env_steps[env_idx]
        if len(scores) == 0:
            print(f"⚠ Env {env_idx}: No completions recorded")
            all_consistent = False
            continue
        
        first_gym_score = scores[0]
        first_gym_steps = steps[0]
        cli_score = recorded_games[env_idx]['scores']
        cli_steps = recorded_games[env_idx]['steps']
        
        score_match = first_gym_score[0] == cli_score[0] and first_gym_score[1] == cli_score[1]
        steps_match = first_gym_steps == cli_steps
        
        if not score_match or not steps_match:
            all_consistent = False
            print(f"❌ Env {env_idx} mismatch: gym={first_gym_score} in {first_gym_steps} steps, CLI={cli_score} in {cli_steps} steps")
        else:
            print(f"✓ Env {env_idx} matches CLI: {first_gym_score} in {first_gym_steps} steps")
    
    # Second check: all completions of each env produce same scores
    print("\nComparing multiple completions per env...")
    for env_idx in range(num_envs):
        scores = env_scores[env_idx]
        if len(scores) == 0:
            continue
            
        baseline = scores[0]
        env_consistent = True
        
        for i, score in enumerate(scores[1:], 1):
            if score[0] != baseline[0] or score[1] != baseline[1]:
                env_consistent = False
                all_consistent = False
                print(f"❌ Env {env_idx} mismatch: Run 1={baseline}, Run {i+1}={score}")
        
        if env_consistent:
            print(f"✓ Env {env_idx} consistent across {len(scores)} completions: {baseline}")

    if not all_consistent:
        pytest.fail("Selective reset consistency check failed")
        
    print("\n✅ Selective reset consistency verified!")
    env.shutdown()


if __name__ == "__main__":
    # Run both tests sequentially
    try:
        test_bulk_reset_consistency()
        print("\n" + "="*80)
        print("Starting next test...")
        print("="*80 + "\n")
        # Note: JVM restart issue might prevent running both in same process
        # But let's try, or user can run separately
        print("⚠ NOTE: Running both tests in same process might fail due to JVM restart limitation.")
        print("If second test fails with JVM error, run them separately.")
        test_selective_reset_consistency()
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
