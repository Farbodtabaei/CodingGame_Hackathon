"""
Multi-mode consistency test.

Tests that CLI, Python Gym, GameRunner, and CLI concurrent modes all produce
identical scores for the same games.

Run as pytest (10 games):
    pytest tests/test_consistency_multi_mode.py

Run standalone with custom game count:
    python tests/test_consistency_multi_mode.py [num_games]
"""

import pytest
import json
import subprocess
import random
import time
import sys
from pathlib import Path
import codingame_gym.cggym as gym
from codingame_gym.game_runner import GameRunner


# Configuration
LEAGUE_LEVEL = 5
CELLULARENA_DIR = Path("/Users/fzinno/dev/git/hackathon25/cellularena")
BOT_PATH = CELLULARENA_DIR / "bots" / "julien.py"
JAR_PATH = CELLULARENA_DIR / "target" / "backtrack.jar"


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
        cwd=str(CELLULARENA_DIR),
        capture_output=True,
        text=True
    )
    
    lines = [line for line in result.stdout.split('\n') if not line.startswith('WARNING')]
    json_output = '\n'.join(lines)
    
    try:
        data = json.loads(json_output)
        game_result = data['results'][0]
        return {
            'scores': [game_result['scores']['0'], game_result['scores']['1']],
            'actions': game_result['outputs']
        }
    except Exception as e:
        return {'error': str(e)}


def replay_python_gym(seed, cli_actions):
    """Replay actions through Python Gym."""
    player0_outputs = cli_actions['0']
    player1_outputs = cli_actions['1']
    num_turns = min(len(player0_outputs), len(player1_outputs))
    
    actions = []
    for turn in range(num_turns):
        p0_lines = [line for line in player0_outputs[turn].split('\n') if line.strip()]
        p1_lines = [line for line in player1_outputs[turn].split('\n') if line.strip()]
        actions.append([
            p0_lines if p0_lines else ['WAIT'],
            p1_lines if p1_lines else ['WAIT']
        ])
    
    env = gym.make('backtrack')
    env.reset(league_level=LEAGUE_LEVEL, random_seeds=seed)
    
    for turn_actions in actions:
        done = env.step(turn_actions)
        if done:
            break
    
    scores = env.get_scores()
    return [scores[0], scores[1]]


def run_gamerunner_batch(seeds_batch):
    """Run games via GameRunner."""
    runner = GameRunner('backtrack')
    results = runner.run_games(
        num_games=len(seeds_batch),
        league_level=LEAGUE_LEVEL,
        agent_1_cmd=f"python3 {BOT_PATH}",
        agent_2_cmd=f"python3 {BOT_PATH}",
        seeds=seeds_batch,
        debug=False
    )
    return [[r['0'], r['1']] for r in results]


def run_cli_concurrent_batch(seeds_batch):
    """Test CLI in concurrent mode."""
    cmd = [
        "java", "-jar", str(JAR_PATH),
        "-l", str(LEAGUE_LEVEL),
        "-n", str(len(seeds_batch)),
        "-t", "-c",
        "-p", f"python3 {BOT_PATH}", f"python3 {BOT_PATH}"
    ]
    
    for seed in seeds_batch:
        cmd.extend(["-s", str(seed)])
    
    result = subprocess.run(cmd, cwd=str(CELLULARENA_DIR), capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"CLI concurrent failed: {result.stderr[:200]}")
    
    lines = [line for line in result.stdout.split('\n') if not line.startswith('WARNING')]
    json_output = '\n'.join(lines)
    
    if not json_output.strip():
        raise Exception("CLI concurrent produced no output")
    
    try:
        data = json.loads(json_output)
        return [[game['scores']['0'], game['scores']['1']] for game in data['results']]
    except Exception as e:
        raise Exception(f"Failed to parse CLI concurrent results: {e}")


def run_consistency_test(num_games=10):
    """
    Run the full multi-mode consistency test.
    
    Returns: dict with test results and statistics
    """
    # Generate test seeds
    random.seed(42)
    test_seeds = [random.randint(0, 999999) for _ in range(num_games)]
    
    print(f"{'='*80}")
    print(f"Multi-Mode Consistency Test Suite")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Games: {num_games}")
    print(f"  League level: {LEAGUE_LEVEL}")
    print(f"  Bot: julien.py vs julien.py")
    print(f"  Seeds: {test_seeds[:5]} ... {test_seeds[-5:]}")
    print(f"{'='*80}\n")
    
    # Phase 1: CLI
    print("Phase 1: Recording games via CLI...")
    print("-" * 80)
    cli_results = []
    cli_start = time.time()
    
    for i, seed in enumerate(test_seeds):
        if (i + 1) % 10 == 0:
            print(f"  Recording game {i+1}/{num_games}...")
        
        result = run_cli_game(seed)
        cli_results.append({'seed': seed, 'cli_result': result})
    
    cli_time = time.time() - cli_start
    print(f"✓ CLI games completed in {cli_time:.1f}s\n")
    
    cli_errors = sum(1 for r in cli_results if 'error' in r['cli_result'])
    if cli_errors > 0:
        print(f"⚠ {cli_errors} CLI games had errors\n")
    
    # Phase 2: Python Gym replay
    print("Phase 2: Replaying games through Python Gym...")
    print("-" * 80)
    python_start = time.time()
    mismatches = []
    python_errors = []
    
    for i, test in enumerate(cli_results):
        if (i + 1) % 10 == 0:
            print(f"  Replaying game {i+1}/{num_games}...")
        
        if 'error' in test['cli_result']:
            continue
        
        seed = test['seed']
        cli_scores = test['cli_result']['scores']
        cli_actions = test['cli_result']['actions']
        
        try:
            python_scores = replay_python_gym(seed, cli_actions)
            if python_scores[0] != cli_scores[0] or python_scores[1] != cli_scores[1]:
                mismatches.append({
                    'seed': seed,
                    'cli_scores': cli_scores,
                    'python_scores': python_scores
                })
        except Exception as e:
            python_errors.append({'seed': seed, 'error': str(e)})
    
    python_time = time.time() - python_start
    print(f"✓ Python Gym replay completed in {python_time:.1f}s\n")
    
    # Phase 3: GameRunner
    print("Phase 3: Testing via GameRunner (subprocess mode)...")
    print("-" * 80)
    gamerunner_start = time.time()
    gamerunner_mismatches = []
    gamerunner_errors = []
    
    for batch_start in range(0, len(test_seeds), 10):
        batch_end = min(batch_start + 10, len(test_seeds))
        batch_seeds = test_seeds[batch_start:batch_end]
        
        if batch_end % 10 == 0 or batch_end == len(test_seeds):
            print(f"  Testing games {batch_start+1}-{batch_end}/{num_games}...")
        
        try:
            gamerunner_scores = run_gamerunner_batch(batch_seeds)
            
            for i, seed in enumerate(batch_seeds):
                cli_idx = test_seeds.index(seed)
                if 'error' in cli_results[cli_idx]['cli_result']:
                    continue
                
                cli_scores = cli_results[cli_idx]['cli_result']['scores']
                gr_scores = gamerunner_scores[i]
                
                if gr_scores[0] != cli_scores[0] or gr_scores[1] != cli_scores[1]:
                    gamerunner_mismatches.append({
                        'seed': seed,
                        'cli_scores': cli_scores,
                        'gamerunner_scores': gr_scores
                    })
        except Exception as e:
            for seed in batch_seeds:
                gamerunner_errors.append({'seed': seed, 'error': str(e)})
    
    gamerunner_time = time.time() - gamerunner_start
    print(f"✓ GameRunner testing completed in {gamerunner_time:.1f}s\n")
    
    # Phase 4: CLI Concurrent
    print("Phase 4: Testing CLI in concurrent mode...")
    print("-" * 80)
    cli_concurrent_start = time.time()
    cli_concurrent_mismatches = []
    cli_concurrent_errors = []
    
    for batch_start in range(0, len(test_seeds), 10):
        batch_end = min(batch_start + 10, len(test_seeds))
        batch_seeds = test_seeds[batch_start:batch_end]
        
        if batch_end % 10 == 0 or batch_end == len(test_seeds):
            print(f"  Testing games {batch_start+1}-{batch_end}/{num_games}...")
        
        try:
            cli_concurrent_scores = run_cli_concurrent_batch(batch_seeds)
            
            for i, seed in enumerate(batch_seeds):
                cli_idx = test_seeds.index(seed)
                if 'error' in cli_results[cli_idx]['cli_result']:
                    continue
                
                cli_scores = cli_results[cli_idx]['cli_result']['scores']
                concurrent_scores = cli_concurrent_scores[i]
                
                if concurrent_scores[0] != cli_scores[0] or concurrent_scores[1] != cli_scores[1]:
                    cli_concurrent_mismatches.append({
                        'seed': seed,
                        'cli_sequential': cli_scores,
                        'cli_concurrent': concurrent_scores
                    })
        except Exception as e:
            for seed in batch_seeds:
                cli_concurrent_errors.append({'seed': seed, 'error': str(e)})
    
    cli_concurrent_time = time.time() - cli_concurrent_start
    print(f"✓ CLI concurrent testing completed in {cli_concurrent_time:.1f}s\n")
    
    # Generate report
    print("\n" + "="*80)
    print("CONSISTENCY TEST REPORT")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Total games: {num_games}")
    print(f"  League level: {LEAGUE_LEVEL}")
    print(f"  Bot: julien.py vs julien.py")
    print(f"\nExecution Times:")
    print(f"  CLI sequential: {cli_time:.1f}s ({cli_time/num_games:.3f}s per game)")
    print(f"  Python replay: {python_time:.1f}s ({python_time/num_games:.3f}s per game)")
    print(f"  GameRunner: {gamerunner_time:.1f}s ({gamerunner_time/num_games:.3f}s per game)")
    print(f"  CLI concurrent: {cli_concurrent_time:.1f}s ({cli_concurrent_time/num_games:.3f}s per game)")
    print(f"\nResults Summary:")
    
    total_mismatches = len(mismatches) + len(gamerunner_mismatches) + len(cli_concurrent_mismatches)
    total_errors = cli_errors + len(python_errors) + len(gamerunner_errors) + len(cli_concurrent_errors)
    successful_games = num_games - total_errors
    
    print(f"  Successfully tested: {successful_games}/{num_games} games")
    print(f"  CLI sequential errors: {cli_errors}")
    print(f"  Python replay errors: {len(python_errors)}")
    print(f"  GameRunner errors: {len(gamerunner_errors)}")
    print(f"  CLI concurrent errors: {len(cli_concurrent_errors)}")
    print(f"  Python replay mismatches: {len(mismatches)}")
    print(f"  GameRunner mismatches: {len(gamerunner_mismatches)}")
    print(f"  CLI concurrent mismatches: {len(cli_concurrent_mismatches)}")
    
    if total_mismatches == 0 and total_errors == 0:
        print(f"\n✅ PERFECT CONSISTENCY")
        print(f"All {num_games} games produced identical scores across all modes:")
        print(f"  - CLI (sequential)")
        print(f"  - CLI (concurrent)")
        print(f"  - Python Gym (replay)")
        print(f"  - GameRunner (subprocess)")
    elif total_mismatches == 0 and successful_games > 0:
        print(f"\n✅ CONSISTENCY VERIFIED")
        print(f"All {successful_games} successfully tested games produced identical scores!")
        if total_errors > 0:
            print(f"Note: {total_errors} games had errors (not consistency issues)")
    else:
        print(f"\n❌ CONSISTENCY FAILURES DETECTED")
    
    # Sample scores
    print(f"\nScore Distribution (sample of 10 games):")
    print("-" * 80)
    sample_games = [r for r in cli_results if 'error' not in r['cli_result']][:10]
    for game in sample_games:
        scores = game['cli_result']['scores']
        print(f"  Seed {game['seed']}: Player0={scores[0]}, Player1={scores[1]}")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)
    
    return {
        'total_mismatches': total_mismatches,
        'total_errors': total_errors,
        'successful_games': successful_games,
        'num_games': num_games
    }


@pytest.mark.slow
@pytest.mark.integration
def test_multi_mode_consistency_quick():
    """
    Quick multi-mode consistency test (10 games).
    
    Tests that CLI, Python Gym, GameRunner, and CLI concurrent modes
    all produce identical scores.
    """
    result = run_consistency_test(num_games=10)
    
    assert result['total_errors'] == 0, \
        f"Test had {result['total_errors']} errors"
    assert result['total_mismatches'] == 0, \
        f"Test had {result['total_mismatches']} consistency failures"


if __name__ == "__main__":
    # Allow running standalone with custom game count
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_consistency_test(num_games=num_games)
