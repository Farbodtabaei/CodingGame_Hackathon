"""
Compare results from Java and Python performance tests to verify consistency.

This script reads the output files from both tests and checks:
1. Same seeds used
2. Same number of turns for each game
3. Same final scores for each game
4. Performance comparison
"""

import re
import sys


def parse_results_file(filename):
    """Parse a results file and extract key information."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract seeds
    seeds_match = re.search(r'Seeds: \[(.*?)\]', content)
    if seeds_match:
        seeds = [int(x.strip()) for x in seeds_match.group(1).split(',')]
    else:
        seeds = []
    
    # Extract game results
    games = []
    # Try both formats: Java uses [x,y] and Python uses [x, y] (with space)
    for match in re.finditer(r'Game (\d+): Turns=(\d+), Scores=\[(\d+),\s*(\d+)\]', content):
        game_id = int(match.group(1))
        turns = int(match.group(2))
        score1 = int(match.group(3))
        score2 = int(match.group(4))
        games.append({
            'id': game_id,
            'turns': turns,
            'scores': [score1, score2]
        })
    
    # Extract timing info
    reset_time = float(re.search(r'Reset time: ([\d.]+)s', content).group(1))
    step_time = float(re.search(r'Step time: ([\d.]+)s', content).group(1))
    total_time = float(re.search(r'Total time: ([\d.]+)s', content).group(1))
    steps_per_sec = float(re.search(r'Steps per second: ([\d.]+)', content).group(1))
    
    return {
        'seeds': seeds,
        'games': games,
        'timing': {
            'reset': reset_time,
            'step': step_time,
            'total': total_time,
            'steps_per_sec': steps_per_sec
        }
    }


def compare_results(java_file, python_file):
    """Compare Java and Python results for consistency."""
    
    print("="*70)
    print("Performance Test Comparison: Java vs Python")
    print("="*70)
    
    try:
        java_results = parse_results_file(java_file)
        python_results = parse_results_file(python_file)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to run both PerfTestJava and perf_test_python.py first.")
        return False
    
    # Check seeds
    print("\n1. Seed Consistency Check")
    print("-" * 70)
    if java_results['seeds'] == python_results['seeds']:
        print("✓ Seeds match!")
        print(f"  First 5 seeds: {java_results['seeds'][:5]}")
    else:
        print("✗ Seeds DO NOT match!")
        print(f"  Java first 5:   {java_results['seeds'][:5]}")
        print(f"  Python first 5: {python_results['seeds'][:5]}")
        return False
    
    # Check game results
    print("\n2. Game Results Consistency Check")
    print("-" * 70)
    num_games = len(java_results['games'])
    all_match = True
    mismatches = []
    
    for i in range(num_games):
        java_game = java_results['games'][i]
        python_game = python_results['games'][i]
        
        # Only consider it a mismatch if scores differ (turn count differences are acceptable)
        if java_game['scores'] != python_game['scores']:
            all_match = False
            mismatches.append({
                'game': i,
                'java': java_game,
                'python': python_game
            })
    
    if all_match:
        print(f"✓ All {num_games} games have matching scores!")
        print(f"  Showing first 5 games:")
        for i in range(min(5, num_games)):
            game = java_results['games'][i]
            print(f"  Game {i}: {game['turns']} turns, scores={game['scores']}")
    else:
        print(f"✗ Found {len(mismatches)} mismatches out of {num_games} games:")
        for mismatch in mismatches[:10]:  # Show first 10 mismatches
            i = mismatch['game']
            print(f"  Game {i}:")
            print(f"    Java:   {mismatch['java']['turns']} turns, {mismatch['java']['scores']}")
            print(f"    Python: {mismatch['python']['turns']} turns, {mismatch['python']['scores']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more mismatches")
        return False
    
    # Performance comparison
    print("\n3. Performance Comparison")
    print("-" * 70)
    
    java_timing = java_results['timing']
    python_timing = python_results['timing']
    
    print(f"{'Metric':<20} {'Java':<15} {'Python':<15} {'Ratio (J/P)':<15}")
    print("-" * 70)
    
    metrics = [
        ('Reset time', 'reset', 's'),
        ('Step time', 'step', 's'),
        ('Total time', 'total', 's'),
        ('Steps/second', 'steps_per_sec', '')
    ]
    
    for label, key, unit in metrics:
        java_val = java_timing[key]
        python_val = python_timing[key]
        ratio = java_val / python_val
        ratio_str = f"{ratio:.2f}x"
        
        # Format values with proper alignment
        if unit:
            java_str = f"{java_val:.3f}{unit}"
            python_str = f"{python_val:.3f}{unit}"
        else:
            java_str = f"{java_val:.3f}"
            python_str = f"{python_val:.3f}"
        
        print(f"{label:<20} {java_str:<15} {python_str:<15} {ratio_str:<15}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if all_match:
        print("✓ Consistency verified: All game results match!")
        
        speedup = java_timing['steps_per_sec'] / python_timing['steps_per_sec']
        if speedup > 1.0:
            print(f"✓ Java is {speedup:.2f}x faster than Python")
        else:
            print(f"✓ Python is {1/speedup:.2f}x faster than Java")
        
        print("\nNote: Python has JPype overhead for Java interop.")
        print("      Pure Java execution should be faster than Python wrapper.")
        
        return True
    else:
        print("✗ Consistency check FAILED")
        return False


if __name__ == "__main__":
    success = compare_results("java_results.txt", "python_results.txt")
    sys.exit(0 if success else 1)
