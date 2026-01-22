"""
Python Performance Test for CodingGame Gym

This test runs 100 games with the same seeds as the Java test to compare performance
and verify consistency of results.
"""

from codingame_gym import cggym as gym
import numpy as np
import time
import random
import sys


def main():
    NUM_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    WARMUP_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # Default 100 warmup steps
    # USE_SHARED_BUFFER parameter removed - always enabled now
    LEAGUE_LEVEL = 5
    MASTER_SEED = 42
    
    print(f"Python Performance Test: {NUM_GAMES} games at league level {LEAGUE_LEVEL}")
    print(f"Master seed: {MASTER_SEED}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"ByteBuffer shared memory: always enabled\n")
    
    # Read seeds from Java-generated file
    import os
    seeds_file = os.path.expanduser("~/dev/git/hackathon25/cellularena/seeds.txt")
    seeds = []
    if os.path.exists(seeds_file):
        try:
            with open(seeds_file, 'r') as f:
                seeds = [int(line.strip()) for line in f.readlines()]
            print(f"Loaded {len(seeds)} seeds from seeds.txt")
        except FileNotFoundError:
            pass
    
    if not seeds or len(seeds) < NUM_GAMES:
        # Generate seeds if file doesn't exist or doesn't have enough
        if seeds:
            print(f"Only {len(seeds)} seeds in seeds.txt, need {NUM_GAMES}")
        random.seed(MASTER_SEED)
        seeds = [random.randint(0, 999999) for _ in range(NUM_GAMES)]
        print(f"Generated {len(seeds)} seeds")
    else:
        # Only use the first NUM_GAMES seeds
        seeds = seeds[:NUM_GAMES]
        print(f"Using first {NUM_GAMES} seeds from seeds.txt")
    
    print("First 10 seeds:", seeds[:10])
    
    # Create multi-environment (ByteBuffer always enabled)
    env = gym.make('backtrack', num_envs=NUM_GAMES)
    
    # JVM Warmup Phase - run many steps to let JIT optimize hot paths
    warmup_time = 0.0
    if WARMUP_STEPS > 0:
        print(f"\n{'='*60}")
        print(f"JVM Warmup: Running {WARMUP_STEPS} steps...")
        print(f"{'='*60}")
        warmup_start = time.perf_counter()
        
        # Use the same env but reset with different seeds for warmup
        warmup_seeds = [MASTER_SEED + 1000000 + i for i in range(NUM_GAMES)]
        env.reset(random_seeds=warmup_seeds, league_level=LEAGUE_LEVEL)
        
        # Run warmup steps to heat up JIT - use the actual game actions
        for _ in range(WARMUP_STEPS):
            actions = [[['AUTOPLACE 0 0 29 19'], ['AUTOPLACE 5 5 25 15']] for _ in range(NUM_GAMES)]
            env.step(actions)
        
        warmup_time = time.perf_counter() - warmup_start
        warmup_total_steps = WARMUP_STEPS * NUM_GAMES
        print(f"Warmup complete: {warmup_total_steps} total steps in {warmup_time:.3f}s")
        print(f"Warmup performance: {warmup_total_steps / warmup_time:.1f} steps/sec")
        print(f"{'='*60}\n")
    
    # Start timing the actual test (after warmup)
    start_time = time.perf_counter()
    
    # Reset with actual test seeds - this is now timed
    before_reset = time.perf_counter()
    env.reset(random_seeds=seeds, league_level=LEAGUE_LEVEL)
    observations = env.observations  # Access via property if needed
    after_reset = time.perf_counter()
    
    print(f"Reset complete in {after_reset - before_reset:.3f}s")
    
    # Use fixed AUTOPLACE actions for all steps (identical to Java test - no parsing overhead)
    fixed_actions = [[['AUTOPLACE 0 0 29 19'], ['AUTOPLACE 5 5 25 15']] for _ in range(NUM_GAMES)]
    
    total_steps = 0
    
    before_steps = time.perf_counter()
    
    all_done = False
    while not all_done:
        dones = env.step(fixed_actions)
        observations = env.observations  # Access via property if needed

        total_steps += 1
        
        all_done = all(dones)
    
    after_steps = time.perf_counter()
    end_time = time.perf_counter()
    
    # Get final scores
    scores = env.get_scores()
    
    # Write results to file
    with open("python_results.txt", "w") as f:
        f.write(f"Seeds: {seeds}\n")
        f.write(f"League Level: {LEAGUE_LEVEL}\n")
        f.write(f"Total Games: {NUM_GAMES}\n")
        f.write(f"Total Steps: {total_steps}\n")
        f.write(f"\nResults:\n")
        
        for i in range(NUM_GAMES):
            f.write(f"Game {i}: Scores={scores[i]}\n")
        
        reset_time = after_reset - before_reset
        step_time = after_steps - before_steps
        total_time = end_time - start_time
        
        f.write(f"\nTiming:\n")
        f.write(f"Warmup time: {warmup_time:.3f}s\n")
        f.write(f"Reset time: {reset_time:.3f}s\n")
        f.write(f"Step time: {step_time:.3f}s\n")
        f.write(f"Total time: {total_time:.3f}s\n")
        actual_total_steps = total_steps * NUM_GAMES
        f.write(f"Steps per second: {actual_total_steps / step_time:.1f}\n")
    
    # Print summary
    reset_time = after_reset - before_reset
    step_time = after_steps - before_steps
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("Python Performance Results")
    print("="*60)
    if warmup_time > 0:
        print(f"Warmup time:       {warmup_time:.3f}s")
    print(f"Reset time:        {reset_time:.3f}s")
    print(f"Step time:         {step_time:.3f}s")
    print(f"Total time:        {total_time:.3f}s")
    actual_total_steps = total_steps * NUM_GAMES
    print(f"Total steps:       {actual_total_steps}")
    print(f"Steps per second:  {actual_total_steps / step_time:.1f}")
    print("="*60)
    
    # Show first 5 game results
    print("\nFirst 5 game results:")
    for i in range(min(5, NUM_GAMES)):
        print(f"Game {i}: scores={scores[i]}")
    
    env.shutdown()
    print("\nResults written to python_results.txt")


if __name__ == "__main__":
    main()
