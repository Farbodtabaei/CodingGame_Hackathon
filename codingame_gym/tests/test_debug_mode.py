"""
Debug mode tests for visualizing games with replays.

These tests demonstrate how to run games in debug mode for visual inspection.
They will start a local web server to view game replays.

Run with: pytest tests/test_debug_mode.py -v -s
"""

import pytest
import jpype
from pathlib import Path
import sys

# Import gym and game runner
sys.path.insert(0, str(Path(__file__).parent.parent))
import codingame_gym.cggym as gym
from codingame_gym.game_runner import GameRunner


@pytest.fixture
def cellularena_dir():
    """Path to the cellularena Java project."""
    return Path.home() / "dev/git/hackathon25/cellularena"


@pytest.mark.debug
@pytest.mark.manual
def test_debug_single_game_cli(cellularena_dir):
    """
    Test running a single game in debug mode using CLI approach.
    
    This spawns the Java MultiplayerGameRunner directly with agent commands.
    A web server will start on localhost:8888 to view the replay.
    
    NOTE: This test blocks waiting for Ctrl+C. The game runs and serves a replay.
    To run: pytest tests/test_debug_mode.py::test_debug_single_game_cli -v -s
    Then open http://localhost:8888 in your browser
    Press Ctrl+C when done viewing
    """
    # Skip if JVM already running
    if jpype.isJVMStarted():
        pytest.skip("JVM already started by another test - cannot run debug mode")
    
    print("\n" + "="*70)
    print("Debug Mode: Single Game with CLI")
    print("="*70)
    print("Starting game with visual replay server...")
    print("Open http://localhost:8888 in your browser to view the game")
    print("Press Ctrl+C to stop the server when done watching")
    print("="*70 + "\n")
    
    # Import Java classes
    jpype.startJVM(classpath=[str(cellularena_dir / "target" / "backtrack.jar")])
    
    MultiplayerGameRunner = jpype.JClass("com.codingame.gameengine.runner.MultiplayerGameRunner")
    Long = jpype.JClass("java.lang.Long")
    
    # Create game runner
    gameRunner = MultiplayerGameRunner()
    gameRunner.setLeagueLevel(3)
    gameRunner.setSeed(Long(1539742955359708000))
    
    # Add bots
    player1_bot = cellularena_dir / "bots" / "julien.py"
    player2_bot = cellularena_dir / "config" / "Boss.py"
    
    gameRunner.addAgent(f"python {player1_bot}", "Player 1")
    gameRunner.addAgent(f"python {player2_bot}", "Player 2 (Boss)")
    
    # Start with web server on port 8888
    try:
        gameRunner.start(8888)
        print("\n✓ Game completed. Check the replay in your browser.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        jpype.shutdownJVM()
@pytest.mark.debug
@pytest.mark.manual
def test_debug_game_runner():
    """
    Test running a game in debug mode using GameRunner class.
    
    This uses the Python GameRunner wrapper which provides a cleaner API.
    A web server will start on localhost to view the replay.
    
    NOTE: This test blocks waiting for user interaction with the replay server.
    To run: pytest tests/test_debug_mode.py::test_debug_game_runner -v -s
    Then check output for localhost URL to view the game
    """
    # Skip if JVM already running
    if jpype.isJVMStarted():
        pytest.skip("JVM already started by another test - cannot run debug mode")
    
    print("\n" + "="*70)
    print("Debug Mode: GameRunner Wrapper")
    print("="*70)
    print("Starting game with visual replay server...")
    print("Check output for localhost URL to view the game")
    print("="*70 + "\n")
    
    # Create GameRunner
    runner = GameRunner('backtrack')
    
    # Define bot paths
    cellularena_dir = Path.home() / "dev/git/hackathon25/cellularena"
    player1_bot = cellularena_dir / "bots" / "julien.py"
    player2_bot = cellularena_dir / "config" / "Boss.py"
    
    # Run single game in debug mode
    try:
        result = runner.run_games(
            num_games=1,
            league_level=3,
            agent_1_cmd=f"python {player1_bot}",
            agent_2_cmd=f"python {player2_bot}",
            seeds=[1539742955359708000],  # Python int will be converted to Java Long by GameRunner
            debug=True  # This enables visual replay server
        )
        
        if result is None:
            print("\n✓ Game completed in debug mode")
            print("Check the replay in your browser (URL shown above)")
            print("Press Ctrl+C when done viewing")
            # Keep server running
            import time
            while True:
                time.sleep(1)
        else:
            print(f"\n✓ Game completed with scores: {result[0]}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")


@pytest.mark.debug
def test_debug_multiple_seeds_comparison():
    """
    Test running multiple games with different seeds for comparison.
    
    This runs 5 games with different seeds and shows the results.
    No visual replay - just score comparison.
    
    To run: pytest tests/test_debug_mode.py::test_debug_multiple_seeds_comparison -v -s
    """
    # Skip if JVM already running
    if jpype.isJVMStarted():
        pytest.skip("JVM already started by another test - cannot run debug mode")
    
    print("\n" + "="*70)
    print("Debug Mode: Multiple Seeds Comparison")
    print("="*70)
    
    # Create GameRunner
    runner = GameRunner('backtrack')
    
    # Define bot paths
    cellularena_dir = Path.home() / "dev/git/hackathon25/cellularena"
    player1_bot = cellularena_dir / "bots" / "julien.py"
    player2_bot = cellularena_dir / "config" / "Boss.py"
    
    # Test with multiple seeds
    test_seeds = [12345, 67890, 11111, 22222, 33333]
    
    print(f"Running {len(test_seeds)} games with seeds: {test_seeds}")
    print()
    
    scores = runner.run_games(
        num_games=len(test_seeds),
        league_level=3,
        agent_1_cmd=f"python {player1_bot}",
        agent_2_cmd=f"python {player2_bot}",
        seeds=test_seeds,
        debug=False  # No visual replay for batch runs
    )
    
    # Analyze results
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    
    player1_wins = sum(1 for s in scores if s[0] > s[1])
    player2_wins = sum(1 for s in scores if s[1] > s[0])
    ties = sum(1 for s in scores if s[0] == s[1])
    
    print(f"\nPlayer 1 wins: {player1_wins}")
    print(f"Player 2 wins: {player2_wins}")
    print(f"Ties: {ties}")
    
    print("\nDetailed scores:")
    for i, (seed, score) in enumerate(zip(test_seeds, scores)):
        winner = "Player 1" if score[0] > score[1] else "Player 2" if score[1] > score[0] else "Tie"
        print(f"  Game {i+1} (seed={seed}): {score[0]:3d} - {score[1]:3d}  [{winner}]")
    
    print("="*70)


if __name__ == "__main__":
    # Allow running directly: python test_debug_mode.py
    pytest.main([__file__, '-v', '-s'])
