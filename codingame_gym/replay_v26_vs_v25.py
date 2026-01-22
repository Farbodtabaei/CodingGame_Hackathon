#!/usr/bin/env python3
"""Replay a single V26 vs V25 game in debug mode (visual replay server).

This is meant for manual inspection of *specific losing seeds*.

Examples (PowerShell):
    ./.venv/Scripts/python.exe ./replay_v26_vs_v25.py --seed 522919 --league 5 --v26-as p1
    ./.venv/Scripts/python.exe ./replay_v26_vs_v25.py --seed 30838 --league 5 --v26-as p0

Notes:
- Debug mode serves a replay server (typically http://localhost:8888).
- The process will keep running until you press Ctrl+C.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from codingame_gym.game_runner import GameRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--league", type=int, default=5)
    ap.add_argument(
        "--v26-as",
        choices=["p0", "p1"],
        default="p0",
        help="Whether V26 plays as player 0 (first) or player 1 (second).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    v26 = root / "strategic_bot_v26.py"
    v25 = root / "strategic_bot_v25.py"

    if args.v26_as == "p0":
        agent1 = f"python {v26.as_posix()}"
        agent2 = f"python {v25.as_posix()}"
        label = "V26 (P0) vs V25 (P1)"
    else:
        agent1 = f"python {v25.as_posix()}"
        agent2 = f"python {v26.as_posix()}"
        label = "V25 (P0) vs V26 (P1)"

    print("=" * 80)
    print("DEBUG REPLAY")
    print(label)
    print(f"seed={args.seed} league={args.league}")
    print("A replay server will start (usually http://localhost:8888).")
    print("Press Ctrl+C here when done watching.")
    print("=" * 80)

    runner = GameRunner("backtrack")

    # debug=True will only run one game and start replay server.
    runner.run_games(
        num_games=1,
        league_level=args.league,
        agent_1_cmd=agent1,
        agent_2_cmd=agent2,
        seeds=[args.seed],
        debug=True,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
