#!/usr/bin/env python3
"""Run exactly one game and print the score dict as JSON.

This is used by evaluators in "safe" mode to enforce a per-game timeout
by running each game in a separate Python process.
"""

from __future__ import annotations

import argparse
import json

from codingame_gym.game_runner import GameRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="backtrack")
    ap.add_argument("--league", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--agent1", type=str, required=True)
    ap.add_argument("--agent2", type=str, required=True)
    args = ap.parse_args()

    runner = GameRunner(args.env)
    scores = runner.run_games(
        num_games=1,
        league_level=args.league,
        agent_1_cmd=args.agent1,
        agent_2_cmd=args.agent2,
        seeds=[args.seed],
        debug=False,
    )
    if not scores:
        raise SystemExit("No scores returned")

    print(json.dumps(scores[0]))


if __name__ == "__main__":
    main()
