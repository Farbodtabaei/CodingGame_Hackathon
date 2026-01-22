#!/usr/bin/env python3
"""Run a batch of games and print scores as JSON.

Why: Starting the JVM is expensive. For tuning loops, running many games in one
process is much faster than calling `run_one_game.py` repeatedly.

Output: JSON list of score dicts (one per game), same schema as `run_one_game.py`.

Example:
    python ./run_batch_games.py --league 5 --agent1 "python strategic_bot_v27.py" \
        --agent2 "python strategic_bot_v25.py" --seeds 16000,16001,16002
"""

from __future__ import annotations

import argparse
import json
from typing import List

from codingame_gym.game_runner import GameRunner


def _parse_seeds_csv(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_seeds_file(path: str) -> List[int]:
    seeds: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(int(line))
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="backtrack")
    ap.add_argument("--league", type=int, required=True)
    ap.add_argument("--agent1", type=str, required=True)
    ap.add_argument("--agent2", type=str, required=True)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seeds", type=str, default="", help="Comma-separated seeds")
    g.add_argument("--seeds-file", type=str, default="", help="Text file with one seed per line")

    ap.add_argument("--max-games", type=int, default=0, help="If set, only run first N seeds")
    args = ap.parse_args()

    seeds = _parse_seeds_csv(args.seeds) if args.seeds else _parse_seeds_file(args.seeds_file)
    if int(args.max_games) > 0:
        seeds = seeds[: int(args.max_games)]

    if not seeds:
        raise SystemExit("No seeds provided")

    runner = GameRunner(args.env)
    scores = runner.run_games(
        num_games=len(seeds),
        league_level=int(args.league),
        agent_1_cmd=str(args.agent1),
        agent_2_cmd=str(args.agent2),
        seeds=seeds,
        debug=False,
    )
    if scores is None:
        raise SystemExit("No scores returned")

    print(json.dumps(scores))


if __name__ == "__main__":
    main()
