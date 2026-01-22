"""\
STRATEGIC BOT V22 (Tunable)
===========================

This bot is a thin, parameter-driven wrapper around `strategic_bot_v21.py`.

Goal: make iterative coefficient tuning easy by accepting a JSON params file and
overriding all major multipliers/thresholds without editing code.

Usage:
  python strategic_bot_v22.py
  python strategic_bot_v22.py --params '{"OPP_TRACK_TRAVERSAL_PENALTY": 0.4}'
  python strategic_bot_v22.py --params-file best_params.json

Notes:
- Default behavior matches V21.
- Designed for local tuning/evaluation harnesses; for CodinGame submission you can
  copy the resulting tuned constants back into V21 (or inline V21 into V22).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from strategic_bot_v21 import StrategicBotV21


def _load_params(params_json: Optional[str], params_file: Optional[str]) -> Dict[str, Any]:
    if params_json and params_file:
        raise SystemExit("Use only one of --params or --params-file")

    if params_file:
        p = Path(params_file)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit("--params-file must contain a JSON object")
        return data

    if params_json:
        data = json.loads(params_json)
        if not isinstance(data, dict):
            raise SystemExit("--params must be a JSON object")
        return data

    return {}


class StrategicBotV22(StrategicBotV21):
    """V21 + parameter overrides."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        # Pathing cost model
        "OPP_TRACK_TRAVERSAL_PENALTY": StrategicBotV21.OPP_TRACK_TRAVERSAL_PENALTY,
        "NEUTRAL_TRACK_TRAVERSAL_PENALTY": StrategicBotV21.NEUTRAL_TRACK_TRAVERSAL_PENALTY,
        "SELF_HARM_ALPHA": StrategicBotV21.SELF_HARM_ALPHA,

        # Prediction knobs
        "PRED_TOP_K": StrategicBotV21.PRED_TOP_K,
        "PRED_WEIGHT_SCALE": StrategicBotV21.PRED_WEIGHT_SCALE,
        "MY_TRACK_BLOCKING_PENALTY": StrategicBotV21.MY_TRACK_BLOCKING_PENALTY,
        "NEUTRAL_TRACK_PENALTY": StrategicBotV21.NEUTRAL_TRACK_PENALTY,
        "PRED_OVERRIDE_THRESHOLD": StrategicBotV21.PRED_OVERRIDE_THRESHOLD,
        "PRED_COMPARE_MARGIN": StrategicBotV21.PRED_COMPARE_MARGIN,
        "PRED_OVERRIDE_BEHIND_FOR_2HIT": StrategicBotV21.PRED_OVERRIDE_BEHIND_FOR_2HIT,
        "PRED_DISABLE_OVERRIDE_WHEN_AHEAD": StrategicBotV21.PRED_DISABLE_OVERRIDE_WHEN_AHEAD,
        "PRED_EXTRA_MARGIN_IF_BASE_1HIT": StrategicBotV21.PRED_EXTRA_MARGIN_IF_BASE_1HIT,
        "PRED_ALLOW_OVERRIDE_IF_BASE_1HIT_BEHIND": StrategicBotV21.PRED_ALLOW_OVERRIDE_IF_BASE_1HIT_BEHIND,
        "OPP_TOWN_NEAR_DIST": StrategicBotV21.OPP_TOWN_NEAR_DIST,

        # Action switching
        "SWITCH_THRESHOLD": 1.5,

        # EV / scoring weights (from V21 `calc_connection_ev`)
        "SPEED_COST_T1": 10,
        "SPEED_COST_T2": 20,
        "SPEED_MULT_T1": 1.5,
        "SPEED_MULT_T2": 1.2,
        "SPEED_MULT_T3": 0.9,
        "EFFICIENCY_SCALE": 100.0,
        "TOWN_ID_BASE_BONUS_MULT": 50.0,
        "CONSOLIDATION_BONUS_PER_REGION": 25.0,
        "OVERLAP_BONUS_PER_MY_CELL": 4.0,
        "JUNCTION_BONUS_PER_HIT": 2.0,
        "HUB_BONUS": 35.0,
        "GIFT_PENALTY_OPP_CELL": 7.0,
        "GIFT_PENALTY_NEUTRAL_CELL": 2.0,

        # Bias response (from `detect_opponent_bias`)
        "BIAS_MULT_HIGH": -5.0,
        "BIAS_MULT_LOW": 6.0,
        "BIAS_MULT_NEUTRAL": -3.0,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.params: Dict[str, Any] = dict(self.DEFAULT_PARAMS)
        if params:
            for k, v in params.items():
                self.params[str(k)] = v

        self._apply_params()

    def _apply_params(self) -> None:
        # Override attributes used throughout V21 via `self.<NAME>` lookups.
        for k, v in self.params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def detect_opponent_bias(self):
        # Copy/paste of V21 logic but with tunable multipliers.
        if len(self.opp_score_history) < 6 or len(self.my_score_history) < 6:
            return

        opp_rate = (self.opp_score_history[-1] - self.opp_score_history[-6]) / 5
        my_rate = (self.my_score_history[-1] - self.my_score_history[-6]) / 5

        if opp_rate <= 0:
            return

        avg_opp_town_id = 0.0
        avg_all_town_id = 0.0

        if self.towns:
            avg_all_town_id = sum(t.id for t in self.towns) / len(self.towns)

        opp_town_ids = []
        for (x, y), owner in self.cell_tracks.items():
            if owner != self.opp_id:
                continue
            if (x, y) in self.town_cells:
                continue
            # closest town by Manhattan
            best = None
            best_d = 10**9
            for t in self.towns:
                d = abs(t.x - x) + abs(t.y - y)
                if d < best_d:
                    best_d = d
                    best = t
            if best is not None:
                opp_town_ids.append(best.id)

        if opp_town_ids:
            avg_opp_town_id = sum(opp_town_ids) / len(opp_town_ids)
        else:
            avg_opp_town_id = avg_all_town_id

        if avg_all_town_id <= 0:
            return

        if avg_opp_town_id > avg_all_town_id * 1.15:
            if self.detected_opp_bias != "high":
                self.detected_opp_bias = "high"
                self.my_bias_multiplier = float(self.params["BIAS_MULT_HIGH"])
        elif avg_opp_town_id < avg_all_town_id * 0.85:
            if self.detected_opp_bias != "low":
                self.detected_opp_bias = "low"
                self.my_bias_multiplier = float(self.params["BIAS_MULT_LOW"])
        else:
            if self.detected_opp_bias != "neutral":
                self.detected_opp_bias = "neutral"
                self.my_bias_multiplier = float(self.params["BIAS_MULT_NEUTRAL"])

    def calc_connection_ev(self, path, from_id: int = 0, to_id: int = 0) -> float:
        if not path:
            return 0.0

        cost = self.calc_path_cost(path)
        if cost <= 0 or cost >= 10**8:
            return 0.0

        turns_to_complete = (cost + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN
        turns_left = self.TOTAL_TURNS - self.turn
        scoring_turns = max(0, turns_left - turns_to_complete)
        if scoring_turns <= 0:
            return 0.0

        my_cells, opp_cells, neutral_cells, empty_cells = self._count_track_owners_on_path(path)
        expected_points = empty_cells * scoring_turns

        t1 = float(self.params["SPEED_COST_T1"])
        t2 = float(self.params["SPEED_COST_T2"])
        if cost <= t1:
            speed_multiplier = float(self.params["SPEED_MULT_T1"])
        elif cost <= t2:
            speed_multiplier = float(self.params["SPEED_MULT_T2"])
        else:
            speed_multiplier = float(self.params["SPEED_MULT_T3"])

        weighted_points = expected_points * speed_multiplier
        efficiency = weighted_points / max(cost, 1)

        town_id_bonus = 0.0
        if from_id > 0 and to_id > 0:
            base_bonus = (from_id + to_id) * float(self.params["TOWN_ID_BASE_BONUS_MULT"])
            town_id_bonus = base_bonus * float(self.my_bias_multiplier)

        regions_in_path = set(self.region_ids[y][x] for x, y in path if self.region_ids[y][x] != -1)
        shared_regions = regions_in_path & self.my_track_regions
        consolidation_bonus = float(len(shared_regions)) * float(self.params["CONSOLIDATION_BONUS_PER_REGION"])

        overlap_bonus = float(my_cells) * float(self.params["OVERLAP_BONUS_PER_MY_CELL"])

        junction_hits = sum(1 for x, y in path if (x, y) in self.town_adjacent_cells)
        junction_bonus = float(junction_hits) * float(self.params["JUNCTION_BONUS_PER_HIT"])

        hub_bonus = 0.0
        if self.hub_town_id is not None and (from_id == self.hub_town_id or to_id == self.hub_town_id):
            hub_bonus = float(self.params["HUB_BONUS"])

        gifting_penalty = (
            (float(opp_cells) * float(self.params["GIFT_PENALTY_OPP_CELL"]) + float(neutral_cells) * float(self.params["GIFT_PENALTY_NEUTRAL_CELL"]))
            * speed_multiplier
        )

        efficiency_scale = float(self.params["EFFICIENCY_SCALE"])
        return weighted_points + efficiency * efficiency_scale + town_id_bonus + consolidation_bonus + overlap_bonus + junction_bonus + hub_bonus - gifting_penalty

    def get_action(self) -> str:
        # Same as V21, but with tunable switch threshold.
        actions = []

        continue_ev = self.calc_continue_ev()
        new_candidates = self.get_connection_candidates()
        best_new = new_candidates[0] if new_candidates else None
        best_new_ev = best_new[3] if best_new else 0.0
        best_new_path = best_new[2] if best_new else None

        switch_threshold = float(self.params["SWITCH_THRESHOLD"])

        if self.active_build and self.active_build_path:
            if best_new and best_new_ev > continue_ev * switch_threshold:
                from_id, to_id, path, _ = best_new
                self.active_build = (from_id, to_id)
                self.active_build_path = path

            from_id, to_id = self.active_build
            from_town = self.town_by_id[from_id]
            to_town = self.town_by_id[to_id]
            refreshed = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y)
            if refreshed:
                self.active_build_path = refreshed

            actions.extend(self.build_path_two_ended_stable(from_town, to_town))

            if not actions and best_new:
                from_id, to_id, path, _ = best_new
                self.active_build = (from_id, to_id)
                self.active_build_path = path
                from_town = self.town_by_id[from_id]
                to_town = self.town_by_id[to_id]
                actions.extend(self.build_path_two_ended_stable(from_town, to_town))

        elif best_new:
            from_id, to_id, path, _ = best_new
            self.active_build = (from_id, to_id)
            self.active_build_path = path
            from_town = self.town_by_id[from_id]
            to_town = self.town_by_id[to_id]
            actions.extend(self.build_path_two_ended_stable(from_town, to_town))

        if self.turn >= 1:
            target = self.get_disruption_target(best_next_path=best_new_path)
            if target is not None:
                actions.append(f"DISRUPT {target}")

        return ";".join(actions) if actions else "WAIT"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--params", type=str, default=None)
    ap.add_argument("--params-file", type=str, default=None)
    ap.add_argument("--dump-default-params", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    ns, _ = ap.parse_known_args(argv)

    if ns.help:
        print(__doc__.strip())
        print()
        print("CLI:")
        print("  --params JSON          Inline JSON object")
        print("  --params-file PATH     JSON file containing object")
        print("  --dump-default-params  Print DEFAULT_PARAMS as JSON")
        return 0

    if ns.dump_default_params:
        print(json.dumps(StrategicBotV22.DEFAULT_PARAMS, indent=2, sort_keys=True))
        return 0

    params = _load_params(ns.params, ns.params_file)

    bot = StrategicBotV22(params=params)
    bot.read_init()
    while True:
        bot.read_turn()
        print(bot.get_action())
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
