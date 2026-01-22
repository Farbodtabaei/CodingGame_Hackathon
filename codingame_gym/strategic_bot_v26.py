"""\
STRATEGIC BOT V25 (Standalone, Tuned + Smart Densify)
==============================

This file is intentionally self-contained for easy CodinGame submission.

Base: V21 standalone (V20 build/pathing + conservative predictive disruption).
Tuning: inlines the best hyperparameters found by `tune_v22.py` (run v22_20251213_085747).
Add-on: densifies the owned network more intelligently by spending leftover budget on
low-cost placements that also improve future connection options (town-adjacent / near
unfinished connection endpoints), while staying conservative about risk.
"""

import sys
import traceback
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Set
import heapq
import itertools
import os
import json


class Town:
    def __init__(self, town_id: int, x: int, y: int, desired_connections: List[int]):
        self.id = town_id
        self.x = x
        self.y = y
        self.desired = desired_connections


class StrategicBotV26:
    # --- Core rules ---
    POINTS_PER_TURN = 3
    INSTABILITY_TO_INK = 4
    TOTAL_TURNS = 100

    # --- Pathing cost model (copied from current V20) ---
    OPP_TRACK_TRAVERSAL_PENALTY = 0.35
    NEUTRAL_TRACK_TRAVERSAL_PENALTY = 0.20
    SELF_HARM_ALPHA = 1.15

    # --- V21 prediction knobs ---
    PRED_TOP_K = 7
    PRED_WEIGHT_SCALE = 0.35

    # Opponent route estimation penalties
    MY_TRACK_BLOCKING_PENALTY = 2.25
    NEUTRAL_TRACK_PENALTY = 0.45

    # Conservative override rules
    PRED_OVERRIDE_THRESHOLD = 4.5
    PRED_COMPARE_MARGIN = 1.0
    PRED_OVERRIDE_BEHIND_FOR_2HIT = 160
    PRED_DISABLE_OVERRIDE_WHEN_AHEAD = 150
    PRED_EXTRA_MARGIN_IF_BASE_1HIT = 2.75
    PRED_ALLOW_OVERRIDE_IF_BASE_1HIT_BEHIND = 85

    # Only predict routes involving towns near existing opponent tracks
    OPP_TOWN_NEAR_DIST = 7

    # --- V26: Stronger opponent-path prediction ---
    # Previous versions only predicted opponent connections when both towns were near
    # existing opponent tracks; strong bots can start connections far from current tracks,
    # which made opp_threat/choke under-react early. We now prefilter by cheap heuristic
    # distance, then run opponent pathfinding on the top-N candidates.
    OPP_PRED_PREFILTER_TOP_N = 24

    # --- V25: Anti-dense (opponent densify blocking) ---
    ANTI_DENSE_ENABLED = True
    ANTI_DENSE_ALPHA = 0.55
    ANTI_DENSE_MAX_CELL_COST = 3
    ANTI_DENSE_IGNORE_IF_AHEAD_BY = 120

    # Use leftover budget to deny opponent cheap densify cells.
    ANTI_DENSE_BLOCK_PLACEMENTS = 2
    ANTI_DENSE_BLOCK_MAX_INSTABILITY = 2
    ANTI_DENSE_POISON_REGION = True
    ANTI_DENSE_POISON_MAX_CELL_COST = 2
    ANTI_DENSE_POISON_MAX_DIST_TO_FRONTIER = 3

    # --- V25: Opponent-path choke placement ---
    CHOKE_ENABLED = True
    CHOKE_TOP_K_PATHS = 4
    CHOKE_MAX_PLACEMENTS = 1
    CHOKE_MAX_CELL_COST = 2
    CHOKE_MIN_DELTA_COST = 0.35
    CHOKE_DISABLE_WHEN_AHEAD = 140
    CHOKE_START_TURN = 2

    # --- Densify (V24 baseline) ---
    DENSE_MAX_EXTRA_PLACEMENTS = 2
    DENSE_MAX_INSTABILITY = 2
    DENSE_AVOID_OPP_REGIONS = True
    # --- V25: Targeted spend along best connection path ---
    SUPPORT_ENABLED = True
    SUPPORT_MAX_PLACEMENTS = 2
    SUPPORT_MAX_CELL_COST = 2
    SUPPORT_MAX_INSTABILITY = 2

    # --- V25: Targeted densify fallback (non-random) ---
    TARGETED_DENSE_ENABLED = True
    TARGETED_DENSE_MAX_PLACEMENTS = 1
    TARGETED_DENSE_MAX_CELL_COST = 2
    TARGETED_DENSE_MAX_INSTABILITY = 2
    TARGETED_DENSE_MAX_DIST_TO_MY_TRACK = 3

    # --- V25: When to densify (avoid over-investing) ---
    DENSIFY_ONLY_WHEN_AHEAD_BY = 35
    DENSIFY_ONLY_WHEN_OPP_THREAT_LEQ = 55.0
    DENSIFY_ONLY_WHEN_BEST_NEW_EV_LEQ = 35.0

    # --- V25 experimental densify knobs (currently disabled / inert) ---
    DENSE_BASE_MAX_INSTABILITY = 2
    DENSE_ALLOW_INSTABILITY_IN_OWN_REGION = 2
    DENSE_TOWN_ADJ_BONUS = 0.0
    DENSE_NEAR_TOWN_RADIUS = 6
    DENSE_NEAR_TOWN_BONUS_SCALE = 0.0
    DENSE_SKIP_WHEN_ACTIVE_BUILD = False


    def __init__(self):
        self.my_id = 0
        self.opp_id = 1
        self.width = 0
        self.height = 0
        self.towns: List[Town] = []
        self.town_by_id: Dict[int, Town] = {}
        self.region_ids: List[List[int]] = []
        self.cell_types: List[List[int]] = []

        self.turn = 0
        self.my_score = 0
        self.opp_score = 0

        self.completed_connections: Set[Tuple[int, int]] = set()
        self.region_instability: Dict[int, int] = defaultdict(int)
        self.region_inked: Set[int] = set()
        self.cell_tracks: Dict[Tuple[int, int], int] = {}
        self.town_regions: Set[int] = set()
        self.town_cells: Set[Tuple[int, int]] = set()

        self.active_build: Optional[Tuple[int, int]] = None
        self.active_build_path: Optional[List[Tuple[int, int]]] = None

        self.my_score_history: List[int] = []
        self.opp_score_history: List[int] = []

        self.detected_opp_bias: Optional[str] = 'high'
        self.my_bias_multiplier: float = -5.0

        self.opp_track_regions: Set[int] = set()
        self.my_track_regions: Set[int] = set()

        self.active_weight_by_region_my: Dict[int, float] = defaultdict(float)
        self.active_weight_by_region_opp: Dict[int, float] = defaultdict(float)
        self.active_weight_by_region_neutral: Dict[int, float] = defaultdict(float)

        self.hub_town_id: Optional[int] = None
        self.town_adjacent_cells: Set[Tuple[int, int]] = set()

        self.path_cache: Dict[Tuple, Optional[List[Tuple[int, int]]]] = {}
        # Pathfinding mode flag (see find_path): when enabled, treat opponent tracks as
        # non-traversable to recover buildable routes if a cheapest path would rely on
        # opponent-owned tracks.
        self._pf_block_opp_tracks: bool = False

        self._pred_opp_scored_paths_turn: int = -1
        self._pred_opp_scored_paths: List[Tuple[float, List[Tuple[int, int]]]] = []

        # --- Opponent disruption inference (from instability deltas) ---
        self._prev_region_instability: Dict[int, int] = {}
        self._prev_region_inked: Set[int] = set()
        self._last_disrupt_target: Optional[int] = None
        self._opp_disrupt_count: Dict[int, int] = defaultdict(int)
        self._opp_disrupt_last_turn: Dict[int, int] = {}

        # --- Opponent build-region inference (from newly placed opp tracks) ---
        self._prev_opp_tracks: Set[Tuple[int, int]] = set()
        self._opp_recent_build_region: Optional[int] = None
        self._opp_recent_build_turn: int = -1

        # --- Disrupt stickiness ---
        self._disrupt_lock_region: Optional[int] = None
        self._disrupt_lock_until_turn: int = -1

        self._trace_enabled = bool(os.environ.get("V26_TRACE", ""))
        self._trace_file = os.environ.get("V26_TRACE_FILE", "v26_trace.log")
        self._trace_seed = os.environ.get("V26_TRACE_SEED", "")
        self._trace_as = os.environ.get("V26_TRACE_AS", "")

        # --- Dataset logging (for training small NN helpers) ---
        self._dataset_enabled = bool(os.environ.get("V26_DATASET", ""))
        self._dataset_file = os.environ.get("V26_DATASET_FILE", "v26_dataset.jsonl")
        self._dataset_kind = os.environ.get("V26_DATASET_KIND", "disrupt")
        self._dataset_seed = os.environ.get("V26_DATASET_SEED", self._trace_seed)
        self._dataset_as = os.environ.get("V26_DATASET_AS", self._trace_as)
        try:
            self._dataset_max_turn = int(os.environ.get("V26_DATASET_MAX_TURN", "0"))
        except Exception:
            self._dataset_max_turn = 0

        # --- Oracle forcing (used by labeling scripts) ---
        # Force the disrupt choice at a specific turn for deterministic reruns.
        # If set, this overrides only the DISRUPT decision, not track placement.
        self._force_disrupt_turn = os.environ.get("V26_FORCE_DISRUPT_TURN", "")
        self._force_disrupt_region = os.environ.get("V26_FORCE_DISRUPT_REGION", "")

    def _trace(self, msg: str) -> None:
        if not self._trace_enabled:
            return
        try:
            with open(self._trace_file, "a", encoding="utf-8") as f:
                f.write(msg.rstrip("\n") + "\n")
        except Exception:
            pass

    def _dataset_write(self, obj: dict) -> None:
        if not self._dataset_enabled:
            return
        if self._dataset_max_turn and self.turn > self._dataset_max_turn:
            return
        try:
            with open(self._dataset_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _disrupt_candidate_features(self, region_id: int, turns_left: int) -> Optional[dict]:
        if region_id in self.region_inked or region_id in self.town_regions:
            return None
        instability = int(self.region_instability.get(region_id, 0))
        if instability >= self.INSTABILITY_TO_INK:
            return None
        hits_to_ink = int(self.INSTABILITY_TO_INK - instability)

        anti_dense = self._opp_dense_potential_by_region() if self.ANTI_DENSE_ENABLED else {}
        opp_w = float(self.active_weight_by_region_opp.get(region_id, 0.0))
        my_w = float(self.active_weight_by_region_my.get(region_id, 0.0))
        neu_w = float(self.active_weight_by_region_neutral.get(region_id, 0.0))
        pot = float(anti_dense.get(region_id, 0.0)) if anti_dense else 0.0

        base = (opp_w - self.SELF_HARM_ALPHA * my_w + 0.15 * neu_w) + pot * float(self.ANTI_DENSE_ALPHA)
        score = -1e9
        if base > 0:
            score = (base * float(turns_left)) / float(max(hits_to_ink, 1))
            if hits_to_ink == 1:
                score *= 2.0
            elif hits_to_ink >= 2:
                score *= 0.65

        pressured = float(self._opp_disrupt_pressure(region_id))
        recent_build = 1 if (self._opp_recent_build_region == region_id and (self.turn - self._opp_recent_build_turn) <= 2) else 0

        return {
            "rid": int(region_id),
            "inst": int(instability),
            "hits": int(hits_to_ink),
            "opp_w": float(opp_w),
            "my_w": float(my_w),
            "neu_w": float(neu_w),
            "anti_dense": float(pot),
            "pressure": float(pressured),
            "recent_build": int(recent_build),
            "score": float(score),
        }

    def _get_disruption_candidates_for_dataset(
        self, best_next_path: Optional[List[Tuple[int, int]]] = None, limit: int = 12
    ) -> List[dict]:
        turns_left = int(self.TOTAL_TURNS - self.turn)
        avoid: Set[int] = set()
        if self.active_build_path:
            for x, y in self.active_build_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)
        if best_next_path:
            for x, y in best_next_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)

        anti_dense = self._opp_dense_potential_by_region() if self.ANTI_DENSE_ENABLED else {}
        region_ids: Set[int] = set(
            list(self.active_weight_by_region_opp.keys())
            + list(self.active_weight_by_region_my.keys())
            + list(self.active_weight_by_region_neutral.keys())
            + list(self.opp_track_regions)
            + (list(anti_dense.keys()) if anti_dense else [])
        )

        # Include predicted opponent path regions (often where strong bots care).
        for _v, path in self._predict_opponent_scored_paths()[: max(1, self.PRED_TOP_K)]:
            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    region_ids.add(rid)

        feats: List[dict] = []
        for rid in region_ids:
            if rid in avoid:
                continue
            d = self._disrupt_candidate_features(int(rid), turns_left)
            if d is not None and d.get("score", -1e9) > 0:
                feats.append(d)

        feats.sort(key=lambda x: (float(x.get("score", 0.0)), -int(x.get("hits", 9)), -float(x.get("opp_w", 0.0))), reverse=True)
        return feats[: int(max(1, limit))]

    # -------------------- Input --------------------

    def read_init(self):
        lines = []
        lines.append(input())
        width = int(input())
        height = int(input())
        lines.append(str(width))
        lines.append(str(height))
        for _ in range(height * width):
            lines.append(input())
        town_count = int(input())
        lines.append(str(town_count))
        for _ in range(town_count):
            lines.append(input())
        self.read_init_lines(lines)

    def read_init_lines(self, lines: List[str]):
        it = iter(lines)
        self.my_id = int(next(it))
        self.opp_id = 1 - self.my_id
        self.width = int(next(it))
        self.height = int(next(it))

        self.region_ids = []
        self.cell_types = []
        for _ in range(self.height):
            region_row = []
            type_row = []
            for _ in range(self.width):
                try:
                    inputs = next(it).split()
                    region_id = int(inputs[0])
                    cell_type = int(inputs[1]) if len(inputs) > 1 else 0
                    region_row.append(region_id)
                    type_row.append(cell_type)
                except (ValueError, IndexError, StopIteration):
                    region_row.append(-1)
                    type_row.append(0)
            self.region_ids.append(region_row)
            self.cell_types.append(type_row)

        try:
            town_count = int(next(it))
        except StopIteration:
            town_count = 0

        self.towns = []
        self.town_by_id = {}
        self.town_cells.clear()

        for _ in range(town_count):
            try:
                parts = next(it).split()
            except StopIteration:
                break
            if len(parts) < 3:
                continue

            town_id = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            desired_str = parts[3] if len(parts) > 3 else "-"

            desired = []
            if desired_str not in ("", "-", "x", "X"):
                for d in desired_str.split(","):
                    d = d.strip()
                    if d.isdigit():
                        desired.append(int(d))

            t = Town(town_id, x, y, desired)
            self.towns.append(t)
            self.town_by_id[town_id] = t
            self.town_cells.add((x, y))

        self.town_regions = set()
        for t in self.towns:
            if 0 <= t.y < self.height and 0 <= t.x < self.width:
                self.town_regions.add(self.region_ids[t.y][t.x])

        indeg = defaultdict(int)
        for t in self.towns:
            for d in t.desired:
                indeg[d] += 1
        if self.towns:
            self.hub_town_id = max(self.towns, key=lambda t: (len(t.desired) + indeg[t.id], -t.id)).id
        else:
            self.hub_town_id = None

        self.town_adjacent_cells.clear()
        for t in self.towns:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = t.x + dx, t.y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.town_cells:
                        self.town_adjacent_cells.add((nx, ny))

        self.turn = 0
        self.my_score = 0
        self.opp_score = 0
        self.active_build = None
        self.active_build_path = None
        self.path_cache.clear()
        self.my_score_history.clear()
        self.opp_score_history.clear()

        self._pred_opp_scored_paths_turn = -1
        self._pred_opp_scored_paths = []

        self._prev_region_instability = {}
        self._prev_region_inked = set()
        self._last_disrupt_target = None
        self._opp_disrupt_count = defaultdict(int)
        self._opp_disrupt_last_turn = {}
        self._disrupt_lock_region = None
        self._disrupt_lock_until_turn = -1

    def _opp_disrupt_pressure(self, region_id: int) -> float:
        """Heuristic: how likely this region is to be inked soon by opponent."""
        if region_id in self.region_inked or region_id in self.town_regions:
            return 0.0
        c = float(min(self._opp_disrupt_count.get(region_id, 0), 6))
        if c <= 0:
            return 0.0
        last = int(self._opp_disrupt_last_turn.get(region_id, -10**9))
        age = max(0, int(self.turn) - last)
        # Fast decay: if it wasn't hit recently, treat as low pressure.
        decay = 0.55 ** (age / 4.0)
        return float(c * decay)

    def _infer_and_record_opp_disrupts(self) -> None:
        # Compare region instability deltas turn-over-turn.
        if not self._prev_region_instability and self.turn <= 1:
            self._prev_region_instability = dict(self.region_instability)
            self._prev_region_inked = set(self.region_inked)
            return

        regions = set(self._prev_region_instability.keys()) | set(self.region_instability.keys())
        # Attribute at most 1 disruption to us (we can only disrupt once/turn).
        our_target = self._last_disrupt_target

        for rid in regions:
            prev_i = int(self._prev_region_instability.get(rid, 0))
            cur_i = int(self.region_instability.get(rid, 0))
            if cur_i <= prev_i:
                continue
            delta = cur_i - prev_i

            # If we disrupted this region last turn, subtract one hit.
            if our_target is not None and rid == our_target and delta > 0:
                delta -= 1
            if delta <= 0:
                continue
            self._opp_disrupt_count[rid] += int(delta)
            self._opp_disrupt_last_turn[rid] = int(self.turn)

        # Also: if a region just became inked (and it wasn't previously), treat it as pressured.
        # IMPORTANT: avoid falsely attributing our own disruption-caused inks to the opponent.
        newly_inked = set(self.region_inked) - set(self._prev_region_inked)
        for rid in newly_inked:
            if our_target is not None and rid == our_target:
                # We disrupted this region last turn; if it inked now, it's very likely
                # due to our action (or at least not safe to attribute to opponent).
                continue
            self._opp_disrupt_count[rid] += 1
            self._opp_disrupt_last_turn[rid] = int(self.turn)

        self._prev_region_instability = dict(self.region_instability)
        self._prev_region_inked = set(self.region_inked)

    def read_turn(self):
        lines = [input(), input()]
        for _ in range(self.height * self.width):
            lines.append(input())
        self.read_turn_lines(lines)

    def read_turn_lines(self, lines: List[str]):
        it = iter(lines)
        try:
            self.my_score = int(next(it))
            self.opp_score = int(next(it))
        except StopIteration:
            return

        self.turn += 1
        self.my_score_history.append(self.my_score)
        self.opp_score_history.append(self.opp_score)

        self.completed_connections.clear()
        self.region_instability.clear()
        self.region_inked.clear()
        self.cell_tracks.clear()
        self.opp_track_regions.clear()
        self.my_track_regions.clear()

        self.active_weight_by_region_my.clear()
        self.active_weight_by_region_opp.clear()
        self.active_weight_by_region_neutral.clear()

        cur_opp_tracks: Set[Tuple[int, int]] = set()

        for y in range(self.height):
            for x in range(self.width):
                try:
                    inputs = next(it).split()
                    if len(inputs) < 3:
                        continue
                    track_owner = int(inputs[0])
                    instability = int(inputs[1])
                    inked = inputs[2] != "0"
                    connections_str = inputs[3] if len(inputs) > 3 else "x"
                except (StopIteration, IndexError, ValueError):
                    continue

                region_id = self.region_ids[y][x]
                if region_id == -1:
                    continue

                if track_owner != -1:
                    self.cell_tracks[(x, y)] = track_owner
                    if track_owner == self.opp_id:
                        self.opp_track_regions.add(region_id)
                        cur_opp_tracks.add((x, y))
                    elif track_owner == self.my_id:
                        self.my_track_regions.add(region_id)

                self.region_instability[region_id] = max(self.region_instability[region_id], instability)
                if inked:
                    self.region_inked.add(region_id)

                if connections_str not in ("x", "-", ""):
                    pairs = []
                    for conn in connections_str.split(','):
                        conn = conn.strip()
                        if '-' not in conn:
                            continue
                        try:
                            a, b = conn.split('-')
                            t1, t2 = int(a), int(b)
                            pairs.append((t1, t2))
                            self.completed_connections.add((min(t1, t2), max(t1, t2)))
                        except ValueError:
                            continue

                    if pairs and region_id not in self.town_regions and region_id not in self.region_inked:
                        w = float(len(pairs))
                        if track_owner == self.my_id:
                            self.active_weight_by_region_my[region_id] += w
                        elif track_owner == self.opp_id:
                            self.active_weight_by_region_opp[region_id] += w
                        elif track_owner == 2:
                            self.active_weight_by_region_neutral[region_id] += w

        # Infer where opponent is actively building (based on new opp tracks this turn).
        newly_added_opp = cur_opp_tracks - self._prev_opp_tracks
        if newly_added_opp:
            region_counts: Dict[int, int] = defaultdict(int)
            for x, y in newly_added_opp:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.town_regions or rid in self.region_inked:
                    continue
                region_counts[rid] += 1
            if region_counts:
                best_rid = max(region_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
                self._opp_recent_build_region = int(best_rid)
                self._opp_recent_build_turn = int(self.turn)
        self._prev_opp_tracks = set(cur_opp_tracks)

        self._detect_opponent_behavior()

        if self.turn % 10 == 0:
            self.path_cache.clear()

        if self.active_build:
            conn_id = (min(self.active_build[0], self.active_build[1]), max(self.active_build[0], self.active_build[1]))
            if conn_id in self.completed_connections:
                self.active_build = None
                self.active_build_path = None

        # Infer opponent disruption pressure from instability deltas.
        self._infer_and_record_opp_disrupts()

    # -------------------- Placement rules --------------------

    def can_place_track(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if (x, y) in self.town_cells:
            return False

        region_id = self.region_ids[y][x]
        if region_id == -1:
            return False
        if region_id in self.region_inked:
            return False

        instability = self.region_instability.get(region_id, 0)
        if instability >= self.INSTABILITY_TO_INK:
            return False

        if (x, y) in self.cell_tracks:
            return False

        return True

    def _is_existing_connection_cell(self, x: int, y: int) -> bool:
        return (x, y) in self.town_cells or (x, y) in self.cell_tracks

    # -------------------- Pathfinding --------------------

    def _empty_cell_place_cost(self, x: int, y: int) -> int:
        return int(self.cell_types[y][x] + 1)

    def _get_cell_cost(self, x: int, y: int) -> float:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return float('inf')

        if (x, y) in self.town_cells:
            return 0.0

        region_id = self.region_ids[y][x]
        if region_id == -1 or region_id in self.region_inked:
            return float('inf')

        instability = self.region_instability.get(region_id, 0)
        if instability >= self.INSTABILITY_TO_INK:
            return float('inf')

        if (x, y) in self.cell_tracks:
            owner = self.cell_tracks[(x, y)]
            if owner == self.my_id:
                rid = self.region_ids[y][x]
                p = self._opp_disrupt_pressure(rid) if rid != -1 else 0.0
                return 0.0 + 0.30 * p
            if owner == self.opp_id:
                        if self._pf_block_opp_tracks:
                            return float('inf')
                        rid = self.region_ids[y][x]
                        p = self._opp_disrupt_pressure(rid) if rid != -1 else 0.0
                        return float(self.OPP_TRACK_TRAVERSAL_PENALTY) + 0.35 * p
            return float(self.NEUTRAL_TRACK_TRAVERSAL_PENALTY)

        base_cost = self._empty_cell_place_cost(x, y)
        if base_cost > self.POINTS_PER_TURN:
            return float('inf')

        instability_penalty = instability * 2

        # If opponent is repeatedly disrupting this region, treat it as risky:
        # it may be inked soon, invalidating longer build plans through it.
        rid = self.region_ids[y][x]
        p = self._opp_disrupt_pressure(rid) if rid != -1 else 0.0
        risk_penalty = 1.10 * p

        return float(base_cost + instability_penalty + risk_penalty)

    def find_path(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        max_nodes: int = 1400,
        use_cache: bool = True,
        block_opp_tracks: bool = False,
    ) -> Optional[List[Tuple[int, int]]]:
        cache_key = (x0, y0, x1, y1, bool(block_opp_tracks))
        if use_cache and cache_key in self.path_cache:
            return self.path_cache[cache_key]

        prev_block = bool(self._pf_block_opp_tracks)
        self._pf_block_opp_tracks = bool(block_opp_tracks)

        try:
            def heuristic(x, y):
                return abs(x - x1) + abs(y - y1)

            order = itertools.count()
            open_set = [(0.0, 0.0, next(order), x0, y0)]
            came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
            g_score: Dict[Tuple[int, int], float] = {(x0, y0): 0.0}
            nodes_expanded = 0

            while open_set and nodes_expanded < max_nodes:
                _, g, _, x, y = heapq.heappop(open_set)
                nodes_expanded += 1

                if x == x1 and y == y1:
                    path: List[Tuple[int, int]] = []
                    cur = (x, y)
                    while cur in came_from:
                        path.append(cur)
                        cur = came_from[cur]
                    path.append((x0, y0))
                    result = list(reversed(path))
                    if use_cache:
                        self.path_cache[cache_key] = result
                    return result

                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    step_cost = self._get_cell_cost(nx, ny)
                    if step_cost == float('inf'):
                        continue

                    tentative_g = g + step_cost
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = (x, y)
                        g_score[(nx, ny)] = tentative_g
                        f = tentative_g + heuristic(nx, ny)
                        heapq.heappush(open_set, (f, tentative_g, next(order), nx, ny))

            if use_cache:
                self.path_cache[cache_key] = None
            return None
        finally:
            self._pf_block_opp_tracks = prev_block

    def find_shortest_path_bfs(self, x0: int, y0: int, x1: int, y1: int) -> Optional[List[Tuple[int, int]]]:
        start = (x0, y0)
        goal = (x1, y1)
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue

                region_id = self.region_ids[ny][nx]
                if region_id in self.region_inked:
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    # -------------------- Cost / EV --------------------

    def calc_path_cost(self, path: List[Tuple[int, int]]) -> int:
        total = 0
        for x, y in path:
            if (x, y) in self.town_cells:
                continue
            if (x, y) in self.cell_tracks:
                continue
            total += self._empty_cell_place_cost(x, y)
        return total

    def calc_remaining_cost(self, path: List[Tuple[int, int]]) -> int:
        return self.calc_path_cost(path)

    def _count_track_owners_on_path(self, path: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        my_cells = 0
        opp_cells = 0
        neutral_cells = 0
        empty_cells = 0
        for x, y in path:
            if (x, y) in self.town_cells:
                continue
            if (x, y) in self.cell_tracks:
                owner = self.cell_tracks[(x, y)]
                if owner == self.my_id:
                    my_cells += 1
                elif owner == self.opp_id:
                    opp_cells += 1
                elif owner == 2:
                    neutral_cells += 1
            else:
                empty_cells += 1
        return my_cells, opp_cells, neutral_cells, empty_cells

    def calc_connection_ev(self, path: List[Tuple[int, int]], from_id: int = 0, to_id: int = 0) -> float:
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

        speed_multiplier = 1.5 if cost <= 10 else (1.2 if cost <= 17 else 0.7568338133671997)
        weighted_points = expected_points * speed_multiplier
        efficiency = weighted_points / max(cost, 1)

        town_id_bonus = 0.0
        if from_id > 0 and to_id > 0:
            base_bonus = (from_id + to_id) * 53.71401146671039
            town_id_bonus = base_bonus * self.my_bias_multiplier

        regions_in_path = set(self.region_ids[y][x] for x, y in path if self.region_ids[y][x] != -1)
        shared_regions = regions_in_path & self.my_track_regions
        consolidation_bonus = float(len(shared_regions) * 26.88965731068639)

        # Opponent disruption pressure penalty: avoid committing EV to regions
        # that the opponent is actively trying to ink.
        risk = 0.0
        for rid in regions_in_path:
            if rid in self.town_regions or rid in self.region_inked:
                continue
            risk += self._opp_disrupt_pressure(rid)
        disrupt_risk_penalty = 800.0 * risk

        overlap_bonus = float(my_cells * 4)

        junction_hits = sum(1 for x, y in path if (x, y) in self.town_adjacent_cells)
        junction_bonus = float(junction_hits * 2.164151903532464)

        hub_bonus = 0.0
        if self.hub_town_id is not None and (from_id == self.hub_town_id or to_id == self.hub_town_id):
            hub_bonus = 35.0

        gifting_penalty = float((opp_cells * 7 + neutral_cells * 2) * speed_multiplier)

        return (
            weighted_points
            + efficiency * 100
            + town_id_bonus
            + consolidation_bonus
            + overlap_bonus
            + junction_bonus
            + hub_bonus
            - gifting_penalty
            - disrupt_risk_penalty
        )

    def calc_continue_ev(self) -> float:
        if not self.active_build or not self.active_build_path:
            return 0.0

        path = self.active_build_path
        blocked = sum(1 for x, y in path if self.region_ids[y][x] in self.region_inked)
        if blocked > len(path) * 0.5:
            return 0.0

        from_id, to_id = self.active_build
        base = self.calc_connection_ev(path, from_id=from_id, to_id=to_id)

        # Small stickiness term to avoid thrashing when EVs are close.
        my_cells, _, _, _ = self._count_track_owners_on_path(path)
        progress_ratio = my_cells / max(len(path), 1)
        sunk_cost_bonus = progress_ratio * 50.0

        return float(base + sunk_cost_bonus)

    # -------------------- Build execution --------------------

    def _advance_from_start(self, path: List[Tuple[int, int]], left: int, right: int) -> int:
        while left + 1 <= right:
            x, y = path[left + 1]
            if self._is_existing_connection_cell(x, y):
                left += 1
                continue
            break
        return left

    def _advance_from_end(self, path: List[Tuple[int, int]], left: int, right: int) -> int:
        while right - 1 >= left:
            x, y = path[right - 1]
            if self._is_existing_connection_cell(x, y):
                right -= 1
                continue
            break
        return right

    def build_path_two_ended_stable(self, from_town: Town, to_town: Town) -> List[str]:
        actions: List[str] = []
        budget = self.POINTS_PER_TURN

        # Use cached pathfinding here (same as candidate refresh) to avoid
        # oscillating between multiple equivalent paths.
        # Use fresh routing for actual build placement so we can adapt to the
        # evolving board (ink/instability/tracks).
        path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
        if not path or len(path) < 2:
            return actions

        left = 0
        right = len(path) - 1
        left = self._advance_from_start(path, left, right)
        right = self._advance_from_end(path, left, right)

        while budget > 0:
            left = self._advance_from_start(path, left, right)
            right = self._advance_from_end(path, left, right)

            if left >= right - 1:
                break

            feasible = []

            sx, sy = path[left + 1]
            if self.can_place_track(sx, sy):
                c = self._empty_cell_place_cost(sx, sy)
                if c <= budget:
                    rid = self.region_ids[sy][sx]
                    inst = self.region_instability.get(rid, 0) if rid != -1 else 0
                    danger = 1 if inst >= (self.INSTABILITY_TO_INK - 1) else 0
                    feasible.append((danger, inst, c, 0, sx, sy))

            ex, ey = path[right - 1]
            if self.can_place_track(ex, ey):
                c = self._empty_cell_place_cost(ex, ey)
                if c <= budget:
                    rid = self.region_ids[ey][ex]
                    inst = self.region_instability.get(rid, 0) if rid != -1 else 0
                    danger = 1 if inst >= (self.INSTABILITY_TO_INK - 1) else 0
                    feasible.append((danger, inst, c, 1, ex, ey))

            if not feasible:
                # Cached paths can become stale as instability/ink evolves; force a
                # recompute when the current path can't advance.
                new_path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
                new_path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
                if (not new_path or new_path == path) and any(
                    self.cell_tracks.get((x, y)) == self.opp_id for (x, y) in path
                ):
                    # If the cheapest route relies on opponent tracks, try again blocking
                    # them so we can find a buildable alternative.
                    new_path = self.find_path(
                        from_town.x,
                        from_town.y,
                        to_town.x,
                        to_town.y,
                        use_cache=False,
                        block_opp_tracks=True,
                    )
                if not new_path or new_path == path:
                    break
                path = new_path
                left = 0
                right = len(path) - 1
                left = self._advance_from_start(path, left, right)
                right = self._advance_from_end(path, left, right)
                continue

            # Prefer not to place into near-ink regions (instability==3) unless forced,
            # to avoid self-trapping and losing late-game build options.
            feasible.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            _danger, _inst, cost, side, x, y = feasible[0]
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(cost)
            self.cell_tracks[(x, y)] = self.my_id
            if side == 0:
                left += 1
            else:
                right -= 1

        return actions

    def _path_has_affordable_step(self, path: List[Tuple[int, int]]) -> bool:
        for x, y in path:
            if not self.can_place_track(x, y):
                continue
            return self._empty_cell_place_cost(x, y) <= self.POINTS_PER_TURN
        return False

    # -------------------- V25: Smart densify --------------------

    def _remaining_budget_after_actions(self, actions: List[str]) -> int:
        spent = 0
        for a in actions:
            if not a.startswith("PLACE_TRACKS"):
                continue
            parts = a.split()
            if len(parts) != 3:
                continue
            try:
                x = int(parts[1])
                y = int(parts[2])
            except ValueError:
                continue
            spent += self._empty_cell_place_cost(x, y)
        return max(0, self.POINTS_PER_TURN - spent)

    def _dense_candidate_score(self, x: int, y: int) -> float:
        # V24: Prefer cheap cells that create junctions with existing own network.
        cost = self._empty_cell_place_cost(x, y)
        region_id = self.region_ids[y][x]
        instability = self.region_instability.get(region_id, 0)

        adj_my = 0
        adj_opp = 0
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if (nx, ny) in self.town_cells:
                adj_my += 1
                continue
            owner = self.cell_tracks.get((nx, ny))
            if owner == self.my_id:
                adj_my += 1
            elif owner == self.opp_id:
                adj_opp += 1

        # Cost is king; also reward junction formation.
        score = 0.0
        score += (3 - cost) * 10.0
        score += adj_my * 6.0
        if adj_my >= 2:
            score += 12.0

        # Prefer stable regions.
        score += (self.DENSE_MAX_INSTABILITY - instability) * 2.5

        # Slightly prefer consolidating inside existing owned regions.
        if region_id in self.my_track_regions:
            score += 3.0

        # Avoid creating adjacency to opponent tracks (helps them latch on).
        score -= adj_opp * 4.0

        return score

    def _densify_network(self, budget: int) -> List[str]:
        if budget <= 0:
            return []

        actions: List[str] = []
        placed = 0

        # Candidate set = neighbors of own tracks and towns.
        frontier: Set[Tuple[int, int]] = set()
        anchors = set(self.town_cells)
        anchors.update([pos for pos, owner in self.cell_tracks.items() if owner == self.my_id])

        for ax, ay in anchors:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        while budget > 0 and placed < self.DENSE_MAX_EXTRA_PLACEMENTS:
            best = None
            best_score = -1e9

            for x, y in list(frontier):
                if not self.can_place_track(x, y):
                    continue
                c = self._empty_cell_place_cost(x, y)
                if c > budget:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                if self.region_instability.get(rid, 0) > self.DENSE_MAX_INSTABILITY:
                    continue
                if self.DENSE_AVOID_OPP_REGIONS and rid in self.opp_track_regions:
                    continue

                s = self._dense_candidate_score(x, y)
                if s > best_score:
                    best_score = s
                    best = (x, y, c)

            if best is None:
                break

            x, y, c = best
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(c)
            placed += 1

            # Update local state so next pick can build around the new cell.
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

            # Expand frontier from new cell.
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        return actions

    # -------------------- V21 prediction helpers --------------------

    def _town_near_opp_tracks(self, town: Town, opp_cells: List[Tuple[int, int]]) -> bool:
        if not opp_cells:
            return False
        tx, ty = town.x, town.y
        dmax = self.OPP_TOWN_NEAR_DIST
        for x, y in opp_cells:
            if abs(x - tx) + abs(y - ty) <= dmax:
                return True
        return False

    def _get_cell_cost_for_opponent_path(self, x: int, y: int) -> float:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return float('inf')

        if (x, y) in self.town_cells:
            return 0.0

        region_id = self.region_ids[y][x]
        if region_id == -1 or region_id in self.region_inked:
            return float('inf')

        instability = self.region_instability.get(region_id, 0)
        if instability >= self.INSTABILITY_TO_INK:
            return float('inf')

        if (x, y) in self.cell_tracks:
            owner = self.cell_tracks[(x, y)]
            if owner == self.opp_id:
                return 0.0
            if owner == self.my_id:
                return float(self.MY_TRACK_BLOCKING_PENALTY)
            return float(self.NEUTRAL_TRACK_PENALTY)

        base_cost = int(self.cell_types[y][x] + 1)
        if base_cost > self.POINTS_PER_TURN:
            return float('inf')

        return float(base_cost + instability * 2)

    def _find_opponent_path(self, x0: int, y0: int, x1: int, y1: int, max_nodes: int = 1200) -> Optional[List[Tuple[int, int]]]:
        def heuristic(x, y):
            return abs(x - x1) + abs(y - y1)

        order = itertools.count()
        open_set = [(0.0, 0.0, next(order), x0, y0)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(x0, y0): 0.0}
        nodes_expanded = 0

        while open_set and nodes_expanded < max_nodes:
            _, g, _, x, y = heapq.heappop(open_set)
            nodes_expanded += 1

            if x == x1 and y == y1:
                path = []
                cur = (x, y)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append((x0, y0))
                return list(reversed(path))

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                step_cost = self._get_cell_cost_for_opponent_path(nx, ny)
                if step_cost == float('inf'):
                    continue

                tentative_g = g + step_cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (x, y)
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, tentative_g, next(order), nx, ny))

        return None

    def _opp_candidate_value(self, path: List[Tuple[int, int]]) -> float:
        if not path:
            return 0.0

        cost = 0
        empty_cells = 0
        for x, y in path:
            if (x, y) in self.town_cells:
                continue
            if (x, y) in self.cell_tracks:
                continue
            c = int(self.cell_types[y][x] + 1)
            if c > self.POINTS_PER_TURN:
                return 0.0
            cost += c
            empty_cells += 1

        if cost <= 0:
            return 0.0

        turns_to_complete = (cost + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN
        turns_left = self.TOTAL_TURNS - self.turn
        scoring_turns = max(0, turns_left - turns_to_complete)
        if scoring_turns <= 0:
            return 0.0

        return float((empty_cells * scoring_turns) / max(cost, 1) * 100.0)

    def _predict_opponent_scored_paths(self) -> List[Tuple[float, List[Tuple[int, int]]]]:
        if self._pred_opp_scored_paths_turn == self.turn:
            return self._pred_opp_scored_paths

        scored_paths: List[Tuple[float, List[Tuple[int, int]]]] = []

        # Heuristic prefilter: consider only the closest uncompleted desired connections.
        # This keeps runtime stable while avoiding the "near opp tracks" blind spot.
        pref: List[Tuple[int, int, int]] = []  # (manhattan, from_id, to_id)
        for town in self.towns:
            for dest_id in town.desired:
                conn_id = (min(town.id, dest_id), max(town.id, dest_id))
                if conn_id in self.completed_connections:
                    continue
                dest = self.town_by_id.get(dest_id)
                if dest is None:
                    continue
                md = abs(town.x - dest.x) + abs(town.y - dest.y)
                pref.append((int(md), int(town.id), int(dest_id)))

        pref.sort(key=lambda t: (t[0], t[1], t[2]))
        pref = pref[: int(self.OPP_PRED_PREFILTER_TOP_N)]

        for _md, from_id, to_id in pref:
            a = self.town_by_id.get(int(from_id))
            b = self.town_by_id.get(int(to_id))
            if a is None or b is None:
                continue

            path = self._find_opponent_path(a.x, a.y, b.x, b.y)
            if not path or len(path) < 2:
                continue

            v = float(self._opp_candidate_value(path))
            if v <= 0:
                continue

            # If we observed the opponent recently building in a region that appears on
            # this path, boost it slightly (helps lock onto their real intent).
            if self._opp_recent_build_region is not None and (self.turn - self._opp_recent_build_turn) <= 2:
                rid0 = int(self._opp_recent_build_region)
                if rid0 != -1 and any(self.region_ids[y][x] == rid0 for x, y in path if (x, y) not in self.town_cells):
                    v *= 1.12

            scored_paths.append((v, path))

        scored_paths.sort(key=lambda t: -t[0])
        self._pred_opp_scored_paths = scored_paths[: self.PRED_TOP_K]
        self._pred_opp_scored_paths_turn = self.turn
        return self._pred_opp_scored_paths

    def _predict_opponent_region_weights(self) -> Dict[int, float]:
        top = self._predict_opponent_scored_paths()
        if not top:
            return {}

        pred: Dict[int, float] = defaultdict(float)
        for v, path in top:
            denom = float(max(len(path) - 1, 1))
            per_cell = v / denom
            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                region_id = self.region_ids[y][x]
                if region_id == -1:
                    continue
                if region_id in self.town_regions or region_id in self.region_inked:
                    continue
                pred[region_id] += per_cell

        return pred

    def _choke_opponent_paths(self, budget: int) -> List[str]:
        if not self.CHOKE_ENABLED or budget <= 0:
            return []
        if self.turn < self.CHOKE_START_TURN:
            return []
        if (self.my_score - self.opp_score) >= self.CHOKE_DISABLE_WHEN_AHEAD:
            return []

        top = self._predict_opponent_scored_paths()
        if not top:
            return []

        top = top[: self.CHOKE_TOP_K_PATHS]

        best_cells: List[Tuple[float, int, int, int]] = []
        # (score, cost, x, y)
        for v, path in top:
            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                if not self.can_place_track(x, y):
                    continue
                cost = self._empty_cell_place_cost(x, y)
                if cost > budget or cost > self.CHOKE_MAX_CELL_COST:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.town_regions or rid in self.region_inked:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability >= self.INSTABILITY_TO_INK:
                    continue

                empty_cost = float(cost + instability * 2)
                delta = float(self.MY_TRACK_BLOCKING_PENALTY) - empty_cost
                if delta < self.CHOKE_MIN_DELTA_COST:
                    continue

                # Prefer high-value paths, large delta, and low placement cost.
                s = float(v) * delta / float(max(cost, 1))
                best_cells.append((s, int(cost), x, y))

        if not best_cells:
            return []

        best_cells.sort(reverse=True)
        actions: List[str] = []
        placed = 0
        used: Set[Tuple[int, int]] = set()

        for s, cost, x, y in best_cells:
            if placed >= self.CHOKE_MAX_PLACEMENTS:
                break
            if budget < cost:
                continue
            if (x, y) in used:
                continue
            if not self.can_place_track(x, y):
                continue
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(cost)
            placed += 1
            used.add((x, y))
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

        return actions

    def _opp_threat_value(self) -> float:
        top = self._predict_opponent_scored_paths()
        return float(top[0][0]) if top else 0.0

    def _support_path_placement_score(self, x: int, y: int, cost: int, end_a: Tuple[int, int], end_b: Tuple[int, int]) -> float:
        rid = self.region_ids[y][x]
        instability = self.region_instability.get(rid, 0)

        da = abs(x - end_a[0]) + abs(y - end_a[1])
        db = abs(x - end_b[0]) + abs(y - end_b[1])
        d = min(da, db)

        adj_opp = 0
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if self.cell_tracks.get((nx, ny)) == self.opp_id:
                adj_opp += 1

        score = 0.0
        score += (3 - cost) * 8.0
        score += max(0.0, 10.0 - float(d))
        score += (self.SUPPORT_MAX_INSTABILITY - instability) * 2.0
        score -= adj_opp * 3.0
        return score

    def _support_best_path(self, budget: int, path: Optional[List[Tuple[int, int]]]) -> List[str]:
        if not self.SUPPORT_ENABLED or budget <= 0 or not path or len(path) < 2:
            return []

        end_a = path[0]
        end_b = path[-1]
        actions: List[str] = []
        placed = 0

        while budget > 0 and placed < self.SUPPORT_MAX_PLACEMENTS:
            best = None
            best_score = -1e9

            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                if not self.can_place_track(x, y):
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability > self.SUPPORT_MAX_INSTABILITY:
                    continue
                c = self._empty_cell_place_cost(x, y)
                if c > budget or c > self.SUPPORT_MAX_CELL_COST:
                    continue

                s = self._support_path_placement_score(x, y, c, end_a, end_b)
                if s > best_score:
                    best_score = s
                    best = (x, y, c)

            if best is None:
                break

            x, y, c = best
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(c)
            placed += 1
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

        return actions

    def _targeted_densify(self, budget: int, focus: Set[Tuple[int, int]]) -> List[str]:
        if not self.TARGETED_DENSE_ENABLED or budget <= 0 or not focus:
            return []

        actions: List[str] = []
        placed = 0

        my_track_positions = [pos for pos, owner in self.cell_tracks.items() if owner == self.my_id]

        frontier: Set[Tuple[int, int]] = set()
        for fx, fy in focus:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        def dist_to_focus(x: int, y: int) -> int:
            # Focus sets are small; linear scan is fine.
            best = 10**9
            for fx, fy in focus:
                d = abs(x - fx) + abs(y - fy)
                if d < best:
                    best = d
                    if best <= 1:
                        break
            return int(best)

        while budget > 0 and placed < self.TARGETED_DENSE_MAX_PLACEMENTS:
            best = None
            best_score = -1e9

            for x, y in list(frontier):
                if not self.can_place_track(x, y):
                    continue
                c = self._empty_cell_place_cost(x, y)
                if c > budget or c > self.TARGETED_DENSE_MAX_CELL_COST:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                if self.region_instability.get(rid, 0) > self.TARGETED_DENSE_MAX_INSTABILITY:
                    continue
                if self.DENSE_AVOID_OPP_REGIONS and rid in self.opp_track_regions:
                    continue

                if self.TARGETED_DENSE_MAX_DIST_TO_MY_TRACK >= 0 and my_track_positions:
                    near_my = False
                    for tx, ty in my_track_positions:
                        if abs(tx - x) + abs(ty - y) <= self.TARGETED_DENSE_MAX_DIST_TO_MY_TRACK:
                            near_my = True
                            break
                    if not near_my:
                        continue

                # Score: prefer cheap, near focus, and not adjacent to opponent.
                instability = self.region_instability.get(rid, 0)
                d = dist_to_focus(x, y)

                adj_my = 0
                adj_opp = 0
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    if (nx, ny) in self.town_cells:
                        adj_my += 1
                        continue
                    owner = self.cell_tracks.get((nx, ny))
                    if owner == self.my_id:
                        adj_my += 1
                    elif owner == self.opp_id:
                        adj_opp += 1

                s = 0.0
                s += (3 - c) * 9.0
                s += max(0.0, 8.0 - float(d))
                s += adj_my * 3.5
                s -= adj_opp * 4.5
                s += (self.TARGETED_DENSE_MAX_INSTABILITY - instability) * 2.0

                if s > best_score:
                    best_score = s
                    best = (x, y, c)

            if best is None:
                break

            x, y, c = best
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(c)
            placed += 1
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        return actions

    # -------------------- Disruption (V20 baseline + V21 override) --------------------

    def _get_disruption_target_v20(self) -> Optional[int]:
        turns_left = self.TOTAL_TURNS - self.turn

        anti_dense = self._opp_dense_potential_by_region() if self.ANTI_DENSE_ENABLED else {}

        def net_score(region_id: int) -> float:
            opp_w = self.active_weight_by_region_opp.get(region_id, 0.0)
            my_w = self.active_weight_by_region_my.get(region_id, 0.0)
            neu_w = self.active_weight_by_region_neutral.get(region_id, 0.0)

            instability = self.region_instability.get(region_id, 0)
            hits_to_ink = self.INSTABILITY_TO_INK - instability

            base = (opp_w - self.SELF_HARM_ALPHA * my_w + 0.15 * neu_w)
            if anti_dense:
                base += float(anti_dense.get(region_id, 0.0)) * float(self.ANTI_DENSE_ALPHA)
            if base <= 0:
                return -1e9

            score = (base * turns_left) / max(hits_to_ink, 1)
            if hits_to_ink == 1:
                score *= 2.0
            elif hits_to_ink >= 2:
                score *= 0.65
            return float(score)

        immediate: List[Tuple[int, float]] = []
        for region_id in set(
            list(self.active_weight_by_region_opp.keys())
            + list(self.active_weight_by_region_my.keys())
            + list(self.active_weight_by_region_neutral.keys())
            + (list(anti_dense.keys()) if anti_dense else [])
        ):
            if region_id in self.region_inked or region_id in self.town_regions:
                continue
            instability = self.region_instability.get(region_id, 0)
            if instability >= self.INSTABILITY_TO_INK:
                continue
            if (self.INSTABILITY_TO_INK - instability) == 1:
                s = net_score(region_id)
                if s > 0:
                    immediate.append((region_id, s))
        if immediate:
            return max(immediate, key=lambda t: t[1])[0]

        candidates: Dict[int, float] = {}

        for region_id in set(list(self.active_weight_by_region_opp.keys()) + list(self.opp_track_regions) + (list(anti_dense.keys()) if anti_dense else [])):
            if region_id in self.region_inked or region_id in self.town_regions:
                continue
            instability = self.region_instability.get(region_id, 0)
            if instability >= self.INSTABILITY_TO_INK:
                continue

            score = net_score(region_id)
            if score > 0:
                candidates[region_id] = max(candidates.get(region_id, 0.0), score)

        if not candidates:
            for region_id in self.opp_track_regions:
                if region_id in self.region_inked or region_id in self.town_regions:
                    continue
                instability = self.region_instability.get(region_id, 0)
                if instability >= self.INSTABILITY_TO_INK:
                    continue

                my_w = self.active_weight_by_region_my.get(region_id, 0.0)
                if my_w > 0:
                    continue

                opp_tracks = sum(
                    1
                    for (x, y), owner in self.cell_tracks.items()
                    if owner == self.opp_id and self.region_ids[y][x] == region_id
                )
                if opp_tracks <= 0:
                    continue

                hits_to_ink = self.INSTABILITY_TO_INK - instability
                candidates[region_id] = (opp_tracks * turns_left * 0.25) / max(hits_to_ink, 1)

        if not candidates:
            return None

        return max(candidates.items(), key=lambda kv: kv[1])[0]

    def _opp_dense_potential_by_region(self) -> Dict[int, float]:
        # Estimate how much the opponent can densify next turns in each region by
        # counting cheap, placeable empty frontier cells adjacent to opponent tracks.
        # Higher => better disrupt target to slow V24-style densification.
        if (self.my_score - self.opp_score) >= self.ANTI_DENSE_IGNORE_IF_AHEAD_BY:
            return {}

        potentials: Dict[int, float] = defaultdict(float)
        seen_cells: Set[Tuple[int, int]] = set()

        for (x, y), owner in self.cell_tracks.items():
            if owner != self.opp_id:
                continue
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in self.town_cells:
                    continue
                if (nx, ny) in seen_cells:
                    continue
                if not self.can_place_track(nx, ny):
                    continue
                cost = self._empty_cell_place_cost(nx, ny)
                if cost > self.ANTI_DENSE_MAX_CELL_COST:
                    continue

                rid = self.region_ids[ny][nx]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability >= self.INSTABILITY_TO_INK:
                    continue

                seen_cells.add((nx, ny))
                # Prefer blocking very cheap placements.
                potentials[rid] += float(4 - cost)

        return potentials

    def _block_opponent_densify(self, budget: int) -> List[str]:
        if budget <= 0 or not self.ANTI_DENSE_ENABLED:
            return []
        if (self.my_score - self.opp_score) >= self.ANTI_DENSE_IGNORE_IF_AHEAD_BY:
            return []

        potentials = self._opp_dense_potential_by_region()
        if not potentials:
            return []

        # Consider only top few regions to keep this cheap.
        top_regions = sorted(potentials.items(), key=lambda kv: -kv[1])[:6]

        max_places = min(self.ANTI_DENSE_BLOCK_PLACEMENTS, max(0, budget))
        if max_places <= 0:
            return []

        frontier: List[Tuple[float, int, int, int, int]] = []
        # (score, cost, y, x, region)
        for rid, pot in top_regions:
            if rid in self.region_inked or rid in self.town_regions:
                continue
            instability = self.region_instability.get(rid, 0)
            if instability >= self.INSTABILITY_TO_INK or instability > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                continue

            # Enumerate cheap, placeable empty cells adjacent to opponent tracks in this region.
            for (x, y), owner in self.cell_tracks.items():
                if owner != self.opp_id:
                    continue
                if self.region_ids[y][x] != rid:
                    continue
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    if (nx, ny) in self.town_cells:
                        continue
                    if not self.can_place_track(nx, ny):
                        continue
                    cost = self._empty_cell_place_cost(nx, ny)
                    if cost > budget or cost > self.ANTI_DENSE_MAX_CELL_COST:
                        continue
                    rr = self.region_ids[ny][nx]
                    if rr != rid or rr == -1 or rr in self.region_inked or rr in self.town_regions:
                        continue
                    inst2 = self.region_instability.get(rr, 0)
                    if inst2 >= self.INSTABILITY_TO_INK or inst2 > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                        continue

                    # Prefer blocking cheaper cells in higher-potential regions.
                    s = float(pot) * 2.0 + float(4 - cost)
                    frontier.append((s, cost, ny, nx, rid))

        # Region-poison candidates: placing a single cheap track anywhere in the region
        # makes V24 avoid densifying there (it avoids opponent-track regions).
        poison: List[Tuple[float, int, int, int, int]] = []
        if self.ANTI_DENSE_POISON_REGION:
            for rid, pot in top_regions:
                if rid in self.region_inked or rid in self.town_regions:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability >= self.INSTABILITY_TO_INK or instability > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                    continue

                best_cell = None
                best_cost = 10**9
                # Keep poison local: look for a cheap cell within a small radius of the
                # opponent's frontier in this region.
                frontier_seeds: Set[Tuple[int, int]] = set()
                for (ox, oy), owner in self.cell_tracks.items():
                    if owner != self.opp_id:
                        continue
                    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                        sx, sy = ox + dx, oy + dy
                        if not (0 <= sx < self.width and 0 <= sy < self.height):
                            continue
                        if (sx, sy) in self.town_cells:
                            continue
                        if self.region_ids[sy][sx] != rid:
                            continue
                        if not self.can_place_track(sx, sy):
                            continue
                        frontier_seeds.add((sx, sy))

                if not frontier_seeds:
                    continue

                r = int(self.ANTI_DENSE_POISON_MAX_DIST_TO_FRONTIER)
                scanned: Set[Tuple[int, int]] = set()
                for sx, sy in frontier_seeds:
                    for dx in range(-r, r + 1):
                        rem = r - abs(dx)
                        for dy in range(-rem, rem + 1):
                            x, y = sx + dx, sy + dy
                            if not (0 <= x < self.width and 0 <= y < self.height):
                                continue
                            if (x, y) in scanned or (x, y) in self.town_cells:
                                continue
                            scanned.add((x, y))
                            if self.region_ids[y][x] != rid:
                                continue
                            if not self.can_place_track(x, y):
                                continue
                            c = self._empty_cell_place_cost(x, y)
                            if c < best_cost:
                                best_cost = c
                                best_cell = (x, y)
                                if best_cost <= 1:
                                    break
                        if best_cost <= 1:
                            break
                    if best_cost <= 1:
                        break

                if best_cell is None:
                    continue
                if best_cost > budget or best_cost > self.ANTI_DENSE_POISON_MAX_CELL_COST:
                    continue

                # Prefer poisoning very cheap cells in high-potential regions.
                x, y = best_cell
                s = float(pot) * 3.0 + float(5 - best_cost)
                poison.append((s, int(best_cost), y, x, rid))

        if not frontier and not poison:
            return []

        frontier.sort(reverse=True)
        poison.sort(reverse=True)
        actions: List[str] = []
        placed = 0
        used: Set[Tuple[int, int]] = set()

        # Mix both sources: try best poison first, then best adjacency block.
        merged = poison[:]
        merged.extend(frontier)
        merged.sort(reverse=True)

        for s, cost, ny, nx, rid in merged:
            if placed >= max_places:
                break
            if budget < cost:
                continue
            if (nx, ny) in used:
                continue
            if not self.can_place_track(nx, ny):
                continue

            actions.append(f"PLACE_TRACKS {nx} {ny}")
            budget -= int(cost)
            placed += 1
            used.add((nx, ny))
            self.cell_tracks[(nx, ny)] = self.my_id
            if rid != -1:
                self.my_track_regions.add(rid)

        return actions

    def get_disruption_target(self, best_next_path: Optional[List[Tuple[int, int]]] = None) -> Optional[int]:
        # If we recently committed to ink a region (1-2 hits away), keep hitting it
        # briefly to avoid wasting ink by switching every turn.
        if self._disrupt_lock_region is not None and self.turn <= self._disrupt_lock_until_turn:
            rid = self._disrupt_lock_region
            if rid not in self.region_inked and rid not in self.town_regions:
                inst = self.region_instability.get(rid, 0)
                hits = self.INSTABILITY_TO_INK - inst
                if 1 <= hits <= 2:
                    return rid
            self._disrupt_lock_region = None
            self._disrupt_lock_until_turn = -1

        v20_target = self._get_disruption_target_v20()

        # When comfortably ahead, avoid "creative" overrides; keep the reliable baseline.
        score_delta = self.my_score - self.opp_score
        if score_delta >= self.PRED_DISABLE_OVERRIDE_WHEN_AHEAD:
            return v20_target

        base_hits = None
        if v20_target is not None:
            base_instability = self.region_instability.get(v20_target, 0)
            base_hits = self.INSTABILITY_TO_INK - base_instability
            # If baseline is already an immediate 1-hit ink and we're not behind,
            # don't override it.
            if base_hits == 1 and score_delta >= -self.PRED_ALLOW_OVERRIDE_IF_BASE_1HIT_BEHIND:
                return v20_target

        avoid: Set[int] = set()
        if self.active_build_path:
            for x, y in self.active_build_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)

        if best_next_path:
            for x, y in best_next_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)

        # If opponent just placed tracks in a region, it is often the strongest
        # immediate disruption target (especially vs strong build-focused bots).
        allow_2hit = (-score_delta) >= self.PRED_OVERRIDE_BEHIND_FOR_2HIT
        if self._opp_recent_build_region is not None and (self.turn - self._opp_recent_build_turn) <= 2:
            rid = int(self._opp_recent_build_region)
            if rid not in self.region_inked and rid not in self.town_regions and rid not in avoid:
                inst = self.region_instability.get(rid, 0)
                if inst < self.INSTABILITY_TO_INK:
                    hits = self.INSTABILITY_TO_INK - inst
                    if hits == 1 or (allow_2hit and hits == 2 and base_hits != 1):
                        if rid in self.opp_track_regions or self.active_weight_by_region_opp.get(rid, 0.0) > 0.0:
                            if self.active_weight_by_region_my.get(rid, 0.0) <= 0.8:
                                return rid

        pred_opp = self._predict_opponent_region_weights()

        best_region: Optional[int] = None
        best_w = 0.0

        # (allow_2hit computed above)

        for region_id, w in pred_opp.items():
            if w <= 0:
                continue
            if region_id in self.region_inked or region_id in self.town_regions or region_id in avoid:
                continue
            instability = self.region_instability.get(region_id, 0)
            if instability >= self.INSTABILITY_TO_INK:
                continue
            hits = (self.INSTABILITY_TO_INK - instability)
            if hits != 1:
                if not (allow_2hit and hits == 2):
                    continue
            # Never override a baseline 1-hit ink with a 2-hit ink.
            if base_hits == 1 and hits == 2:
                continue

            if region_id not in self.opp_track_regions and region_id not in self.active_weight_by_region_opp:
                continue
            if self.active_weight_by_region_my.get(region_id, 0.0) > 0.8:
                continue

            if w > best_w:
                best_w = float(w)
                best_region = region_id

        if best_region is not None:
            base_w = float(pred_opp.get(v20_target, 0.0)) if v20_target is not None else 0.0
            required = max(self.PRED_OVERRIDE_THRESHOLD, base_w + self.PRED_COMPARE_MARGIN)
            if base_hits == 1:
                required = max(required, base_w + self.PRED_COMPARE_MARGIN + self.PRED_EXTRA_MARGIN_IF_BASE_1HIT)
            if best_w >= required:
                return best_region

        return v20_target

    # -------------------- Candidates --------------------

    def get_connection_candidates(self) -> List[Tuple[int, int, List[Tuple[int, int]], float]]:
        candidates = []

        for town in self.towns:
            for dest_id in town.desired:
                conn_id = (min(town.id, dest_id), max(town.id, dest_id))
                if conn_id in self.completed_connections:
                    continue
                if dest_id not in self.town_by_id:
                    continue

                dest = self.town_by_id[dest_id]
                # Candidates must reflect the current board (tracks/ink/instability);
                # stale cached paths can mis-rank EVs and cause bad switches.
                path = self.find_path(town.x, town.y, dest.x, dest.y, use_cache=False)
                if not path or len(path) < 2:
                    continue

                if not self._path_has_affordable_step(path):
                    # If the cheapest path runs through opponent tracks (unplaceable),
                    # retry while blocking opponent tracks to find a buildable route.
                    if any(self.cell_tracks.get((x, y)) == self.opp_id for (x, y) in path):
                        path2 = self.find_path(
                            town.x,
                            town.y,
                            dest.x,
                            dest.y,
                            use_cache=False,
                            block_opp_tracks=True,
                        )
                        if path2 and len(path2) >= 2 and self._path_has_affordable_step(path2):
                            path = path2
                        else:
                            continue
                    else:
                        continue

                if not self._path_has_affordable_step(path):
                    continue

                ev = self.calc_connection_ev(path, from_id=town.id, to_id=dest_id)
                if ev > 0:
                    candidates.append((town.id, dest_id, path, ev))

        candidates.sort(key=lambda x: -x[3])
        return candidates


    def _detect_opponent_behavior(self):
        if self.turn < 5:
            return
        if self.detected_opp_bias is not None and self.turn < 15:
            return

        towns_with_opp_tracks = set()
        for (x, y), owner in self.cell_tracks.items():
            if owner != self.opp_id:
                continue
            min_dist = 10**9
            nearest = None
            for town in self.towns:
                dist = abs(town.x - x) + abs(town.y - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = town.id
            if nearest is not None and min_dist < 10:
                towns_with_opp_tracks.add(nearest)

        if len(towns_with_opp_tracks) < 2:
            return

        avg_opp_town_id = sum(towns_with_opp_tracks) / len(towns_with_opp_tracks)
        avg_all_town_id = sum(t.id for t in self.towns) / len(self.towns)

        if avg_opp_town_id > avg_all_town_id * 1.15:
            if self.detected_opp_bias != 'high':
                self.detected_opp_bias = 'high'
                self.my_bias_multiplier = -5.0
        elif avg_opp_town_id < avg_all_town_id * 0.85:
            if self.detected_opp_bias != 'low':
                self.detected_opp_bias = 'low'
                self.my_bias_multiplier = 6.0
        else:
            if self.detected_opp_bias != 'neutral':
                self.detected_opp_bias = 'neutral'
                self.my_bias_multiplier = -3.0

    # -------------------- Action selection --------------------

    # --- V26: Unified planning (logic upgrade) ---
    PLAN_TOP_K = 18

    # Denial gating: only spend on choke / anti-dense when threat is real.
    DENY_MIN_OPP_THREAT_FOR_CHOKE = 18.0
    DENY_MIN_POTENTIAL_SUM_FOR_ANTI_DENSE = 5.0

    def _choose_primary_path(
        self,
        continue_ev: float,
        best_new: Optional[Tuple[int, int, List[Tuple[int, int]], float]],
    ) -> Optional[List[Tuple[int, int]]]:
        best_new_ev = best_new[3] if best_new else 0.0
        best_new_path = best_new[2] if best_new else None

        if not (self.active_build and self.active_build_path):
            if best_new_path:
                from_id, to_id, path, _ = best_new
                self.active_build = (from_id, to_id)
                self.active_build_path = path
                self._trace(
                    f"turn={self.turn} seed={self._trace_seed} as={self._trace_as} choose=new init {from_id}->{to_id} new_ev={best_new_ev:.2f}"
                )
                return path
            return None

        # Refresh current active path if possible.
        from_id, to_id = self.active_build
        from_town = self.town_by_id.get(from_id)
        to_town = self.town_by_id.get(to_id)
        if from_town and to_town:
            refreshed = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
            if refreshed:
                old_path = self.active_build_path
                if not old_path:
                    self.active_build_path = refreshed
                else:
                    old_can = self._path_has_affordable_step(old_path)
                    new_can = self._path_has_affordable_step(refreshed)
                    old_cost = self.calc_remaining_cost(old_path)
                    new_cost = self.calc_remaining_cost(refreshed)

                    # Avoid oscillating between near-equivalent routes: only switch
                    # the execution path when it materially improves remaining cost
                    # or when the previous path can't advance this turn.
                    if (not old_can and new_can) or (new_cost + 1 < old_cost):
                        self.active_build_path = refreshed

        active_path = self.active_build_path
        if not active_path:
            if best_new_path:
                from_id, to_id, path, _ = best_new
                self.active_build = (from_id, to_id)
                self.active_build_path = path
                return path
            return None

        # Progress-aware switch threshold.
        my_cells, _, _, _ = self._count_track_owners_on_path(active_path)
        path_len = max(len(active_path) - 1, 1)
        progress = float(my_cells) / float(path_len)
        switch_threshold = 1.20 + 0.90 * progress

        # As P1 (second player), path churn tends to be punished more; be slightly
        # more conservative about switching even if a new EV looks better.
        if self.my_id == 1:
            switch_threshold *= 1.10
            if (self.my_score - self.opp_score) < 0:
                switch_threshold *= 0.97

        if best_new_path and best_new_ev > continue_ev * switch_threshold:
            from_id, to_id, path, _ = best_new
            cur_from, cur_to = self.active_build
            self.active_build = (from_id, to_id)
            self.active_build_path = path
            self._trace(
                f"turn={self.turn} seed={self._trace_seed} as={self._trace_as} switch {cur_from}->{cur_to} -> {from_id}->{to_id} cont_ev={continue_ev:.2f} new_ev={best_new_ev:.2f} progress={progress:.2f} thr={switch_threshold:.2f}"
            )
            return path

        return active_path

    def _first_affordable_step_from_end(
        self, path: List[Tuple[int, int]], from_start: bool, budget: int
    ) -> Optional[Tuple[int, int, int]]:
        if not path or budget <= 0:
            return None

        it = path[1:] if from_start else list(reversed(path[:-1]))
        for x, y in it:
            if (x, y) in self.town_cells:
                continue
            if not self.can_place_track(x, y):
                continue
            c = self._empty_cell_place_cost(x, y)
            if c <= budget:
                return (x, y, int(c))
        return None

    def _plan_track_placements(
        self,
        budget: int,
        primary_path: Optional[List[Tuple[int, int]]],
        best_new_path: Optional[List[Tuple[int, int]]],
        best_new_ev: float,
        continue_ev: float,
        allow_build: bool = True,
    ) -> List[str]:
        if budget <= 0:
            return []

        opp_threat = self._opp_threat_value()
        score_delta = self.my_score - self.opp_score

        denial_boost = 1.0
        if score_delta < 0 or opp_threat > 40.0:
            denial_boost = 1.35
        if score_delta < -120:
            denial_boost = 1.55

        build_boost = 1.0
        if best_new_ev > 0 or continue_ev > 0:
            build_boost = 1.20

        densify_boost = 1.0
        if score_delta >= 60 and opp_threat <= 25.0:
            densify_boost = 1.15

        candidates: List[Tuple[float, int, int, int, str]] = []
        used_cells: Set[Tuple[int, int]] = set()

        def add_candidate(score: float, cost: int, x: int, y: int, tag: str) -> None:
            if cost <= 0 or cost > budget:
                return
            if (x, y) in used_cells:
                return
            used_cells.add((x, y))
            candidates.append((float(score), int(cost), int(x), int(y), tag))

        path_ev = max(float(best_new_ev), float(continue_ev))
        if allow_build and primary_path and len(primary_path) >= 2:
            a = self._first_affordable_step_from_end(primary_path, from_start=True, budget=budget)
            b = self._first_affordable_step_from_end(primary_path, from_start=False, budget=budget)
            if a:
                x, y, c = a
                add_candidate((220.0 + 0.06 * path_ev) * build_boost, c, x, y, "advance")
            if b:
                x, y, c = b
                add_candidate((215.0 + 0.06 * path_ev) * build_boost, c, x, y, "advance")

            end_a = primary_path[0]
            end_b = primary_path[-1]
            for x, y in primary_path:
                if (x, y) in self.town_cells:
                    continue
                if not self.can_place_track(x, y):
                    continue

                # Avoid wasting tempo on mid-path filler: only consider cells near endpoints
                # or that immediately attach to our existing network.
                d_end = min(
                    abs(x - end_a[0]) + abs(y - end_a[1]),
                    abs(x - end_b[0]) + abs(y - end_b[1]),
                )
                if d_end > 2:
                    adj_my = False
                    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in self.town_cells:
                            adj_my = True
                            break
                        if self.cell_tracks.get((nx, ny)) == self.my_id:
                            adj_my = True
                            break
                    if not adj_my:
                        continue

                c = self._empty_cell_place_cost(x, y)
                if c > budget or c > self.SUPPORT_MAX_CELL_COST:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                inst = self.region_instability.get(rid, 0)
                if inst > self.SUPPORT_MAX_INSTABILITY:
                    continue
                s = self._support_path_placement_score(x, y, c, end_a, end_b)
                add_candidate((60.0 + 10.0 * s) * build_boost, int(c), x, y, "support")

        if (
            self.CHOKE_ENABLED
            and budget > 0
            and self.turn >= self.CHOKE_START_TURN
            and opp_threat >= self.DENY_MIN_OPP_THREAT_FOR_CHOKE
        ):
            if (self.my_score - self.opp_score) < self.CHOKE_DISABLE_WHEN_AHEAD:
                top = self._predict_opponent_scored_paths()[: self.CHOKE_TOP_K_PATHS]
                for v, path in top:
                    for x, y in path:
                        if (x, y) in self.town_cells:
                            continue
                        if not self.can_place_track(x, y):
                            continue
                        c = self._empty_cell_place_cost(x, y)
                        if c > budget or c > self.CHOKE_MAX_CELL_COST:
                            continue
                        rid = self.region_ids[y][x]
                        if rid == -1 or rid in self.town_regions or rid in self.region_inked:
                            continue
                        inst = self.region_instability.get(rid, 0)
                        if inst >= self.INSTABILITY_TO_INK:
                            continue
                        empty_cost = float(c + inst * 2)
                        delta = float(self.MY_TRACK_BLOCKING_PENALTY) - empty_cost
                        if delta < self.CHOKE_MIN_DELTA_COST:
                            continue
                        choke_score = float(v) * delta / float(max(c, 1))
                        choke_score = min(choke_score, 10.0)
                        add_candidate((30.0 * choke_score) * denial_boost, int(c), x, y, "choke")

        if (
            self.ANTI_DENSE_ENABLED
            and budget > 0
            and (self.my_score - self.opp_score) < self.ANTI_DENSE_IGNORE_IF_AHEAD_BY
        ):
            potentials = self._opp_dense_potential_by_region()
            if potentials and sum(potentials.values()) >= self.DENY_MIN_POTENTIAL_SUM_FOR_ANTI_DENSE:
                top_regions = sorted(potentials.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
                for rid, pot in top_regions:
                    if rid in self.region_inked or rid in self.town_regions:
                        continue
                    inst = self.region_instability.get(rid, 0)
                    if inst >= self.INSTABILITY_TO_INK or inst > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                        continue
                    for (ox, oy), owner in self.cell_tracks.items():
                        if owner != self.opp_id:
                            continue
                        if self.region_ids[oy][ox] != rid:
                            continue
                        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                            nx, ny = ox + dx, oy + dy
                            if not (0 <= nx < self.width and 0 <= ny < self.height):
                                continue
                            if (nx, ny) in self.town_cells:
                                continue
                            if not self.can_place_track(nx, ny):
                                continue
                            c = self._empty_cell_place_cost(nx, ny)
                            if c > budget or c > self.ANTI_DENSE_MAX_CELL_COST:
                                continue
                            rr = self.region_ids[ny][nx]
                            if rr != rid or rr == -1 or rr in self.region_inked or rr in self.town_regions:
                                continue
                            inst2 = self.region_instability.get(rr, 0)
                            if inst2 >= self.INSTABILITY_TO_INK or inst2 > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                                continue
                            # Cap region potential contribution to avoid over-spending on denial.
                            pot_c = min(float(pot), 10.0)
                            s = (pot_c * 6.0 + float(4 - c) * 2.5) * float(self.ANTI_DENSE_ALPHA)
                            add_candidate(s * denial_boost, int(c), nx, ny, "anti_dense")

        anchors = set(self.town_cells)
        anchors.update([pos for pos, owner in self.cell_tracks.items() if owner == self.my_id])
        frontier: Set[Tuple[int, int]] = set()
        for ax, ay in anchors:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        for x, y in frontier:
            if not self.can_place_track(x, y):
                continue
            c = self._empty_cell_place_cost(x, y)
            if c > budget:
                continue
            rid = self.region_ids[y][x]
            if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                continue
            if self.region_instability.get(rid, 0) > self.DENSE_MAX_INSTABILITY:
                continue
            if self.DENSE_AVOID_OPP_REGIONS and rid in self.opp_track_regions:
                continue
            s = self._dense_candidate_score(x, y)
            add_candidate((5.5 * s) * densify_boost, int(c), x, y, "densify")

        if not candidates:
            return []

        candidates.sort(key=lambda t: (-t[0], t[1], t[4], t[3], t[2]))
        candidates = candidates[: self.PLAN_TOP_K]

        best_score = -1e18
        best_set: Tuple[int, ...] = ()

        coord = [(x, y) for _, _, x, y, _ in candidates]

        def synergy(idxs: Tuple[int, ...]) -> float:
            if len(idxs) < 2:
                return 0.0
            bonus = 0.0
            for i in range(len(idxs)):
                x1, y1 = coord[idxs[i]]
                for j in range(i + 1, len(idxs)):
                    x2, y2 = coord[idxs[j]]
                    if abs(x1 - x2) + abs(y1 - y2) == 1:
                        bonus += 7.0
            return bonus

        n = len(candidates)
        for i in range(n):
            s1, c1, *_ = candidates[i]
            if c1 > budget:
                continue
            if s1 > best_score:
                best_score = s1
                best_set = (i,)
            for j in range(i + 1, n):
                s2, c2, *_ = candidates[j]
                if c1 + c2 > budget:
                    continue
                score2 = s1 + s2 + synergy((i, j))
                if score2 > best_score:
                    best_score = score2
                    best_set = (i, j)
                for k in range(j + 1, n):
                    s3, c3, *_ = candidates[k]
                    if c1 + c2 + c3 > budget:
                        continue
                    score3 = s1 + s2 + s3 + synergy((i, j, k))
                    if score3 > best_score:
                        best_score = score3
                        best_set = (i, j, k)

        if not best_set:
            return []

        chosen = [candidates[i] for i in best_set]
        chosen.sort(key=lambda t: (t[1], t[3], t[2], t[4]))

        actions: List[str] = []
        spent = 0
        for _, cost, x, y, _tag in chosen:
            if spent + cost > budget:
                continue
            if not self.can_place_track(x, y):
                continue
            actions.append(f"PLACE_TRACKS {x} {y}")
            spent += int(cost)
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

        return actions

    def get_action(self) -> str:
        actions: List[str] = []

        continue_ev = self.calc_continue_ev()
        new_candidates = self.get_connection_candidates()
        best_new = new_candidates[0] if new_candidates else None
        best_new_ev = best_new[3] if best_new else 0.0
        best_new_path = best_new[2] if best_new else None

        # Step 1: Advance a real connection first (tempo). This restores V25's
        # reliable scoring behavior and prevents V26 from over-spending on filler.
        primary_path = self._choose_primary_path(float(continue_ev), best_new)

        if self._trace_enabled:
            opp_threat = self._opp_threat_value()
            contested_regions = len(self.opp_track_regions & self.my_track_regions)
            inked_regions = len(self.region_inked)
            self._trace(
                f"turn={self.turn} seed={self._trace_seed} as={self._trace_as} pre cont_ev={float(continue_ev):.2f} best_new_ev={float(best_new_ev):.2f} active={self.active_build} opp_threat={opp_threat:.2f} contested={contested_regions} inked={inked_regions}"
            )

        if self.active_build and self.active_build_path:
            from_id, to_id = self.active_build
            from_town = self.town_by_id.get(from_id)
            to_town = self.town_by_id.get(to_id)
            if from_town and to_town:
                actions.extend(self.build_path_two_ended_stable(from_town, to_town))

        if not actions:
            # If the current active build can't place anything this turn, try other
            # high-EV connection candidates instead of immediately falling back to
            # planner-only spending.
            def _conn_key(a: int, b: int) -> Tuple[int, int]:
                return (min(a, b), max(a, b))

            cur_key: Optional[Tuple[int, int]] = None
            if self.active_build:
                cur_key = _conn_key(int(self.active_build[0]), int(self.active_build[1]))

            for from_id2, to_id2, path2, _ev2 in new_candidates:
                if cur_key is not None and _conn_key(int(from_id2), int(to_id2)) == cur_key:
                    continue
                if not path2 or len(path2) < 2:
                    continue
                if not self._path_has_affordable_step(path2):
                    continue

                self.active_build = (int(from_id2), int(to_id2))
                self.active_build_path = path2
                ft2 = self.town_by_id.get(int(from_id2))
                tt2 = self.town_by_id.get(int(to_id2))
                if not ft2 or not tt2:
                    continue
                actions.extend(self.build_path_two_ended_stable(ft2, tt2))
                if actions:
                    primary_path = path2
                    break

        if not actions and best_new:
            # No active build and no feasible candidate (or candidates exhausted):
            # keep the prior behavior of attempting best_new.
            from_id, to_id, path, _ = best_new
            if path and len(path) >= 2 and self._path_has_affordable_step(path):
                self.active_build = (int(from_id), int(to_id))
                self.active_build_path = path
                from_town = self.town_by_id.get(int(from_id))
                to_town = self.town_by_id.get(int(to_id))
                if from_town and to_town:
                    actions.extend(self.build_path_two_ended_stable(from_town, to_town))

        # If we failed to place any track on a connection this turn, fall back to
        # the planner with allow_build=True so we don't spend the entire turn on
        # denial/densify while making zero progress.
        built_any = any(a.startswith("PLACE_TRACKS") for a in actions)

        # Step 2: Use remaining budget for denial/densify (V26 planner), then keep
        # V25's anti-dense poison and gating as a safety net.
        budget_left = self._remaining_budget_after_actions(actions)
        if budget_left > 0:
            if not built_any:
                actions.extend(
                    self._plan_track_placements(
                        budget=budget_left,
                        primary_path=primary_path,
                        best_new_path=best_new_path,
                        best_new_ev=float(best_new_ev),
                        continue_ev=float(continue_ev),
                        allow_build=True,
                    )
                )
                budget_left = self._remaining_budget_after_actions(actions)
            else:
                # If we already made progress on a real connection this turn,
                # prefer using any leftover budget to strengthen that path (tempo)
                # before doing denial/densify. This mirrors V25's reliable behavior
                # and helps on seeds where pure denial spending falls behind.
                support_path = self.active_build_path if self.active_build_path else best_new_path
                if support_path:
                    actions.extend(self._support_best_path(budget_left, support_path))
                    budget_left = self._remaining_budget_after_actions(actions)

                opp_threat = self._opp_threat_value()
                score_delta = self.my_score - self.opp_score

                # If the planner has no denial candidates (only densify), reserve a point
                # so V25's cheap anti-dense block still fires reliably.
                reserve_for_block = 0
                if (
                    self.ANTI_DENSE_ENABLED
                    and budget_left >= 1
                    and score_delta < self.ANTI_DENSE_IGNORE_IF_AHEAD_BY
                    and any(owner == self.opp_id for owner in self.cell_tracks.values())
                ):
                    potentials = self._opp_dense_potential_by_region()
                    pot_sum = float(sum(potentials.values())) if potentials else 0.0
                    if (
                        opp_threat < self.DENY_MIN_OPP_THREAT_FOR_CHOKE
                        and pot_sum < self.DENY_MIN_POTENTIAL_SUM_FOR_ANTI_DENSE
                    ):
                        reserve_for_block = 1

                plan_budget = budget_left - reserve_for_block
                if plan_budget > 0:
                    actions.extend(
                        self._plan_track_placements(
                            budget=plan_budget,
                            primary_path=primary_path,
                            best_new_path=best_new_path,
                            best_new_ev=float(best_new_ev),
                            continue_ev=float(continue_ev),
                            allow_build=False,
                        )
                    )
                    budget_left = self._remaining_budget_after_actions(actions)

        if budget_left > 0:
            actions.extend(self._block_opponent_densify(budget_left))
            budget_left = self._remaining_budget_after_actions(actions)

        if budget_left > 0:
            opp_threat = self._opp_threat_value()
            score_delta = self.my_score - self.opp_score

            if (
                score_delta >= self.DENSIFY_ONLY_WHEN_AHEAD_BY
                and opp_threat <= self.DENSIFY_ONLY_WHEN_OPP_THREAT_LEQ
                and best_new_ev <= self.DENSIFY_ONLY_WHEN_BEST_NEW_EV_LEQ
            ):
                actions.extend(self._densify_network(budget_left))
            else:
                focus: Set[Tuple[int, int]] = set()

                def add_neighborhood(p: Tuple[int, int], r: int) -> None:
                    px, py = p
                    for dx in range(-r, r + 1):
                        rem = r - abs(dx)
                        for dy in range(-rem, rem + 1):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                focus.add((nx, ny))

                if self.active_build_path and len(self.active_build_path) >= 2:
                    add_neighborhood(self.active_build_path[0], 2)
                    add_neighborhood(self.active_build_path[-1], 2)
                elif best_new_path and len(best_new_path) >= 2:
                    add_neighborhood(best_new_path[0], 2)
                    add_neighborhood(best_new_path[-1], 2)
                actions.extend(self._targeted_densify(budget_left, focus))

        if self.turn >= 1:
            target = self.get_disruption_target(best_next_path=best_new_path)

            # Optional oracle forcing (for labeling): override only the disrupt decision.
            forced = False
            forced_skip = False
            force_target: Optional[int] = None
            try:
                if str(self._force_disrupt_turn).strip() and int(self._force_disrupt_turn) == int(self.turn):
                    s = str(self._force_disrupt_region).strip().lower()
                    if s in ("", "none", "null", "skip"):
                        forced_skip = True
                        forced = True
                    else:
                        force_target = int(s)
                        forced = True
            except Exception:
                force_target = None
                forced = False
                forced_skip = False

            if forced:
                if forced_skip:
                    target = None
                elif force_target is not None:
                    if force_target not in self.region_inked and force_target not in self.town_regions:
                        inst_f = int(self.region_instability.get(int(force_target), 0))
                        if inst_f < self.INSTABILITY_TO_INK:
                            target = int(force_target)

            if self._dataset_enabled and str(self._dataset_kind).strip().lower() == "disrupt":
                cand = self._get_disruption_candidates_for_dataset(best_next_path=best_new_path, limit=12)
                if target is not None and all(int(c["rid"]) != int(target) for c in cand):
                    extra = self._disrupt_candidate_features(int(target), int(self.TOTAL_TURNS - self.turn))
                    if extra is not None:
                        cand.append(extra)

                active_from = int(self.active_build[0]) if self.active_build else -1
                active_to = int(self.active_build[1]) if self.active_build else -1
                progress = 0.0
                if self.active_build_path:
                    my_cells, _, _, _ = self._count_track_owners_on_path(self.active_build_path)
                    path_len = max(len(self.active_build_path) - 1, 1)
                    progress = float(my_cells) / float(path_len)

                self._dataset_write(
                    {
                        "kind": "disrupt",
                        "seed": str(self._dataset_seed) if str(self._dataset_seed) else None,
                        "as": str(self._dataset_as) if str(self._dataset_as) else None,
                        "turn": int(self.turn),
                        "score_my": int(self.my_score),
                        "score_opp": int(self.opp_score),
                        "score_delta": int(self.my_score - self.opp_score),
                        "opp_threat": float(self._opp_threat_value()),
                        "continue_ev": float(continue_ev),
                        "best_new_ev": float(best_new_ev),
                        "active_from": int(active_from),
                        "active_to": int(active_to),
                        "progress": float(progress),
                        "contested": int(len(self.opp_track_regions & self.my_track_regions)),
                        "inked": int(len(self.region_inked)),
                        "candidates": cand,
                        "chosen": int(target) if target is not None else None,
                        "forced": bool(forced),
                        "force_region": int(force_target) if force_target is not None else None,
                        "force_skip": bool(forced_skip),
                    }
                )
            if target is not None:
                score_delta = self.my_score - self.opp_score
                inst = int(self.region_instability.get(int(target), 0))
                hits = int(self.INSTABILITY_TO_INK - inst)

                locked = (
                    self._disrupt_lock_region is not None
                    and int(self.turn) <= int(self._disrupt_lock_until_turn)
                    and int(target) == int(self._disrupt_lock_region)
                )

                opp_presence = (
                    int(target) in self.opp_track_regions
                    or float(self.active_weight_by_region_opp.get(int(target), 0.0)) > 0.15
                )
                anti_dense_pot = 0.0
                if not opp_presence and self.ANTI_DENSE_ENABLED:
                    anti_dense_pot = float(self._opp_dense_potential_by_region().get(int(target), 0.0))

                allow_disrupt = True
                # Avoid slow inks unless they're clearly relevant or we're very behind.
                if hits >= 3 and score_delta > -80 and (not opp_presence) and anti_dense_pot < 3.0:
                    allow_disrupt = False

                # Oracle forcing should always execute the requested disrupt (unless forced_skip).
                if forced and not forced_skip:
                    allow_disrupt = True

                if locked:
                    allow_disrupt = True

                if allow_disrupt:
                    actions.append(f"DISRUPT {target}")

                    # Record for next-turn opponent disruption inference.
                    self._last_disrupt_target = int(target)

                    # If this is a near-ink region, lock briefly to finish it.
                    if target not in self.region_inked and target not in self.town_regions:
                        if 1 <= hits <= 2:
                            self._disrupt_lock_region = int(target)
                            self._disrupt_lock_until_turn = int(self.turn) + 2
                else:
                    self._last_disrupt_target = None
            else:
                self._last_disrupt_target = None

        out = ";".join(actions) if actions else "WAIT"
        if self._trace_enabled:
            placed = sum(1 for a in actions if a.startswith("PLACE_TRACKS"))
            self._trace(
                f"turn={self.turn} seed={self._trace_seed} as={self._trace_as} out placed={placed} built_any={built_any} action={out}"
            )
        return out


StrategicBot = StrategicBotV26


def main():
    bot = StrategicBotV26()
    bot.read_init()
    while True:
        try:
            bot.read_turn()
            action = bot.get_action()
        except EOFError:
            break
        except Exception as e:
            # Never crash: engine can hang if a player stops responding.
            try:
                print(f"V26 ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            except Exception:
                pass
            try:
                with open("v26_crash.log", "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"turn={getattr(bot, 'turn', '?')} my={getattr(bot, 'my_score', '?')} opp={getattr(bot, 'opp_score', '?')}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            action = "WAIT"

        print(action)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
