"""\
STRATEGIC BOT V24 (Standalone, Tuned + Dense Network)
==============================

This file is intentionally self-contained for easy CodinGame submission.

Base: V21 standalone (V20 build/pathing + conservative predictive disruption).
Tuning: inlines the best hyperparameters found by `tune_v22.py` (run v22_20251213_085747).
Add-on: attempts to densify the owned track network by spending leftover budget on
low-cost adjacent placements in safe regions.
"""

import sys
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Set
import heapq
import itertools


class Town:
    def __init__(self, town_id: int, x: int, y: int, desired_connections: List[int]):
        self.id = town_id
        self.x = x
        self.y = y
        self.desired = desired_connections


class StrategicBotV24:
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

    # --- V24 dense-network knobs ---
    DENSE_MAX_EXTRA_PLACEMENTS = 2
    DENSE_MAX_INSTABILITY = 2
    DENSE_AVOID_OPP_REGIONS = True

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

        self._detect_opponent_behavior()

        if self.turn % 10 == 0:
            self.path_cache.clear()

        if self.active_build:
            conn_id = (min(self.active_build[0], self.active_build[1]), max(self.active_build[0], self.active_build[1]))
            if conn_id in self.completed_connections:
                self.active_build = None
                self.active_build_path = None

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
                return 0.0
            if owner == self.opp_id:
                return float(self.OPP_TRACK_TRAVERSAL_PENALTY)
            return float(self.NEUTRAL_TRACK_TRAVERSAL_PENALTY)

        base_cost = self._empty_cell_place_cost(x, y)
        if base_cost > self.POINTS_PER_TURN:
            return float('inf')

        instability_penalty = instability * 2
        return float(base_cost + instability_penalty)

    def find_path(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        max_nodes: int = 1400,
        use_cache: bool = True,
    ) -> Optional[List[Tuple[int, int]]]:
        cache_key = (x0, y0, x1, y1)
        if use_cache and cache_key in self.path_cache:
            return self.path_cache[cache_key]

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

        overlap_bonus = float(my_cells * 4)

        junction_hits = sum(1 for x, y in path if (x, y) in self.town_adjacent_cells)
        junction_bonus = float(junction_hits * 2.164151903532464)

        hub_bonus = 0.0
        if self.hub_town_id is not None and (from_id == self.hub_town_id or to_id == self.hub_town_id):
            hub_bonus = 35.0

        gifting_penalty = float((opp_cells * 7 + neutral_cells * 2) * speed_multiplier)

        return weighted_points + efficiency * 100 + town_id_bonus + consolidation_bonus + overlap_bonus + junction_bonus + hub_bonus - gifting_penalty

    def calc_continue_ev(self) -> float:
        if not self.active_build or not self.active_build_path:
            return 0.0

        path = self.active_build_path
        blocked = sum(1 for x, y in path if self.region_ids[y][x] in self.region_inked)
        if blocked > len(path) * 0.5:
            return 0.0

        remaining_cost = self.calc_remaining_cost(path)
        turns_to_complete = (remaining_cost + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN

        turns_left = self.TOTAL_TURNS - self.turn
        scoring_turns = max(0, turns_left - turns_to_complete)
        if scoring_turns <= 0:
            return 0.0

        _, _, _, empty_cells = self._count_track_owners_on_path(path)
        expected_points = empty_cells * scoring_turns

        my_cells, _, _, _ = self._count_track_owners_on_path(path)
        progress_ratio = my_cells / max(len(path), 1)
        sunk_cost_bonus = progress_ratio * 50

        return expected_points + sunk_cost_bonus

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
                    feasible.append((c, 0, sx, sy))

            ex, ey = path[right - 1]
            if self.can_place_track(ex, ey):
                c = self._empty_cell_place_cost(ex, ey)
                if c <= budget:
                    feasible.append((c, 1, ex, ey))

            if not feasible:
                new_path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
                if not new_path or new_path == path:
                    break
                path = new_path
                left = 0
                right = len(path) - 1
                left = self._advance_from_start(path, left, right)
                right = self._advance_from_end(path, left, right)
                continue

            feasible.sort(key=lambda t: (t[0], t[1]))
            cost, side, x, y = feasible[0]
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

    # -------------------- V24: Dense network --------------------

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
        # Prefer cheap cells that create junctions with existing own network.
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

    def _predict_opponent_region_weights(self) -> Dict[int, float]:
        scored_paths: List[Tuple[float, List[Tuple[int, int]]]] = []

        opp_cells = [pos for pos, owner in self.cell_tracks.items() if owner == self.opp_id]
        if not opp_cells:
            return {}

        for town in self.towns:
            if not self._town_near_opp_tracks(town, opp_cells):
                continue
            for dest_id in town.desired:
                conn_id = (min(town.id, dest_id), max(town.id, dest_id))
                if conn_id in self.completed_connections:
                    continue
                if dest_id not in self.town_by_id:
                    continue

                dest = self.town_by_id[dest_id]
                if not self._town_near_opp_tracks(dest, opp_cells):
                    continue

                path = self._find_opponent_path(town.x, town.y, dest.x, dest.y)
                if not path or len(path) < 2:
                    continue

                v = self._opp_candidate_value(path)
                if v <= 0:
                    continue

                scored_paths.append((v, path))

        scored_paths.sort(key=lambda t: -t[0])
        top = scored_paths[: self.PRED_TOP_K]

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

    # -------------------- Disruption (V20 baseline + V21 override) --------------------

    def _get_disruption_target_v20(self) -> Optional[int]:
        turns_left = self.TOTAL_TURNS - self.turn

        def net_score(region_id: int) -> float:
            opp_w = self.active_weight_by_region_opp.get(region_id, 0.0)
            my_w = self.active_weight_by_region_my.get(region_id, 0.0)
            neu_w = self.active_weight_by_region_neutral.get(region_id, 0.0)

            instability = self.region_instability.get(region_id, 0)
            hits_to_ink = self.INSTABILITY_TO_INK - instability

            base = (opp_w - self.SELF_HARM_ALPHA * my_w + 0.15 * neu_w)
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

        for region_id in set(list(self.active_weight_by_region_opp.keys()) + list(self.opp_track_regions)):
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

    def get_disruption_target(self, best_next_path: Optional[List[Tuple[int, int]]] = None) -> Optional[int]:
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

        pred_opp = self._predict_opponent_region_weights()

        best_region: Optional[int] = None
        best_w = 0.0

        allow_2hit = (-score_delta) >= self.PRED_OVERRIDE_BEHIND_FOR_2HIT

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
                path = self.find_path(town.x, town.y, dest.x, dest.y)
                if not path or len(path) < 2:
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

    def get_action(self) -> str:
        actions: List[str] = []

        continue_ev = self.calc_continue_ev()
        new_candidates = self.get_connection_candidates()
        best_new = new_candidates[0] if new_candidates else None
        best_new_ev = best_new[3] if best_new else 0.0
        best_new_path = best_new[2] if best_new else None

        switch_threshold = 1.5

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

        # V24: if we have leftover budget, place extra tracks to densify the network.
        budget_left = self._remaining_budget_after_actions(actions)
        if budget_left > 0:
            actions.extend(self._densify_network(budget_left))

        if self.turn >= 1:
            target = self.get_disruption_target(best_next_path=best_new_path)
            if target is not None:
                actions.append(f"DISRUPT {target}")

        return ";".join(actions) if actions else "WAIT"


StrategicBot = StrategicBotV24


def main():
    bot = StrategicBotV24()
    bot.read_init()
    while True:
        bot.read_turn()
        print(bot.get_action())
        sys.stdout.flush()


if __name__ == "__main__":
    main()
