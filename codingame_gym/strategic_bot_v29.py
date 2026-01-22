"""\
STRATEGIC BOT V29 (Standalone, From Scratch)
==========================================

A self-contained bot for the Backtrack CodinGame Gym environment.

Constraints:
- This implementation is written fresh and does not copy code from V25/V28 or
  earlier bots.

High-level approach:
- Maintain a single active "connection plan" (town A -> town B) chosen by a
  simple EV heuristic.
- Build the plan from both ends each turn, spending up to 3 points.
- Spend leftover points to densify near our frontier.
- Choose a disruption target to ink regions that appear important to the
  opponent (tracks + activity), preferring regions that are 1 hit from inking.

Notes:
- The authoritative rules are enforced by the engine; this bot is conservative
  about legality (town cells, inked regions, instability >= 4, occupied cells).
"""

from __future__ import annotations

import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import heapq


Point = Tuple[int, int]
Conn = Tuple[int, int]


@dataclass(frozen=True)
class Town:
    tid: int
    x: int
    y: int
    wants: Tuple[int, ...]


@dataclass
class Plan:
    a: int
    b: int
    path: List[Point]


class StrategicBotV29:
    POINTS_PER_TURN = 3
    INK_AT = 4
    MAX_TURNS = 100

    STEP_ON_OPP = 0.40
    STEP_ON_NEUTRAL = 0.20

    # Opponent path model (used for prediction / blocking)
    OPP_STEP_ON_ME = 2.40
    OPP_STEP_ON_NEUTRAL = 0.35
    OPP_NEAR_TRACK_DIST = 10

    def __init__(self) -> None:
        self.me = 0
        self.opp = 1

        self.w = 0
        self.h = 0

        self.region: List[List[int]] = []
        self.base_cost: List[List[int]] = []

        self.towns: Dict[int, Town] = {}
        self.town_cells: Set[Point] = set()
        self.town_regions: Set[int] = set()

        self.turn = 0
        self.my_score = 0
        self.opp_score = 0

        self.owner: Dict[Point, int] = {}
        self.inst_by_region: Dict[int, int] = defaultdict(int)
        self.inked_regions: Set[int] = set()
        self.completed: Set[Conn] = set()
        self.opp_tracks_by_region: Dict[int, int] = defaultdict(int)
        self.my_tracks_by_region: Dict[int, int] = defaultdict(int)
        self.opp_activity_by_region: Dict[int, float] = defaultdict(float)

        self.plan: Optional[Plan] = None

    def read_init(self) -> None:
        lines: List[str] = []
        lines.append(input())
        width = int(input())
        height = int(input())
        lines.append(str(width))
        lines.append(str(height))
        for _ in range(width * height):
            lines.append(input())
        town_count = int(input())
        lines.append(str(town_count))
        for _ in range(town_count):
            lines.append(input())
        self._read_init_lines(lines)

    def _read_init_lines(self, lines: Sequence[str]) -> None:
        it = iter(lines)
        self.me = int(next(it))
        self.opp = 1 - self.me
        self.w = int(next(it))
        self.h = int(next(it))

        self.region = [[-1 for _ in range(self.w)] for _ in range(self.h)]
        self.base_cost = [[0 for _ in range(self.w)] for _ in range(self.h)]

        for y in range(self.h):
            for x in range(self.w):
                parts = next(it).split()
                self.region[y][x] = int(parts[0])
                self.base_cost[y][x] = int(parts[1]) if len(parts) > 1 else 0

        tcount = int(next(it))
        self.towns.clear()
        self.town_cells.clear()

        for _ in range(tcount):
            parts = next(it).split()
            tid = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            wants_raw = parts[3] if len(parts) > 3 else "-"
            wants: List[int] = []
            if wants_raw not in ("-", "x", "X", ""):
                for token in wants_raw.split(","):
                    token = token.strip()
                    if token.isdigit():
                        wants.append(int(token))
            self.towns[tid] = Town(tid=tid, x=x, y=y, wants=tuple(wants))
            self.town_cells.add((x, y))

        self.town_regions = set()
        for t in self.towns.values():
            if 0 <= t.x < self.w and 0 <= t.y < self.h:
                self.town_regions.add(self.region[t.y][t.x])

        self.turn = 0
        self.plan = None

    def read_turn(self) -> None:
        lines: List[str] = [input(), input()]
        for _ in range(self.w * self.h):
            lines.append(input())
        self._read_turn_lines(lines)

    def _read_turn_lines(self, lines: Sequence[str]) -> None:
        it = iter(lines)
        self.my_score = int(next(it))
        self.opp_score = int(next(it))
        self.turn += 1

        self.owner.clear()
        self.inst_by_region = defaultdict(int)
        self.inked_regions = set()
        self.completed = set()
        self.opp_tracks_by_region = defaultdict(int)
        self.my_tracks_by_region = defaultdict(int)
        self.opp_activity_by_region = defaultdict(float)

        for y in range(self.h):
            for x in range(self.w):
                parts = next(it).split()
                if len(parts) < 3:
                    continue

                try:
                    who = int(parts[0])
                    inst = int(parts[1])
                except ValueError:
                    continue

                inked = parts[2] != "0"
                conns = parts[3] if len(parts) > 3 else "x"

                rid = self.region[y][x]
                if rid == -1:
                    continue

                if inked:
                    self.inked_regions.add(rid)
                self.inst_by_region[rid] = max(self.inst_by_region[rid], inst)

                if who != -1:
                    self.owner[(x, y)] = who
                    if who == self.opp:
                        self.opp_tracks_by_region[rid] += 1
                    elif who == self.me:
                        self.my_tracks_by_region[rid] += 1

                if conns not in ("x", "-", ""):
                    pairs = 0
                    for token in conns.split(","):
                        token = token.strip()
                        if "-" not in token:
                            continue
                        a, b = token.split("-", 1)
                        if a.isdigit() and b.isdigit():
                            self.completed.add(self._norm_conn(int(a), int(b)))
                            pairs += 1
                    if pairs > 0 and who == self.opp and rid not in self.inked_regions and rid not in self.town_regions:
                        self.opp_activity_by_region[rid] += float(pairs)

        if self.plan is not None and self._norm_conn(self.plan.a, self.plan.b) in self.completed:
            self.plan = None

    @staticmethod
    def _norm_conn(a: int, b: int) -> Conn:
        return (a, b) if a < b else (b, a)

    def _in_bounds(self, p: Point) -> bool:
        x, y = p
        return 0 <= x < self.w and 0 <= y < self.h

    def _place_cost(self, p: Point) -> int:
        x, y = p
        return int(self.base_cost[y][x] + 1)

    def _can_place(self, p: Point) -> bool:
        if not self._in_bounds(p):
            return False
        if p in self.town_cells or p in self.owner:
            return False
        x, y = p
        rid = self.region[y][x]
        if rid == -1 or rid in self.inked_regions or rid in self.town_regions:
            return False
        if self.inst_by_region.get(rid, 0) >= self.INK_AT:
            return False
        return self._place_cost(p) <= self.POINTS_PER_TURN

    def _blocked(self, p: Point) -> bool:
        if not self._in_bounds(p):
            return True
        x, y = p
        rid = self.region[y][x]
        if rid == -1 or rid in self.inked_regions:
            return True
        if self.inst_by_region.get(rid, 0) >= self.INK_AT:
            return True
        if p not in self.town_cells and p not in self.owner and self._place_cost(p) > self.POINTS_PER_TURN:
            return True
        return False

    def _step_cost(self, p: Point) -> float:
        if self._blocked(p):
            return float("inf")
        if p in self.town_cells:
            return 0.0
        if p in self.owner:
            who = self.owner[p]
            if who == self.me:
                return 0.0
            if who == self.opp:
                return self.STEP_ON_OPP
            return self.STEP_ON_NEUTRAL
        x, y = p
        rid = self.region[y][x]
        inst = self.inst_by_region.get(rid, 0)
        return float(self._place_cost(p) + 2 * inst)

    def _opp_step_cost(self, p: Point) -> float:
        # A path cost model approximating how the opponent will route.
        if self._blocked(p):
            return float("inf")
        if p in self.town_cells:
            return 0.0
        if p in self.owner:
            who = self.owner[p]
            if who == self.opp:
                return 0.0
            if who == self.me:
                return self.OPP_STEP_ON_ME
            return self.OPP_STEP_ON_NEUTRAL
        x, y = p
        rid = self.region[y][x]
        inst = self.inst_by_region.get(rid, 0)
        return float(self._place_cost(p) + 2 * inst)

    def _astar(self, start: Point, goal: Point, limit: int = 2500) -> Optional[List[Point]]:
        if start == goal:
            return [start]

        def h(p: Point) -> int:
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        pq: List[Tuple[float, float, int, Point]] = []
        push_id = 0
        heapq.heappush(pq, (float(h(start)), 0.0, push_id, start))
        push_id += 1

        came: Dict[Point, Point] = {}
        g: Dict[Point, float] = {start: 0.0}
        expanded = 0

        while pq and expanded < limit:
            _, gcur, _, cur = heapq.heappop(pq)
            expanded += 1
            if cur == goal:
                out: List[Point] = [cur]
                while cur in came:
                    cur = came[cur]
                    out.append(cur)
                out.reverse()
                return out

            cx, cy = cur
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nxt = (cx + dx, cy + dy)
                step = self._step_cost(nxt)
                if step == float("inf"):
                    continue
                ng = gcur + step
                if ng < g.get(nxt, float("inf")):
                    came[nxt] = cur
                    g[nxt] = ng
                    heapq.heappush(pq, (ng + float(h(nxt)), ng, push_id, nxt))
                    push_id += 1

        return None

    def _astar_opp(self, start: Point, goal: Point, limit: int = 2200) -> Optional[List[Point]]:
        if start == goal:
            return [start]

        def h(p: Point) -> int:
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        pq: List[Tuple[float, float, int, Point]] = []
        push_id = 0
        heapq.heappush(pq, (float(h(start)), 0.0, push_id, start))
        push_id += 1

        came: Dict[Point, Point] = {}
        g: Dict[Point, float] = {start: 0.0}
        expanded = 0

        while pq and expanded < limit:
            _, gcur, _, cur = heapq.heappop(pq)
            expanded += 1
            if cur == goal:
                out: List[Point] = [cur]
                while cur in came:
                    cur = came[cur]
                    out.append(cur)
                out.reverse()
                return out

            cx, cy = cur
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nxt = (cx + dx, cy + dy)
                step = self._opp_step_cost(nxt)
                if step == float("inf"):
                    continue
                ng = gcur + step
                if ng < g.get(nxt, float("inf")):
                    came[nxt] = cur
                    g[nxt] = ng
                    heapq.heappush(pq, (ng + float(h(nxt)), ng, push_id, nxt))
                    push_id += 1

        return None

    def _town_near_any_opp_track(self, town: Town, opp_cells: List[Point]) -> bool:
        if not opp_cells:
            return False
        tx, ty = town.x, town.y
        dmax = self.OPP_NEAR_TRACK_DIST
        for x, y in opp_cells:
            if abs(x - tx) + abs(y - ty) <= dmax:
                return True
        return False

    def _predict_opp_paths_topk(self, k: int = 6) -> List[Tuple[float, List[Point]]]:
        # Predict a handful of opponent connection paths likely to be pursued.
        opp_cells = [p for p, who in self.owner.items() if who == self.opp]
        if not opp_cells:
            return []

        turns_left = max(0, self.MAX_TURNS - self.turn)
        if turns_left <= 0:
            return []

        scored: List[Tuple[float, List[Point]]] = []

        for a in self.towns.values():
            if not self._town_near_any_opp_track(a, opp_cells):
                continue
            for b_id in a.wants:
                if b_id not in self.towns:
                    continue
                if self._norm_conn(a.tid, b_id) in self.completed:
                    continue
                b = self.towns[b_id]
                if not self._town_near_any_opp_track(b, opp_cells):
                    continue

                path = self._astar_opp((a.x, a.y), (b.x, b.y))
                if not path or len(path) < 2:
                    continue

                # Rough value: future scoring from empty cells after completion.
                cost_pts = 0
                empty_cells = 0
                inst_sum = 0.0
                for x, y in path:
                    if (x, y) in self.town_cells:
                        continue
                    rid = self.region[y][x]
                    inst_sum += float(self.inst_by_region.get(rid, 0))
                    if (x, y) not in self.owner:
                        c = self._place_cost((x, y))
                        if c > self.POINTS_PER_TURN:
                            cost_pts = 10**9
                            break
                        cost_pts += c
                        empty_cells += 1

                if cost_pts <= 0 or cost_pts >= 10**8:
                    continue

                t_finish = (cost_pts + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN
                scoring_turns = max(0, turns_left - t_finish)
                if scoring_turns <= 0:
                    continue

                value = float(empty_cells * scoring_turns)
                value -= inst_sum * 0.75
                value = value / float(cost_pts)
                scored.append((value, path))

        scored.sort(key=lambda t: -t[0])
        return scored[:k]

    def _opp_weights(self) -> Tuple[Dict[Point, float], Dict[int, float]]:
        # Returns (cell_weight, region_weight)
        top = self._predict_opp_paths_topk(k=6)
        cell_w: Dict[Point, float] = defaultdict(float)
        region_w: Dict[int, float] = defaultdict(float)

        for v, path in top:
            if not path:
                continue
            denom = float(max(1, len(path) - 1))
            per = float(v) / denom
            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region[y][x]
                if rid == -1 or rid in self.inked_regions or rid in self.town_regions:
                    continue
                p = (x, y)
                cell_w[p] += per
                region_w[rid] += per

        return dict(cell_w), dict(region_w)

    def _path_cost_points(self, path: Sequence[Point]) -> int:
        total = 0
        for p in path:
            if p in self.town_cells or p in self.owner:
                continue
            total += self._place_cost(p)
        return total

    def _path_empty_cells(self, path: Sequence[Point]) -> int:
        return sum(1 for p in path if p not in self.town_cells and p not in self.owner)

    def _choose_plan(self) -> Optional[Plan]:
        turns_left = max(0, self.MAX_TURNS - self.turn)
        if turns_left <= 2:
            return None

        deg = {tid: len(t.wants) for tid, t in self.towns.items()}
        best_score = -1e18
        best_plan: Optional[Plan] = None

        for a in self.towns.values():
            for b_id in a.wants:
                if b_id not in self.towns:
                    continue
                if self._norm_conn(a.tid, b_id) in self.completed:
                    continue
                b = self.towns[b_id]
                path = self._astar((a.x, a.y), (b.x, b.y))
                if not path or len(path) < 2:
                    continue

                cost = self._path_cost_points(path)
                if cost <= 0:
                    continue

                empties = self._path_empty_cells(path)
                t_finish = (cost + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN
                scoring_turns = max(0, turns_left - t_finish)

                ev = float(empties * scoring_turns)

                inst_sum = 0.0
                opp_touch = 0
                for x, y in path:
                    if (x, y) in self.town_cells:
                        continue
                    rid = self.region[y][x]
                    inst_sum += float(self.inst_by_region.get(rid, 0))
                    if self.owner.get((x, y)) == self.opp:
                        opp_touch += 1

                ev -= inst_sum * 0.8
                ev -= float(opp_touch) * 25.0
                ev += float(deg.get(a.tid, 0) + deg.get(b_id, 0)) * 12.0

                score = ev / float(cost)
                if score > best_score:
                    best_score = score
                    best_plan = Plan(a=a.tid, b=b_id, path=path)

        return best_plan

    def _build_actions(self, plan: Plan) -> List[str]:
        budget = self.POINTS_PER_TURN
        actions: List[str] = []

        i = 0
        j = len(plan.path) - 1

        while budget > 0 and i < j:
            candidates: List[Tuple[int, int, Point]] = []

            # front
            while i <= j and (plan.path[i] in self.town_cells or plan.path[i] in self.owner):
                i += 1
            if i <= j and self._can_place(plan.path[i]):
                c = self._place_cost(plan.path[i])
                if c <= budget:
                    candidates.append((c, 0, plan.path[i]))

            # back
            while j >= i and (plan.path[j] in self.town_cells or plan.path[j] in self.owner):
                j -= 1
            if j >= i and self._can_place(plan.path[j]):
                c = self._place_cost(plan.path[j])
                if c <= budget:
                    candidates.append((c, 1, plan.path[j]))

            if not candidates:
                break

            candidates.sort(key=lambda t: (t[0], t[1]))
            c, side, p = candidates[0]
            actions.append(f"PLACE_TRACKS {p[0]} {p[1]}")
            budget -= int(c)
            self.owner[p] = self.me

            if side == 0:
                i += 1
            else:
                j -= 1

        return actions

    def _densify_actions(self, budget: int) -> List[str]:
        if budget <= 0:
            return []

        frontier: Set[Point] = set()
        anchors = [p for p, who in self.owner.items() if who == self.me]
        anchors.extend(self.town_cells)

        for ax, ay in anchors:
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                frontier.add((ax + dx, ay + dy))

        def cell_score(p: Point) -> float:
            x, y = p
            rid = self.region[y][x]
            inst = self.inst_by_region.get(rid, 0)
            cost = self._place_cost(p)

            adj_me = 0
            adj_opp = 0
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                q = (x + dx, y + dy)
                who = self.owner.get(q)
                if who == self.me or q in self.town_cells:
                    adj_me += 1
                elif who == self.opp:
                    adj_opp += 1

            s = 0.0
            s += float(adj_me) * 7.0
            s -= float(adj_opp) * 6.0
            s += (3.0 - float(cost)) * 6.0
            s -= float(inst) * 2.25
            return s

        actions: List[str] = []
        while budget > 0:
            best_p: Optional[Point] = None
            best_s = -1e18

            for p in list(frontier):
                if not self._in_bounds(p) or not self._can_place(p):
                    continue
                c = self._place_cost(p)
                if c > budget:
                    continue
                s = cell_score(p)
                if s > best_s:
                    best_s = s
                    best_p = p

            if best_p is None:
                break

            c = self._place_cost(best_p)
            actions.append(f"PLACE_TRACKS {best_p[0]} {best_p[1]}")
            budget -= int(c)
            self.owner[best_p] = self.me

            bx, by = best_p
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                frontier.add((bx + dx, by + dy))

        return actions

    def _regions_on_plan(self) -> Set[int]:
        if self.plan is None:
            return set()
        s: Set[int] = set()
        for x, y in self.plan.path:
            if (x, y) in self.town_cells:
                continue
            rid = self.region[y][x]
            if rid != -1:
                s.add(rid)
        return s

    def _choose_disrupt(self) -> Optional[int]:
        turns_left = max(0, self.MAX_TURNS - self.turn)
        if turns_left <= 0:
            return None

        avoid = self._regions_on_plan()
        candidates = set(self.opp_tracks_by_region.keys()) | set(self.opp_activity_by_region.keys())

        best_rid = None
        best_score = 0.0

        for rid in candidates:
            if rid in self.inked_regions or rid in self.town_regions:
                continue
            inst = int(self.inst_by_region.get(rid, 0) or 0)
            if inst >= self.INK_AT:
                continue
            hits = self.INK_AT - inst
            if hits <= 0:
                continue

            opp_tracks = float(self.opp_tracks_by_region.get(rid, 0))
            opp_act = float(self.opp_activity_by_region.get(rid, 0.0))

            s = opp_tracks * 1.6 + opp_act * 6.0

            if hits == 1:
                s *= 3.0
            elif hits == 2:
                s *= 1.2
            else:
                s *= 0.55

            if rid in avoid:
                s *= 0.35

            s *= min(1.0, float(turns_left) / 25.0 + 0.2)

            if s > best_score:
                best_score = s
                best_rid = rid

        return best_rid

    def _candidate_cells_from_plan(self, limit: int = 6) -> List[Point]:
        if self.plan is None or not self.plan.path:
            return []
        path = self.plan.path

        out: List[Point] = []
        i = 0
        j = len(path) - 1
        while (i <= j) and len(out) < limit:
            if i <= j:
                p = path[i]
                if p not in self.town_cells and p not in self.owner and self._can_place(p):
                    out.append(p)
                i += 1
            if len(out) >= limit:
                break
            if j >= i:
                p = path[j]
                if p not in self.town_cells and p not in self.owner and self._can_place(p):
                    out.append(p)
                j -= 1
        return out

    def _candidate_cells_from_frontier(self, limit: int = 8) -> List[Point]:
        # A lightweight variant of densify selection: return top-scoring frontier cells.
        frontier: Set[Point] = set()
        anchors = [p for p, who in self.owner.items() if who == self.me]
        anchors.extend(self.town_cells)
        for ax, ay in anchors:
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                frontier.add((ax + dx, ay + dy))

        scored: List[Tuple[float, Point]] = []
        for p in frontier:
            if not self._in_bounds(p) or not self._can_place(p):
                continue
            x, y = p
            rid = self.region[y][x]
            inst = self.inst_by_region.get(rid, 0)
            cost = self._place_cost(p)

            adj_me = 0
            adj_opp = 0
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                q = (x + dx, y + dy)
                who = self.owner.get(q)
                if who == self.me or q in self.town_cells:
                    adj_me += 1
                elif who == self.opp:
                    adj_opp += 1

            s = 0.0
            s += float(adj_me) * 7.0
            s -= float(adj_opp) * 6.0
            s += (3.0 - float(cost)) * 6.0
            s -= float(inst) * 2.25
            scored.append((s, p))

        scored.sort(key=lambda t: -t[0])
        return [p for _, p in scored[:limit]]

    def _candidate_cells_choke(self, opp_cell_w: Dict[Point, float], limit: int = 8) -> List[Point]:
        # Cells that likely lie on opponent future paths.
        scored: List[Tuple[float, Point]] = []
        for p, w in opp_cell_w.items():
            if w <= 0:
                continue
            if not self._in_bounds(p) or not self._can_place(p):
                continue
            c = self._place_cost(p)
            if c > self.POINTS_PER_TURN:
                continue
            score = float(w) * (10.0 if c == 1 else (6.0 if c == 2 else 3.5))
            scored.append((score, p))
        scored.sort(key=lambda t: -t[0])
        return [p for _, p in scored[:limit]]

    def _enumerate_bundles(self, cells: List[Point], max_cells: int = 3, max_cost: int = 3, limit: int = 260) -> List[List[Point]]:
        # Enumerate distinct bundles of up to `max_cells` placements.
        uniq: List[Point] = []
        seen: Set[Point] = set()
        for p in cells:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        bundles: List[List[Point]] = [[]]
        n = min(len(uniq), 12)

        def cost_of(bundle: List[Point]) -> int:
            return sum(self._place_cost(p) for p in bundle)

        for i in range(n):
            p1 = uniq[i]
            if not self._can_place(p1):
                continue
            c1 = self._place_cost(p1)
            if c1 <= max_cost:
                bundles.append([p1])
            if max_cells >= 2:
                for j in range(i + 1, n):
                    p2 = uniq[j]
                    if p2 == p1 or not self._can_place(p2):
                        continue
                    c2 = c1 + self._place_cost(p2)
                    if c2 <= max_cost:
                        bundles.append([p1, p2])
                    if max_cells >= 3:
                        for k in range(j + 1, n):
                            p3 = uniq[k]
                            if p3 in (p1, p2) or not self._can_place(p3):
                                continue
                            c3 = c2 + self._place_cost(p3)
                            if c3 <= max_cost:
                                bundles.append([p1, p2, p3])
                            if len(bundles) >= limit:
                                return bundles

        return bundles

    def _candidate_disrupts(self, opp_region_w: Dict[int, float]) -> List[Optional[int]]:
        # Include baseline disrupt plus a few oracle-like candidates.
        out: List[Optional[int]] = [None]
        base = self._choose_disrupt()
        if base is not None:
            out.append(int(base))

        ranked = sorted(opp_region_w.items(), key=lambda kv: -kv[1])
        for rid, _ in ranked[:6]:
            out.append(int(rid))

        # Dedup + validate
        seen: Set[Optional[int]] = set()
        valid: List[Optional[int]] = []
        for rid in out:
            if rid in seen:
                continue
            seen.add(rid)
            if rid is None:
                valid.append(None)
                continue
            if rid in self.inked_regions or rid in self.town_regions:
                continue
            inst = int(self.inst_by_region.get(rid, 0) or 0)
            if inst >= self.INK_AT:
                continue
            valid.append(int(rid))
        return valid

    def get_action(self) -> str:
        # Refresh plan.
        if self.plan is not None:
            a = self.towns.get(self.plan.a)
            b = self.towns.get(self.plan.b)
            if a is None or b is None:
                self.plan = None
            else:
                refreshed = self._astar((a.x, a.y), (b.x, b.y))
                if refreshed is None:
                    self.plan = None
                else:
                    self.plan.path = refreshed

        if self.plan is None:
            self.plan = self._choose_plan()

        # Opponent prediction weights for blocking/disruption.
        opp_cell_w, opp_region_w = self._opp_weights()

        # Candidate placements.
        plan_cells = self._candidate_cells_from_plan(limit=8)
        frontier_cells = self._candidate_cells_from_frontier(limit=6)
        choke_cells = self._candidate_cells_choke(opp_cell_w, limit=4)

        candidates = []
        candidates.extend(plan_cells)
        candidates.extend(choke_cells)
        candidates.extend(frontier_cells)

        # Rank candidates by a blended desirability.
        plan_index: Dict[Point, int] = {}
        if self.plan is not None:
            for idx, p in enumerate(self.plan.path):
                plan_index[p] = idx

        def base_cell_value(p: Point) -> float:
            v = 0.0
            v += float(opp_cell_w.get(p, 0.0)) * 60.0
            if p in plan_index:
                # Prefer endpoints first.
                idx = plan_index[p]
                end_dist = min(idx, max(0, len(self.plan.path) - 1 - idx)) if self.plan is not None else 0
                v += 260.0 / float(1 + end_dist)
            x, y = p
            rid = self.region[y][x]
            v -= float(self.inst_by_region.get(rid, 0)) * 6.0
            v -= float(self._place_cost(p)) * 2.0
            return v

        candidates = list(dict.fromkeys([p for p in candidates if self._can_place(p)]))
        candidates.sort(key=base_cell_value, reverse=True)

        bundles = self._enumerate_bundles(candidates, max_cells=3, max_cost=self.POINTS_PER_TURN, limit=260)
        disrupts = self._candidate_disrupts(opp_region_w)

        avoid_regions = self._regions_on_plan()

        plan_placeable = bool(plan_index) and any(self._can_place(p) for p in plan_index.keys())

        def eval_bundle(bundle: List[Point], disrupt: Optional[int]) -> float:
            score = 0.0
            spent = 0
            local_owner_added: Set[Point] = set()
            plan_hits = 0

            for p in bundle:
                c = self._place_cost(p)
                spent += c

                # Block predicted opponent paths.
                score += float(opp_cell_w.get(p, 0.0)) * 80.0

                # Progress on our plan.
                if p in plan_index:
                    plan_hits += 1
                    idx = plan_index[p]
                    end_dist = min(idx, max(0, len(self.plan.path) - 1 - idx)) if self.plan is not None else 0
                    score += 260.0 / float(1 + end_dist)

                x, y = p
                rid = self.region[y][x]
                inst = float(self.inst_by_region.get(rid, 0))
                score -= inst * 7.5

                # Prefer cheap cells.
                score += (3.0 - float(c)) * 14.0

                # Local adjacency bonus (approx).
                adj_me = 0
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    q = (x + dx, y + dy)
                    if q in local_owner_added:
                        adj_me += 1
                    else:
                        who = self.owner.get(q)
                        if who == self.me or q in self.town_cells:
                            adj_me += 1
                score += float(adj_me) * 10.0
                local_owner_added.add(p)

            # Encourage using full budget slightly.
            score += float(spent) * 2.5
            score -= float(self.POINTS_PER_TURN - spent) * 1.0

            # If we have a viable plan placement, avoid wasting turns purely blocking.
            if plan_placeable and spent > 0 and plan_hits == 0:
                score -= 240.0

            if disrupt is not None:
                rid = int(disrupt)
                inst = int(self.inst_by_region.get(rid, 0) or 0)
                hits = int(self.INK_AT) - inst
                if hits > 0:
                    w = float(opp_region_w.get(rid, 0.0))
                    score += w * (380.0 if hits == 1 else (190.0 if hits == 2 else 95.0))
                    score += float(self.opp_tracks_by_region.get(rid, 0)) * 14.0
                    score += float(self.opp_activity_by_region.get(rid, 0.0)) * 45.0
                    if hits == 1:
                        score += 420.0
                    if rid in avoid_regions:
                        score *= 0.72

            return score

        best = None
        best_score = -1e18
        for bundle in bundles:
            if plan_placeable and bundle and plan_index:
                if not any(p in plan_index for p in bundle):
                    continue
            for d in disrupts:
                s = eval_bundle(bundle, d)
                if s > best_score:
                    best_score = s
                    best = (bundle, d)

        # Fallback to old greedy behavior if something goes wrong.
        if best is None:
            actions: List[str] = []
            if self.plan is not None:
                actions.extend(self._build_actions(self.plan))
            spent = 0
            for a in actions:
                if a.startswith("PLACE_TRACKS"):
                    _, xs, ys = a.split()
                    spent += self._place_cost((int(xs), int(ys)))
            rem = max(0, self.POINTS_PER_TURN - spent)
            if rem > 0:
                actions.extend(self._densify_actions(rem))
            if self.turn >= 1:
                rid = self._choose_disrupt()
                if rid is not None:
                    actions.append(f"DISRUPT {rid}")
            return ";".join(actions) if actions else "WAIT"

        bundle, d = best

        # Execute chosen bundle.
        actions_out: List[str] = []
        spent = 0
        for p in bundle:
            if not self._can_place(p):
                continue
            c = self._place_cost(p)
            if spent + c > self.POINTS_PER_TURN:
                continue
            actions_out.append(f"PLACE_TRACKS {p[0]} {p[1]}")
            spent += c
            self.owner[p] = self.me

        # Fill remaining budget with a tiny greedy densify.
        rem = max(0, self.POINTS_PER_TURN - spent)
        if rem > 0:
            actions_out.extend(self._densify_actions(rem))

        if self.turn >= 1 and d is not None:
            actions_out.append(f"DISRUPT {int(d)}")

        return ";".join(actions_out) if actions_out else "WAIT"


StrategicBot = StrategicBotV29


def main() -> None:
    bot = StrategicBotV29()
    bot.read_init()
    while True:
        try:
            bot.read_turn()
            print(bot.get_action())
            sys.stdout.flush()
        except EOFError:
            break
        except Exception as e:
            try:
                print(f"V29 ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            except Exception:
                pass
            try:
                with open("v29_crash.log", "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(
                        f"turn={getattr(bot, 'turn', '?')} my={getattr(bot, 'my_score', '?')} opp={getattr(bot, 'opp_score', '?')}\n"
                    )
                    f.write(traceback.format_exc())
            except Exception:
                pass
            print("WAIT")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
