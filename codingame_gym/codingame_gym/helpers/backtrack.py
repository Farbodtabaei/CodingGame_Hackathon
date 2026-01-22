"""Utility helpers for parsing Backtrack metadata and building simple heuristics."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Town:
    town_id: int
    x: int
    y: int
    desired: List[int]


@dataclass
class MapSnapshot:
    width: int
    height: int
    cell_types: List[int]
    region_ids: List[int]
    towns: List[Town]

    def town_lookup(self) -> Dict[int, Town]:
        return {town.town_id: town for town in self.towns}


def _parse_cells(values: Sequence[str], width: int, height: int) -> Tuple[List[int], List[int]]:
    if len(values) != width * height:
        raise ValueError("cell metadata length mismatches width*height")

    cell_types: List[int] = []
    region_ids: List[int] = []
    for entry in values:
        parts = entry.split()
        if not parts:
            raise ValueError("empty cell entry in global_info")
        cell_types.append(int(parts[0]))
        region_ids.append(int(parts[1]) if len(parts) > 1 else -1)
    return cell_types, region_ids


def parse_global_info(raw_lines: Sequence[str]) -> MapSnapshot:
    if len(raw_lines) < 3:
        raise ValueError("global_info missing width/height header")

    width = int(raw_lines[1])
    height = int(raw_lines[2])

    cell_start = 3
    cell_end = cell_start + width * height
    if cell_end > len(raw_lines):
        raise ValueError("global_info truncated before cell metadata ended")

    cell_types, region_ids = _parse_cells(raw_lines[cell_start:cell_end], width, height)

    if cell_end >= len(raw_lines):
        raise ValueError("global_info missing town count entry")
    num_towns = int(raw_lines[cell_end])

    cursor = cell_end + 1
    towns: List[Town] = []
    for _ in range(num_towns):
        if cursor >= len(raw_lines):
            raise ValueError("global_info truncated while reading towns")
        parts = raw_lines[cursor].split(maxsplit=3)
        if len(parts) < 3:
            raise ValueError(f"malformed town entry: {raw_lines[cursor]}")
        tail = parts[3].strip() if len(parts) == 4 else ""
        desired: List[int] = []
        if tail and tail.lower() != "x":
            desired = [int(token) for token in tail.split(',') if token]
        towns.append(
            Town(
                town_id=int(parts[0]),
                x=int(parts[1]),
                y=int(parts[2]),
                desired=desired,
            )
        )
        cursor += 1

    return MapSnapshot(width=width, height=height, cell_types=cell_types, region_ids=region_ids, towns=towns)


def iter_desired_connections(snapshot: MapSnapshot) -> Iterable[Tuple[Town, Town]]:
    lookup = snapshot.town_lookup()
    for town in snapshot.towns:
        for target_id in town.desired:
            target = lookup.get(target_id)
            if target:
                yield town, target


def pick_random_connection(snapshot: MapSnapshot, rng: Optional[random.Random] = None) -> Optional[Tuple[Town, Town]]:
    rng = rng or random.Random()
    pairs = list(iter_desired_connections(snapshot))
    if not pairs:
        return None
    return rng.choice(pairs)


class AutoplaceAgent:
    """Chooses a desired town pair and keeps issuing AUTOPLACE commands."""

    def __init__(self, snapshot: MapSnapshot, *, repeat: bool = False, rng: Optional[random.Random] = None):
        self.snapshot = snapshot
        self.repeat = repeat
        self.rng = rng or random.Random()
        self._plan = pick_random_connection(snapshot, self.rng)
        self._issued_once = False

    @property
    def plan(self) -> Optional[Tuple[Town, Town]]:
        return self._plan

    def plan_summary(self) -> Optional[dict]:
        if not self._plan:
            return None
        src, dst = self._plan
        return {
            "source_id": src.town_id,
            "target_id": dst.town_id,
            "source_coords": (src.x, src.y),
            "target_coords": (dst.x, dst.y),
        }

    def next_actions(self) -> List[str]:
        if not self._plan:
            return ["WAIT"]

        src, dst = self._plan
        command = f"AUTOPLACE {src.x} {src.y} {dst.x} {dst.y}"

        if not self._issued_once:
            self._issued_once = True
            return [command]

        if self.repeat:
            return [command]

        return ["WAIT"]
