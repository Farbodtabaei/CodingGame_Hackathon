"""Lightweight replay buffer utilities for Codingame RL experiments."""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class Transition:
    env_idx: int
    episode: int
    step: int
    seed: int
    observation: np.ndarray
    action: List[List[str]]
    reward: float
    done: bool
    next_observation: Optional[np.ndarray]
    scores: Sequence[float]
    plan: Optional[dict]

    def to_serializable(self, include_observations: bool) -> dict:
        payload = {
            "env_idx": self.env_idx,
            "episode": self.episode,
            "step": self.step,
            "seed": self.seed,
            "actions": self.action,
            "reward": self.reward,
            "done": self.done,
            "scores": list(self.scores),
            "plan": self.plan,
        }
        if include_observations:
            payload["observation"] = self.observation.tolist()
            if self.next_observation is not None:
                payload["next_observation"] = self.next_observation.tolist()
        return payload


class ReplayBuffer:
    """Cyclic buffer that stores recent transitions and can dump to JSONL."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            raise ValueError("not enough samples in buffer")
        return random.sample(self._buffer, batch_size)

    def dump_jsonl(self, path: Path, *, include_observations: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            for transition in self._buffer:
                fp.write(json.dumps(transition.to_serializable(include_observations)))
                fp.write("\n")

    def clear(self) -> None:
        self._buffer.clear()
