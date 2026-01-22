"""Utilities for consuming JSONL rollout data in training pipelines."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np


@dataclass
class RolloutSample:
    """Single step sampled from rl_scaffold or replay-buffer dumps."""

    env_idx: int
    episode: int
    step: int
    seed: int
    actions: List[List[str]]
    scores: List[float]
    reward: float
    done: bool
    plan: Optional[dict]
    observation: Optional[np.ndarray]
    next_observation: Optional[np.ndarray]

    @staticmethod
    def from_payload(payload: dict) -> "RolloutSample":
        def _maybe_array(field: str) -> Optional[np.ndarray]:
            data = payload.get(field)
            if data is None:
                return None
            return np.asarray(data, dtype=np.float32)

        return RolloutSample(
            env_idx=int(payload.get("env_idx", 0)),
            episode=int(payload.get("episode", 0)),
            step=int(payload.get("step", 0)),
            seed=int(payload.get("seed", 0)),
            actions=[list(player or []) for player in (payload.get("actions") or [])],
            scores=list(payload.get("scores", [])),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
            plan=payload.get("plan"),
            observation=_maybe_array("observation"),
            next_observation=_maybe_array("next_observation"),
        )


def iter_jsonl(paths: Sequence[Path], *, strict: bool = True) -> Iterator[RolloutSample]:
    """Yield `RolloutSample` objects from one or many JSONL files."""

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    if strict:
                        raise RuntimeError(f"{path}:{line_no} invalid JSON: {exc}") from exc
                    print(f"Warning: skipping malformed row {path}:{line_no}: {exc}")
                    continue
                yield RolloutSample.from_payload(payload)


def to_numpy_batch(samples: Sequence[RolloutSample]) -> dict:
    """Stack a list of samples into numpy arrays for training."""

    if not samples:
        raise ValueError("samples list is empty")

    rewards = np.asarray([sample.reward for sample in samples], dtype=np.float32)
    dones = np.asarray([sample.done for sample in samples], dtype=np.bool_)
    scores = np.asarray([sample.scores for sample in samples], dtype=np.float32)

    if samples[0].observation is None or samples[0].next_observation is None:
        raise ValueError("observations are missing in samples; rerun rl_scaffold with --record-observations")

    obs = np.stack([sample.observation for sample in samples])  # type: ignore[arg-type]
    next_obs = np.stack([sample.next_observation for sample in samples])  # type: ignore[arg-type]

    return {
        "observations": obs,
        "next_observations": next_obs,
        "rewards": rewards,
        "dones": dones,
        "scores": scores,
        "actions": [sample.actions for sample in samples],
        "plans": [sample.plan for sample in samples],
    }
