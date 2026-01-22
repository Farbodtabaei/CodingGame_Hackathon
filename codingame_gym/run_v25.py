#!/usr/bin/env python3
"""Wrapper to run V25 with runtime-config overrides.

This allows self-play (V25 vs V25) with different parameter sets by launching the
same bot code with different configs.

Examples (PowerShell):
    ./.venv/Scripts/python.exe ./run_v25.py --set SUPPORT_MAX_PLACEMENTS=3
    ./.venv/Scripts/python.exe ./run_v25.py --config ./configs/v25_a.json

Config rules:
- Only overrides existing UPPERCASE attributes on StrategicBotV25.
- Values come from JSON (bool/int/float/str) or from --set parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _coerce_scalar(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(ch in s for ch in [".", "e", "E"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _load_config(path: str | None, inline_json: str | None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    env_json = os.environ.get("V25_CONFIG_JSON")
    if env_json:
        try:
            cfg.update(json.loads(env_json))
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid V25_CONFIG_JSON: {e}")

    if inline_json:
        try:
            cfg.update(json.loads(inline_json))
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --json: {e}")

    if path:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    if not isinstance(cfg, dict):
        raise SystemExit("Config must be a JSON object")

    return cfg


def _apply_overrides(bot_cls: type, cfg: Dict[str, Any]) -> None:
    for key, value in cfg.items():
        if not isinstance(key, str):
            continue
        if not key.isupper():
            continue
        if not hasattr(bot_cls, key):
            continue
        setattr(bot_cls, key, value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    ap.add_argument("--json", type=str, default=None, help="Inline JSON object of overrides")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a single key, e.g. --set SUPPORT_MAX_PLACEMENTS=3 (repeatable)",
    )
    args = ap.parse_args()

    cfg = _load_config(args.config, args.json)

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"Invalid --set '{item}', expected KEY=VALUE")
        k, v = item.split("=", 1)
        cfg[k.strip()] = _coerce_scalar(v)

    # Import after parsing so argparse errors don't get swallowed by bot exception guards.
    import strategic_bot_v25 as v25

    _apply_overrides(v25.StrategicBotV25, cfg)

    # Delegate to the real bot entrypoint.
    v25.main()


if __name__ == "__main__":
    main()
