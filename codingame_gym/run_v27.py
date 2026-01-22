#!/usr/bin/env python3
"""Wrapper to run V27 with runtime-config overrides.

This allows self-play (V27 vs V27) or tuning (V27 vs others) with different
parameter sets by launching the same bot code with different configs.

Examples (PowerShell):
    ./.venv/Scripts/python.exe ./run_v27.py --set ANTI_DENSE_ALPHA=0.8
    ./.venv/Scripts/python.exe ./run_v27.py --config ./configs/v27_a.json

Config rules:
- Only overrides existing UPPERCASE attributes on StrategicBotV27.
- Values come from JSON (bool/int/float/str) or from --set parsing.

Default gameplay is unchanged when no config is provided.
"""

from __future__ import annotations

import argparse
import json
import os
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

    env_json = os.environ.get("V27_CONFIG_JSON")
    if env_json:
        try:
            cfg.update(json.loads(env_json))
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid V27_CONFIG_JSON: {e}")

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
            file_cfg = json.load(f)

        # Optional config layering: load a base config first, then override with
        # keys from the current file.
        base_path = file_cfg.get("BASE_CONFIG") if isinstance(file_cfg, dict) else None
        if isinstance(base_path, str) and base_path.strip():
            bp = Path(base_path)
            if not bp.is_absolute():
                # First try relative to the config file location, then fall back
                # to the current working directory.
                bp1 = (p.parent / bp).resolve()
                bp2 = (Path.cwd() / bp).resolve()
                bp = bp1 if bp1.exists() else bp2
            if not bp.exists():
                raise SystemExit(f"Base config file not found: {bp}")
            with bp.open("r", encoding="utf-8") as bf:
                base_cfg = json.load(bf)
            if not isinstance(base_cfg, dict):
                raise SystemExit("BASE_CONFIG must point to a JSON object")
            if isinstance(file_cfg, dict):
                merged = dict(base_cfg)
                merged.update({k: v for k, v in file_cfg.items() if k != "BASE_CONFIG"})
                file_cfg = merged

        if not isinstance(file_cfg, dict):
            raise SystemExit("Config file must be a JSON object")

        cfg.update(file_cfg)

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
        help="Override a single key, e.g. --set ANTI_DENSE_ALPHA=0.8 (repeatable)",
    )
    args = ap.parse_args()

    cfg = _load_config(args.config, args.json)

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"Invalid --set '{item}', expected KEY=VALUE")
        k, v = item.split("=", 1)
        cfg[k.strip()] = _coerce_scalar(v)

    # Import after parsing so argparse errors don't get swallowed by bot exception guards.
    import strategic_bot_v27 as v27

    _apply_overrides(v27.StrategicBotV27, cfg)

    # Delegate to the real bot entrypoint.
    v27.main()


if __name__ == "__main__":
    main()
