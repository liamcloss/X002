"""Mooner state helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from trading_bot.paths import mooner_state_path


def load_mooner_states(base_dir: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    """Load persisted Mooner states for attaching context to other pipelines."""

    path = mooner_state_path(base_dir)
    if not path.exists():
        logger.debug("Mooner state file missing: %s", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load Mooner states from %s: %s", path, exc)
        return []

    if not isinstance(payload, dict):
        logger.warning("Invalid Mooner state payload at %s; expected dict.", path)
        return []

    states: list[dict[str, Any]] = []
    for ticker, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        states.append(
            {
                "ticker": ticker,
                "state": entry.get("state"),
                "context": entry.get("context"),
                "as_of": entry.get("state_since"),
            }
        )
    return states
