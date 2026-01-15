"""Schema migration support for persisted state."""

from __future__ import annotations

import logging
from typing import Any

from trading_bot.state import schema

logger = logging.getLogger("trading_bot")


def migrate(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate a state payload to the latest schema version."""

    if not isinstance(state, dict):
        raise TypeError("State payload must be a dict.")

    version = state.get("version")
    if version is None:
        logger.warning("State missing version; assuming schema v%s.", schema.SCHEMA_VERSION)
        version = schema.SCHEMA_VERSION
        state["version"] = version

    if version != schema.SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported state schema version {version}. "
            f"Expected {schema.SCHEMA_VERSION}."
        )

    return _ensure_v1_keys(state)


def _ensure_v1_keys(state: dict[str, Any]) -> dict[str, Any]:
    defaults = schema.default_state()
    for key in ("pullbacks", "alerts", "cooldowns", "history", "paper_reports"):
        if key not in state or not isinstance(state.get(key), type(defaults[key])):
            state[key] = defaults[key]
    state["version"] = schema.SCHEMA_VERSION
    return state
