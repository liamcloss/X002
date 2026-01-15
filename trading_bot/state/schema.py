"""Schema definition for the trading bot state engine."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1

State = dict[str, Any]


def default_state() -> State:
    """Return a new, empty state payload with the current schema version."""

    return {
        "version": SCHEMA_VERSION,
        "pullbacks": {},
        "alerts": {},
        "cooldowns": {},
        "history": [],
    }
