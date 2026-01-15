"""State engine package for persistent pullback tracking."""

from __future__ import annotations

from trading_bot.state.migrate import migrate
from trading_bot.state.schema import SCHEMA_VERSION, default_state
from trading_bot.state.store import (
    add_cooldown,
    add_pullback,
    add_trading_days,
    apply_invalidation_cooldown,
    cleanup_expired_cooldowns,
    increment_pullback_day,
    invalidate_pullback,
    in_cooldown,
    is_alerted,
    load_state,
    mark_alerted,
    remove_pullback,
    reset_pullback,
    save_state,
)

__all__ = [
    "SCHEMA_VERSION",
    "add_cooldown",
    "add_pullback",
    "add_trading_days",
    "apply_invalidation_cooldown",
    "cleanup_expired_cooldowns",
    "default_state",
    "increment_pullback_day",
    "invalidate_pullback",
    "in_cooldown",
    "is_alerted",
    "load_state",
    "mark_alerted",
    "migrate",
    "remove_pullback",
    "reset_pullback",
    "save_state",
]
