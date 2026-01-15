"""State persistence and mutation helpers for pullbacks, alerts, and cooldowns."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from trading_bot.state import schema
from trading_bot.state.migrate import migrate as migrate_payload

logger = logging.getLogger("trading_bot")

MAX_ACTIVE_PULLBACKS = 10
PULLBACK_EXPIRY_DAYS = 3
INVALIDATION_REASON = "Invalidated (>5% below high)"
RESET_REASON = "Reset due to new breakout"
REPLACED_REASON = "Replaced due to cap limit"
EXPIRED_REASON = "Expired (3 days in pullback)"


State = dict[str, Any]


def load_state() -> State:
    """Load state from disk, creating a default file if missing."""

    state_path = _state_path()
    if not state_path.exists():
        state = schema.default_state()
        save_state(state)
        return state

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"State file is invalid JSON: {state_path}") from exc

    state = migrate_payload(payload)
    save_state(state)
    return state


def save_state(state: State) -> None:
    """Persist state to disk atomically."""

    state_path = _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, sort_keys=True)
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, state_path)


def add_pullback(state: State, ticker: str, breakout_high: float, today_date: date | str) -> None:
    """Add a new pullback, enforcing the max active pullback cap."""

    pullbacks = state.setdefault("pullbacks", {})
    entry = {
        "ticker": ticker,
        "breakout_high": float(breakout_high),
        "entered_at": _format_date(today_date),
        "days_in_pullback": 1,
    }
    pullbacks[ticker] = entry

    if len(pullbacks) > MAX_ACTIVE_PULLBACKS:
        oldest_ticker = _oldest_pullback_ticker(pullbacks)
        if oldest_ticker:
            remove_pullback(state, oldest_ticker, REPLACED_REASON)


def increment_pullback_day(state: State, ticker: str) -> None:
    """Increment the days-in-pullback counter and expire stale entries."""

    pullback = state.get("pullbacks", {}).get(ticker)
    if not pullback:
        logger.warning("Cannot increment pullback day; %s not tracked.", ticker)
        return

    pullback["days_in_pullback"] = int(pullback.get("days_in_pullback", 0)) + 1
    if pullback["days_in_pullback"] >= PULLBACK_EXPIRY_DAYS:
        remove_pullback(state, ticker, EXPIRED_REASON)


def remove_pullback(state: State, ticker: str, reason: str) -> None:
    """Remove a pullback and record its removal in history."""

    pullbacks = state.get("pullbacks", {})
    if ticker not in pullbacks:
        logger.warning("Pullback for %s not found; nothing to remove.", ticker)
        return

    pullbacks.pop(ticker, None)
    _remove_alert(state, ticker)
    _record_history(state, ticker, "REMOVED", reason)


def invalidate_pullback(state: State, ticker: str, today_date: date | str) -> None:
    """Invalidate a pullback and apply the cooldown window."""

    remove_pullback(state, ticker, INVALIDATION_REASON)
    apply_invalidation_cooldown(state, ticker, today_date)


def reset_pullback(state: State, ticker: str, today_date: date | str) -> None:
    """Reset a pullback due to a new breakout and apply cooldown."""

    remove_pullback(state, ticker, RESET_REASON)
    apply_invalidation_cooldown(state, ticker, today_date)


def mark_alerted(state: State, ticker: str, today_date: date | str) -> None:
    """Mark a ticker as alerted and active."""

    alerts = state.setdefault("alerts", {})
    alerts[ticker] = {
        "alerted_at": _format_date(today_date),
        "status": "ACTIVE",
    }


def is_alerted(state: State, ticker: str) -> bool:
    """Return True if the ticker has an active alert."""

    entry = state.get("alerts", {}).get(ticker)
    return bool(entry and entry.get("status") == "ACTIVE")


def add_cooldown(state: State, ticker: str, until_date: date | str) -> None:
    """Add or update a cooldown entry for a ticker."""

    cooldowns = state.setdefault("cooldowns", {})
    cooldowns[ticker] = {"cooldown_until": _format_date(until_date)}


def in_cooldown(state: State, ticker: str, today_date: date | str) -> bool:
    """Return True if the ticker is currently in cooldown."""

    entry = state.get("cooldowns", {}).get(ticker)
    if not entry:
        return False
    cooldown_until = _maybe_parse_date(entry.get("cooldown_until"))
    if not cooldown_until:
        return False
    return _parse_date(today_date) <= cooldown_until


def cleanup_expired_cooldowns(state: State, today_date: date | str) -> None:
    """Remove cooldowns that have expired prior to the given date."""

    cooldowns = state.get("cooldowns", {})
    cutoff = _parse_date(today_date)
    expired = [
        ticker
        for ticker, entry in cooldowns.items()
        if _maybe_parse_date(entry.get("cooldown_until"))
        and _maybe_parse_date(entry.get("cooldown_until")) < cutoff
    ]
    for ticker in expired:
        cooldowns.pop(ticker, None)


def _state_path() -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "state" / "state.json"


def _format_date(value: date | str) -> str:
    parsed = _parse_date(value)
    return parsed.isoformat()


def _parse_date(value: date | str | None) -> date:
    if isinstance(value, date):
        return value
    if not value:
        raise ValueError("Date value is required.")
    return datetime.strptime(value, "%Y-%m-%d").date()


def _oldest_pullback_ticker(pullbacks: dict[str, dict[str, Any]]) -> str | None:
    if not pullbacks:
        return None

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[date, str]:
        ticker, payload = item
        entered_at = _safe_parse_date(payload.get("entered_at"))
        return entered_at, ticker

    return min(pullbacks.items(), key=sort_key)[0]


def _safe_parse_date(value: str | None) -> date:
    parsed = _maybe_parse_date(value)
    return parsed if parsed else date.min


def _maybe_parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _record_history(state: State, ticker: str, action: str, reason: str) -> None:
    history = state.setdefault("history", [])
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ticker": ticker,
            "action": action,
            "reason": reason,
        }
    )


def _remove_alert(state: State, ticker: str) -> None:
    alerts = state.get("alerts", {})
    if ticker in alerts:
        alerts.pop(ticker, None)


def add_trading_days(start_date: date | str, days: int) -> date:
    """Return a date that is N trading days after the start date."""

    current = _parse_date(start_date)
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def apply_invalidation_cooldown(state: State, ticker: str, today_date: date | str) -> None:
    """Apply a five-trading-day cooldown for invalidations/resets."""

    cooldown_until = add_trading_days(today_date, 5)
    add_cooldown(state, ticker, cooldown_until)
