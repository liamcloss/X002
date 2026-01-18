"""Mooner call-out emission utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from trading_bot import config
from trading_bot.mooner.mooner_state_engine import MoonerState

CALLOUTS_FILENAME = 'MoonerCallouts.json'
STATE_TRACK_FILENAME = 'mooner_state.json'


def emit_mooner_callouts(
    base_dir: Path,
    snapshots: Iterable[dict],
    logger: logging.Logger,
) -> list[dict]:
    """Emit callouts for FIRING mooner states."""

    snapshots_list = [snap for snap in snapshots if isinstance(snap, dict)]
    previous_state = _load_previous_state(base_dir, logger)
    callouts: list[dict] = []
    current_state: dict[str, dict] = {}

    for snapshot in snapshots_list:
        ticker = _normalize_ticker(snapshot.get("ticker"))
        if not ticker:
            continue
        state = snapshot.get("state")
        context = snapshot.get("context") or "Mooner regime detected."
        as_of = snapshot.get("as_of") or datetime.now(timezone.utc).date().isoformat()
        current_state[ticker] = {
            "state": state,
            "as_of": as_of,
            "context": context,
        }
        if state != MoonerState.FIRING.value:
            continue

        previous = previous_state.get(ticker, {})
        was_firing = previous.get("state") == MoonerState.FIRING.value
        if was_firing and not config.MOONER_EMIT_WHILE_FIRING:
            continue

        callouts.append(
            {
                "ticker": ticker,
                "state": MoonerState.FIRING.value,
                "detected_on": as_of,
                "context": context,
                "severity": "RARE_EVENT",
            }
        )

    _write_state_track(base_dir, current_state, logger)
    _write_callouts(base_dir, callouts, logger)
    return callouts


def load_mooner_callouts(base_dir: Path, logger: logging.Logger) -> list[dict]:
    """Load mooner callouts if present."""

    path = base_dir / CALLOUTS_FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read mooner callouts from %s: %s", path, exc)
        return []

    if isinstance(payload, dict):
        callouts = payload.get("callouts", [])
    elif isinstance(payload, list):
        callouts = payload
    else:
        return []
    return [item for item in callouts if isinstance(item, dict)]


def _load_previous_state(base_dir: Path, logger: logging.Logger) -> dict[str, dict]:
    path = base_dir / "state" / STATE_TRACK_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read mooner state track from %s: %s", path, exc)
        return {}
    if isinstance(payload, dict):
        return {key: value for key, value in payload.items() if isinstance(value, dict)}
    return {}


def _write_state_track(base_dir: Path, state: dict[str, dict], logger: logging.Logger) -> None:
    path = base_dir / "state" / STATE_TRACK_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    except OSError as exc:
        logger.warning("Failed to write mooner state track to %s: %s", path, exc)


def _write_callouts(base_dir: Path, callouts: list[dict], logger: logging.Logger) -> None:
    path = base_dir / CALLOUTS_FILENAME
    try:
        payload: object = callouts
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except OSError as exc:
        logger.warning("Failed to write mooner callouts to %s: %s", path, exc)


def _normalize_ticker(value: object | None) -> str:
    if value is None:
        return ''
    return str(value).strip().upper()


__all__ = [
    'emit_mooner_callouts',
    'load_mooner_callouts',
]
