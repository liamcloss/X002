"""Mooner subset selection helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_bot import config
from trading_bot.universe.active import ensure_active_column

SUBSET_FILENAME = 'MoonerSubset.json'


def select_mooner_subset(base_dir: Path, logger: logging.Logger) -> list[str]:
    """Select a deterministic subset of the main universe for mooner monitoring."""

    configured = [ticker.strip().upper() for ticker in config.MOONER_SUBSET_TICKERS if ticker]
    if not configured:
        return []

    max_size = max(1, int(config.MOONER_SUBSET_MAX))
    selected = configured[:max_size]

    universe_path = base_dir / "universe" / "clean" / "universe.parquet"
    if not universe_path.exists():
        logger.warning("Universe file missing; mooner subset will rely on configured tickers.")
        return selected

    universe = ensure_active_column(universe_path, logger)
    if universe.empty or "ticker" not in universe.columns:
        logger.warning("Universe data unavailable; mooner subset will rely on configured tickers.")
        return selected

    active = universe[universe["active"].fillna(False).astype(bool)]
    active_tickers = {str(ticker).strip().upper() for ticker in active["ticker"].tolist()}
    filtered = [ticker for ticker in selected if ticker in active_tickers]

    if not filtered:
        logger.warning("Mooner subset empty after filtering inactive tickers.")
    return filtered


def write_mooner_subset(base_dir: Path, tickers: list[str], logger: logging.Logger) -> Path | None:
    """Persist the mooner subset to disk."""

    if not tickers:
        return None

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "max_size": int(config.MOONER_SUBSET_MAX),
        "tickers": tickers,
    }
    path = base_dir / SUBSET_FILENAME
    try:
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except OSError as exc:
        logger.warning("Failed to write mooner subset to %s: %s", path, exc)
        return None
    return path


def load_mooner_subset(base_dir: Path, logger: logging.Logger) -> list[str]:
    """Load a previously written mooner subset."""

    path = base_dir / SUBSET_FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read mooner subset from %s: %s", path, exc)
        return []

    if isinstance(payload, dict):
        tickers = payload.get("tickers", [])
    elif isinstance(payload, list):
        tickers = payload
    else:
        return []

    return [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]


__all__ = [
    'load_mooner_subset',
    'select_mooner_subset',
    'write_mooner_subset',
]
