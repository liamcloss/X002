"""Parquet-backed cache for yfinance OHLCV data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

MAX_TRADING_DAYS = 250


def ensure_prices_dir(base_dir: Path) -> Path:
    """Ensure the data/prices directory exists and return it."""

    prices_dir = base_dir / "data" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    return prices_dir


def cache_path(prices_dir: Path, ticker: str) -> Path:
    """Build a Windows-safe cache path for the given ticker."""

    safe_ticker = ticker.upper().replace("/", "_")
    return prices_dir / f"{safe_ticker}.parquet"


def load_cache(prices_dir: Path, ticker: str, logger: logging.Logger) -> pd.DataFrame:
    """Load cached data if present, handling corruption."""

    path = cache_path(prices_dir, ticker)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:  # noqa: BLE001 - controlled logging
        logger.error("Cache file for %s is unreadable (%s). Rebuilding.", ticker, exc)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete corrupt cache file: %s", path)
        return pd.DataFrame()


def last_cached_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """Return the most recent cached date or None."""

    if df.empty:
        return None
    df = df.sort_index()
    last = df.index.max()
    return pd.to_datetime(last).normalize()


def update_cache(
    prices_dir: Path,
    ticker: str,
    new_data: pd.DataFrame,
    logger: logging.Logger,
    existing: pd.DataFrame | None = None,
    last_date: pd.Timestamp | None = None,
    max_trading_days: int = MAX_TRADING_DAYS,
) -> None:
    """Merge new data into the cache and persist to Parquet."""

    if existing is None:
        existing = load_cache(prices_dir, ticker, logger)
    if last_date is None:
        last_date = last_cached_date(existing)

    incoming = new_data.copy()
    incoming.index = pd.to_datetime(incoming.index)
    incoming = incoming.sort_index()

    if last_date is not None:
        incoming = incoming[incoming.index.normalize() > last_date]

    if incoming.empty and existing.empty:
        logger.info("No data to write for %s; cache remains empty.", ticker)
        return

    combined = pd.concat([existing, incoming]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    if not combined.empty:
        combined = combined.tail(max_trading_days)

    path = cache_path(prices_dir, ticker)
    combined.to_parquet(path)
    logger.info("Updated cache for %s (%s rows).", ticker, combined.shape[0])


def calculate_fetch_start(last_date: pd.Timestamp | None, lookback_days: int = 400) -> datetime:
    """Determine the fetch start date based on cache freshness."""

    if last_date is None:
        return datetime.now() - timedelta(days=lookback_days)
    return (last_date + pd.Timedelta(days=1)).to_pydatetime()
