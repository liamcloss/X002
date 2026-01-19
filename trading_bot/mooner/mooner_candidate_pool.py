"""Mooner candidate pool discovery helpers."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from trading_bot import config
from trading_bot.market_data import cache
from trading_bot.paths import mooner_output_path
from trading_bot.universe.active import ensure_active_column

FX_TO_GBP = {
    "USD": 0.8,
    "GBP": 1.0,
    "EUR": 0.87,
    "CAD": 0.59,
    "CHF": 0.84,
    "AUD": 0.57,
    "JPY": 0.0069,
}
DEFAULT_FX = 1.0


def build_candidate_pool(base_dir: Path, logger: logging.Logger) -> list[str]:
    """Discover the broad Mooner candidate pool."""

    prices_dir = base_dir / "data" / "prices"
    universe = _load_universe(base_dir, logger)
    active_tickers = _extract_active_tickers(universe)

    pool: list[str] = []
    as_of_date: date | None = None

    for entry in active_tickers:
        ticker = entry["ticker"]
        currency = entry.get("currency_code", "GBP") or "GBP"
        df = cache.load_cache(prices_dir, ticker, logger)
        if df.empty:
            continue
        df = df.sort_index()
        last_date = _last_cached_date(df)
        if last_date:
            as_of_date = last_date if as_of_date is None else max(as_of_date, last_date)
        if _qualifies_candidate(df, currency):
            pool.append(ticker)

    _write_candidate_pool(base_dir, pool, as_of_date)
    logger.info("Mooner candidate pool built (%s tickers).", len(pool))
    return pool


def _load_universe(base_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    path = base_dir / "universe" / "clean" / "universe.parquet"
    return ensure_active_column(path, logger)


def _extract_active_tickers(df: pd.DataFrame) -> list[dict[str, str]]:
    tickers: list[dict[str, str]] = []
    if df.empty or "ticker" not in df.columns:
        return tickers
    for _, row in df.iterrows():
        ticker = _normalize(row.get("ticker"))
        if not ticker:
            continue
        tickers.append(
            {
                "ticker": ticker,
                "currency_code": _normalize(row.get("currency_code")) or "GBP",
            }
        )
    return tickers


def _qualifies_candidate(df: pd.DataFrame, currency_code: str) -> bool:
    if len(df) < 60:
        return False

    close = _get_series(df, ("close", "Close"))
    volume = _get_series(df, ("volume", "Volume"))
    if close is None or volume is None:
        return False
    close = close.dropna()
    volume = volume.dropna()
    if len(close) < 60 or len(volume) < 60:
        return False

    avg_volume_60 = volume.tail(60).mean()
    avg_price_60 = close.tail(60).mean()
    if not pd.notna(avg_volume_60) or not pd.notna(avg_price_60):
        return False

    liquidity_ok = (
        avg_volume_60 >= config.MOONER_CANDIDATE_VOLUME_THRESHOLD
        and _to_gbp(avg_volume_60 * avg_price_60, currency_code)
        >= config.MOONER_CANDIDATE_DOLLAR_VOLUME_GBP
    )

    atr20 = _compute_atr(df, 20)
    atr60 = _compute_atr(df, 60)
    atr20_latest = _last_valid(atr20)
    atr60_latest = _last_valid(atr60)
    atr_high_enough = False
    compression = False
    if atr20_latest and atr60_latest:
        atr_high_enough = _percentile(atr20.tail(252), atr20_latest) >= config.MOONER_ATR_PERCENTILE
        compression = atr20_latest <= config.MOONER_ATR_COMPRESSION_RATIO * atr60_latest

    avg_volume_30 = volume.tail(30).mean()
    volume_last = volume.iloc[-1]
    surge = False
    if pd.notna(avg_volume_30) and avg_volume_30 > 0 and pd.notna(volume_last):
        surge = volume_last >= config.MOONER_CANDIDATE_SURGE_MULTIPLIER * avg_volume_30

    return liquidity_ok or atr_high_enough or compression or surge


def _get_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    for key in candidates:
        if key in df.columns:
            return pd.to_numeric(df[key], errors="coerce")
    return None


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = _get_series(df, ("high", "High"))
    low = _get_series(df, ("low", "Low"))
    close = _get_series(df, ("close", "Close"))
    if high is None or low is None or close is None:
        return pd.Series(dtype="float64")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _percentile(series: pd.Series, value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    valid = series.dropna()
    if valid.empty:
        return 0.0
    return float((valid < value).sum()) / max(len(valid), 1)


def _last_valid(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    for value in reversed(series.tolist()):
        if pd.notna(value):
            return float(value)
    return None


def _to_gbp(value: float, currency: str) -> float:
    rate = FX_TO_GBP.get(currency.upper(), DEFAULT_FX)
    return value * rate


def _write_candidate_pool(base_dir: Path, tickers: Iterable[str], as_of: date | None) -> None:
    payload = {
        "as_of": as_of.isoformat() if as_of else date.today().isoformat(),
        "tickers": sorted(set(tickers)),
    }
    path = mooner_output_path(base_dir, CANDIDATE_POOL_FILENAME)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.upper() if text else None


def _last_cached_date(df: pd.DataFrame) -> date | None:
    if df.empty:
        return None
    idx = pd.to_datetime(df.index)
    return idx.max().date()
