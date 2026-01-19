"""Mooner subset ranking helpers."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from trading_bot import config
from trading_bot.market_data import cache
from trading_bot.paths import mooner_output_path
SUBSET_FILENAME = "MoonerSubset.json"


def rank_mooner_subset(base_dir: Path, universe: Iterable[str], logger: logging.Logger) -> list[str]:
    """Rank the Mooner universe and select the top subset."""

    prices_dir = base_dir / "data" / "prices"
    scored: list[tuple[float, float, int, str]] = []
    as_of_date: date | None = None

    for ticker in universe:
        df = cache.load_cache(prices_dir, ticker, logger)
        if df.empty:
            continue
        df = df.sort_index()
        last_date = _last_cached_date(df)
        if last_date:
            as_of_date = last_date if as_of_date is None else max(as_of_date, last_date)
        entry = _score_ticker(df)
        if entry is None:
            continue
        atr_ratio, rel_strength, structure_penalty = entry
        scored.append((atr_ratio, -rel_strength, structure_penalty, ticker))

    scored.sort()
    subset_limit = max(1, int(config.MOONER_SUBSET_MAX))
    subset = [ticker for *_ignored, ticker in scored[:subset_limit]]
    _write_subset(base_dir, subset, as_of_date)
    logger.info("Mooner subset ranked (%s tickers).", len(subset))
    return subset


def _score_ticker(df: pd.DataFrame) -> tuple[float, float, int] | None:
    close = _get_series(df, ("close", "Close"))
    high = _get_series(df, ("high", "High"))
    low = _get_series(df, ("low", "Low"))
    if close is None or close.empty:
        return None
    close = close.dropna()
    if len(close) < 90:
        return None

    atr20 = _compute_atr(df, 20)
    atr60 = _compute_atr(df, 60)
    atr20_latest = _last_valid(atr20)
    atr60_latest = _last_valid(atr60)
    if atr20_latest is None or atr60_latest in (None, 0):
        return None
    atr_ratio = atr20_latest / max(atr60_latest, 1e-9)

    rel_strength = _relative_strength(close, config.MOONER_REL_STRENGTH_LOOKBACK)
    structure_penalty = _structure_penalty(close)

    return atr_ratio, rel_strength, structure_penalty


def _relative_strength(close: pd.Series, lookback: int) -> float:
    if len(close) <= lookback:
        start = close.iloc[0]
    else:
        start = close.iloc[-lookback]
    if start <= 0:
        return 0.0
    return float(close.iloc[-1] / start)


def _structure_penalty(close: pd.Series) -> int:
    ma50 = close.rolling(window=50, min_periods=10).mean()
    recent = min(len(close), 30)
    penalty = 0
    for idx in range(-recent, 0):
        price = close.iloc[idx]
        ma_value = ma50.iloc[idx]
        if pd.notna(price) and pd.notna(ma_value) and price < ma_value:
            penalty += 1
    return penalty


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


def _get_series(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series | None:
    for key in keys:
        if key in df.columns:
            return pd.to_numeric(df[key], errors="coerce")
    return None


def _last_cached_date(df: pd.DataFrame) -> date | None:
    if df.empty:
        return None
    idx = pd.to_datetime(df.index)
    return idx.max().date()


def _last_valid(series: pd.Series) -> float | None:
    if series.empty:
        return None
    for value in reversed(series.tolist()):
        if pd.notna(value):
            return float(value)
    return None


def _write_subset(base_dir: Path, tickers: Iterable[str], as_of: date | None) -> None:
    payload = {
        "as_of": as_of.isoformat() if as_of else date.today().isoformat(),
        "tickers": list(tickers),
    }
    path = mooner_output_path(base_dir, SUBSET_FILENAME)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
