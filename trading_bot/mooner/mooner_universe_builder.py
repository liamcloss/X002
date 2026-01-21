"""Mooner universe construction helper."""

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
from trading_bot.universe.active import ensure_active_column
from trading_bot.mooner.mooner_candidate_pool import FX_TO_GBP, DEFAULT_FX

UNIVERSE_FILENAME = "MoonerUniverse.json"


def build_mooner_universe(base_dir: Path, candidates: Iterable[str], logger: logging.Logger) -> list[str]:
    """Filter the candidate pool into the Mooner universe."""

    prices_dir = base_dir / "data" / "prices"
    currency_map = _build_currency_map(base_dir, logger)
    universe: list[str] = []
    as_of_date: date | None = None

    for ticker in candidates:
        df = cache.load_cache(prices_dir, ticker, logger)
        if df.empty:
            continue
        df = df.sort_index()
        last_date = _last_cached_date(df)
        if last_date:
            as_of_date = last_date if as_of_date is None else max(as_of_date, last_date)
        if _passes_universe(df, currency_map.get(ticker, "GBP")):
            universe.append(ticker)

    _write_universe(base_dir, universe, as_of_date)
    logger.info("Mooner universe built (%s tickers).", len(universe))
    return universe


def _passes_universe(df: pd.DataFrame, currency_code: str) -> bool:
    if len(df) < 250:
        return False

    close = _get_series(df, ("close", "Close"))
    if close is None or close.empty:
        return False
    close = close.dropna()
    if len(close) < 250:
        return False

    price = close.iloc[-1]
    if pd.isna(price):
        return False
    
    mooner_config = config.CONFIG["mooner"]
    price_gbp = _to_gbp(price, currency_code)
    if price_gbp < mooner_config["price_min_gbp"]:
        return False

    ma200 = close.rolling(window=200, min_periods=200).mean()
    if ma200.isna().all():
        return False
    last_ma200 = _last_valid(ma200)
    prev_ma200 = _nth_last_valid(ma200, 10)
    if last_ma200 is None or prev_ma200 is None or last_ma200 - prev_ma200 < 0:
        return False

    drawdown = _max_drawdown(close.tail(250))
    if drawdown > mooner_config["drawdown_max"]:
        return False

    high90 = close.tail(90).max()
    if high90 > price * (1 + mooner_config["resistance_buffer"]):
        return False

    return True


def _build_currency_map(base_dir: Path, logger: logging.Logger) -> dict[str, str]:
    path = base_dir / "universe" / "clean" / "universe.parquet"
    df = ensure_active_column(path, logger)
    mapping: dict[str, str] = {}
    if df.empty or "ticker" not in df.columns:
        return mapping
    for _, row in df.iterrows():
        ticker = _normalize(row.get("ticker"))
        if not ticker:
            continue
        mapping[ticker] = _normalize(row.get("currency_code")) or "GBP"
    return mapping


def _write_universe(base_dir: Path, tickers: Iterable[str], as_of: date | None) -> None:
    payload = {
        "as_of": as_of.isoformat() if as_of else date.today().isoformat(),
        "tickers": sorted(set(tickers)),
    }
    path = mooner_output_path(base_dir, UNIVERSE_FILENAME)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _get_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for key in candidates:
        if key in df.columns:
            return pd.to_numeric(df[key], errors="coerce")
    return None


def _last_cached_date(df: pd.DataFrame) -> date | None:
    if df.empty:
        return None
    idx = pd.to_datetime(df.index)
    return idx.max().date()


def _normalize(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.upper() if text else None


def _to_gbp(value: float, currency: str) -> float:
    rate = FX_TO_GBP.get(currency.upper(), DEFAULT_FX)
    return value * rate


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 1.0
    cumulative_max = series.cummax()
    drawdowns = (cumulative_max - series) / cumulative_max
    drawdowns = drawdowns.replace([pd.NA, pd.NaT], 0.0)
    return float(drawdowns.max())


def _last_valid(series: pd.Series) -> float | None:
    if series.empty:
        return None
    for value in reversed(series.tolist()):
        if pd.notna(value):
            return float(value)
    return None


def _nth_last_valid(series: pd.Series, n: int) -> float | None:
    if series.empty:
        return None
    vals = [value for value in reversed(series.tolist()) if pd.notna(value)]
    if len(vals) < n:
        return None
    return float(vals[n - 1])
