"""Mooner regime state classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import pandas as pd

from trading_bot import config
from trading_bot.market_data import cache
from trading_bot.paths import mooner_state_path


class MoonerState(str, Enum):
    DORMANT = "DORMANT"
    WARMING = "WARMING"
    ARMED = "ARMED"
    FIRING = "FIRING"


@dataclass(frozen=True)
class MoonerStateSnapshot:
    ticker: str
    state: MoonerState
    state_since: str
    context: str
    previous_state: str | None


def evaluate_mooner_states(
    base_dir: Path,
    tickers: Iterable[str],
    logger: logging.Logger,
) -> tuple[list[MoonerStateSnapshot], dict[str, dict[str, str]]]:
    """Evaluate each mooner ticker for its current regime."""

    previous_states = _load_states(base_dir)
    new_states: dict[str, dict[str, str]] = {}
    snapshots: list[MoonerStateSnapshot] = []
    today = datetime.now(timezone.utc).date().isoformat()

    prices_dir = base_dir / "data" / "prices"

    for ticker in tickers:
        df = cache.load_cache(prices_dir, ticker, logger)
        if df.empty:
            continue
        df = df.sort_index()
        state, context = _classify_state(df)
        prev_entry = previous_states.get(ticker)
        prev_state_name = prev_entry.get("state") if prev_entry else None
        state_since = prev_entry.get("state_since") if prev_state_name == state.value else today

        new_states[ticker] = {
            "state": state.value,
            "state_since": state_since,
            "context": context,
        }
        snapshots.append(
            MoonerStateSnapshot(
                ticker=ticker,
                state=state,
                state_since=state_since,
                context=context,
                previous_state=prev_state_name,
            )
        )

    _write_states(base_dir, new_states)
    return snapshots, previous_states


def _classify_state(df: pd.DataFrame) -> Tuple[MoonerState, str]:
    close = _get_series(df, ("close", "Close"))
    high = _get_series(df, ("high", "High"))
    low = _get_series(df, ("low", "Low"))
    volume = _get_series(df, ("volume", "Volume"))
    if close is None or close.empty or high is None or low is None or volume is None:
        return MoonerState.DORMANT, "Insufficient data"

    close = close.dropna()
    high = high.dropna()
    low = low.dropna()
    volume = volume.dropna()
    if len(close) < 90:
        return MoonerState.DORMANT, "Insufficient history"

    atr20 = _compute_atr(df, 20)
    atr60 = _compute_atr(df, 60)
    atr20_latest = _last_valid(atr20)
    atr60_latest = _last_valid(atr60)
    atr20_prev = _nth_last_valid(atr20, 2)

    ma50 = close.rolling(window=50, min_periods=30).mean()
    ma200 = close.rolling(window=200, min_periods=200).mean()
    ma50_latest = _last_valid(ma50)
    ma50_prev = _nth_last_valid(ma50, 2)
    ma200_latest = _last_valid(ma200)
    price = close.iloc[-1]

    high30 = high.tail(30).max()
    low30 = low.tail(30).min()
    range_30 = high30 - low30
    range_pct = range_30 / max(low30, 1.0)

    avg_volume_30 = volume.tail(30).mean()
    volume_latest = volume.iloc[-1]

    atr_rising = _is_atr_rising(atr20)
    ma50_rising = ma50_prev is not None and ma50_latest is not None and ma50_latest > ma50_prev

    if (
        _is_firing(
            price=price,
            high30=high30,
            atr20=atr20,
            volume_latest=volume_latest,
            avg_volume_30=avg_volume_30,
        )
    ):
        return MoonerState.FIRING, "Breakout beyond the recent range with increasing ATR and volume."

    if (
        _is_armed(
            price=price,
            ma50_latest=ma50_latest,
            ma200_latest=ma200_latest,
            atr20_latest=atr20_latest,
            atr60_latest=atr60_latest,
            atr_rising=atr_rising,
            high=high,
        )
    ):
        return MoonerState.ARMED, "ATR compression complete with clean structure."

    if (
        _is_warming(
            atr20_latest=atr20_latest,
            atr60_latest=atr60_latest,
            range_pct=range_pct,
            price=price,
            ma50_latest=ma50_latest,
            ma50_rising=ma50_rising,
        )
    ):
        return MoonerState.WARMING, "Compression confirmed; price above rising 50d MA."

    if _is_dormant(atr20_latest=atr20_latest, atr20_prev=atr20_prev, price=price, ma50_latest=ma50_latest, high30=high30, low30=low30):
        return MoonerState.DORMANT, "Quiet ATR; price range-bound below the 50d MA."

    return MoonerState.DORMANT, "Default dormant state."


def _is_firing(
    *,
    price: float,
    high30: float,
    atr20: pd.Series,
    volume_latest: float,
    avg_volume_30: float,
) -> bool:
    if pd.isna(price) or avg_volume_30 <= 0:
        return False
    if price <= high30:
        return False
    if volume_latest < config.MOONER_STATE_VOLUME_MULTIPLIER * avg_volume_30:
        return False
    return _is_atr_rising(atr20, days=config.MOONER_STATE_ATR_RISE_DAYS)


def _is_armed(
    *,
    price: float,
    ma50_latest: float | None,
    ma200_latest: float | None,
    atr20_latest: float | None,
    atr60_latest: float | None,
    atr_rising: bool,
    high: pd.Series,
) -> bool:
    if (
        ma50_latest is None
        or ma200_latest is None
        or atr20_latest is None
        or atr60_latest is None
    ):
        return False
    if not (ma50_latest > ma200_latest and price > ma50_latest):
        return False
    if atr20_latest >= 0.65 * atr60_latest:
        return False
    if not atr_rising:
        return False
    resistance = high.tail(60).max()
    if resistance > price * 1.08:
        return False
    return True


def _is_warming(
    *,
    atr20_latest: float | None,
    atr60_latest: float | None,
    range_pct: float,
    price: float,
    ma50_latest: float | None,
    ma50_rising: bool,
) -> bool:
    if atr20_latest is None or atr60_latest is None or ma50_latest is None:
        return False
    if atr20_latest > 0.75 * atr60_latest:
        return False
    if range_pct > 0.10:
        return False
    if price <= ma50_latest:
        return False
    return ma50_rising


def _is_dormant(
    *,
    atr20_latest: float | None,
    atr20_prev: float | None,
    price: float,
    ma50_latest: float | None,
    high30: float,
    low30: float,
) -> bool:
    if atr20_latest is None or atr20_prev is None or ma50_latest is None:
        return False
    atr_flat_or_falling = atr20_latest <= atr20_prev
    inside_range = low30 <= price <= high30
    below_ma50 = price <= ma50_latest
    return atr_flat_or_falling and inside_range and below_ma50


def _is_atr_rising(atr20: pd.Series, days: int = 3) -> bool:
    values = [value for value in atr20.dropna().tolist()[-days:]]
    if len(values) < days:
        return False
    return all(
        later > earlier
        for earlier, later in zip(values, values[1:])
    )


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
    values = [value for value in reversed(series.tolist()) if pd.notna(value)]
    if len(values) < n:
        return None
    return float(values[n - 1])


def _load_states(base_dir: Path) -> dict[str, dict[str, str]]:
    path = mooner_state_path(base_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def _write_states(base_dir: Path, states: dict[str, dict[str, str]]) -> None:
    path = mooner_state_path(base_dir)
    path.write_text(json.dumps(states, indent=2), encoding="utf-8")
