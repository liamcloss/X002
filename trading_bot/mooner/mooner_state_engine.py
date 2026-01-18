"""Mooner regime classification engine."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from trading_bot.market_data import cache

STATES_FILENAME = 'MoonerStates.json'


class MoonerState(str, Enum):
    DORMANT = 'DORMANT'
    WARMING = 'WARMING'
    ARMED = 'ARMED'
    FIRING = 'FIRING'


@dataclass(frozen=True)
class MoonerStateSnapshot:
    ticker: str
    state: MoonerState
    as_of: str
    context: str
    metrics: dict[str, float | None]


def evaluate_mooner_states(
    prices_dir: Path,
    tickers: list[str],
    logger: logging.Logger,
) -> list[MoonerStateSnapshot]:
    """Evaluate mooner regime state for each ticker."""

    snapshots: list[MoonerStateSnapshot] = []
    for ticker in tickers:
        df = cache.load_cache(prices_dir, ticker, logger)
        snapshot = _evaluate_single_ticker(ticker, df, logger)
        if snapshot:
            snapshots.append(snapshot)
    return snapshots


def write_mooner_states(base_dir: Path, snapshots: list[MoonerStateSnapshot]) -> Path | None:
    """Persist the mooner regime states."""

    if not snapshots:
        return None

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "states": [
            {
                "ticker": snapshot.ticker,
                "state": snapshot.state.value,
                "as_of": snapshot.as_of,
                "context": snapshot.context,
                "metrics": snapshot.metrics,
            }
            for snapshot in snapshots
        ],
    }
    path = base_dir / STATES_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def load_mooner_states(base_dir: Path, logger: logging.Logger) -> list[dict]:
    """Load mooner state output if present."""

    path = base_dir / STATES_FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read mooner states from %s: %s", path, exc)
        return []

    if isinstance(payload, dict):
        states = payload.get("states", [])
    elif isinstance(payload, list):
        states = payload
    else:
        return []
    return [state for state in states if isinstance(state, dict)]


def _evaluate_single_ticker(
    ticker: str,
    df: pd.DataFrame,
    logger: logging.Logger,
) -> MoonerStateSnapshot | None:
    if df.empty:
        logger.warning("No price data available for mooner ticker %s.", ticker)
        return None

    df = df.sort_index()
    ohlc = _resolve_ohlc(df)
    if ohlc is None:
        logger.warning("Missing OHLC data for mooner ticker %s.", ticker)
        return None

    high, low, close = ohlc
    if len(close) < 210:
        logger.warning("Insufficient history for mooner ticker %s.", ticker)
        return MoonerStateSnapshot(
            ticker=ticker,
            state=MoonerState.DORMANT,
            as_of=_as_of_date(df),
            context="Insufficient history; defaulting to dormant.",
            metrics={},
        )

    atr20 = _atr(high, low, close, 20).iloc[-1]
    atr60 = _atr(high, low, close, 60).iloc[-1]
    ma50 = close.rolling(window=50).mean().iloc[-1]
    ma200 = close.rolling(window=200).mean().iloc[-1]
    ma50_prev = close.rolling(window=50).mean().iloc[-11] if len(close) >= 210 else ma50
    ma50_slope = ma50 - ma50_prev

    range_20 = _range_pct(high, low, close, 20)
    range_40 = _range_pct(high, low, close, 40)

    prior_high_60 = high.shift(1).rolling(window=60).max().iloc[-1]
    prior_high_20 = high.shift(1).rolling(window=20).max().iloc[-1]

    last_close = close.iloc[-1]
    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    close_position = _close_position(last_close, last_high, last_low)

    metrics = {
        "atr20": float(atr20) if pd.notna(atr20) else None,
        "atr60": float(atr60) if pd.notna(atr60) else None,
        "range_20d_pct": range_20,
        "range_40d_pct": range_40,
        "ma50": float(ma50) if pd.notna(ma50) else None,
        "ma200": float(ma200) if pd.notna(ma200) else None,
        "ma50_slope": float(ma50_slope) if pd.notna(ma50_slope) else None,
        "prior_high_20d": float(prior_high_20) if pd.notna(prior_high_20) else None,
        "prior_high_60d": float(prior_high_60) if pd.notna(prior_high_60) else None,
        "close": float(last_close),
    }

    state, context = _classify_state(
        atr20=atr20,
        atr60=atr60,
        range_20=range_20,
        range_40=range_40,
        close=last_close,
        ma50=ma50,
        ma200=ma200,
        ma50_slope=ma50_slope,
        prior_high_60=prior_high_60,
        prior_high_20=prior_high_20,
        close_position=close_position,
    )

    return MoonerStateSnapshot(
        ticker=ticker,
        state=state,
        as_of=_as_of_date(df),
        context=context,
        metrics=metrics,
    )


def _resolve_ohlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    columns = {str(col).lower(): col for col in df.columns}
    high = _get_series(df, columns, ("high", "high_price", "h"))
    low = _get_series(df, columns, ("low", "low_price", "l"))
    close = _get_series(df, columns, ("close", "close_price", "c", "adj close", "adj_close"))
    if high is None or low is None or close is None:
        return None
    return high, low, close


def _get_series(
    df: pd.DataFrame,
    columns: dict[str, str],
    names: tuple[str, ...],
) -> pd.Series | None:
    for name in names:
        if name in columns:
            return df[columns[name]]
    return None


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window).mean()


def _range_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> float:
    high_roll = high.rolling(window=window).max().iloc[-1]
    low_roll = low.rolling(window=window).min().iloc[-1]
    last_close = close.iloc[-1]
    if pd.isna(high_roll) or pd.isna(low_roll) or last_close == 0:
        return 0.0
    return float((high_roll - low_roll) / last_close)


def _close_position(close: float, high: float, low: float) -> float:
    if high == low:
        return 1.0
    return float((close - low) / (high - low))


def _classify_state(
    *,
    atr20: float,
    atr60: float,
    range_20: float,
    range_40: float,
    close: float,
    ma50: float,
    ma200: float,
    ma50_slope: float,
    prior_high_60: float,
    prior_high_20: float,
    close_position: float,
) -> tuple[MoonerState, str]:
    atr_compression = atr20 < atr60 if pd.notna(atr20) and pd.notna(atr60) else False
    atr_expansion = atr20 > atr60 if pd.notna(atr20) and pd.notna(atr60) else False
    trend_aligned = close > ma50 > ma200 if pd.notna(ma50) and pd.notna(ma200) else False
    ma50_rising = ma50_slope > 0 if pd.notna(ma50_slope) else False
    above_ma50 = close > ma50 if pd.notna(ma50) else False
    air_above = False
    if pd.notna(prior_high_60) and prior_high_60 > 0:
        air_above = ((prior_high_60 - close) / close) <= 0.02
    breakout = False
    if pd.notna(prior_high_60) and prior_high_60 > 0:
        breakout = close > prior_high_60

    if breakout and atr_expansion and close_position >= 0.7:
        return MoonerState.FIRING, "Breakout above range with volatility expansion."

    if atr_compression and trend_aligned and air_above:
        return MoonerState.ARMED, "Compression complete with trend alignment and clear overhead."

    if atr_compression and range_20 <= 0.10 and above_ma50 and ma50_rising:
        return MoonerState.WARMING, "ATR compression with tight range and rising 50D MA."

    if range_40 <= 0.10 and not trend_aligned:
        return MoonerState.DORMANT, "Range-bound with muted volatility."

    return MoonerState.DORMANT, "No actionable mooner regime detected."


def _as_of_date(df: pd.DataFrame) -> str:
    if df.empty:
        return datetime.now(timezone.utc).date().isoformat()
    last_ts = df.index.max()
    if isinstance(last_ts, pd.Timestamp):
        return last_ts.date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


__all__ = [
    'MoonerState',
    'MoonerStateSnapshot',
    'evaluate_mooner_states',
    'load_mooner_states',
    'write_mooner_states',
]
