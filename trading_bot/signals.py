"""Signal generation for breakout continuation with retrace entry."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalResult:
    """Signal result containing ranked candidates."""

    signals: pd.DataFrame


def generate_signals(market_data: pd.DataFrame) -> SignalResult:
    """Generate trading signals.

    TODO: Replace stub logic with validated strategy rules.
    """

    if market_data.empty:
        return SignalResult(signals=pd.DataFrame())

    frame = market_data.sort_values(["symbol", "date"]).copy()
    frame["sma_10"] = (
        frame.groupby("symbol")["close"]
        .transform(lambda series: series.rolling(window=10, min_periods=5).mean())
    )
    frame["high_20"] = (
        frame.groupby("symbol")["close"]
        .transform(lambda series: series.rolling(window=20, min_periods=10).max())
    )
    latest = frame.groupby("symbol").tail(1).copy()

    latest["retrace_pct"] = (latest["high_20"] - latest["close"]) / latest["high_20"]
    latest["trend_pct"] = (
        latest["close"]
        / frame.groupby("symbol")["close"].transform(lambda series: series.shift(10))
        - 1
    )

    candidates = latest[
        (latest["retrace_pct"].between(0.01, 0.03))
        & (latest["close"] >= latest["high_20"] * 0.97)
        & (latest["close"] >= latest["sma_10"])
    ].copy()

    if candidates.empty:
        return SignalResult(signals=pd.DataFrame())

    candidates["score"] = (
        (1 - candidates["retrace_pct"]) * 100 + candidates["trend_pct"].fillna(0) * 50
    )
    candidates["signal"] = "BREAKOUT_RETRACE"
    output = candidates.sort_values("score", ascending=False)[
        ["symbol", "close", "score", "signal"]
    ].rename(columns={"close": "entry_price"})

    return SignalResult(signals=output.reset_index(drop=True))
