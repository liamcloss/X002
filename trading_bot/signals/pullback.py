"""Pullback detection logic."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"close", "high"}


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    prepared.columns = [str(col).lower() for col in prepared.columns]
    return prepared


def detect_pullback(df: pd.DataFrame) -> dict:
    """
    Returns:
    {
        "in_pullback": bool,
        "pullback_pct": float,
        "breakout_high": float,
        "days_since_high": int
    }
    """

    prepared = _prepare_df(df)
    if prepared.empty or not REQUIRED_COLUMNS.issubset(prepared.columns):
        return {
            "in_pullback": False,
            "pullback_pct": np.nan,
            "breakout_high": np.nan,
            "days_since_high": np.nan,
        }

    if len(prepared) < 20:
        return {
            "in_pullback": False,
            "pullback_pct": np.nan,
            "breakout_high": np.nan,
            "days_since_high": np.nan,
        }

    close = prepared["close"]
    high = prepared["high"]

    high_20d = high.rolling(window=20).max()
    last_idx = prepared.index[-1]
    close_last = float(close.loc[last_idx])
    high_20d_last = float(high_20d.loc[last_idx])

    recent_high_window = high.tail(20)
    days_since_high = (len(recent_high_window) - 1) - int(
        np.argmax(recent_high_window.values)
    )

    pullback_pct = (high_20d_last - close_last) / high_20d_last if high_20d_last else np.nan

    in_pullback = (
        days_since_high <= 4
        and 0.01 <= pullback_pct <= 0.05
    )

    return {
        "in_pullback": bool(in_pullback),
        "pullback_pct": float(pullback_pct),
        "breakout_high": float(high_20d_last),
        "days_since_high": int(days_since_high),
    }
