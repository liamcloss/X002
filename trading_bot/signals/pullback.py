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

    last_high_window = high.tail(20)
    if last_high_window.isna().any():
        return {
            "in_pullback": False,
            "pullback_pct": np.nan,
            "breakout_high": np.nan,
            "days_since_high": np.nan,
        }

    high_values = last_high_window.values
    high_target = float(np.max(high_values)) if len(high_values) else np.nan
    if np.isnan(high_target):
        return {
            "in_pullback": False,
            "pullback_pct": np.nan,
            "breakout_high": np.nan,
            "days_since_high": np.nan,
        }

    last_high_idx = int(np.where(high_values == high_target)[0][-1])
    days_since_high = (len(high_values) - 1) - last_high_idx
    close_last = float(close.iloc[-1])
    pullback_pct = (high_target - close_last) / high_target if high_target else np.nan

    in_pullback = (
        days_since_high <= 4
        and 0.01 <= pullback_pct <= 0.05
    )

    return {
        "in_pullback": bool(in_pullback),
        "pullback_pct": float(pullback_pct),
        "breakout_high": float(high_target),
        "days_since_high": int(days_since_high),
    }
