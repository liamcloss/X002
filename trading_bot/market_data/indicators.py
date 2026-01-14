"""Stateless indicator helpers for market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma20(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Compute the 20-day simple moving average."""

    return df[column].rolling(window=20, min_periods=20).mean()


def ma50(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Compute the 50-day simple moving average."""

    return df[column].rolling(window=50, min_periods=50).mean()


def high_20(df: pd.DataFrame, column: str = "High") -> pd.Series:
    """Compute the 20-day rolling high."""

    return df[column].rolling(window=20, min_periods=20).max()


def high_50(df: pd.DataFrame, column: str = "High") -> pd.Series:
    """Compute the 50-day rolling high."""

    return df[column].rolling(window=50, min_periods=50).max()


def avg_volume_20(df: pd.DataFrame, column: str = "Volume") -> pd.Series:
    """Compute the 20-day average volume."""

    return df[column].rolling(window=20, min_periods=20).mean()


def atr(df: pd.DataFrame) -> pd.Series:
    """Compute the 14-day Average True Range (ATR)."""

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    ranges = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return pd.Series(ranges, index=df.index).rolling(window=14, min_periods=14).mean()
