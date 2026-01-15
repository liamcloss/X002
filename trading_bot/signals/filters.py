"""Signal filters for candidate selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"close", "high", "volume"}


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prepared = df.copy()
    prepared.columns = [str(col).lower() for col in prepared.columns]
    return prepared


def _has_required_columns(df: pd.DataFrame) -> bool:
    return REQUIRED_COLUMNS.issubset(df.columns)


def apply_filters(
    df: pd.DataFrame,
    max_pct_from_20d_high: float = 0.01,
) -> pd.DataFrame:
    """
    Input:
        df contains OHLCV + indicators for a single ticker
        max_pct_from_20d_high controls the maximum pullback from the 20d high
    Output:
        Returns a row with computed metrics if all filters pass,
        otherwise returns empty dataframe.
    """

    prepared = _prepare_df(df)
    if prepared.empty or not _has_required_columns(prepared):
        return prepared.iloc[0:0].copy()

    if len(prepared) < 50:
        return prepared.iloc[0:0].copy()

    close = prepared["close"]
    high = prepared["high"]
    volume = prepared["volume"]

    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    high_20d = high.rolling(window=20).max()
    volume_avg_20d = volume.rolling(window=20).mean()

    last_idx = prepared.index[-1]
    close_last = float(close.loc[last_idx])
    ma20_last = float(ma20.loc[last_idx])
    ma50_last = float(ma50.loc[last_idx])
    high_20d_last = float(high_20d.loc[last_idx])
    volume_avg_last = float(volume_avg_20d.loc[last_idx])
    volume_last = float(volume.loc[last_idx])

    recent_high_window = high.tail(20)
    if recent_high_window.isna().any():
        return prepared.iloc[0:0].copy()
    high_values = recent_high_window.values
    high_target = high_values.max()
    if pd.isna(high_target):
        return prepared.iloc[0:0].copy()
    last_high_idx = int(np.where(high_values == high_target)[0][-1])
    days_since_20d_high = (len(high_values) - 1) - last_high_idx

    volume_multiple = volume_last / volume_avg_last if volume_avg_last else np.nan
    momentum_5d = (close_last / float(close.shift(5).loc[last_idx]) - 1) if len(prepared) > 5 else np.nan
    extension_from_ma50 = (close_last - ma50_last) / ma50_last if ma50_last else np.nan
    pct_from_20d_high = (high_20d_last - close_last) / high_20d_last if high_20d_last else np.nan

    trend_ok = close_last > ma20_last and ma20_last > ma50_last
    breakout_ok = (
        pct_from_20d_high <= max_pct_from_20d_high
        and days_since_20d_high <= 4
    )
    volume_ok = volume_multiple >= 1.5
    momentum_ok = momentum_5d >= 0.03
    extension_ok = extension_from_ma50 <= 0.20

    if not (trend_ok and breakout_ok and volume_ok and momentum_ok and extension_ok):
        return prepared.iloc[0:0].copy()

    result = prepared.tail(1).copy()
    result["close_price"] = close_last
    result["ma20"] = ma20_last
    result["ma50"] = ma50_last
    result["high_20d"] = high_20d_last
    result["days_since_20d_high"] = days_since_20d_high
    result["volume_avg_20d"] = volume_avg_last
    result["volume_multiple"] = volume_multiple
    result["momentum_5d"] = momentum_5d
    result["extension_from_ma50"] = extension_from_ma50
    result["pct_from_20d_high"] = pct_from_20d_high

    return result
