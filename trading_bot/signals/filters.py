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

    close_last = float(close.iloc[-1])
    last_20_close = close.tail(20)
    last_50_close = close.tail(50)
    last_20_high = high.tail(20)
    last_20_volume = volume.tail(20)

    if (
        last_20_close.isna().any()
        or last_50_close.isna().any()
        or last_20_high.isna().any()
        or last_20_volume.isna().any()
        or pd.isna(close_last)
    ):
        return prepared.iloc[0:0].copy()

    ma20_last = float(last_20_close.mean())
    ma50_last = float(last_50_close.mean())
    high_20d_last = float(last_20_high.max())
    volume_avg_last = float(last_20_volume.mean())
    volume_last = float(volume.iloc[-1])

    high_values = last_20_high.values
    high_target = float(np.max(high_values)) if len(high_values) else np.nan
    if pd.isna(high_target):
        return prepared.iloc[0:0].copy()
    last_high_idx = int(np.where(high_values == high_target)[0][-1])
    days_since_20d_high = (len(high_values) - 1) - last_high_idx

    volume_multiple = volume_last / volume_avg_last if volume_avg_last else np.nan
    momentum_5d = (close_last / float(close.iloc[-6]) - 1) if len(prepared) > 5 else np.nan
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
