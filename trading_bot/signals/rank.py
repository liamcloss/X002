"""Candidate ranking logic."""

from __future__ import annotations

import pandas as pd


WEIGHTS = {
    "volume_multiple": 0.22,
    "pct_from_20d_high": 0.13,
    "momentum_5d": 0.17,
    "rr": 0.20,
    "stop_pct": 0.10,
    "spread_pct": 0.10,
    "atr_pct": 0.08,
}


def _min_max_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([0.0] * len(series), index=series.index)
    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)
    normalized = (series - min_val) / (max_val - min_val)
    if not higher_is_better:
        normalized = 1 - normalized
    return normalized


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe sorted descending by final score.
    """

    if df.empty:
        return df.copy()

    required = {
        "volume_multiple",
        "pct_from_20d_high",
        "momentum_5d",
        "rr",
        "stop_pct",
        "spread_pct",
        "atr_pct",
    }
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns for ranking")

    ranked = df.copy()

    ranked["volume_score"] = _min_max_normalize(
        ranked["volume_multiple"], higher_is_better=True
    )
    ranked["closeness_score"] = _min_max_normalize(
        ranked["pct_from_20d_high"], higher_is_better=False
    )
    ranked["momentum_score"] = _min_max_normalize(
        ranked["momentum_5d"], higher_is_better=True
    )
    ranked["rr_score"] = _min_max_normalize(ranked["rr"], higher_is_better=True)
    ranked["stop_score"] = _min_max_normalize(
        ranked["stop_pct"], higher_is_better=False
    )
    ranked["spread_score"] = _min_max_normalize(
        ranked["spread_pct"], higher_is_better=False
    )
    ranked["atr_score"] = _min_max_normalize(
        ranked["atr_pct"], higher_is_better=False
    )

    ranked["score"] = (
        ranked["volume_score"] * WEIGHTS["volume_multiple"]
        + ranked["closeness_score"] * WEIGHTS["pct_from_20d_high"]
        + ranked["momentum_score"] * WEIGHTS["momentum_5d"]
        + ranked["rr_score"] * WEIGHTS["rr"]
        + ranked["stop_score"] * WEIGHTS["stop_pct"]
        + ranked["spread_score"] * WEIGHTS["spread_pct"]
        + ranked["atr_score"] * WEIGHTS["atr_pct"]
    )

    ranked = ranked.sort_values(by="score", ascending=False)
    return ranked
