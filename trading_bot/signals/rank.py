"""Candidate ranking logic."""

from __future__ import annotations

import pandas as pd


WEIGHTS = {
    "volume_multiple": 0.50,
    "pct_from_20d_high": 0.25,
    "momentum_5d": 0.25,
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

    required = {"volume_multiple", "pct_from_20d_high", "momentum_5d"}
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

    ranked["score"] = (
        ranked["volume_score"] * WEIGHTS["volume_multiple"]
        + ranked["closeness_score"] * WEIGHTS["pct_from_20d_high"]
        + ranked["momentum_score"] * WEIGHTS["momentum_5d"]
    )

    ranked = ranked.sort_values(by="score", ascending=False)
    return ranked
