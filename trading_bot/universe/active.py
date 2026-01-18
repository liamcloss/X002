"""Universe active flag helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def ensure_active_column(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Universe file missing: {path}')

    df = pd.read_parquet(path)
    if df.empty:
        return df

    if 'active' not in df.columns:
        df = df.copy()
        df['active'] = _infer_active(df)
        df.to_parquet(path, index=False)
        logger.info('Added active column to universe file: %s', path)

    return df


def deactivate_universe_ticker(
    path: Path,
    ticker: str,
    logger: logging.Logger,
) -> bool:
    df = ensure_active_column(path, logger)
    if df.empty or 'ticker' not in df.columns:
        return False

    ticker_upper = str(ticker).strip().upper()
    if not ticker_upper:
        return False

    mask = df['ticker'].astype(str).str.upper() == ticker_upper
    if not mask.any():
        return False

    if df.loc[mask, 'active'].astype(bool).eq(False).all():
        return False

    df.loc[mask, 'active'] = False
    df.to_parquet(path, index=False)
    logger.warning('Deactivated %s in universe after repeated data failures.', ticker_upper)
    return True


def _infer_active(df: pd.DataFrame) -> pd.Series:
    if 'max_open_quantity' in df.columns:
        series = pd.to_numeric(df['max_open_quantity'], errors='coerce')
        return series.fillna(1).astype(float) > 0
    return pd.Series([True] * len(df), index=df.index)


__all__ = [
    'deactivate_universe_ticker',
    'ensure_active_column',
]
