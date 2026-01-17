"""Yfinance client wrapper with batching and retry handling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

BATCH_SIZE = 10


class YFinanceError(RuntimeError):
    """Base exception for yfinance failures."""


@dataclass
class TickerDownloadError(YFinanceError):
    """Controlled exception for per-ticker download failures."""

    ticker: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.ticker}: {self.message}"


def download_batch(
    tickers: Iterable[str],
    start: datetime | None,
    end: datetime | None,
    logger: logging.Logger,
    retries: int = 3,
    backoff_seconds: int = 2,
) -> pd.DataFrame:
    """Download a batch of tickers from yfinance with retry logic."""

    tickers_list = list(tickers)
    if not tickers_list:
        raise YFinanceError("No tickers provided for batch download.")

    attempt = 0
    last_error: Exception | None = None
    while attempt < retries:
        try:
            logger.debug(
                "Fetching yfinance batch (attempt %s/%s): %s",
                attempt + 1,
                retries,
                tickers_list,
            )
            data = yf.download(
                tickers=tickers_list,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if data is None or data.empty:
                raise YFinanceError("yfinance returned empty dataset for batch.")
            return data
        except Exception as exc:  # noqa: BLE001 - controlled error reporting
            last_error = exc
            attempt += 1
            if attempt >= retries:
                break
            sleep_for = backoff_seconds**attempt
            logger.warning(
                "Batch download failed (attempt %s/%s). Retrying in %ss: %s",
                attempt,
                retries,
                sleep_for,
                exc,
            )
            time.sleep(sleep_for)

    raise YFinanceError(f"Batch download failed after {retries} retries: {last_error}")


def extract_ticker_data(batch_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract per-ticker OHLCV data or raise a controlled exception."""

    if batch_data is None or batch_data.empty:
        raise TickerDownloadError(ticker=ticker, message="Batch data was empty.")

    if isinstance(batch_data.columns, pd.MultiIndex):
        if ticker not in batch_data.columns.get_level_values(0):
            raise TickerDownloadError(ticker=ticker, message="Ticker missing from batch data.")
        ticker_data = batch_data.xs(ticker, axis=1, level=0, drop_level=True)
    else:
        ticker_data = batch_data.copy()

    ticker_data = ticker_data.dropna(how="all")
    if ticker_data.empty:
        raise TickerDownloadError(ticker=ticker, message="Ticker data was empty.")

    ticker_data.index = pd.to_datetime(ticker_data.index)
    return ticker_data

from src.market_data.yfinance_client import get_quote  # noqa: E402,F401
