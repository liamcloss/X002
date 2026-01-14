"""Market data refresh orchestrator for yfinance."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from trading_bot.logging_setup import setup_logging
from trading_bot.market_data import cache, yfinance_client

BATCH_SIZE = 10


def run() -> None:
    """Run the market data refresh pipeline."""

    base_dir = Path(__file__).resolve().parents[2]
    logger = _ensure_logger(base_dir)

    universe_path = base_dir / "universe" / "clean" / "universe.parquet"
    if not universe_path.exists():
        logger.error("Universe file missing: %s", universe_path)
        return

    tickers = _load_active_tickers(universe_path, logger)
    if not tickers:
        logger.warning("No active tickers found in universe.")
        return

    random.shuffle(tickers)
    prices_dir = cache.ensure_prices_dir(base_dir)

    total = len(tickers)
    successes = 0
    failures = 0

    for batch_index, batch in enumerate(_batches(tickers, BATCH_SIZE), start=1):
        logger.info("Starting batch %s with %s tickers.", batch_index, len(batch))
        batch_payload = _prepare_batch(batch, prices_dir, logger)
        if not batch_payload:
            logger.info("Batch %s: all tickers up to date.", batch_index)
            successes += len(batch)
            continue

        min_start = min(payload["start"] for payload in batch_payload.values())
        end = _last_business_day() + pd.Timedelta(days=1)

        try:
            batch_data = yfinance_client.download_batch(
                tickers=list(batch_payload.keys()),
                start=min_start,
                end=end.to_pydatetime(),
                logger=logger,
            )
        except yfinance_client.YFinanceError as exc:
            logger.error("Batch %s failed: %s", batch_index, exc)
            failures += len(batch_payload)
            continue

        for ticker, payload in batch_payload.items():
            try:
                ticker_data = yfinance_client.extract_ticker_data(batch_data, ticker)
                cache.update_cache(
                    prices_dir=prices_dir,
                    ticker=ticker,
                    new_data=ticker_data,
                    logger=logger,
                    existing=payload["existing"],
                    last_date=payload["last_date"],
                )
                successes += 1
            except yfinance_client.TickerDownloadError as exc:
                logger.warning("Skipping %s due to data error: %s", ticker, exc)
                failures += 1

        logger.info("Finished batch %s.", batch_index)

    logger.info(
        "Market data refresh complete. Total: %s, Success: %s, Failed: %s",
        total,
        successes,
        failures,
    )


def _ensure_logger(base_dir: Path) -> logging.Logger:
    logger = logging.getLogger("trading_bot")
    if not logger.handlers:
        return setup_logging(base_dir / "logs")
    return logger


def _load_active_tickers(universe_path: Path, logger: logging.Logger) -> list[str]:
    df = pd.read_parquet(universe_path)

    if "active" in df.columns:
        df = df[df["active"].fillna(False)]
    else:
        logger.warning("Universe missing 'active' column; defaulting to all rows.")

    if "ticker" not in df.columns:
        raise RuntimeError("Universe parquet missing required 'ticker' column.")

    tickers = df["ticker"].dropna().astype(str).unique().tolist()
    return tickers


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _prepare_batch(
    tickers: list[str],
    prices_dir: Path,
    logger: logging.Logger,
) -> dict[str, dict[str, object]]:
    batch_payload: dict[str, dict[str, object]] = {}
    last_business_day = _last_business_day()

    for ticker in tickers:
        existing = cache.load_cache(prices_dir, ticker, logger)
        last_date = cache.last_cached_date(existing)
        if last_date is not None and last_date >= last_business_day:
            logger.info("%s already up to date (last cached %s).", ticker, last_date.date())
            continue

        start = cache.calculate_fetch_start(last_date)
        batch_payload[ticker] = {
            "start": start,
            "existing": existing,
            "last_date": last_date,
        }

    return batch_payload


def _last_business_day() -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    if today.dayofweek >= 5:
        today -= pd.tseries.offsets.BDay(1)
    return today


def main() -> None:
    run()


if __name__ == "__main__":
    main()
