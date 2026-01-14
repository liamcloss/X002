"""Market data refresh orchestrator for yfinance."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from trading_bot.logging_setup import setup_logging
from trading_bot.market_data import cache, yfinance_client

BATCH_SIZE = 10
RATE_LIMIT_DELAY_RANGE = (1.0, 2.0)
BURST_BATCHES = 3
BURST_COOLDOWN_SECONDS = 4.0
RETRY_STATE_FILE = "prices_retry_state.json"
MAX_CONSECUTIVE_FAILURES = 3
MAX_DEFERRALS = 3
DEFERRAL_WINDOW = (7, 14)


def run() -> None:
    """Run the market data refresh pipeline."""

    base_dir = Path(__file__).resolve().parents[2]
    logger = _ensure_logger(base_dir)
    universe_path = base_dir / "universe" / "clean" / "universe.parquet"
    if not universe_path.exists():
        logger.error("Universe file missing: %s", universe_path)
        return

    retry_state_path = base_dir / "state" / RETRY_STATE_FILE
    retry_state = _load_retry_state(retry_state_path)

    tickers = _load_active_tickers(universe_path, logger)
    if not tickers:
        logger.warning("No active tickers found in universe.")
        return

    random.shuffle(tickers)
    prices_dir = cache.ensure_prices_dir(base_dir)

    total = len(tickers)
    successes = 0
    failures = 0
    last_business_day = _last_business_day()

    batches = list(_batches(tickers, BATCH_SIZE))
    total_batches = len(batches)
    for batch_index, batch in enumerate(batches, start=1):
        logger.info("Starting batch %s with %s tickers.", batch_index, len(batch))
        current_time = datetime.now()
        batch_payload = _prepare_batch(
            batch,
            prices_dir,
            logger,
            retry_state,
            current_time,
            last_business_day,
        )

        if not batch_payload:
            logger.info("Batch %s: all tickers up to date or deferred.", batch_index)
            successes += len(batch)
            _enforce_rate_limit(batch_index, total_batches, logger)
            continue

        min_start = min(payload["start"] for payload in batch_payload.values())
        end = last_business_day + pd.Timedelta(days=1)
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
            for alias in batch_payload:
                _record_failure(alias, retry_state, retry_state_path, logger, datetime.now())
            _enforce_rate_limit(batch_index, total_batches, logger)
            continue

        for alias, payload in batch_payload.items():
            base_ticker = payload["base_ticker"]
            try:
                ticker_data = yfinance_client.extract_ticker_data(batch_data, alias)
                cache.update_cache(
                    prices_dir=prices_dir,
                    ticker=base_ticker,
                    new_data=ticker_data,
                    logger=logger,
                    existing=payload["existing"],
                    last_date=payload["last_date"],
                )
                successes += 1
                _record_success(alias, retry_state, retry_state_path)
            except yfinance_client.TickerDownloadError as exc:
                logger.warning(
                    "Skipping %s (%s) due to data error: %s", base_ticker, alias, exc
                )
                failures += 1
                _record_failure(alias, retry_state, retry_state_path, logger, datetime.now())
        logger.info("Finished batch %s.", batch_index)
        _enforce_rate_limit(batch_index, total_batches, logger)

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


def _load_retry_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_retry_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _record_success(alias: str, state: dict[str, dict[str, Any]], path: Path) -> None:
    if alias in state:
        del state[alias]
        _save_retry_state(path, state)


def _record_failure(
    alias: str,
    state: dict[str, dict[str, Any]],
    path: Path,
    logger: logging.Logger,
    now: datetime,
) -> None:
    entry = state.setdefault(alias, {})
    last_failure = _parse_iso(entry.get("last_failure"))
    delta_days = (
        (now.date() - last_failure.date()).days if last_failure else None
    )
    if delta_days == 1:
        entry["consecutive_days"] = entry.get("consecutive_days", 0) + 1
    else:
        entry["consecutive_days"] = 1
    entry["last_failure"] = now.isoformat()
    entry["last_attempt"] = now.isoformat()

    if entry["consecutive_days"] >= MAX_CONSECUTIVE_FAILURES:
        deferral_days = random.randint(*DEFERRAL_WINDOW)
        entry["next_try"] = (now + timedelta(days=deferral_days)).isoformat()
        entry["consecutive_days"] = 0
        entry["deferral_count"] = entry.get("deferral_count", 0) + 1
        entry["last_deferral"] = now.isoformat()
        logger.info(
            "Deferring %s for %s days after %s consecutive failures.",
            alias,
            deferral_days,
            MAX_CONSECUTIVE_FAILURES,
        )
        if entry["deferral_count"] >= MAX_DEFERRALS:
            entry["blacklisted"] = True
            logger.warning(
                "%s blacklisted after %s deferrals.", alias, entry["deferral_count"]
            )
    _save_retry_state(path, state)


def _select_market_identifier(row: pd.Series, fallback: str) -> str:
    for column in ("short_name", "isin"):
        candidate = _normalize_identifier(row.get(column))
        if candidate:
            return candidate
    return fallback


def _normalize_identifier(value: object | None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _load_active_tickers(universe_path: Path, logger: logging.Logger) -> list[dict[str, str]]:
    df = pd.read_parquet(universe_path)

    if "active" in df.columns:
        df = df[df["active"].fillna(False)]
    else:
        logger.warning("Universe missing 'active' column; defaulting to all rows.")

    if "ticker" not in df.columns:
        raise RuntimeError("Universe parquet missing required 'ticker' column.")

    tickers: list[dict[str, str]] = []
    seen_aliases: set[str] = set()
    for _, row in df.iterrows():
        base_ticker = _normalize_identifier(row.get("ticker"))
        if not base_ticker:
            logger.warning("Skipping universe row with missing ticker: %s", row.to_dict())
            continue
        alias = _select_market_identifier(row, base_ticker)
        normalized_alias = alias.upper()
        if normalized_alias in seen_aliases:
            logger.warning("Skipping duplicate market identifier %s for %s", alias, base_ticker)
            continue
        seen_aliases.add(normalized_alias)
        tickers.append({"ticker": base_ticker, "alias": alias})
    return tickers


def _batches(items: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _prepare_batch(
    tickers: list[dict[str, str]],
    prices_dir: Path,
    logger: logging.Logger,
    retry_state: dict[str, dict[str, Any]],
    current_time: datetime,
    last_business_day: pd.Timestamp,
) -> dict[str, dict[str, object]]:
    batch_payload: dict[str, dict[str, object]] = {}

    for entry in tickers:
        base_ticker = entry["ticker"]
        alias = entry["alias"]
        info = retry_state.get(alias, {})
        if info.get("blacklisted"):
            logger.debug("Skipping %s because it is blacklisted.", alias)
            continue
        if not _should_attempt(info, current_time):
            logger.debug("Deferring %s until %s.", alias, info.get("next_try"))
            continue
        existing = cache.load_cache(prices_dir, base_ticker, logger)
        last_date = cache.last_cached_date(existing)
        if last_date is not None and last_date >= last_business_day:
            logger.info("%s already up to date (last cached %s).", base_ticker, last_date.date())
            continue
        start = cache.calculate_fetch_start(last_date)
        batch_payload[alias] = {
            "base_ticker": base_ticker,
            "start": start,
            "existing": existing,
            "last_date": last_date,
        }
    return batch_payload


def _should_attempt(info: dict[str, Any], current_time: datetime) -> bool:
    next_try = _parse_iso(info.get("next_try"))
    if next_try and current_time < next_try:
        return False
    last_attempt = _parse_iso(info.get("last_attempt"))
    if last_attempt and last_attempt.date() == current_time.date():
        return False
    return True


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _enforce_rate_limit(batch_index: int, total_batches: int, logger: logging.Logger) -> None:
    if batch_index >= total_batches:
        return
    if batch_index % BURST_BATCHES == 0:
        delay = BURST_COOLDOWN_SECONDS
    else:
        delay = random.uniform(*RATE_LIMIT_DELAY_RANGE)
    logger.debug("Sleeping %.1fs to remain within yfinance rate limits.", delay)
    time.sleep(delay)


def _last_business_day() -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    if today.dayofweek >= 5:
        today -= pd.tseries.offsets.BDay(1)
    return today


def main() -> None:
    run()


if __name__ == "__main__":
    main()
