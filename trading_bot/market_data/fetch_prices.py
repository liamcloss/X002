"""Market data refresh orchestrator for yfinance."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from trading_bot.logging_setup import setup_logging
from trading_bot.market_data import cache, yfinance_client
from trading_bot.symbols import yfinance_symbol
from trading_bot.universe.active import deactivate_universe_ticker, ensure_active_column

RETRY_STATE_FILE = "prices_retry_state.json"
MARKET_STATE_FILE = "market_data_state.json"
LOCK_FILE = "market_data.lock"
SYMBOL_MAP_FILE = 'market_symbol_map.json'

BATCH_SIZE = 20
RATE_LIMIT_DELAY_RANGE = (0.6, 1.2)
BURST_BATCHES = 4
BURST_COOLDOWN_SECONDS = 3.0
ADAPTIVE_PENALTY_STEP = 0.4
ADAPTIVE_PENALTY_DECAY = 0.2
ADAPTIVE_MAX_PENALTY = 3.0
RETRY_STATE_FILE = "prices_retry_state.json"
MAX_CONSECUTIVE_FAILURES = 3
MAX_DEFERRALS = 3
DEFERRAL_WINDOW = (7, 14)


@dataclass
class AdaptiveRateLimiter:
    base_delay_range: tuple[float, float]
    burst_batches: int
    burst_cooldown_seconds: float
    penalty: float = 0.0
    consecutive_failures: int = 0

    def record_success(self) -> None:
        if self.consecutive_failures:
            self.consecutive_failures = 0
        self.penalty = max(0.0, self.penalty - ADAPTIVE_PENALTY_DECAY)

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        penalty_step = ADAPTIVE_PENALTY_STEP * self.consecutive_failures
        self.penalty = min(ADAPTIVE_MAX_PENALTY, self.penalty + penalty_step)

    def next_delay(self, batch_index: int) -> tuple[float, float]:
        if self.burst_batches and batch_index % self.burst_batches == 0:
            base_delay = self.burst_cooldown_seconds
        else:
            base_delay = random.uniform(*self.base_delay_range)
        delay = base_delay * (1.0 + self.penalty)
        return base_delay, delay


def run() -> None:
    """Run the market data refresh pipeline."""

    base_dir = Path(__file__).resolve().parents[2]
    logger = _ensure_logger(base_dir)
    run_started_at = datetime.now(timezone.utc)
    run_start_monotonic = time.monotonic()

    universe_path = base_dir / "universe" / "clean" / "universe.parquet"
    retry_state_path = base_dir / "state" / RETRY_STATE_FILE
    retry_state = _load_retry_state(retry_state_path)
    market_state_path = base_dir / "state" / MARKET_STATE_FILE
    market_state = _load_market_state(market_state_path)
    lock_path = base_dir / "state" / LOCK_FILE
    symbol_map_path = base_dir / 'state' / SYMBOL_MAP_FILE
    symbol_map = _load_symbol_map(symbol_map_path)
    last_run_ts = market_state.get("last_run")
    last_duration = market_state.get("last_duration_seconds")
    if last_run_ts:
        logger.info(
            "Previous market refresh: %s (%ss)",
            last_run_ts,
            f"{last_duration:.1f}" if isinstance(last_duration, (int, float)) else "unknown",
        )

    acquired = _record_run_start(market_state_path, lock_path, run_started_at, logger)
    if not acquired:
        logger.warning('Market data refresh already running; skipping this run.')
        return

    failed = False
    completed = False
    try:
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
        processed_total = 0
        last_business_day = _last_business_day()
        batches = list(_batches(tickers, BATCH_SIZE))
        total_batches = len(batches)
        batch_durations: list[float] = []

        logger.info(
            "Refreshing market data for %s tickers across %s batches.", total, total_batches
        )

        rate_limiter = AdaptiveRateLimiter(
            base_delay_range=RATE_LIMIT_DELAY_RANGE,
            burst_batches=BURST_BATCHES,
            burst_cooldown_seconds=BURST_COOLDOWN_SECONDS,
        )

        for batch_index, batch in enumerate(batches, start=1):
            batch_start = time.monotonic()
            processed_before_batch = processed_total
            logger.info("Starting batch %s with %s tickers.", batch_index, len(batch))
            current_time = datetime.now()
            batch_payload = _prepare_batch(
                batch,
                prices_dir,
                logger,
                retry_state,
                current_time,
                last_business_day,
                symbol_map,
            )
            _save_symbol_map(symbol_map_path, symbol_map)

            if not batch_payload:
                logger.info("Batch %s: all tickers up to date or deferred.", batch_index)
                processed_total += len(batch)
                _finalize_batch(
                    batch_index,
                    total_batches,
                    batch_start,
                    batch_durations,
                    processed_total,
                    total,
                    successes,
                    failures,
                    processed_total - processed_before_batch,
                    rate_limiter,
                    batch_requested=False,
                    batch_failed=False,
                    logger=logger,
                )
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
                processed_total += len(batch_payload)
                now = datetime.now()
                for alias, payload in batch_payload.items():
                    _record_failure(
                        alias,
                        payload["base_ticker"],
                        retry_state,
                        retry_state_path,
                        logger,
                        now,
                        universe_path,
                        ticker_failure=False,
                    )
                _finalize_batch(
                    batch_index,
                    total_batches,
                    batch_start,
                    batch_durations,
                    processed_total,
                    total,
                    successes,
                    failures,
                    processed_total - processed_before_batch,
                    rate_limiter,
                    batch_requested=True,
                    batch_failed=True,
                    logger=logger,
                )
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
                    processed_total += 1
                    _record_success(alias, retry_state, retry_state_path)
                except yfinance_client.TickerDownloadError as exc:
                    logger.warning(
                        "Skipping %s (%s) due to data error: %s", base_ticker, alias, exc
                    )
                    failures += 1
                    processed_total += 1
                    _record_failure(
                        alias,
                        base_ticker,
                        retry_state,
                        retry_state_path,
                        logger,
                        datetime.now(),
                        universe_path,
                        ticker_failure=True,
                    )
            _finalize_batch(
                batch_index,
                total_batches,
                batch_start,
                batch_durations,
                processed_total,
                total,
                successes,
                failures,
                processed_total - processed_before_batch,
                rate_limiter,
                batch_requested=True,
                batch_failed=False,
                logger=logger,
            )

        run_duration = time.monotonic() - run_start_monotonic
        logger.info(
            "Market data refresh complete. Total: %s, Success: %s, Failed: %s, Processed: %s",
            total,
            successes,
            failures,
            processed_total,
        )
        logger.info("Total refresh duration: %.1fs", run_duration)
        completed = True
    except Exception as exc:  # noqa: BLE001 - ensure status file reflects crash
        failed = True
        logger.exception("Market data refresh crashed: %s", exc)
        raise
    finally:
        if acquired:
            run_duration = time.monotonic() - run_start_monotonic
            _record_run_finish(market_state_path, run_started_at, run_duration, failed, completed)
            _clear_lock(lock_path, logger)


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


def _load_market_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_market_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _record_run_start(
    state_path: Path,
    lock_path: Path,
    run_started_at: datetime,
    logger: logging.Logger,
) -> bool:
    existing_lock = _read_lock(lock_path)
    if existing_lock:
        logger.warning("Market data lock already present (started %s).", existing_lock)
        return False
    _write_lock(lock_path, run_started_at)
    state = _load_market_state(state_path)
    state["status"] = "running"
    state["run_started_at"] = run_started_at.isoformat()
    _save_market_state(state_path, state)
    return True


def _record_run_finish(
    state_path: Path,
    run_started_at: datetime,
    run_duration: float,
    failed: bool,
    completed: bool,
) -> None:
    state = _load_market_state(state_path)
    state["status"] = "failed" if failed else "idle"
    state["run_started_at"] = run_started_at.isoformat()
    finished_at = datetime.now(timezone.utc).isoformat()
    state["run_finished_at"] = finished_at
    if not failed and completed:
        state["last_run"] = finished_at
        state["last_duration_seconds"] = round(run_duration, 1)
    _save_market_state(state_path, state)


def _read_lock(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_lock(path: Path, run_started_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_started_at.isoformat(), encoding="utf-8")


def _clear_lock(path: Path, logger: logging.Logger) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Failed to clear market data lock %s: %s", path, exc)


def _load_symbol_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def _save_symbol_map(path: Path, symbol_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(symbol_map, indent=2), encoding='utf-8')


def _record_success(alias: str, state: dict[str, dict[str, Any]], path: Path) -> None:
    if alias in state:
        del state[alias]
        _save_retry_state(path, state)


def _record_failure(
    alias: str,
    base_ticker: str,
    state: dict[str, dict[str, Any]],
    path: Path,
    logger: logging.Logger,
    now: datetime,
    universe_path: Path,
    ticker_failure: bool,
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

    if ticker_failure:
        last_ticker_failure = _parse_iso(entry.get('last_ticker_failure'))
        delta_ticker_days = (
            (now.date() - last_ticker_failure.date()).days if last_ticker_failure else None
        )
        if delta_ticker_days == 1:
            entry['consecutive_ticker_days'] = entry.get('consecutive_ticker_days', 0) + 1
        else:
            entry['consecutive_ticker_days'] = 1
        entry['last_ticker_failure'] = now.isoformat()

        if entry['consecutive_ticker_days'] >= MAX_CONSECUTIVE_FAILURES:
            if not entry.get('deactivated'):
                if deactivate_universe_ticker(universe_path, base_ticker, logger):
                    entry['deactivated'] = True

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
    short_name = _normalize_identifier(row.get('short_name'))
    return yfinance_symbol(fallback, short_name=short_name)


def _legacy_market_identifier(row: pd.Series, fallback: str) -> str:
    for column in ('short_name', 'isin'):
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
    df = ensure_active_column(universe_path, logger)
    if df.empty:
        return []
    df = df[df["active"].fillna(False)]

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
        legacy_alias = _legacy_market_identifier(row, base_ticker)
        normalized_alias = alias.upper()
        if normalized_alias in seen_aliases:
            logger.warning("Skipping duplicate market identifier %s for %s", alias, base_ticker)
            continue
        seen_aliases.add(normalized_alias)
        tickers.append({"ticker": base_ticker, "alias": alias, "legacy_alias": legacy_alias})
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
    symbol_map: dict[str, str],
) -> dict[str, dict[str, object]]:
    batch_payload: dict[str, dict[str, object]] = {}

    for entry in tickers:
        base_ticker = entry["ticker"]
        alias = entry["alias"]
        legacy_alias = entry.get('legacy_alias', '')
        previous_alias = symbol_map.get(base_ticker) or legacy_alias
        alias_changed = bool(
            previous_alias and previous_alias.upper() != alias.upper()
        )
        if alias_changed:
            logger.warning(
                'Market identifier changed for %s: %s -> %s. Rebuilding cache.',
                base_ticker,
                previous_alias,
                alias,
            )
        symbol_map[base_ticker] = alias
        info = retry_state.get(alias, {})
        if info.get("blacklisted"):
            logger.debug("Skipping %s because it is blacklisted.", alias)
            continue
        if not _should_attempt(info, current_time):
            logger.debug("Deferring %s until %s.", alias, info.get("next_try"))
            continue
        if alias_changed:
            existing = pd.DataFrame()
            last_date = None
        else:
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


def _enforce_rate_limit(
    batch_index: int,
    total_batches: int,
    logger: logging.Logger,
    rate_limiter: AdaptiveRateLimiter,
    batch_failed: bool,
) -> None:
    if batch_index >= total_batches:
        return
    if batch_failed:
        rate_limiter.record_failure()
    else:
        rate_limiter.record_success()
    base_delay, delay = rate_limiter.next_delay(batch_index)
    logger.debug(
        'Sleeping %.1fs (base %.1fs, penalty %.2f, streak %s) to respect yfinance limits.',
        delay,
        base_delay,
        rate_limiter.penalty,
        rate_limiter.consecutive_failures,
    )
    time.sleep(delay)


def _finalize_batch(
    batch_index: int,
    total_batches: int,
    batch_start: float,
    batch_durations: list[float],
    processed_total: int,
    total_tickers: int,
    successes: int,
    failures: int,
    batch_processed: int,
    rate_limiter: AdaptiveRateLimiter | None,
    batch_requested: bool,
    batch_failed: bool,
    logger: logging.Logger,
) -> None:
    duration = time.monotonic() - batch_start
    batch_durations.append(duration)
    remaining = max(total_batches - batch_index, 0)
    avg_duration = sum(batch_durations) / len(batch_durations)
    eta = remaining * avg_duration
    logger.info(
        "Batch %s/%s completed in %.1fs (avg %.1fs, ETA %.1fs). "
        "Batch processed %s tickers; overall %s/%s completed (success %s, fail %s).",
        batch_index,
        total_batches,
        duration,
        avg_duration,
        eta,
        batch_processed,
        processed_total,
        total_tickers,
        successes,
        failures,
    )
    if rate_limiter and batch_requested:
        _enforce_rate_limit(batch_index, total_batches, logger, rate_limiter, batch_failed)


def _last_business_day() -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    if today.dayofweek >= 5:
        today -= pd.tseries.offsets.BDay(1)
    return today


def main() -> None:
    run()


if __name__ == "__main__":
    main()
