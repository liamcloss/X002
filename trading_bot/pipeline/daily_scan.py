"""Daily scan orchestrator for the trading bot."""

from __future__ import annotations

import copy
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from trading_bot import config
from trading_bot.market_data import cache
from trading_bot.market_data.fetch_prices import run as refresh_market_data
from trading_bot.messaging.format_message import format_daily_scan_message, format_error_message
from trading_bot.messaging import telegram_client
from trading_bot.paper.paper_engine import (
    maybe_send_weekly_summary,
    open_paper_trade,
    process_open_trades,
)
from trading_bot.pretrade.setup_candidates import write_setup_candidates
from trading_bot.run_state import finish_run, start_run
from trading_bot.signals.filters import apply_filters
from trading_bot.signals.pullback import detect_pullback
from trading_bot.signals.rank import rank_candidates
from trading_bot.signals.risk_geometry import find_risk_geometry
from trading_bot.symbols import tradingview_symbol
from trading_bot.state import (
    add_pullback,
    cleanup_expired_cooldowns,
    increment_pullback_day,
    invalidate_pullback,
    is_alerted,
    load_state,
    mark_alerted,
    reset_pullback,
    save_state,
    in_cooldown,
)
from trading_bot.universe.active import ensure_active_column


_CURRENCY_SYMBOLS = {
    "GBP": "£",
    "USD": "$",
    "EUR": "€",
    "JPY": "¥",
    "CHF": "CHF ",
    "CAD": "C$",
    "AUD": "A$",
}

ALERT_CANDIDATE_LIMIT = 3


def run_daily_scan(dry_run: bool = False) -> None:
    """
    Executes the entire daily pipeline.
    """

    logger = logging.getLogger("trading_bot")
    base_dir = Path(__file__).resolve().parents[2]
    run_handle = start_run(base_dir, 'scan', logger)
    if not run_handle.acquired:
        return
    failed = False
    completed = False
    try:
        _run_pipeline(dry_run=dry_run, logger=logger)
        completed = True
    except Exception as exc:  # noqa: BLE001 - controlled error reporting
        failed = True
        logger.exception("Daily scan failed: %s", exc)
        error_text = format_error_message(str(exc))
        if dry_run:
            print(error_text)
            return
        try:
            telegram_client.send_error(error_text)
        except Exception as send_exc:  # noqa: BLE001 - log and continue
            logger.exception("Failed to send Telegram error message: %s", send_exc)
    finally:
        finish_run(run_handle, logger, failed=failed, completed=completed)


def _run_pipeline(dry_run: bool, logger: logging.Logger) -> None:
    base_dir = Path(__file__).resolve().parents[2]
    today = date.today()
    today_str = today.isoformat()

    universe_df = _load_universe(base_dir, logger)
    if universe_df.empty:
        logger.warning("Universe is empty; daily scan aborted.")
        return

    prices_dir = _ensure_cache_initialized(base_dir, logger)
    if prices_dir is None:
        return

    logger.info("Refreshing market data cache.")
    refresh_market_data()

    state = load_state()
    working_state = copy.deepcopy(state)
    cleanup_expired_cooldowns(working_state, today)

    ticker_map = _build_ticker_map(universe_df)

    candidate_rows: list[dict[str, float | str]] = []
    candidate_metadata: dict[str, dict[str, float]] = {}
    data_as_of = None
    skipped = 0

    for ticker, payload in ticker_map.items():
        try:
            price_df = cache.load_cache(prices_dir, ticker, logger)
            if price_df.empty:
                logger.warning("No cached data for %s; skipping.", ticker)
                skipped += 1
                continue

            price_df = price_df.sort_index()
            last_date = cache.last_cached_date(price_df)
            if last_date is not None:
                data_as_of = max(data_as_of, last_date) if data_as_of else last_date

            pullback = detect_pullback(price_df)
            _update_pullback_state(working_state, ticker, pullback, today)

            filtered = apply_filters(price_df, max_pct_from_20d_high=0.05)
            if filtered.empty:
                continue

            if not pullback.get("in_pullback"):
                continue

            if in_cooldown(working_state, ticker, today) or is_alerted(working_state, ticker):
                continue

            last_row = filtered.iloc[-1]
            price = float(last_row.get("close_price", last_row.get("close", 0.0)))
            geometry = find_risk_geometry(price)
            if not geometry:
                continue

            candidate_rows.append(
                {
                    "ticker": ticker,
                    "volume_multiple": float(last_row.get("volume_multiple", 0.0)),
                    "pct_from_20d_high": float(last_row.get("pct_from_20d_high", 0.0)),
                    "momentum_5d": float(last_row.get("momentum_5d", 0.0)),
                }
            )
            candidate_metadata[ticker] = {
                "price": price,
                "stop_pct": float(geometry["stop_pct"]),
                "target_pct": float(geometry["target_pct"]),
                "rr": float(geometry["rr"]),
            }
        except Exception as exc:  # noqa: BLE001 - continue on per-ticker failure
            logger.warning("Skipping %s due to data error: %s", ticker, exc, exc_info=True)
            skipped += 1

    pretrade_limit = _pretrade_candidate_limit()
    setup_candidates, ranked_count = _rank_and_build_candidates(
        candidate_rows,
        candidate_metadata,
        ticker_map,
        limit=pretrade_limit,
    )
    alert_candidates = setup_candidates[:ALERT_CANDIDATE_LIMIT]

    if alert_candidates and not dry_run:
        for candidate in alert_candidates:
            mark_alerted(working_state, candidate["ticker"], today)

    data_as_of_str = data_as_of.isoformat() if data_as_of else "Unknown"
    generated_at = datetime.now().isoformat(timespec='seconds')
    generated_at_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    message = format_daily_scan_message(
        date=today_str,
        mode=config.MODE,
        candidates=alert_candidates,
        scanned_count=len(ticker_map),
        valid_count=ranked_count,
        data_as_of=data_as_of_str,
        generated_at=generated_at,
        dry_run=dry_run,
    )
    if skipped:
        logger.warning("%s instruments skipped due to data errors.", skipped)
        message = f"{message}\n\nSome instruments skipped due to data errors"

    if dry_run:
        print(message)
        return

    try:
        write_setup_candidates(
            base_dir,
            setup_candidates,
            mode=config.MODE,
            data_as_of=data_as_of_str,
            generated_at=generated_at_utc,
        )
        logger.info('SetupCandidates.json updated.')
    except OSError as exc:
        logger.warning('Failed to write SetupCandidates.json: %s', exc)

    process_open_trades(today_str)
    maybe_send_weekly_summary(working_state, today, logger)
    if alert_candidates:
        open_paper_trade(alert_candidates[0], today_str)

    telegram_client.send_message(message)
    save_state(working_state)


def _load_universe(base_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    path = base_dir / "universe" / "clean" / "universe.parquet"
    df = ensure_active_column(path, logger)
    if df.empty:
        return df
    df = df[df["active"].fillna(False).astype(bool)].copy()
    df = df.dropna(subset=["ticker"])
    return df


def _ensure_cache_initialized(base_dir: Path, logger: logging.Logger) -> Path | None:
    prices_dir = base_dir / "data" / "prices"
    if not prices_dir.exists():
        logger.error("Market data cache not initialized. Run Subsystem 2 first.")
        return None
    if not any(prices_dir.glob("*.parquet")):
        logger.error("Market data cache empty. Run Subsystem 2 to initialize OHLCV data.")
        return None
    return prices_dir


def _build_ticker_map(universe_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ticker_map: dict[str, dict[str, Any]] = {}
    for _, row in universe_df.iterrows():
        raw_ticker = str(row.get('ticker', '')).strip()
        if not raw_ticker:
            continue
        ticker_map[raw_ticker.upper()] = {
            'currency_code': str(row.get('currency_code', '')),
            'raw_ticker': raw_ticker,
            'short_name': str(row.get('short_name', '')).strip(),
        }
    return ticker_map


def _update_pullback_state(
    state: dict[str, Any],
    ticker: str,
    pullback: dict[str, Any],
    today: date,
) -> None:
    pullback_pct = float(pullback.get("pullback_pct", 0.0))
    in_pullback = bool(pullback.get("in_pullback"))

    if in_pullback:
        if ticker in state.get("pullbacks", {}):
            increment_pullback_day(state, ticker)
        else:
            add_pullback(state, ticker, float(pullback.get("breakout_high", 0.0)), today)
        return

    if ticker not in state.get("pullbacks", {}):
        return

    if pullback_pct > 0.05:
        invalidate_pullback(state, ticker, today)
    elif pullback_pct < 0.01:
        reset_pullback(state, ticker, today)
    else:
        increment_pullback_day(state, ticker)


def _rank_and_build_candidates(
    candidate_rows: list[dict[str, float | str]],
    candidate_metadata: dict[str, dict[str, float]],
    ticker_map: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        return [], 0

    ranked = rank_candidates(candidate_df)
    ranked = ranked.sort_values(by=["score", "ticker"], ascending=[False, True])
    ranked_count = int(ranked.shape[0])

    top_ranked = ranked.head(limit)
    candidates = []
    for _, row in top_ranked.iterrows():
        key = str(row['ticker'])
        entry = ticker_map[key]
        candidates.append(
            _build_candidate(
                row=row,
                meta=candidate_metadata[key],
                currency_code=str(entry['currency_code']),
                raw_ticker=str(entry.get('raw_ticker', key)),
                short_name=str(entry.get('short_name', '')).strip() or None,
            )
        )
    return candidates, ranked_count


def _pretrade_candidate_limit() -> int:
    limit = getattr(config, 'PRETRADE_CANDIDATE_LIMIT', ALERT_CANDIDATE_LIMIT)
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        return ALERT_CANDIDATE_LIMIT
    return max(ALERT_CANDIDATE_LIMIT, limit_value)


def _build_candidate(
    row: pd.Series,
    meta: dict[str, float],
    currency_code: str,
    raw_ticker: str,
    short_name: str | None,
) -> dict[str, Any]:
    price = float(meta["price"])
    stop_pct = float(meta["stop_pct"])
    target_pct = float(meta["target_pct"])
    position_size = _position_size_for_risk(stop_pct)

    stop_price = price * (1 - stop_pct)
    target_price = price * (1 + target_pct)
    risk_gbp = round(position_size * stop_pct, 2)
    reward_gbp = round(position_size * target_pct, 2)

    ticker = str(row['ticker'])
    display_ticker = short_name or raw_ticker or ticker

    return {
        "ticker": ticker,
        "display_ticker": display_ticker,
        "raw_ticker": raw_ticker,
        "currency_symbol": _currency_symbol(currency_code),
        "momentum_5d": float(row["momentum_5d"]),
        "reason": "Pullback",
        "volume_multiple": round(float(row["volume_multiple"]), 2),
        "price": price,
        "currency_code": currency_code,
        "pct_from_20d_high": float(row["pct_from_20d_high"]),
        "stop_pct": stop_pct,
        "stop_price": stop_price,
        "target_pct": target_pct,
        "target_price": target_price,
        "rr": float(meta["rr"]),
        "position_size": position_size,
        "risk_gbp": risk_gbp,
        "reward_gbp": reward_gbp,
        "tradingview_url": _tradingview_url(raw_ticker),
    }


def _position_size_for_risk(stop_pct: float) -> int:
    if config.MODE == "TEST":
        return int(config.TEST_MODE_POSITION_SIZE)
    if stop_pct <= 0:
        return int(config.LIVE_MODE_POSITION_MIN)
    target_size = config.LIVE_MODE_MAX_RISK / stop_pct
    bounded = max(config.LIVE_MODE_POSITION_MIN, min(config.LIVE_MODE_POSITION_MAX, target_size))
    return int(round(bounded))


def _currency_symbol(currency_code: str) -> str:
    return _CURRENCY_SYMBOLS.get(currency_code.upper(), "")


def _tradingview_url(ticker: str) -> str:
    symbol = tradingview_symbol(ticker)
    if not symbol:
        return ''
    safe_symbol = quote(symbol.replace(' ', ''))
    if ':' in symbol:
        return f'https://www.tradingview.com/chart/?symbol={safe_symbol}'
    return f'https://www.tradingview.com/symbols/{safe_symbol}/'
