"""Daily scan pipeline for signals and Telegram notifications."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from trading_bot import config
from trading_bot.config import DAILY_SCAN_DISPLAY_LIMIT
from trading_bot.constants import Mode
from trading_bot.market_data import cache
from trading_bot.messaging import (
    format_daily_scan_message,
    format_error_message,
    send_error,
    send_message,
)
from trading_bot.mooner import run_mooner_sidecar
from trading_bot.pretrade.setup_candidates import write_setup_candidates
from trading_bot.symbols import (
    tradingview_symbol,
    market_label_for_ticker,
    market_region_for_ticker,
    t212_market_code,
)
from trading_bot.universe.active import ensure_active_column
from trading_bot.signals import apply_filters, detect_pullback, find_risk_geometry, rank_candidates
from trading_bot.state import (
    add_pullback,
    cleanup_expired_cooldowns,
    in_cooldown,
    increment_pullback_day,
    invalidate_pullback,
    is_alerted,
    load_state,
    mark_alerted,
    remove_pullback,
    reset_pullback,
    save_state,
)

MAX_CANDIDATES = max(1, DAILY_SCAN_DISPLAY_LIMIT)

CURRENCY_SYMBOLS = {
    'USD': '$',
    'GBP': 'GBP',
    'GBX': 'GBX',
    'EUR': 'EUR',
    'CAD': 'C$',
    'CHF': 'CHF',
}


def run_daily_scan(dry_run: bool, logger: logging.Logger | None = None) -> None:
    """Run the daily scan pipeline and dispatch Telegram updates."""

    logger = logger or logging.getLogger('trading_bot')
    base_dir = Path(__file__).resolve().parents[1]
    universe_path = base_dir / 'universe' / 'clean' / 'universe.parquet'

    try:
        universe_df = _load_universe(universe_path, logger)
    except Exception as exc:  # noqa: BLE001 - report error to Telegram
        _handle_scan_error(exc, logger, dry_run)
        return

    prices_dir = cache.ensure_prices_dir(base_dir)
    state = load_state()
    today = datetime.now().date()
    if not dry_run:
        cleanup_expired_cooldowns(state, today)

    candidates: list[dict[str, Any]] = []
    scanned_count = 0
    data_as_of: pd.Timestamp | None = None

    for row in universe_df.itertuples(index=False):
        base_ticker = _normalize_text(getattr(row, 'ticker', None))
        if not base_ticker:
            continue
        display_ticker = _normalize_text(getattr(row, 'short_name', None)) or base_ticker
        currency_code = _normalize_text(getattr(row, 'currency_code', None)) or 'USD'
        instrument_type = _normalize_text(getattr(row, 'type', None)) or 'STOCK'

        prices = cache.load_cache(prices_dir, base_ticker, logger)
        if prices.empty:
            continue

        prices = prices.sort_index()
        scanned_count += 1

        last_date = cache.last_cached_date(prices)
        if last_date is not None:
            if data_as_of is None or last_date > data_as_of:
                data_as_of = last_date

        pullback_info = detect_pullback(prices)
        was_pullback = base_ticker in state.get('pullbacks', {})

        filtered = apply_filters(prices)
        candidate_ready = not filtered.empty

        if not dry_run:
            _update_pullback_state(
                state=state,
                ticker=base_ticker,
                pullback_info=pullback_info,
                was_pullback=was_pullback,
                candidate_ready=candidate_ready,
                today=today,
                logger=logger,
            )

        if not candidate_ready:
            continue

        if in_cooldown(state, base_ticker, today):
            logger.info('Skipping %s: in cooldown window.', base_ticker)
            continue
        if is_alerted(state, base_ticker):
            logger.info('Skipping %s: already alerted.', base_ticker)
            continue

        metrics = filtered.iloc[-1].to_dict()
        price = _extract_close_price(metrics, prices)
        if price is None:
            logger.warning('Skipping %s: missing close price.', base_ticker)
            continue

        volume_multiple = _to_float(metrics.get('volume_multiple'))
        pct_from_20d_high = _to_float(metrics.get('pct_from_20d_high'))
        momentum_5d = _to_float(metrics.get('momentum_5d'))
        if volume_multiple is None or pct_from_20d_high is None or momentum_5d is None:
            logger.warning('Skipping %s: incomplete signal metrics.', base_ticker)
            continue

        geometry = find_risk_geometry(price)
        if not geometry:
            logger.info('Skipping %s: no valid risk geometry.', base_ticker)
            continue

        reason = 'Pullback breakout' if was_pullback else 'Breakout'
        region = market_region_for_ticker(base_ticker)
        market_code = t212_market_code(base_ticker)
        market_label = market_label_for_ticker(base_ticker) or market_code
        candidate = _build_candidate_payload(
            base_ticker=base_ticker,
            display_ticker=display_ticker,
            currency_code=currency_code,
            instrument_type=instrument_type,
            price=price,
            volume_multiple=volume_multiple,
            pct_from_20d_high=pct_from_20d_high,
            momentum_5d=momentum_5d,
            geometry=geometry,
            reason=reason,
            mode=config.CONFIG["mode"],
            region=region,
            market_code=market_code,
            market_label=market_label,
        )
        candidates.append(candidate)

        if not dry_run:
            mark_alerted(state, base_ticker, today)
            if was_pullback and base_ticker in state.get('pullbacks', {}):
                reset_pullback(state, base_ticker, today)

    ranked_candidates = _rank_candidates(candidates, logger)
    pretrade_limit = _pretrade_candidate_limit()
    setup_candidates = ranked_candidates[:pretrade_limit]
    delivered = setup_candidates[:MAX_CANDIDATES]
    valid_count = len(ranked_candidates)

    mooner_callouts: list[dict] = []
    try:
        mooner_callouts = run_mooner_sidecar(base_dir, logger)
    except Exception as exc:  # noqa: BLE001 - mooner sidecar must not break scan
        logger.warning("Mooner sidecar failed: %s", exc, exc_info=True)
    message = format_daily_scan_message(
        date=datetime.now().strftime('%Y-%m-%d'),
        mode=config.CONFIG["mode"],
        candidates=delivered,
        scanned_count=scanned_count,
        valid_count=valid_count,
        data_as_of=_format_data_as_of(data_as_of),
        generated_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ'),
        dry_run=dry_run,
        mooner_callouts=mooner_callouts,
    )

    if dry_run:
        logger.info('DRY RUN: scan complete. Telegram output:\n%s', message)
        return

    try:
        write_setup_candidates(
            base_dir,
            setup_candidates,
            mode=config.CONFIG["mode"],
            data_as_of=_format_data_as_of(data_as_of),
            generated_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        )
        logger.info('SetupCandidates.json updated.')
    except OSError as exc:
        logger.warning('Failed to write SetupCandidates.json: %s', exc)

    try:
        send_message(message)
    except Exception as exc:  # noqa: BLE001 - Telegram error handling
        logger.exception('Telegram message failed: %s', exc)
        _handle_scan_error(exc, logger, dry_run=False)
        return

    save_state(state)
    logger.info('Daily scan complete: %s scanned, %s candidates delivered.', scanned_count, len(delivered))


def _load_universe(path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = ensure_active_column(path, logger)
    if df.empty:
        raise RuntimeError('Universe parquet is empty.')

    df = df[df['active'].fillna(False)]

    if 'ticker' not in df.columns:
        raise RuntimeError('Universe parquet missing required \'ticker\' column.')

    logger.info('Loaded universe with %s instruments.', df.shape[0])
    return df


def _update_pullback_state(
    state: dict[str, Any],
    ticker: str,
    pullback_info: dict[str, Any],
    was_pullback: bool,
    candidate_ready: bool,
    today: date,
    logger: logging.Logger,
) -> None:
    in_pullback = bool(pullback_info.get('in_pullback'))
    pullback_pct = pullback_info.get('pullback_pct')
    days_since_high = _to_float(pullback_info.get('days_since_high'))

    if candidate_ready:
        return

    if in_pullback:
        if was_pullback:
            increment_pullback_day(state, ticker)
        else:
            add_pullback(state, ticker, pullback_info.get('breakout_high', 0.0), today)
        return

    if not was_pullback:
        return

    if _is_nan(pullback_pct) or _to_float(pullback_pct) is None:
        invalidate_pullback(state, ticker, today)
        return

    pullback_pct = float(pullback_pct)
    if pullback_pct > 0.05:
        invalidate_pullback(state, ticker, today)
        return

    if days_since_high is not None and days_since_high > 4:
        remove_pullback(state, ticker, 'Expired (pullback window closed)')
        return

    if pullback_pct < 0.01:
        reset_pullback(state, ticker, today)
        return

    logger.debug('Pullback retained for %s; awaiting follow-through.', ticker)


def _rank_candidates(
    candidates: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    df = pd.DataFrame(candidates)
    try:
        ranked = rank_candidates(df)
    except Exception as exc:  # noqa: BLE001 - ranking fallback
        logger.warning('Ranking failed (%s). Using unsorted candidates.', exc)
        return candidates

    return ranked.to_dict(orient='records')


def _pretrade_candidate_limit() -> int:
    limit = config.CONFIG.get("pretrade_candidate_limit", MAX_CANDIDATES)
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        return MAX_CANDIDATES
    return max(MAX_CANDIDATES, limit_value)


def _build_candidate_payload(
    base_ticker: str,
    display_ticker: str,
    currency_code: str,
    instrument_type: str,
    price: float,
    volume_multiple: float,
    pct_from_20d_high: float,
    momentum_5d: float,
    geometry: dict[str, float],
    reason: str,
    mode: str,
    region: str,
    market_code: str | None,
    market_label: str | None,
) -> dict[str, Any]:
    currency_symbol = _currency_symbol(currency_code)
    stop_pct = float(geometry['stop_pct'])
    target_pct = float(geometry['target_pct'])
    rr = float(geometry['rr'])

    position_size = _position_size(stop_pct, mode)
    risk_gbp = round(position_size * stop_pct, 2)
    reward_gbp = round(position_size * target_pct, 2)

    return {
        'ticker': display_ticker,
        'base_ticker': base_ticker,
        'display_ticker': display_ticker,
        'currency_symbol': currency_symbol,
        'currency_code': currency_code,
        'price': round(price, 2),
        'pct_from_20d_high': float(pct_from_20d_high),
        'momentum_5d': float(momentum_5d),
        'volume_multiple': float(volume_multiple),
        'stop_pct': stop_pct,
        'target_pct': target_pct,
        'rr': rr,
        'stop_price': round(price * (1 - stop_pct), 2),
        'target_price': round(price * (1 + target_pct), 2),
        'position_size': round(position_size, 2),
        'risk_gbp': risk_gbp,
        'reward_gbp': reward_gbp,
        'isa_eligible': _infer_isa_eligible(currency_code, instrument_type),
        'tradingview_url': _build_tradingview_url(base_ticker, display_ticker),
        'reason': reason,
        'region': region,
        'market_code': market_code,
        'market_label': market_label,
    }


def _position_size(stop_pct: float, mode: str) -> float:
    if mode == Mode.LIVE.value:
        live_mode_config = config.CONFIG["live_mode"]
        size_by_risk = (
            live_mode_config["max_risk"] / stop_pct
            if stop_pct
            else live_mode_config["position_min"]
        )
        size = min(
            live_mode_config["position_max"],
            max(live_mode_config["position_min"], size_by_risk),
        )
        return float(size)
    return float(config.CONFIG["test_mode"]["position_size"])


def _infer_isa_eligible(currency_code: str, instrument_type: str) -> bool:
    code = (currency_code or '').upper()
    instrument = (instrument_type or '').upper()
    if code not in {'GBP', 'GBX'}:
        return False
    return instrument in {'STOCK', 'ETF'}


def _currency_symbol(currency_code: str) -> str:
    code = (currency_code or '').upper()
    return CURRENCY_SYMBOLS.get(code, '')


def _build_tradingview_url(base_ticker: str, display_ticker: str | None = None) -> str:
    symbol = tradingview_symbol(base_ticker, short_name=display_ticker)
    safe_symbol = quote(symbol.replace(' ', ''))
    if ':' in symbol:
        return f'https://www.tradingview.com/chart/?symbol={safe_symbol}'
    return f'https://www.tradingview.com/symbols/{safe_symbol}/'


def _format_data_as_of(value: pd.Timestamp | None) -> str:
    if value is None:
        return 'Unknown'
    return value.strftime('%Y-%m-%d')


def _extract_close_price(metrics: dict[str, Any], df: pd.DataFrame) -> float | None:
    for key in ('close_price', 'close', 'Close'):
        value = metrics.get(key)
        if value is not None and not _is_nan(value):
            return float(value)

    for column in ('Close', 'close'):
        if column in df.columns:
            series = df[column].dropna()
            if not series.empty:
                return float(series.iloc[-1])
    return None


def _normalize_text(value: object | None) -> str | None:
    if value is None:
        return None
    if _is_nan(value):
        return None
    text = str(value).strip()
    return text or None


def _to_float(value: object | None) -> float | None:
    if value is None or _is_nan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_nan(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _handle_scan_error(exc: Exception, logger: logging.Logger, dry_run: bool) -> None:
    logger.exception('Scan pipeline failed: %s', exc)
    if dry_run:
        return
    try:
        send_error(format_error_message(str(exc)))
    except Exception as send_exc:  # noqa: BLE001 - error reporting fallback
        logger.warning('Failed to send error message via Telegram: %s', send_exc)
