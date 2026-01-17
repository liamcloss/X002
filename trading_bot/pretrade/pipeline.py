"""Pre-trade viability pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.market_data import yfinance_client
from trading_bot.execution.open_trades import get_live_open_trade_count
from trading_bot.symbols import yfinance_symbol
from trading_bot.pretrade.notifier import build_pretrade_message, send_pretrade_message
from trading_bot.pretrade.pretrade_gate import evaluate_pretrade
from trading_bot.pretrade.spread_gate import SpreadGate
from trading_bot.pretrade.spread_sampler import collect_spread_sample, write_spread_report

SETUP_FILENAME = 'SetupCandidates.json'


def run_pretrade(base_dir: Path | None = None, logger: logging.Logger | None = None) -> None:
    """Run the pre-trade viability check against yesterday's setups."""

    logger = logger or logging.getLogger('trading_bot')
    base_dir = base_dir or Path(__file__).resolve().parents[2]

    setup_path = _find_latest_setup_file(base_dir, logger)
    setups = _load_setup_candidates(setup_path, logger)

    try:
        open_trades_count = get_live_open_trade_count(logger=logger)
    except Exception as exc:  # noqa: BLE001 - safe fallback for trade cap
        logger.error('Failed to load live open trades count: %s', exc)
        open_trades_count = 0

    now = datetime.now(timezone.utc)
    checked_at = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    logger.info('MarketDataSource = "yfinance"')
    spread_gate = SpreadGate(logger=logger)

    results: list[dict[str, Any]] = []
    spread_samples: list[dict[str, Any]] = []
    for setup in setups:
        parsed = _parse_setup_candidate(setup)
        if parsed is None:
            results.append(
                _reject_result(
                    symbol=_extract_symbol(setup) or 'UNKNOWN',
                    open_trades_count=open_trades_count,
                    checked_at=checked_at,
                    reason='Invalid setup data',
                )
            )
            continue
        symbol, planned_entry, planned_stop, planned_target = parsed

        display_ticker = _extract_display_ticker(setup)
        quote_symbol = _resolve_quote_symbol(symbol, display_ticker, logger)
        quote = yfinance_client.get_quote(quote_symbol)
        if quote is None:
            results.append(
                _reject_result(
                    symbol=symbol,
                    open_trades_count=open_trades_count,
                    checked_at=checked_at,
                    reason='No live price from yfinance',
                )
            )
            continue

        if quote_symbol != symbol:
            quote = {**quote, 'symbol': symbol}

        bid = quote.get('bid')
        ask = quote.get('ask')
        last = quote.get('last')
        spread = quote.get('spread')
        sample = collect_spread_sample(
            symbol=symbol,
            spread=spread,
            last=last,
            quote_timestamp=quote.get('timestamp') if isinstance(quote, dict) else None,
            checked_at=now,
            logger=logger,
        )
        if sample:
            spread_samples.append(sample)
        is_valid, invalid_reason = spread_gate.evaluate(quote)
        if not is_valid:
            results.append(
                _reject_result(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    last=last,
                    spread=spread,
                    open_trades_count=open_trades_count,
                    checked_at=checked_at,
                    reason=invalid_reason or 'Invalid quote data from yfinance',
                )
            )
            continue

        result = evaluate_pretrade(
            symbol=symbol,
            planned_entry=planned_entry,
            planned_stop=planned_stop,
            planned_target=planned_target,
            live_bid=float(bid),
            live_ask=float(ask),
            open_trades_count=open_trades_count,
            checked_at=checked_at,
        )
        result['last'] = last
        result['spread'] = spread
        results.append(result)

    _write_results(base_dir, results, now, logger)
    try:
        report_path = write_spread_report(base_dir, spread_samples, now, logger)
        if report_path:
            logger.info('Spread report written: %s', report_path)
    except Exception as exc:  # noqa: BLE001 - spread reporting must not break pretrade
        logger.error('Failed to write spread report: %s', exc)

    _print_results_table(results)
    message = build_pretrade_message(results, checked_at)
    send_pretrade_message(message, logger)


def _find_latest_setup_file(base_dir: Path, logger: logging.Logger) -> Path | None:
    candidates = [
        base_dir / SETUP_FILENAME,
        base_dir / 'outputs' / SETUP_FILENAME,
        base_dir / 'data' / SETUP_FILENAME,
    ]
    paths = [path for path in candidates if path.exists()]
    if not paths:
        paths = list(base_dir.rglob(SETUP_FILENAME))
    if not paths:
        logger.error('No %s found in %s.', SETUP_FILENAME, base_dir)
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _load_setup_candidates(path: Path | None, logger: logging.Logger) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        logger.error('Failed to read %s: %s', path, exc)
        return []
    except json.JSONDecodeError as exc:
        logger.error('Invalid JSON in %s: %s', path, exc)
        return []

    candidates = _extract_candidates(payload)
    logger.info('Loaded %s setup candidates from %s.', len(candidates), path)
    return candidates


def _extract_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key, value in payload.items():
        if key.lower() in {'candidates', 'setups', 'ideas', 'setupcandidates', 'setup_candidates'}:
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _parse_setup_candidate(payload: dict[str, Any]) -> tuple[str, float, float, float] | None:
    symbol = _extract_symbol(payload)
    if not symbol:
        return None

    planned_entry = _extract_value(
        payload,
        ('planned_entry', 'plannedEntry', 'entry', 'entry_price', 'entryPrice', 'price'),
    )
    planned_stop = _extract_value(
        payload,
        ('planned_stop', 'plannedStop', 'stop', 'stop_price', 'stopPrice'),
    )
    planned_target = _extract_value(
        payload,
        ('planned_target', 'plannedTarget', 'target', 'target_price', 'targetPrice'),
    )

    if planned_entry is None or planned_stop is None or planned_target is None:
        return None

    return symbol, planned_entry, planned_stop, planned_target


def _extract_symbol(payload: dict[str, Any]) -> str | None:
    for key in ('symbol', 'ticker', 'base_ticker'):
        value = payload.get(key)
        if value:
            return str(value).strip()
    return None


def _extract_display_ticker(payload: dict[str, Any]) -> str | None:
    value = payload.get('display_ticker')
    if value:
        return str(value).strip()
    return None


def _resolve_quote_symbol(
    symbol: str,
    display_ticker: str | None,
    logger: logging.Logger,
) -> str:
    short_name = _select_short_name(symbol, display_ticker)
    mapped = yfinance_symbol(symbol, short_name=short_name)
    if mapped != symbol:
        logger.info('Mapped pretrade symbol %s -> %s', symbol, mapped)
    return mapped


def _select_short_name(symbol: str, display_ticker: str | None) -> str | None:
    if not display_ticker:
        return None
    candidate = display_ticker.strip()
    if not candidate:
        return None
    if ' ' in candidate:
        return None
    if candidate.upper() == symbol.upper():
        return None
    return candidate


def _extract_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            numeric = _to_float(value)
            if numeric is not None:
                return numeric
    return None


def _to_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _reject_result(
    symbol: str,
    open_trades_count: int,
    checked_at: str,
    reason: str,
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
    spread: float | None = None,
) -> dict[str, Any]:
    return {
        'symbol': symbol,
        'last': last,
        'bid': bid,
        'ask': ask,
        'mid_price': None,
        'spread': spread,
        'spread_pct': None,
        'price_drift_pct': None,
        'real_stop_distance_pct': None,
        'real_rr': None,
        'open_trades_count': open_trades_count,
        'status': 'REJECTED',
        'reject_reason': reason,
        'checked_at': checked_at,
    }


def _write_results(
    base_dir: Path,
    results: list[dict[str, Any]],
    timestamp: datetime,
    logger: logging.Logger,
) -> None:
    outputs_dir = base_dir / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = f'pretrade_viability_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
    path = outputs_dir / filename
    try:
        path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    except OSError as exc:
        logger.error('Failed to write pretrade results to %s: %s', path, exc)


def _print_results_table(results: list[dict[str, Any]]) -> None:
    header = 'symbol | last | spread% | status'
    separator = '--- | --- | --- | ---'
    lines = [header, separator]
    for result in results:
        symbol = str(result.get('symbol', ''))
        last = result.get('last')
        spread = result.get('spread')
        status = str(result.get('status', 'REJECTED'))
        last_text = f'{last:.2f}' if isinstance(last, (int, float)) else 'n/a'
        spread_text = f'{spread * 100:.2f}%' if isinstance(spread, (int, float)) else 'n/a'
        lines.append(f'{symbol} | {last_text} | {spread_text} | {status}')
    print('\n'.join(lines))


__all__ = [
    'run_pretrade',
]
