"""Pre-trade viability pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.market_data import yfinance_client
from trading_bot.execution.open_trades import get_live_open_trade_count
from trading_bot.mooner import load_mooner_states
from trading_bot.paths import latest_setup_candidates_path, pretrade_viability_path
from trading_bot.phase import set_phase
from trading_bot.run_state import finish_run, start_run
from trading_bot.symbols import tradingview_symbol, yfinance_symbol
from trading_bot.pretrade.notifier import build_pretrade_messages, send_pretrade_messages
from trading_bot.pretrade.pretrade_gate import evaluate_pretrade
from trading_bot.pretrade.spread_gate import SpreadGate
from trading_bot.pretrade.spread_sampler import collect_spread_sample, write_spread_report

SETUP_FILENAME = 'SetupCandidates.json'


def run_pretrade(base_dir: Path | None = None, logger: logging.Logger | None = None) -> None:
    """Run the pre-trade viability check against yesterday's setups."""

    logger = logger or logging.getLogger('trading_bot')
    set_phase("execution")
    base_dir = base_dir or Path(__file__).resolve().parents[2]
    run_handle = start_run(base_dir, 'pretrade', logger)
    if not run_handle.acquired:
        return
    failed = False
    completed = False

    try:
        scan_lock = base_dir / 'state' / 'scan.lock'
        if scan_lock.exists():
            logger.warning('Scan in progress; pretrade aborted to avoid stale candidates.')
            return

        setup_path = _find_latest_setup_file(base_dir, logger)
        payload, setups = _load_setup_candidates(setup_path, logger)
        if not _validate_snapshot_date(payload, logger):
            logger.error('Pretrade aborted: snapshot is intraday or missing detected date.')
            return
        mooner_states = load_mooner_states(base_dir, logger)
        mooner_map = _build_mooner_map(mooner_states)

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
        for index, setup in enumerate(setups, start=1):
            scan_rank = _extract_scan_rank(setup, index)
            detected_at_utc = setup.get('detected_at_utc')
            detected_close_date = setup.get('detected_close_date')
            detected_price = _extract_detected_price(setup)
            parsed = _parse_setup_candidate(setup)
            if parsed is None:
                reject = _reject_result(
                    symbol=_extract_symbol(setup) or 'UNKNOWN',
                    open_trades_count=open_trades_count,
                    checked_at=checked_at,
                    reason='Invalid setup data',
                    scan_rank=scan_rank,
                    display_ticker=_extract_display_ticker(setup),
                    tradingview_url=_extract_tradingview_url(setup),
                    detected_at_utc=detected_at_utc,
                    detected_close_date=detected_close_date,
                    detected_price=detected_price,
                )
                results.append(
                    _attach_mooner_context(
                        reject,
                        symbol=_extract_symbol(setup) or 'UNKNOWN',
                        display_ticker=_extract_display_ticker(setup),
                        mooner_map=mooner_map,
                    )
                )
                continue
            symbol, planned_entry, planned_stop, planned_target = parsed

            display_ticker = _extract_display_ticker(setup)
            tradingview_url = _extract_tradingview_url(setup)
            if not tradingview_url:
                tradingview_url = _build_tradingview_url(symbol, display_ticker)
            quote_symbol = _resolve_quote_symbol(symbol, display_ticker, logger)
            quote = yfinance_client.get_quote(quote_symbol)
            if quote is None:
                reject = _reject_result(
                    symbol=symbol,
                    open_trades_count=open_trades_count,
                    checked_at=checked_at,
                    reason='No live price from yfinance',
                    scan_rank=scan_rank,
                    display_ticker=display_ticker,
                    tradingview_url=tradingview_url,
                    detected_at_utc=detected_at_utc,
                    detected_close_date=detected_close_date,
                    detected_price=detected_price,
                )
                results.append(
                    _attach_mooner_context(
                        reject,
                        symbol=symbol,
                        display_ticker=display_ticker,
                        mooner_map=mooner_map,
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
                reject = _reject_result(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    last=last,
                    spread=spread,
                    open_trades_count=open_trades_count,
                    checked_at=checked_at,
                    reason=invalid_reason or 'Invalid quote data from yfinance',
                    scan_rank=scan_rank,
                    display_ticker=display_ticker,
                    tradingview_url=tradingview_url,
                    detected_at_utc=detected_at_utc,
                    detected_close_date=detected_close_date,
                    detected_price=detected_price,
                )
                results.append(
                    _attach_mooner_context(
                        reject,
                        symbol=symbol,
                        display_ticker=display_ticker,
                        mooner_map=mooner_map,
                    )
                )
                continue

            detected_price = _extract_detected_price(setup)
            result = evaluate_pretrade(
                symbol=symbol,
                planned_entry=planned_entry,
                planned_stop=planned_stop,
                planned_target=planned_target,
                live_bid=float(bid),
                live_ask=float(ask),
                open_trades_count=open_trades_count,
                checked_at=checked_at,
                detected_price=detected_price,
                current_price=float(last) if last is not None else None,
                mode=config.CONFIG.get("mode"),
                max_move_since_detection=config.LATE_MOVE_PCT_MAX,
            )
            result['last'] = last
            result['spread'] = spread
            result['planned_entry'] = planned_entry
            result['planned_stop'] = planned_stop
            result['planned_target'] = planned_target
            result['detected_at_utc'] = setup.get('detected_at_utc')
            result['detected_close_date'] = setup.get('detected_close_date')
            result['detected_price'] = detected_price
            result['scan_rank'] = scan_rank
            result['display_ticker'] = display_ticker
            result['tradingview_url'] = tradingview_url
            results.append(_attach_mooner_context(result, symbol, display_ticker, mooner_map))

        _assign_exec_ranks(results)
        _write_results(base_dir, results, now, logger)
        try:
            report_path = write_spread_report(base_dir, spread_samples, now, logger)
            if report_path:
                logger.info('Spread report written: %s', report_path)
        except Exception as exc:  # noqa: BLE001 - spread reporting must not break pretrade
            logger.error('Failed to write spread report: %s', exc)

        messages = build_pretrade_messages(results, checked_at)
        _print_results_messages(messages)
        send_pretrade_messages(messages, logger)
        completed = True
    except Exception as exc:  # noqa: BLE001 - ensure run state is updated
        failed = True
        logger.exception('Pre-trade viability check failed: %s', exc)
        raise
    finally:
        finish_run(run_handle, logger, failed=failed, completed=completed)


def _find_latest_setup_file(base_dir: Path, logger: logging.Logger) -> Path | None:
    path = latest_setup_candidates_path(base_dir)
    if path:
        return path
    candidates = [
        base_dir / SETUP_FILENAME,
        base_dir / 'outputs' / SETUP_FILENAME,
        base_dir / 'data' / SETUP_FILENAME,
    ]
    paths = [candidate for candidate in candidates if candidate.exists()]
    if not paths:
        paths = list(base_dir.rglob(SETUP_FILENAME))
    if not paths:
        logger.error('No %s found in %s.', SETUP_FILENAME, base_dir)
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _load_setup_candidates(
    path: Path | None,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        logger.error('Failed to read %s: %s', path, exc)
        return {}, []
    except json.JSONDecodeError as exc:
        logger.error('Invalid JSON in %s: %s', path, exc)
        return {}, []

    candidates = _extract_candidates(payload)
    logger.info('Loaded %s setup candidates from %s.', len(candidates), path)
    return payload if isinstance(payload, dict) else {}, candidates


def _validate_snapshot_date(payload: dict[str, Any], logger: logging.Logger) -> bool:
    # Guardrail: execution must anchor to prior-day frozen snapshots.
    data_as_of = payload.get('data_as_of')
    detected_close_date = _normalize_date_value(data_as_of)
    if not detected_close_date:
        logger.error('Snapshot missing detected_close_date/data_as_of; refusing execution.')
        return False
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if detected_close_date >= today:
        logger.error('Snapshot is intraday (%s); refusing execution.', detected_close_date)
        return False
    return True


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


def _build_mooner_map(states: list[dict]) -> dict[str, dict[str, Any]]:
    mooner_map: dict[str, dict[str, Any]] = {}
    for state in states:
        ticker = _normalize_ticker(state.get('ticker'))
        if not ticker:
            continue
        mooner_map[ticker] = {
            'state': state.get('state'),
            'context': state.get('context'),
            'as_of': state.get('as_of'),
        }
    return mooner_map


def _attach_mooner_context(
    result: dict[str, Any],
    symbol: str,
    display_ticker: str | None,
    mooner_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not mooner_map:
        return result
    ticker_key = _normalize_ticker(display_ticker) or _normalize_ticker(symbol)
    if not ticker_key:
        return result
    mooner_state = mooner_map.get(ticker_key)
    if not mooner_state:
        return result
    if mooner_state.get('state') == 'DORMANT':
        return result
    result['mooner_state'] = mooner_state.get('state')
    result['mooner_context'] = mooner_state.get('context')
    result['mooner_as_of'] = mooner_state.get('as_of')
    return result


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


def _extract_tradingview_url(payload: dict[str, Any]) -> str | None:
    value = payload.get('tradingview_url')
    if value:
        return str(value).strip()
    return None


def _extract_detected_price(payload: dict[str, Any]) -> float | None:
    return _extract_value(payload, ('detected_price', 'price'))


def _build_tradingview_url(symbol: str, display_ticker: str | None) -> str | None:
    if not symbol:
        return None
    tv_symbol = tradingview_symbol(symbol, short_name=display_ticker)
    if not tv_symbol:
        return None
    safe_symbol = quote(tv_symbol.replace(' ', ''))
    if ':' in tv_symbol:
        return f'https://www.tradingview.com/chart/?symbol={safe_symbol}'
    return f'https://www.tradingview.com/symbols/{safe_symbol}/'


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


def _normalize_date_value(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = text[:10]
    if len(candidate) != 10:
        return None
    if candidate[4] != '-' or candidate[7] != '-':
        return None
    if not (candidate[:4].isdigit() and candidate[5:7].isdigit() and candidate[8:10].isdigit()):
        return None
    return candidate


def _reject_result(
    symbol: str,
    open_trades_count: int,
    checked_at: str,
    reason: str,
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
    spread: float | None = None,
    scan_rank: int | None = None,
    display_ticker: str | None = None,
    tradingview_url: str | None = None,
    detected_at_utc: str | None = None,
    detected_close_date: str | None = None,
    detected_price: float | None = None,
) -> dict[str, Any]:
    return {
        'symbol': symbol,
        'display_ticker': display_ticker,
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
        'scan_rank': scan_rank,
        'tradingview_url': tradingview_url,
        'detected_at_utc': detected_at_utc,
        'detected_close_date': detected_close_date,
        'detected_price': detected_price,
        'pct_move_since_detection': None,
        'current_price': None,
    }


def _write_results(
    base_dir: Path,
    results: list[dict[str, Any]],
    timestamp: datetime,
    logger: logging.Logger,
) -> None:
    filename = f'pretrade_viability_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
    path = pretrade_viability_path(base_dir, filename)
    try:
        path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    except OSError as exc:
        logger.error('Failed to write pretrade results to %s: %s', path, exc)


def _print_results_messages(messages: list[str]) -> None:
    if not messages:
        return
    print('\n\n'.join(messages))


def _extract_scan_rank(payload: dict[str, Any], default_rank: int) -> int:
    for key in ('scan_rank', 'rank'):
        if key in payload:
            value = _to_int(payload.get(key))
            if value and value > 0:
                return value
    return default_rank


def _assign_exec_ranks(results: list[dict[str, Any]]) -> None:
    executables = [
        result for result in results if result.get('status') == 'EXECUTABLE'
    ]
    executables.sort(key=lambda item: _sort_rank(item, fallback=10**9))
    for index, result in enumerate(executables, start=1):
        result['exec_rank'] = index


def _sort_rank(result: dict[str, Any], fallback: int) -> int:
    value = _to_int(result.get('scan_rank'))
    if value and value > 0:
        return value
    return fallback


def _to_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _normalize_ticker(value: object | None) -> str:
    if value is None:
        return ''
    return str(value).strip().upper()


__all__ = [
    'run_pretrade',
]
