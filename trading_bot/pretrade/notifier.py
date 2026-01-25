"""Pre-trade Telegram notification helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, quote_plus

from trading_bot.messaging.telegram_client import send_message
from trading_bot.news_links import first_news_link
from trading_bot.pretrade.no_exec_state import increment_no_exec_runs, reset_no_exec_runs
from trading_bot.symbols import tradingview_symbol

ARROW = '→'


def build_pretrade_messages(
    results: Iterable[dict],
    checked_at: str,
    *,
    base_dir: Path | None = None,
) -> list[str]:
    results_list = list(results)
    executables = [item for item in results_list if item.get('status') == 'EXECUTABLE']
    rejected = [item for item in results_list if item.get('status') != 'EXECUTABLE']
    executables.sort(key=_result_sort_rank)
    rejected.sort(key=_result_sort_rank)

    messages: list[str] = []

    if executables:
        for index, result in enumerate(executables, start=1):
            messages.append(_build_executable_message(result, checked_at, index))
        if rejected:
            messages.append(_build_rejections_summary_message(len(rejected), checked_at, total=len(results_list)))
        reset_no_exec_runs(base_dir)
        return messages

    # No executables found
    consecutive = increment_no_exec_runs(base_dir)
    if consecutive >= 3:
        messages.append(_build_rejected_message(rejected, checked_at, total=len(results_list)))
    else:
        messages.append(_build_rejections_summary_message(len(rejected), checked_at, total=len(results_list)))
    return messages


def send_pretrade_messages(messages: Iterable[str], logger: logging.Logger) -> None:
    for message in messages:
        try:
            send_message(message)
        except Exception as exc:  # noqa: BLE001 - do not crash on Telegram failure
            logger.error('Pretrade Telegram notification failed: %s', exc)


def _format_value(value: object | None) -> str:
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.2f}'
    except (TypeError, ValueError):
        return 'n/a'


def _format_percent(value: object | None) -> str:
    if value is None:
        return 'n/a'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if numeric != numeric:
        return 'n/a'
    return f'{numeric:.2f}%'


def _format_number(value: object | None) -> str:
    if value is None:
        return 'n/a'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if numeric != numeric:
        return 'n/a'
    return f'{numeric:.2f}'


def _format_rank(value: object | None, fallback: int | None = None) -> str:
    if value is None:
        return str(fallback) if fallback is not None else 'n/a'
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(fallback) if fallback is not None else 'n/a'
    if numeric <= 0:
        return str(fallback) if fallback is not None else 'n/a'
    return str(numeric)


def _result_sort_rank(result: dict) -> int:
    value = result.get('scan_rank')
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 10**9
    if numeric <= 0:
        return 10**9
    return numeric


def _display_symbol(result: dict) -> str:
    value = result.get('display_ticker') or result.get('symbol') or ''
    return str(value).strip()


def _chart_url(result: dict) -> str | None:
    candidate = result.get('tradingview_url')
    if candidate:
        return str(candidate).strip()
    symbol = result.get('symbol')
    display = result.get('display_ticker')
    if not symbol:
        return None
    tv_symbol = tradingview_symbol(str(symbol), short_name=str(display) if display else None)
    if not tv_symbol:
        return None
    safe_symbol = quote(tv_symbol.replace(' ', ''))
    if ':' in tv_symbol:
        return f'https://www.tradingview.com/chart/?symbol={safe_symbol}'
    return f'https://www.tradingview.com/symbols/{safe_symbol}/'


def _news_url(result: dict) -> str | None:
    query_value = result.get('display_ticker') or result.get('symbol') or ''
    query = str(query_value).strip()
    return first_news_link(query)


def _build_executable_message(result: dict, checked_at: str, index: int) -> str:
    symbol = _display_symbol(result)
    scan_rank = _format_rank(result.get('scan_rank'))
    exec_rank = _format_rank(result.get('exec_rank'), fallback=index)
    entry = _format_value(result.get('planned_entry'))
    stop = _format_value(result.get('planned_stop'))
    target = _format_value(result.get('planned_target'))
    lines = [
        'PRE-TRADE EXECUTABLE',
        f'{exec_rank}. {symbol} (scan #{scan_rank})',
        f'Spread: {_format_percent(result.get("spread_pct"))}',
        f'Drift: {_format_percent(result.get("price_drift_pct"))}',
        f'RR: {_format_number(result.get("real_rr"))}',
        f'Stop: {_format_percent(result.get("real_stop_distance_pct"))}',
        f'Entry: {entry} | Stop: {stop} | Target: {target}',
    ]
    lines.extend(_format_mooner_context(result))
    region = result.get('region')
    market_label = result.get('market_label') or result.get('market_code')
    if region or market_label:
        parts: list[str] = []
        if region:
            parts.append(region)
        if market_label:
            parts.append(str(market_label))
        lines.append(f'Market: {" / ".join(parts)}')
    chart_url = _chart_url(result)
    news_url = _news_url(result)
    lines.append(f'Chart: {chart_url or "n/a"}')
    lines.append(f'News: {news_url or "n/a"}')
    lines.append(f'Checked at: {_format_timestamp(checked_at)}')
    lines.append('Execution is manual. Stops must be placed immediately.')
    return '\n'.join(lines)


def _build_rejected_message(results: list[dict], checked_at: str, total: int) -> str:
    lines = [
        'PRE-TRADE REJECTED',
        f'Rejected: {len(results)} of {total}',
        '',
    ]
    if not results:
        lines.append('None.')
    else:
        for result in results:
            symbol = _display_symbol(result)
            scan_rank = _format_rank(result.get('scan_rank'))
            lines.append(f'{symbol} (scan #{scan_rank}) {ARROW} REJECTED')
            lines.append(f'  Reason: {result.get("reject_reason")}')
            mooner_lines = _format_mooner_context(result)
            if mooner_lines:
                lines.extend([f'  {line}' for line in mooner_lines])
            lines.append('')
    lines.append(f'Checked at: {_format_timestamp(checked_at)}')
    lines.append('Execution is manual. Stops must be placed immediately.')
    return '\n'.join(lines).strip()


def _build_rejections_summary_message(rejected_count: int, checked_at: str, total: int) -> str:
    lines = [
        'PRE-TRADE REJECTIONS SUPPRESSED',
        f'Rejected: {rejected_count} of {total}',
        f'Checked at: {_format_timestamp(checked_at)}',
        'Details logged in outputs/pretrade_viability_<timestamp>.json',
    ]
    return '\n'.join(lines).strip()


def _format_timestamp(value: object | None) -> str:
    if value is None:
        return 'unknown'
    text = str(value)
    try:
        if text.endswith('Z'):
            text = f'{text[:-1]}+00:00'
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime('%Y-%m-%d %H:%M UTC')


def _format_mooner_context(result: dict) -> list[str]:
    state = result.get('mooner_state')
    if not state:
        return []
    context = result.get('mooner_context')
    as_of = result.get('mooner_as_of')
    detail = f'Mooner: {state}'
    if as_of:
        detail = f'{detail} (as of {as_of})'
    if context:
        detail = f'{detail} — {context}'
    return [f'⚠️ {detail} (Informational Only)']


__all__ = [
    'build_pretrade_messages',
    'send_pretrade_messages',
]
