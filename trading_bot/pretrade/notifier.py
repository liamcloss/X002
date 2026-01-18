"""Pre-trade Telegram notification helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

from trading_bot.messaging.telegram_client import send_message

ARROW = '→'


def build_pretrade_message(results: Iterable[dict], checked_at: str) -> str:
    lines = ['PRE-TRADE VIABILITY CHECK', '']

    count = 0
    for result in results:
        count += 1
        symbol = result.get('symbol', '')
        status = result.get('status', 'REJECTED')
        if status == 'EXECUTABLE':
            lines.append(f'{symbol} {ARROW} EXECUTABLE')
            lines.append(f'  Spread: {result["spread_pct"]:.2f}%')
            lines.append(f'  Drift: {result["price_drift_pct"]:.2f}%')
            lines.append(f'  RR: {result["real_rr"]:.2f}')
            lines.append(f'  Stop: {result["real_stop_distance_pct"]:.2f}%')
            entry = _format_value(result.get('planned_entry'))
            stop = _format_value(result.get('planned_stop'))
            target = _format_value(result.get('planned_target'))
            lines.append(f'  Entry: {entry} | Stop: {stop} | Target: {target}')
        else:
            lines.append(f'{symbol} {ARROW} REJECTED')
            lines.append(f'  Reason: {result.get("reject_reason")}')
        lines.append('')

    if count == 0:
        lines.append('No setups to evaluate.')
        lines.append('')

    lines.append(f'Checked at: {_format_timestamp(checked_at)}')
    lines.append('Execution is manual. Stops must be placed immediately.')
    return '\n'.join(lines)


def send_pretrade_message(message: str, logger: logging.Logger) -> None:
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


__all__ = [
    'build_pretrade_message',
    'send_pretrade_message',
]
