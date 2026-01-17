"""Pre-trade Telegram notification helpers."""

from __future__ import annotations

import logging
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
        else:
            lines.append(f'{symbol} {ARROW} REJECTED')
            lines.append(f'  Reason: {result.get("reject_reason")}')
        lines.append('')

    if count == 0:
        lines.append('No setups to evaluate.')
        lines.append('')

    lines.append(f'Checked at: {checked_at}')
    lines.append('Execution is manual. Stops must be placed immediately.')
    return '\n'.join(lines)


def send_pretrade_message(message: str, logger: logging.Logger) -> None:
    try:
        send_message(message)
    except Exception as exc:  # noqa: BLE001 - do not crash on Telegram failure
        logger.error('Pretrade Telegram notification failed: %s', exc)


__all__ = [
    'build_pretrade_message',
    'send_pretrade_message',
]
