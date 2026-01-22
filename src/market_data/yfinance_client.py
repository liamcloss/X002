"""Yfinance quote client."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from trading_bot.phase import get_phase


def get_quote(symbol: str) -> dict[str, Any]:
    """Return a real-time quote snapshot from yfinance."""

    # Guardrail: scanner phase must never hit live pricing.
    if get_phase() == "scanner":
        raise RuntimeError("Live pricing is not permitted during scanner phase.")

    logger = logging.getLogger('trading_bot')
    timestamp = datetime.now(timezone.utc)
    try:
        ticker = yf.Ticker(symbol)
        fast_info = ticker.fast_info or {}
    except Exception as exc:  # noqa: BLE001 - controlled error reporting
        logger.error('yfinance error for %s: %s', symbol, exc)
        return _build_quote(symbol, None, None, None, None, timestamp)

    bid = _to_float(fast_info.get('bid'))
    ask = _to_float(fast_info.get('ask'))
    last = _to_float(
        fast_info.get('last_price')
        or fast_info.get('last')
        or fast_info.get('regular_market_price')
    )

    if bid is None or ask is None:
        try:
            info = ticker.info or {}
        except Exception as exc:  # noqa: BLE001 - controlled error reporting
            logger.error('yfinance info error for %s: %s', symbol, exc)
            info = {}
        if bid is None:
            bid = _to_float(info.get('bid'))
        if ask is None:
            ask = _to_float(info.get('ask'))
        if last is None:
            last = _to_float(
                info.get('regularMarketPrice')
                or info.get('regularMarketPreviousClose')
                or info.get('previousClose')
            )

    if bid is None or ask is None:
        logger.warning('yfinance quote missing bid/ask for %s', symbol)

    spread = None
    if bid is not None and ask is not None and last is not None and last > 0:
        spread = (ask - bid) / last

    return _build_quote(symbol, last, bid, ask, spread, timestamp)


def _build_quote(
    symbol: str,
    last: float | None,
    bid: float | None,
    ask: float | None,
    spread: float | None,
    timestamp: datetime,
) -> dict[str, Any]:
    return {
        'symbol': symbol,
        'last': last,
        'bid': bid,
        'ask': ask,
        'spread': spread,
        'timestamp': timestamp,
    }


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


__all__ = [
    'get_quote',
]
