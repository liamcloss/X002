"""Spread gate for yfinance quotes."""

from __future__ import annotations

import logging

from trading_bot import config


class SpreadGate:
    """Validate yfinance quotes against the spread cap."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger('trading_bot')

    def evaluate(self, quote: dict[str, object]) -> tuple[bool, str | None]:
        return validate_quote(quote, logger=self.logger)


def validate_quote(
    quote: dict[str, object],
    logger: logging.Logger | None = None,
) -> tuple[bool, str | None]:
    spread = quote.get('spread')
    symbol = quote.get('symbol', '')

    if spread is None:
        if logger:
            logger.warning('SpreadGate reject %s: spread unavailable from yfinance.', symbol)
        return False, 'Missing spread data from yfinance'

    try:
        spread_value = float(spread)
    except (TypeError, ValueError):
        if logger:
            logger.warning('SpreadGate reject %s: invalid spread value.', symbol)
        return False, 'Missing spread data from yfinance'

    if spread_value > config.CONFIG["max_spread_pct"]:
        if logger:
            logger.warning(
                'SpreadGate reject %s: spread %.4f > %.4f.',
                symbol,
                spread_value,
                config.CONFIG["max_spread_pct"],
            )
        return (
            False,
            f'Spread too wide ({spread_value * 100:.2f}% > {config.CONFIG["max_spread_pct"] * 100:.2f}%)',
        )

    return True, None


__all__ = [
    'SpreadGate',
    'validate_quote',
]
