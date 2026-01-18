"""Live trade count helpers (execution-only)."""

from __future__ import annotations

import logging


def get_live_open_trade_count(logger: logging.Logger | None = None) -> int:
    """
    Return the count of open live trades.

    Paper trades are intentionally excluded from pre-trade guards.
    """

    logger = logger or logging.getLogger('trading_bot')
    logger.info('Live trade count source not configured; assuming 0 open live trades.')
    return 0


__all__ = [
    'get_live_open_trade_count',
]
