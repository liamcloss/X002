"""Paper trading stubs."""

from __future__ import annotations

import logging

from trading_bot.ideas import TradeIdea


def execute_paper_trades(logger: logging.Logger, ideas: list[TradeIdea]) -> None:
    """Execute paper trades based on ideas.

    TODO: Implement paper trading simulation with safety gates.
    """

    logger.info("Paper trading stub executed for %d ideas.", len(ideas))
