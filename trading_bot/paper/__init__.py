"""Paper trading package."""

from trading_bot.paper.paper_engine import (
    get_open_trade_count,
    maybe_send_weekly_summary,
    open_paper_trade,
    process_open_trades,
)

__all__ = [
    "get_open_trade_count",
    "maybe_send_weekly_summary",
    "open_paper_trade",
    "process_open_trades",
]
