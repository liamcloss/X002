"""Market data retrieval stubs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketData:
    """Placeholder market data container."""

    frame: pd.DataFrame


def get_market_data(symbols: list[str], days: int = 30) -> MarketData:
    """Fetch market data for symbols.

    TODO: Replace this stub with yfinance or broker market data.
    """

    end = datetime.now(timezone.utc)
    dates = [end - timedelta(days=offset) for offset in range(days)]
    dates.sort()

    records = []
    for symbol in symbols:
        prices = np.linspace(100, 120, num=days) + np.random.normal(0, 1, size=days)
        for date, price in zip(dates, prices):
            records.append({"symbol": symbol, "date": date.date(), "close": price})

    frame = pd.DataFrame(records)
    return MarketData(frame=frame)
