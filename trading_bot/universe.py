"""Universe management stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class UniverseSnapshot:
    """Snapshot of the trading universe."""

    symbols: list[str]


def refresh_universe() -> UniverseSnapshot:
    """Refresh the trading universe.

    TODO: Replace this stub with a real universe refresh using broker/API data.
    """

    symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
    return UniverseSnapshot(symbols=symbols)


def universe_to_frame(symbols: Sequence[str]) -> pd.DataFrame:
    """Convert symbols to a DataFrame."""

    return pd.DataFrame({"symbol": list(symbols)})
