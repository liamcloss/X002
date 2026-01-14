"""State persistence stubs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class StateSnapshot:
    """Placeholder state snapshot."""

    last_run: str


def load_state(state_dir: Path) -> StateSnapshot:
    """Load persisted state.

    TODO: Replace with more robust state loading.
    """

    path = state_dir / "state.parquet"
    if not path.exists():
        return StateSnapshot(last_run="")

    frame = pd.read_parquet(path)
    last_run = frame["last_run"].iloc[0] if not frame.empty else ""
    return StateSnapshot(last_run=str(last_run))


def save_state(state_dir: Path, last_run: str) -> None:
    """Persist state to disk."""

    state_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"last_run": [last_run]})
    frame.to_parquet(state_dir / "state.parquet", index=False)
