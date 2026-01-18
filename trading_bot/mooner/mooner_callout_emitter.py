"""Mooner callout publication helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from trading_bot.mooner.mooner_state_engine import (
    MoonerState,
    MoonerStateSnapshot,
)

CALLOUTS_FILENAME = "MoonerCallouts.json"


def emit_mooner_callouts(
    base_dir: Path,
    snapshots: Iterable[MoonerStateSnapshot],
    logger: logging.Logger,
) -> list[dict]:
    """Emit callouts for tickers that transitioned into FIRING."""

    callouts = []
    for snapshot in snapshots:
        if snapshot.state != MoonerState.FIRING:
            continue
        prev_state = snapshot.previous_state
        if prev_state == MoonerState.FIRING.value:
            continue
        callouts.append(
            {
                "ticker": snapshot.ticker,
                "detected_on": snapshot.state_since,
                "context": snapshot.context,
                "severity": "RARE_EVENT",
            }
        )

    _write_callouts(base_dir, callouts)
    logger.info("Mooner callouts emitted (%s entries).", len(callouts))
    return callouts


def _write_callouts(base_dir: Path, callouts: Iterable[dict]) -> None:
    path = base_dir / CALLOUTS_FILENAME
    path.write_text(json.dumps(list(callouts), indent=2), encoding="utf-8")
