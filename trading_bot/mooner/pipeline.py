"""Mooner sidecar pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from trading_bot.mooner.mooner_callout_emitter import emit_mooner_callouts, load_mooner_callouts
from trading_bot.mooner.mooner_state_engine import (
    evaluate_mooner_states,
    load_mooner_states,
    write_mooner_states,
)
from trading_bot.mooner.mooner_subset_selector import (
    load_mooner_subset,
    select_mooner_subset,
    write_mooner_subset,
)


def run_mooner_sidecar(base_dir: Path, logger: logging.Logger) -> list[dict]:
    """Run the mooner subset selection, state engine, and callout emission."""

    tickers = load_mooner_subset(base_dir, logger)
    if not tickers:
        tickers = select_mooner_subset(base_dir, logger)
        write_mooner_subset(base_dir, tickers, logger)

    if not tickers:
        return []

    prices_dir = base_dir / "data" / "prices"
    if not prices_dir.exists():
        logger.warning("Market data cache missing; mooner sidecar skipped.")
        return []

    snapshots = evaluate_mooner_states(prices_dir, tickers, logger)
    write_mooner_states(base_dir, snapshots)
    snapshot_payloads = [
        {
            "ticker": snapshot.ticker,
            "state": snapshot.state.value,
            "as_of": snapshot.as_of,
            "context": snapshot.context,
            "metrics": snapshot.metrics,
        }
        for snapshot in snapshots
    ]
    return emit_mooner_callouts(
        base_dir,
        snapshots=snapshot_payloads,
        logger=logger,
    )


__all__ = [
    'load_mooner_callouts',
    'load_mooner_states',
    'run_mooner_sidecar',
]
