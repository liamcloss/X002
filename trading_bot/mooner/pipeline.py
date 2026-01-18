"""Mooner sidecar orchestration pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from trading_bot.mooner.mooner_candidate_pool import build_candidate_pool
from trading_bot.mooner.mooner_universe_builder import build_mooner_universe
from trading_bot.mooner.mooner_subset_ranker import rank_mooner_subset
from trading_bot.mooner.mooner_state_engine import evaluate_mooner_states
from trading_bot.mooner.mooner_callout_emitter import emit_mooner_callouts


def run_mooner_sidecar(base_dir: Path, logger: logging.Logger) -> list[dict]:
    """Execute the Mooner sidecar pipeline and return any generated callouts."""

    candidate_pool = build_candidate_pool(base_dir, logger)
    universe = build_mooner_universe(base_dir, candidate_pool, logger)
    subset = rank_mooner_subset(base_dir, universe, logger)
    if not subset:
        logger.info("Mooner subset empty; skipping state evaluation.")
        return []

    snapshots, _ = evaluate_mooner_states(base_dir, subset, logger)
    if not snapshots:
        return []

    callouts = emit_mooner_callouts(base_dir, snapshots, logger)
    return callouts


__all__ = [
    "run_mooner_sidecar",
]
