# Mooner Sidecar
This optional capability watches a small, high-conviction subset of the universe for early evidence of a “Mooner regime”—a breakout/expansion situation that is rare and might deserve heightened attention.

## What it does
- **Subset selection** – `trading_bot.mooner.mooner_subset_selector` reads `MOONER_SUBSET_TICKERS` and trims it to `MOONER_SUBSET_MAX`, filtering out instruments marked inactive in `universe/clean/universe.parquet`. Once selected, the subset is cached to `MoonerSubset.json`.
- **State engineering** – `mooner_state_engine` loads historical OHLC data from `data/prices` (via `trading_bot.market_data.cache`) and derives ATR, MA slopes, prior highs, and range statistics. Every ticker is classified into `DORMANT`, `WARMING`, `ARMED`, or `FIRING`, along with human-friendly context and metrics.
- **Callout emission** – `mooner_callout_emitter` compares the current state to the prior run (stored in `state/mooner_state.json`). When a ticker hits `FIRING`, it writes `MoonerCallouts.json` and returns the callout payload unless it was already firing (controlled by `MOONER_EMIT_WHILE_FIRING`).
- **Orchestration** – `run_mooner_sidecar` wires the three stages, skips execution if price data is missing, and always returns whatever callouts were emitted.

## Why it’s useful
- Provides an extra eyes-on signal that is isolated from normal scan/pretrade flows.
- Fully deterministic: no external requests, pure cache-driven metrics.
- Emits JSON artifacts that your ops tooling or Telegram bot can consume to highlight regime changes.

## Architecture

The sidecar now runs as an autonomous companion pipeline:

1. **Mooner Candidate Pool (`MoonerCandidatePool.json`)** – scans every active ticker’s last 60 trading days of yfinance cache for liquidity, volatility, compression, or volume surges and collects ~200–600 names.
2. **Mooner Universe (`MoonerUniverse.json`)** – filters the pool by price floor, 200-day slope, recent drawdown, and lack of nearby resistance so only structurally sound instruments remain.
3. **Mooner Subset (`MoonerSubset.json`)** – ranks the universe by ATR compression, relative strength, and structural cleanliness; selects the top 10 tickers deterministically with no manual overrides.
4. **Mooner State Engine (`MoonerState.json`)** – classifies the subset into `DORMANT`, `WARMING`, `ARMED`, or `FIRING` based on ATR, range, moving averages, and volume rules.
5. **Mooner Callouts (`MoonerCallouts.json`)** – emits a single informational alert when a subset ticker transitions into `FIRING` (context only, no sizing).

Every JSON artifact includes an `as_of` date or `state_since`, making this pipeline fully auditable.

## Running the sidecar
1. Import and call the orchestrator from any context that has access to `base_dir` and a logger, for example:
   ```python
   from trading_bot.mooner import run_mooner_sidecar
   run_mooner_sidecar(base_dir=Path(__file__).resolve().parent, logger=logger)
   ```
2. If the subset cache already exists, the pipeline will reuse it; otherwise it regenerates it based on the configured tickers and universe.
3. After execution, check:
   - `MoonerStates.json` for the full state dump.
   - `MoonerCallouts.json` for the `FIRING` alerts (also returned by `run_mooner_sidecar`).
   - `state/mooner_state.json` for the last emitted status to avoid duplicates.
4. The CLI exposes `python main.py mooner`, which wired into the ops Telegram bot, so you can run the sidecar directly or through `/mooner`.

- ## Suggested integration
- Run it after `market_data` refresh to make sure the cache powering all five stages is fresh.
- `/mooner` (or `python main.py mooner`) writes all five JSON artifacts so you can inspect raw pools, universes, states, and callouts independently.
- Since the sidecar never writes orders, it is safe to schedule before your scan and pretrade jobs; the ops scheduler already runs `/mooner` at 05:45.
- The Telegram callout section in the scan (⚠️ Mooner Call-Outs) consumes `MoonerCallouts.json`, so every FIRING regime surfaces as informational context without affecting execution authority.
### Unhooking Mooner
- Clear `OPS_SCHEDULE_MOONER` or disable `OPS_SCHEDULE_ENABLED` to stop the scheduler, remove `/mooner` from any automated workflows, and delete `Mooner*.json` artifacts to return to the old state.
- Everything else remains untouched: you can still run `python main.py mooner` manually for analysis without feeding the results into the scanner or pretrade gates.

## Observability
- All warnings (missing price cache, bad OHLC, write failures) already bubble to the provided logger.
- The sidecar never modifies live orders or Telegram state—it only produces read-only callouts that you can choose how to surface.
