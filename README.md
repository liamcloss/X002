# X002 Trading Bot

A small CLI-driven trading bot that can scan markets, refresh a universe of instruments, and replay historical data for backtesting. This repository focuses on the core pipeline, data handling, and integrations (yfinance market data, Trading212 account/execution, Telegram). It is designed to be run locally with explicit configuration in place.

## Contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Environment variables](#environment-variables)
- [Running the CLI](#running-the-cli)
- [Pre-trade viability](#pre-trade-viability)
- [Market data and symbol mapping](#market-data-and-symbol-mapping)
- [Universe active flag](#universe-active-flag)
- [Outputs and reports](#outputs-and-reports)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Python** 3.10+ (recommended: 3.11).
- **pip** and **venv** (ships with Python on most systems).
- **Network access** if you intend to pull market data or communicate with Trading212/Telegram.

## Quick start

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Create a .env file with your secrets (see below)
cp .env.example .env  # or create manually

# 4) Run a dry-run scan (no external output)
python main.py scan --dry-run
```

> **Note**: `requirements.txt` includes a Parquet engine (`pyarrow`) so pandas can read/write `.parquet` files used by the cache and universe refresh flows.

## Configuration

Configuration is split between code constants and environment variables.

- **Mode**: update `MODE` in `trading_bot/config.py` to `TEST` or `LIVE`.
- **Storage**: the CLI will auto-create these folders in the repo root if they don't exist: `data/`, `logs/`, `outputs/`, `state/`, `universe/`.
- **Pre-trade settings**: adjust `MAX_SPREAD_PCT`, `PRETRADE_CANDIDATE_LIMIT`, `SPREAD_SAMPLE_LOOKBACK_DAYS`, and `SPREAD_SAMPLE_OPEN_COOLDOWN_MINUTES` in `trading_bot/config.py` to tune risk gates and reporting.

If you run into errors about missing variables or misconfigured mode, double-check the `.env` file and `MODE` value.

## Environment variables

The bot requires a `.env` file with the following variables:

```
T212_API_KEY=your_trading212_key
T212_API_SECRET=your_trading212_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

Create the file in the repository root:

```bash
touch .env
```

If you prefer, create a local template file for convenience:

```bash
cat <<'EOF' > .env.example
T212_API_KEY=
T212_API_SECRET=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
```

## Running the CLI

All commands are exposed via `main.py`:

```bash
python main.py scan --dry-run
python main.py scan
python main.py universe
python main.py pretrade
python main.py replay --days 90
python main.py replay --start-date 2023-01-01 --days 60
```

### Commands

- `scan`: Runs the daily market scan. Use `--dry-run` to avoid messaging/state writes.
- `universe`: Refreshes the Trading212 instrument universe.
- `pretrade`: Evaluates yesterday's setup candidates against live quote rules, prints a console table, writes JSON outputs, and sends one consolidated Telegram summary.
- `replay`: Replays historical data for backtesting. Supports `--days` and `--start-date`.

## Pre-trade viability

The pre-trade check is a second-phase guardrail. It evaluates the latest `SetupCandidates.json` produced by the scan and answers one question: which ideas are still tradable right now under your rules.

- **Quote source**: yfinance only (Trading212 is used for account and execution only).
- **Spread check**: rejects quotes with missing bid/ask or spread over `MAX_SPREAD_PCT`.
- **Rule check**: applies PreTradeGate logic for drift, stop distance, RR, and trade cap.
- **Outputs**:
  - `outputs/pretrade_viability_<timestamp>.json` for full machine-readable results.
  - `outputs/spread_report_<timestamp>.json` with per-instrument spread stats for the last N days.
  - `state/spread_samples.json` for stored spread samples.
- **Sampling window**: spread sampling skips the first `SPREAD_SAMPLE_OPEN_COOLDOWN_MINUTES` after each market open and ignores pre-open sampling for those markets.

`SetupCandidates.json` is written on non-dry-run scans and contains up to `PRETRADE_CANDIDATE_LIMIT` ideas, even if only the top ideas are alerted via Telegram.

## Market data and symbol mapping

- **Market data source**: yfinance is the only quote source used for pre-trade and market data refresh.
- **Trading212 usage**: account balance, positions, and execution only (no price or spread fetching).
- **Symbol mapping**: Trading212 tickers are mapped to yfinance symbols via `trading_bot/symbols.py`, using `display_ticker` when available. The pre-trade guard logs mappings when a symbol is rewritten.

## Universe active flag

The universe refresh writes `universe/clean/universe.parquet` with an `active` column. Scans and market data refreshes only consider active instruments.

- **Inference**: `active` defaults to `max_open_quantity > 0` when missing.
- **Auto-deactivation**: repeated yfinance data failures can mark a ticker inactive in the universe file.
- **Reactivation**: update `active` back to `True` in the parquet or re-run the universe refresh.

## Outputs and reports

- `SetupCandidates.json`: latest scan ideas used by `pretrade` (written on non-dry-run scans).
- `outputs/pretrade_viability_<timestamp>.json`: full pre-trade evaluation results.
- `outputs/spread_report_<timestamp>.json`: per-instrument spread statistics for the lookback window.
- `state/spread_samples.json`: spread sample store used to build reports.
- `logs/<date>.log`: runtime logs from the CLI.

## Project layout

```
.
├── main.py                   # CLI entrypoint
├── outputs/                  # Pre-trade reports and viability JSON
├── state/                    # Runtime state and spread samples
├── src/market_data/          # yfinance quote client
├── src/ops/                  # Operational tooling (see src/ops/README.md)
└── trading_bot/
    ├── backtest/             # Replay/backtest logic
    ├── execution/            # Execution logic
    ├── market_data/          # Data fetching + indicators
    ├── messaging/            # Telegram integration
    ├── pipeline/             # Daily scan orchestration
    ├── pretrade/             # Pre-trade gate + spread sampling
    ├── signals/              # Signal generation + filters
    ├── universe/             # Universe refresh + normalization
    └── ...
```

## Troubleshooting

- **Missing environment variables**: Ensure `.env` exists and includes all four required keys.
- **Invalid MODE**: Set `MODE = "TEST"` or `MODE = "LIVE"` in `trading_bot/config.py`.
- **Parquet engine errors**: Install `pyarrow` (included in `requirements.txt`) so pandas can read/write `.parquet` files used by caching/universe refresh.
- **Dependency errors**: Re-run the `pip install -r requirements.txt` line in [Quick start](#quick-start) inside your venv.
- **All pre-trade setups rejected**: After market close, yfinance often has no bid/ask, so the spread gate rejects. Run pretrade during market hours or review the spread report for realistic caps.
- **Ticker missing in yfinance**: Confirm the Trading212 symbol maps correctly to Yahoo via `trading_bot/symbols.py` and that `display_ticker` is present in `SetupCandidates.json`.
- **Instrument missing from scans**: Check the `active` flag in `universe/clean/universe.parquet` and the auto-deactivation notes above.

---

For additional details about pipelines or configuration, extend the setup instructions above to match your environment.
