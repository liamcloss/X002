mov# X002 Trading Bot

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
- **Daily scan display**: adjust `daily_scan_display_limit` in `config.yaml` to control how many ideas the Telegram scan summary includes (default 10); pre-trade still evaluates the top `PRETRADE_CANDIDATE_LIMIT` setups.
- **Mooner settings**: adjust `MOONER_SUBSET_MAX`, `MOONER_CANDIDATE_VOLUME_THRESHOLD`, `MOONER_CANDIDATE_DOLLAR_VOLUME_GBP`, `MOONER_RESISTANCE_BUFFER`, and the ATR/drawdown thresholds in `trading_bot/config.py` to tweak how regimes are discovered and filtered.

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
python main.py status
python main.py status --verbose
python main.py mooner
python main.py news-scout
python main.py yolo
python main.py replay --days 90
python main.py replay --start-date 2023-01-01 --days 60
```

### Commands

- `scan`: Runs the daily market scan. Use `--dry-run` to avoid messaging/state writes.
- `universe`: Refreshes the Trading212 instrument universe.
- `pretrade`: Evaluates yesterday's setup candidates against live quote rules, prints a console table, writes JSON outputs, and sends one consolidated Telegram summary.
- `status`: Prints the Speculation Edge health summary. The default summary splits command statuses into their own lines for readability, while `--verbose` includes directory paths, lock information, and file timestamps.
- `mooner`: Executes the Mooner regime sidecar and records any `FIRING` callouts for follow-up.
- `news-scout`: Builds the curated link-focused report for the latest setups, storing a JSON snapshot and optionally enriching entries with AI insights (see notes below).
- `yolo`: Runs the weekly penny-stock lottery (idempotent once per week) and prints the latest winner plus the stored alternative contenders so you can spot recurring picks and backup options.
- `replay`: Replays historical data for backtesting. Supports `--days` and `--start-date`.

## Pre-trade viability

The pre-trade check is a second-phase guardrail. It evaluates the latest `SetupCandidates.json` produced by the scan and answers one question: which ideas are still tradable right now under your rules.

- **Quote source**: yfinance only (Trading212 is used for account and execution only).
- **Spread check**: rejects quotes with missing bid/ask or spread over `MAX_SPREAD_PCT`.
- **Rule check**: applies PreTradeGate logic for drift, stop distance, RR, and trade cap.
- **Trade cap**: uses live open trades only; paper trades do not affect the pre-trade guard.
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

## Mooner autonomous pipeline

- **MoonerCandidatePool.json**: broad discovery via liquidity, volume surges, ATR compression, and volatility percentile filters across the last 60 trading days for every active symbol.
- **MoonerUniverse.json**: universe-worthy names remain after enforcing price floors, positive 200-day slopes, capped drawdowns, and no nearby resistance.
- **MoonerSubset.json**: the top 10 universe instruments ranked by ATR compression, relative strength, and structural cleanliness; no manual overrides are allowed.
- **MoonerState.json**: each subset ticker is classified into `DORMANT`, `WARMING`, `ARMED`, or `FIRING` using explicit ATR, range, MA, and volume rules.
- **MoonerCallouts.json**: informational alerts are emitted only when a subset ticker newly transitions into `FIRING`; nothing is traded or sized.

The sidecar runs before the nightly scan, keeps its JSON artifacts under the repo root, and logs its observations so you can audit each stage independently.

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
    ├── yolo_penny_lottery.py  # weekly penny-stock YOLO draw (standalone)
    └── ...
```

## Penny Stock YOLO Lottery

- Standalone weekly lottery: `extra_options/yolo_penny_lottery.py` reads the universe, applies the fixed penny filters, ranks by the YOLO score, and writes the week’s draw to `YOLO_Pick.json` while appending to `YOLO_Ledger.json`.
- Running `python main.py yolo` now executes the lottery (respecting the weekly lock) and prints the stored winner plus its derived alternatives so you can see other high-scoring YOLO candidates and whether the same ticker has been drawn before.
- `YOLO_Pick.json` now includes an `alternatives` array so the summary can report runner-up suggestions and their rationale, while `YOLO_Ledger.json` tracks every logged draw for easy repeat detection.
- It runs once per week (based on Monday) and always suggests a £2 stake; if the module is missing or fails, everything else keeps working exactly the same.
- No interaction with the scanner, Mooner, or pretrade flows—its only outputs are the two JSON artifacts, so you can disable or remove it without affecting the rest of the system.

## Troubleshooting

- **Missing environment variables**: Ensure `.env` exists and includes all four required keys.
- **Invalid MODE**: Set `MODE = "TEST"` or `MODE = "LIVE"` in `trading_bot/config.py`.
- **Parquet engine errors**: Install `pyarrow` (included in `requirements.txt`) so pandas can read/write `.parquet` files used by caching/universe refresh.
- **Dependency errors**: Re-run the `pip install -r requirements.txt` line in [Quick start](#quick-start) inside your venv.
- **All pre-trade setups rejected**: After market close, yfinance often has no bid/ask, so the spread gate rejects. Run pretrade during market hours or review the spread report for realistic caps.
- **Ticker missing in yfinance**: Confirm the Trading212 symbol maps correctly to Yahoo via `trading_bot/symbols.py` and that `display_ticker` is present in `SetupCandidates.json`.
- **Instrument missing from scans**: Check the `active` flag in `universe/clean/universe.parquet` and the auto-deactivation notes above.

## AI / News Scout Notes

- **Current scope**: `news_scout` collects the freshest setups, builds quick links (TradingView, Google News, Reddit, X, Threads), and publishes them as an optional Telegram report. It purposefully stops short of editorial claims such as “pump-and-dump.” Each `news_scout` run now tags the JSON snapshot and CLI summary with an `llm_insight` string whenever the AI hook contributes a sentiment note.
- **Optional AI insights**: Enable the built-in LLM by setting `news_scout.llm_enabled` to `true` in `config.yaml` and pointing `api_key_env` at the environment variable that stores your OpenAI key (defaults to `OPENAI_API_KEY`). The feature is gated and safe to disable by leaving the flag off or omitting the env var. When enabled, the logger records entries like `News scout LLM enrichment requested (model=..., entries=...)`, `News scout LLM call starting (model=...)`, and (if insight data is emitted) `News scout LLM applied insights to X entries: SYMBOLS...` so you can audit every invocation in `logs/<date>.log`. Each insight is surfaced as an `AI:` line in the CLI summary and persists in the JSON snapshot.
- **Next steps for news scout**: Consider augmenting outputs with:
  1. Structured headline metadata (source, timestamp, tone score).
  2. Local sentiment indicators (e.g., Vader) so we can flag “positive/neutral/negative” without external calls.
  3. Keywords or volume anomalies that escalate a candidate into a “high-interest” bucket for Telegram.
