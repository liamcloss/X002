# X002 Trading Bot

A small CLI-driven trading bot that can scan markets, refresh a universe of instruments, and replay historical data for backtesting. This repository focuses on the core pipeline, data handling, and integrations (Trading212 + Telegram). It is designed to be run locally with explicit configuration in place.

## Contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Environment variables](#environment-variables)
- [Running the CLI](#running-the-cli)
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
pip install pandas numpy yfinance requests python-dotenv pyarrow

# 3) Create a .env file with your secrets (see below)
cp .env.example .env  # or create manually

# 4) Run a dry-run scan (no external output)
python main.py scan --dry-run
```

> **Note**: `requirements.txt` includes a Parquet engine (`pyarrow`) so pandas can read/write `.parquet` files used by the cache and universe refresh flows.

## Configuration

Configuration is split between code constants and environment variables.

- **Mode**: update `MODE` in `trading_bot/config.py` to `TEST` or `LIVE`.
- **Storage**: the CLI will auto-create these folders in the repo root if they don't exist: `data/`, `logs/`, `universe/`, `state/`.

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
python main.py replay --days 90
python main.py replay --start-date 2023-01-01 --days 60
```

### Commands

- `scan`: Runs the daily market scan. Use `--dry-run` to avoid messaging/state writes.
- `universe`: Refreshes the Trading212 instrument universe.
- `replay`: Replays historical data for backtesting. Supports `--days` and `--start-date`.

## Project layout

```
.
├── main.py                   # CLI entrypoint
└── trading_bot/
    ├── backtest/             # Replay/backtest logic
    ├── execution/            # Execution logic
    ├── market_data/          # Data fetching + indicators
    ├── messaging/            # Telegram integration
    ├── pipeline/             # Daily scan orchestration
    ├── signals/              # Signal generation + filters
    ├── universe/             # Universe refresh + normalization
    └── ...
```

## Troubleshooting

- **Missing environment variables**: Ensure `.env` exists and includes all four required keys.
- **Invalid MODE**: Set `MODE = "TEST"` or `MODE = "LIVE"` in `trading_bot/config.py`.
- **Parquet engine errors**: Install `pyarrow` (included in `requirements.txt`) so pandas can read/write `.parquet` files used by caching/universe refresh.
- **Dependency errors**: Re-run the `pip install -r requirements.txt` line in [Quick start](#quick-start) inside your venv.

---

If you'd like a `requirements.txt` or `pyproject.toml` added, open an issue or extend the setup instructions above to match your environment.
