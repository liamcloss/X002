# Ops Telegram Command Client

This folder contains a small operational control surface for Speculation Edge. It runs a Telegram bot that accepts whitelisted commands and executes CLI entrypoints locally via subprocess. It does not place trades by itself.

## Prerequisites

- Python 3.10+ with access to the repo root.
- The `python-telegram-bot` package (v20+).
- A Telegram bot token and an allowlist of user IDs.

## Configuration

Environment variables:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ADMIN_USERS=6086802750,123456789
```

Optional ops controls:

```
OPS_PYTHON=.venv/Scripts/python.exe
OPS_OUTPUT_MODE=full        # full|errors|summary|none (summary is the default when invalid)
OPS_SILENT_COMMANDS=pretrade,scan
OPS_NO_REPLY_COMMANDS=pretrade
```

Optional scheduler controls (run jobs on a timetable from the same bot process):

```
OPS_SCHEDULE_ENABLED=1
OPS_SCHEDULE_CHAT_ID=123456789          # where scheduled outputs go (falls back to TELEGRAM_CHAT_ID)
OPS_SCHEDULE_USE_UTC=0                  # 1 = UTC, 0 = local time
OPS_SCHEDULE_UNIVERSE=sun 20:00
OPS_SCHEDULE_MARKET_DATA=daily 00:05
OPS_SCHEDULE_SCAN=daily 06:30
OPS_SCHEDULE_PRETRADE=weekday 08:45; weekday 14:45  # UK pretrade at 08:45, US pretrade at 14:45
OPS_SCHEDULE_MOONER=daily 22:00
OPS_SCHEDULE_NOTIFY_SKIPS=1             # optional: notify if a scheduled job is blocked
OPS_SCHEDULE_SEND_START=0               # optional: send a "started" message
```

Schedule format:
- Use `daily HH:MM` or `HH:MM` for daily runs.
- Use `mon|tue|...|sun HH:MM` for weekly runs.
- Use `weekday`/`weekend` or comma lists like `mon,wed,fri 07:30`.
- Multiple times can be separated with `;` (for example `daily 07:00; daily 12:00`), which allows duplicate commands (like the two pretrade windows) to run independently.

If you want to change the available commands, conflict groups, or log path, update `COMMAND_MAP`, `COMMAND_GROUPS`, `COMMAND_HELP`, or `LOG_PATH` in `src/ops/telegram_command_client.py`.

Install ops dependencies from the repo root:

```bash
pip install -r requirements-ops.txt
```

## Usage

Run from the repo root so relative paths resolve correctly (the client now sets `cwd` automatically):

```bash
python src/ops/telegram_command_client.py
```

The bot uses polling and will keep running until interrupted.

## Command execution

Commands are executed via async subprocesses, so multiple commands can run concurrently. If a command overlaps a running command group (for example, a second `/scan`), the bot responds that the command is already running. The bot also checks for command lock files in `state/` so it won’t start a scan or universe refresh if another process is already running. Output is truncated to `MAX_OUTPUT_CHARS`.
For `/scan`, the bot replies with the output artifacts (SetupCandidates) instead of stdout/log noise. These replies are sent even if `OPS_NO_REPLY_COMMANDS` includes `scan` (unless `OPS_OUTPUT_MODE=none`).
For `/pretrade`, the CLI sends per-setup Telegram messages directly to the chat that invoked the command; the ops bot does not send a summary reply on success.
For `/mooner`, the bot returns a concise summary of the latest Mooner sidecar callouts (stored in `MoonerCallouts.json`) so you can see any `FIRING` regimes without digging through logs. Each run also writes `MoonerCandidatePool.json`, `MoonerUniverse.json`, `MoonerSubset.json`, and `MoonerState.json` so you can audit every stage of the autonomous pipeline.
For `/yolo`, the bot triggers the standalone penny-stock lottery, posts the week's pick (from `YOLO_Pick.json`), and writes the ledger so you keep a history of every draw.
Scheduled jobs use the same conflict and lock checks; if a conflict is detected, the job is skipped (optionally notified via `OPS_SCHEDULE_NOTIFY_SKIPS`).
To keep scheduled `market_data` or `universe` quiet, add them to `OPS_SILENT_COMMANDS` or `OPS_NO_REPLY_COMMANDS`.

If you use a virtual environment, set `OPS_PYTHON` to the correct interpreter (for example, `.venv\Scripts\python.exe` on Windows). The client defaults to the current interpreter.

The client loads `trading_bot/.env` to pick up `TELEGRAM_BOT_TOKEN` and other shared secrets before running any commands.

## Logging

Execution logs are written to `logs/telegram_command_client.log` with timestamps, return codes, and the output (newline-escaped).
Incoming updates are logged with `chat_id`, `chat_title`, and user info so you can set `TELEGRAM_CHAT_ID` without calling `getUpdates` manually.

## Standalone entry points

If you prefer Windows Task Scheduler, run these directly:

```
python main.py universe
python -m trading_bot.market_data.fetch_prices
python main.py scan
python main.py pretrade
python main.py mooner
python main.py status
```

## Notes

- The default `status` command expects a CLI subcommand in `main.py`. If it does not exist, the command will return an error in Telegram.
- `/status` responses include a brief list of running ops jobs before the CLI status output.
- Keep `TELEGRAM_ADMIN_USERS` tight; all allowed commands execute locally.
- Avoid calling `getUpdates` in a browser while the bot is running; it can conflict with polling. Use the log file for chat visibility instead.
