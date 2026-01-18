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
For `/scan` and `/pretrade`, the bot replies with the output artifacts (SetupCandidates or pretrade report) instead of stdout/log noise. These replies are sent even if `OPS_NO_REPLY_COMMANDS` includes `scan` or `pretrade` (unless `OPS_OUTPUT_MODE=none`).

If you use a virtual environment, set `OPS_PYTHON` to the correct interpreter (for example, `.venv\Scripts\python.exe` on Windows). The client defaults to the current interpreter.

The client loads `trading_bot/.env` to pick up `TELEGRAM_BOT_TOKEN` and other shared secrets before running any commands.

## Logging

Execution logs are written to `logs/telegram_command_client.log` with timestamps, return codes, and the output (newline-escaped).
Incoming updates are logged with `chat_id`, `chat_title`, and user info so you can set `TELEGRAM_CHAT_ID` without calling `getUpdates` manually.

## Notes

- The default `status` command expects a CLI subcommand in `main.py`. If it does not exist, the command will return an error in Telegram.
- `/status` responses include a brief list of running ops jobs before the CLI status output.
- Keep `TELEGRAM_ADMIN_USERS` tight; all allowed commands execute locally.
- Avoid calling `getUpdates` in a browser while the bot is running; it can conflict with polling. Use the log file for chat visibility instead.
