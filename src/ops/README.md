# Ops Telegram Command Client

This folder contains a small operational control surface for Speculation Edge. It runs a Telegram bot that accepts whitelisted commands and executes CLI entrypoints locally via subprocess. It does not place trades by itself.

## Prerequisites

- Python 3.10+ with access to the repo root.
- The `python-telegram-bot` package (v20+).
- A Telegram bot token and an allowlist of user IDs.

## Configuration

Open `src/ops/telegram_command_client.py` and update:

- `AUTHORIZED_USERS`: add your Telegram user IDs.
- `COMMAND_MAP`: map Telegram commands to CLI commands.
- `COMMAND_HELP`: update descriptions to match.
- `LOG_PATH`: where command logs are written.

Environment variables:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

Install ops dependencies from the repo root:

```bash
pip install -r requirements-ops.txt
```

## Usage

Run from the repo root so relative paths resolve correctly:

```bash
python src/ops/telegram_command_client.py
```

The bot uses polling and will keep running until interrupted.

## Command execution

Commands are executed via `subprocess.run` with `capture_output=True`, so stdout and stderr are returned to Telegram. Output is truncated to `MAX_OUTPUT_CHARS`.

If you use a virtual environment, ensure `python` in `COMMAND_MAP` points to the correct interpreter (for example, `.venv\Scripts\python.exe` on Windows).

## Logging

Execution logs are written to `logs/telegram_command_client.log` with timestamps, return codes, and the output (newline-escaped).

## Notes

- The default `status` command expects a CLI subcommand in `main.py`. If it does not exist, the command will return an error in Telegram.
- Keep `AUTHORIZED_USERS` tight; all allowed commands execute locally.
