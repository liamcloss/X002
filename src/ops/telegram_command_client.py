"""
This bot is an operational control surface for Speculation Edge. It enables manual,
remote, auditable execution of trading system commands via Telegram. It is not an
automated trading agent.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Iterable

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

AUTHORIZED_USERS = {123456789}

COMMAND_MAP = {
    "scan": ["python", "main.py", "scan"],
    "pretrade": ["python", "main.py", "pretrade"],
    "status": ["python", "main.py", "status"],
}

COMMAND_HELP = {
    "scan": "Run the scan pipeline.",
    "pretrade": "Run pretrade checks.",
    "status": "Show system status.",
    "help": "Show this help message.",
}

LOG_PATH = Path("logs/telegram_command_client.log")
MAX_OUTPUT_CHARS = 3500


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("telegram_command_client")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def user_is_authorized(update: Update, logger: logging.Logger) -> bool:
    user = update.effective_user
    user_id = user.id if user else None
    is_authorized = user_id in AUTHORIZED_USERS
    if not is_authorized:
        logger.warning("Unauthorized user attempt: user_id=%s", user_id)
    return is_authorized


def truncate_output(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return f"{text[:MAX_OUTPUT_CHARS]}\n...output truncated..."


def command_output_to_text(result: subprocess.CompletedProcess[str]) -> str:
    output = (result.stdout or "") + (result.stderr or "")
    if not output.strip():
        output = "(No output)"
    return truncate_output(output)


def command_description_lines(commands: Iterable[str]) -> str:
    lines = ["Allowed commands:"]
    for cmd in commands:
        description = COMMAND_HELP.get(cmd, "")
        lines.append(f"/{cmd} - {description}".rstrip())
    return "\n".join(lines)


async def send_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    if not user_is_authorized(update, logger):
        await update.message.reply_text("Unauthorized user.")
        return
    await update.message.reply_text(command_description_lines(COMMAND_HELP.keys()))


async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    if not user_is_authorized(update, logger):
        await update.message.reply_text("Unauthorized user.")
        return

    command_name = (update.message.text or "").lstrip("/").split()[0]
    command = COMMAND_MAP.get(command_name)
    if not command:
        await update.message.reply_text("Unknown command.")
        logger.info("Unknown command: user_id=%s command=%s", update.effective_user.id, command_name)
        return

    logger.info("Executing command: user_id=%s command=%s", update.effective_user.id, command_name)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = command_output_to_text(result)
    logger.info(
        "Command result: user_id=%s command=%s returncode=%s output=%s",
        update.effective_user.id,
        command_name,
        result.returncode,
        output.replace("\n", "\\n"),
    )
    await update.message.reply_text(output)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    if not user_is_authorized(update, logger):
        await update.message.reply_text("Unauthorized user.")
        return
    await update.message.reply_text("Unknown command.")
    logger.info("Unknown command: user_id=%s text=%s", update.effective_user.id, update.message.text)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    logger = setup_logging()

    application = ApplicationBuilder().token(token).build()
    application.bot_data["logger"] = logger

    application.add_handler(CommandHandler("help", send_help))
    application.add_handler(CommandHandler("scan", handle_command))
    application.add_handler(CommandHandler("pretrade", handle_command))
    application.add_handler(CommandHandler("status", handle_command))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    logger.info("Telegram command client started.")
    application.run_polling()


if __name__ == "__main__":
    main()
