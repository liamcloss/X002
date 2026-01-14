"""Messaging utilities."""

from __future__ import annotations

import logging

import requests

from trading_bot.config import Config


def send_message(logger: logging.Logger, config: Config, message: str) -> None:
    """Send a message via Telegram."""

    url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
    payload = {"chat_id": config.telegram_chat_id, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        logger.info("Telegram message sent.")
    except requests.RequestException as exc:
        logger.error("Failed to send Telegram message: %s", exc)
