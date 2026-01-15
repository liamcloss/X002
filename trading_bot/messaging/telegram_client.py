"""Telegram messaging client."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests


_API_BASE = "https://api.telegram.org"

logger = logging.getLogger("trading_bot")


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _send_message(text: str) -> None:
    token = _get_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_env("TELEGRAM_CHAT_ID")
    url = f"{_API_BASE}/bot{token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    response = requests.post(url, json=payload, timeout=30)
    if not response.ok:
        logger.error(
            "Telegram message failed | status=%s | body=%s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f"Telegram API error {response.status_code}: {response.text}"
        )

    logger.info("Telegram message sent")


def send_message(text: str) -> None:
    """Send a formatted message to Telegram."""

    _send_message(text)


def send_error(text: str) -> None:
    """Send an error message to Telegram."""

    _send_message(text)
