"""Telegram messaging client."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests


_API_BASE = 'https://api.telegram.org'
_MARKDOWN_ESCAPE_PATTERN = re.compile(r'([_*\[\]`])')

logger = logging.getLogger('trading_bot')


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f'Missing required environment variable: {name}')
    return value


def _get_optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _escape_markdown(text: str) -> str:
    if not text:
        return text
    text = text.replace('\\', '\\\\')
    return _MARKDOWN_ESCAPE_PATTERN.sub(r'\\\1', text)


def _send_message(
    text: str,
    chat_id: str | None = None,
    escape_markdown: bool = True,
) -> None:
    token = _get_env('TELEGRAM_BOT_TOKEN')
    target_chat_id = chat_id or _get_env('TELEGRAM_CHAT_ID')
    url = f'{_API_BASE}/bot{token}/sendMessage'
    if escape_markdown:
        text = _escape_markdown(text)
    payload: dict[str, Any] = {
        'chat_id': target_chat_id,
        'text': text,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True,
    }

    response = requests.post(url, json=payload, timeout=30)
    if not response.ok:
        logger.error(
            'Telegram message failed | status=%s | body=%s',
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f'Telegram API error {response.status_code}: {response.text}'
        )

    logger.info('Telegram message sent')


def send_message(text: str, escape_markdown: bool = True) -> None:
    """Send a formatted message to Telegram."""

    _send_message(text, escape_markdown=escape_markdown)


def send_paper_message(text: str, escape_markdown: bool = True) -> None:
    """Send a paper trade message to Telegram."""

    chat_id = _get_optional_env('TELEGRAM_PAPER_CHAT_ID')
    _send_message(text, chat_id=chat_id, escape_markdown=escape_markdown)


def send_error(text: str, escape_markdown: bool = True) -> None:
    """Send an error message to Telegram."""

    _send_message(text, escape_markdown=escape_markdown)
