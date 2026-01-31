"""Telegram messaging client."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests


_API_BASE = 'https://api.telegram.org'
_MARKDOWN_ESCAPE_PATTERN = re.compile(r'([_*\[\]`])')
_DEFAULT_CHAT_ENV = 'TELEGRAM_CHAT_ID'
_CHAT_ALIASES = {
    'TELEGRAM_CHAT_ID': 'alerts',
    'TELEGRAM_PAPER_CHAT_ID': 'paper',
    'OPS_SCHEDULE_CHAT_ID': 'ops-schedule',
}

logger = logging.getLogger('trading_bot')


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f'Missing required environment variable: {name}')
    return value


def _get_optional_env(name: str) -> str | None:
    return os.getenv(name) or None


def _escape_markdown(text: str) -> str:
    if not text:
        return text
    text = text.replace('\\', '\\\\')
    return _MARKDOWN_ESCAPE_PATTERN.sub(r'\\\1', text)


def _friendly_chat_alias(env_name: str | None) -> str:
    if not env_name:
        return 'custom'
    return _CHAT_ALIASES.get(env_name, env_name.lower())


def _context_label(context: str | None, alias: str) -> str:
    return context if context else alias


def _resolve_chat_target(
    chat_id: str | None,
    chat_env_var: str | None,
    required_chat: bool,
) -> tuple[str | None, str]:
    env_name = chat_env_var or _DEFAULT_CHAT_ENV
    if chat_id:
        return chat_id, env_name
    if required_chat:
        return _get_env(env_name), env_name
    return _get_optional_env(env_name), env_name


def _extract_error_reason(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        fallback = response.text.strip()
        return fallback or response.reason or 'Unknown Telegram error'
    description = str(payload.get('description') or '').strip()
    error_code = payload.get('error_code')
    reason_parts: list[str] = []
    if description:
        reason_parts.append(description)
    if error_code is not None:
        reason_parts.append(f'error_code={error_code}')
    if not reason_parts:
        fallback = response.text.strip()
        return fallback or response.reason or 'Unknown Telegram error'
    return ' | '.join(reason_parts)


def _send_message(
    text: str,
    *,
    chat_id: str | None = None,
    escape_markdown: bool = True,
    context: str | None = None,
    chat_env_var: str | None = None,
    required_chat: bool = True,
) -> bool:
    target_chat_id, env_name = _resolve_chat_target(chat_id, chat_env_var, required_chat)
    chat_alias = _friendly_chat_alias(env_name if target_chat_id else chat_env_var or env_name)
    context_label = _context_label(context, chat_alias)
    if not target_chat_id:
        logger.info(
            'Telegram message skipped | context=%s | chat_env=%s not configured',
            context_label,
            env_name,
        )
        return False

    token = _get_env('TELEGRAM_BOT_TOKEN')
    url = f'{_API_BASE}/bot{token}/sendMessage'
    if escape_markdown:
        text = _escape_markdown(text)
    payload: dict[str, Any] = {
        'chat_id': target_chat_id,
        'text': text,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True,
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:
        logger.error(
            'Telegram request failed | context=%s | chat_alias=%s | chat=%s | error=%s',
            context_label,
            chat_alias,
            target_chat_id,
            exc,
        )
        return False

    if not response.ok:
        reason = _extract_error_reason(response)
        logger.error(
            'Telegram message failed | context=%s | chat_alias=%s | chat=%s | status=%s | reason=%s',
            context_label,
            chat_alias,
            target_chat_id,
            response.status_code,
            reason,
        )
        return False

    logger.info(
        'Telegram message sent | context=%s | chat_alias=%s | chat=%s',
        context_label,
        chat_alias,
        target_chat_id,
    )
    return True


def send_message(text: str, escape_markdown: bool = True, context: str | None = None) -> bool:
    """Send a formatted message to Telegram."""

    return _send_message(
        text,
        escape_markdown=escape_markdown,
        context=context or 'alert',
        chat_env_var='TELEGRAM_CHAT_ID',
        required_chat=True,
    )


def send_paper_message(
    text: str,
    escape_markdown: bool = True,
    context: str | None = None,
) -> bool:
    """Send a paper trade message to Telegram."""

    return _send_message(
        text,
        escape_markdown=escape_markdown,
        context=context or 'paper',
        chat_env_var='TELEGRAM_PAPER_CHAT_ID',
        required_chat=False,
    )


def send_error(text: str, escape_markdown: bool = True, context: str | None = None) -> bool:
    """Send an error message to Telegram."""

    return _send_message(
        text,
        escape_markdown=escape_markdown,
        context=context or 'error',
        chat_env_var='TELEGRAM_CHAT_ID',
        required_chat=True,
    )
