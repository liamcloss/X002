"""
This bot is an operational control surface for Speculation Edge. It enables manual,
remote, auditable execution of trading system commands via Telegram. It is not an
automated trading agent.
"""

import asyncio
import html
import itertools
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Set
from urllib.parse import quote, quote_plus

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from trading_bot.mooner import format_mooner_callout_lines
from trading_bot.paths import (
    mooner_output_path,
    mooner_state_path,
    news_scout_output_path,
    pretrade_viability_path,
    setup_candidates_path,
    yolo_output_path,
)
from trading_bot.symbols import tradingview_symbol

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv

ENV_PATH = BASE_DIR / 'trading_bot' / '.env'
load_dotenv(dotenv_path=ENV_PATH)
PYTHON_EXECUTABLE = os.getenv('OPS_PYTHON') or sys.executable
OUTPUT_MODE = os.getenv('OPS_OUTPUT_MODE', 'full').strip().lower()
SILENT_COMMANDS = {
    item.strip().lower()
    for item in os.getenv('OPS_SILENT_COMMANDS', '').split(',')
    if item.strip()
}
NO_REPLY_COMMANDS = {
    item.strip().lower()
    for item in os.getenv('OPS_NO_REPLY_COMMANDS', '').split(',')
    if item.strip()
}

AUTHORIZED_USERS_ENV = 'TELEGRAM_ADMIN_USERS'

COMMAND_MAP = {
    "scan": [PYTHON_EXECUTABLE, "main.py", "scan"],
    "universe": [PYTHON_EXECUTABLE, "main.py", "universe"],
    "pretrade": [PYTHON_EXECUTABLE, "main.py", "pretrade"],
    "status": [PYTHON_EXECUTABLE, "main.py", "status"],
    "market_data": [PYTHON_EXECUTABLE, "-m", "trading_bot.market_data.fetch_prices"],
    "mooner": [PYTHON_EXECUTABLE, "main.py", "mooner"],
    "yolo": [PYTHON_EXECUTABLE, "extra_options/yolo_penny_lottery.py"],
    "news_scout": [PYTHON_EXECUTABLE, "main.py", "news-scout"],
}

COMMAND_HELP = {
    "scan": "Run the scan pipeline.",
    "universe": "Refresh the trading universe.",
    "pretrade": "Run pretrade checks.",
    "status": "Show system status (append 'verbose' for full output).",
    "market_data": "Refresh the market data cache.",
    "mooner": "Run the Mooner sidecar regime watch.",
    "yolo": "Run the penny stock YOLO lottery.",
    "yolo_history": "Display the YOLO ledger history.",
    "news_scout": "Run the news scout summary (alias /news-scout).",
    "rocket": "Alias for /mooner (🚀).",
    "🚀": "Alias for /mooner.",
    "help": "Show this help message.",
}

COMMAND_ALIASES = {
    "news-scout": "news_scout",
    "rocket": "mooner",
    "🚀": "mooner",
}

LOG_PATH = BASE_DIR / 'logs' / 'telegram_command_client.log'
MAX_OUTPUT_CHARS = 3500
MAX_MESSAGE_LOG_CHARS = 200
JOB_COUNTER = itertools.count(1)
PRODUCT_OUTPUT_COMMANDS = {'scan', 'mooner', 'yolo', 'news_scout'}

COMMAND_GROUPS = {
    "scan": {"scan", "ideas", "universe", "market_data"},
    "universe": {"universe", "market_data"},
    "pretrade": {"ideas"},
    "status": set(),
    "market_data": {"market_data", "scan", "universe"},
    "mooner": {"mooner", "scan", "market_data", "universe"},
    "news_scout": {"news_scout"},
    "yolo": {"yolo"},
}
LOCK_GROUPS = {
    "scan": {"scan", "market_data", "universe"},
    "ideas": {"pretrade", "scan"},
    "universe": {"universe", "scan", "market_data"},
    "market_data": {"market_data", "scan", "universe"},
    "mooner": {"mooner", "scan", "market_data", "universe"},
    "news_scout": {"news_scout"},
    "yolo": {"yolo"},
}
LOCK_DIR = BASE_DIR / 'state'
SCHEDULE_ENV_MAP = {
    "universe": "OPS_SCHEDULE_UNIVERSE",
    "market_data": "OPS_SCHEDULE_MARKET_DATA",
    "scan": "OPS_SCHEDULE_SCAN",
    "pretrade": "OPS_SCHEDULE_PRETRADE",
    "mooner": "OPS_SCHEDULE_MOONER",
    "yolo": "OPS_SCHEDULE_YOLO",
}
SCHEDULE_CATCHUP_WINDOW = timedelta(hours=24)
DAY_ALIASES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}
DAY_NAMES = {
    0: "mon",
    1: "tue",
    2: "wed",
    3: "thu",
    4: "fri",
    5: "sat",
    6: "sun",
}


@dataclass
class RunningJob:
    job_id: str
    command_name: str
    args: list[str]
    started_at: datetime
    chat_id: int | None
    user_id: int | None
    username: str | None
    groups: set[str]
    process_id: int | None = None


@dataclass
class LockConflict:
    name: str
    started_at: str | None


@dataclass
class ScheduleSpec:
    command_name: str
    days: set[int]
    hour: int
    minute: int
    second: int
    source: str
    next_run: datetime | None = None


@dataclass
class ScheduleCatchup:
    spec: ScheduleSpec
    expected_at: datetime


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("telegram_command_client")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _truncate_log_text(text: str | None, limit: int = MAX_MESSAGE_LOG_CHARS) -> str:
    if not text:
        return ''
    cleaned = text.replace('\n', '\\n').strip()
    if len(cleaned) <= limit:
        return cleaned
    return f'{cleaned[:limit]}...'


def _format_update_context(update: Update | None) -> str:
    if update is None:
        return 'update=None'
    user = update.effective_user
    chat = update.effective_chat
    message = update.effective_message
    text = _truncate_log_text(message.text or message.caption) if message else ''
    return (
        f'update_id={update.update_id} '
        f'user_id={user.id if user else None} '
        f'username={user.username if user else None} '
        f'chat_id={chat.id if chat else None} '
        f'chat_type={chat.type if chat else None} '
        f'chat_title={chat.title or chat.username if chat else None} '
        f'text={text}'
    )


def user_is_authorized(
    update: Update,
    logger: logging.Logger,
    authorized_users: Set[int],
) -> bool:
    user = update.effective_user
    chat = update.effective_chat
    user_id = user.id if user else None
    is_authorized = user_id in authorized_users
    if not is_authorized:
        logger.warning(
            "Unauthorized user attempt: user_id=%s username=%s chat_id=%s chat_title=%s",
            user_id,
            user.username if user else None,
            chat.id if chat else None,
            chat.title or chat.username if chat else None,
        )
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


def command_summary_text(command_name: str, return_code: int, job_id: str | None = None) -> str:
    prefix = f'Job {job_id}: ' if job_id else ''
    return (
        f'{prefix}Command "{command_name}" finished (exit code {return_code}). '
        'Output suppressed.'
    )


def command_description_lines(commands: Iterable[str]) -> str:
    lines = ["Allowed commands:"]
    for cmd in commands:
        description = COMMAND_HELP.get(cmd, "")
        lines.append(f"/{cmd} - {description}".rstrip())
    return "\n".join(lines)


def _command_groups(command_name: str) -> set[str]:
    groups = COMMAND_GROUPS.get(command_name)
    if groups is None:
        return {command_name}
    return set(groups)


def _build_job(command_name: str, command: list[str], update: Update) -> RunningJob:
    job_id = f'{command_name}-{next(JOB_COUNTER):04d}'
    user = update.effective_user
    chat = update.effective_chat
    return RunningJob(
        job_id=job_id,
        command_name=command_name,
        args=command,
        started_at=datetime.now(timezone.utc),
        chat_id=chat.id if chat else None,
        user_id=user.id if user else None,
        username=user.username if user else None,
        groups=_command_groups(command_name),
    )


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return 'unknown'
    return value.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def _format_lock_timestamp(value: str | None) -> str:
    if not value:
        return 'unknown'
    text = str(value)
    try:
        if text.endswith('Z'):
            text = f'{text[:-1]}+00:00'
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def _format_numeric(value: object | None, decimals: int = 2) -> str:
    if value is None:
        return 'n/a'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if numeric != numeric:  # NaN
        return 'n/a'
    return f'{numeric:.{decimals}f}'


def _format_ratio_percent(value: object | None, decimals: int = 2) -> str:
    if value is None:
        return 'n/a'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if numeric != numeric:  # NaN
        return 'n/a'
    return f'{numeric * 100:.{decimals}f}%'


def _format_percent(value: object | None, decimals: int = 2) -> str:
    if value is None:
        return 'n/a'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if numeric != numeric:  # NaN
        return 'n/a'
    return f'{numeric:.{decimals}f}%'


def _truncate_text(text: str | None, limit: int = 160) -> str:
    if not text:
        return ''
    cleaned = text.replace('\n', ' ').strip()
    if len(cleaned) <= limit:
        return cleaned
    return f'{cleaned[:limit]}...'


def _news_search_url(candidate: dict) -> str | None:
    query = _search_query_from_candidate(candidate)
    return _news_search_url_from_query(query)


def _search_query_from_candidate(candidate: dict) -> str | None:
    for key in ('display_ticker', 'raw_ticker', 'ticker', 'symbol'):
        value = candidate.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return None


def _news_search_url_from_query(query: str | None) -> str | None:
    if not query:
        return None
    return f'https://news.google.com/search?q={quote_plus(f"{query} stock")}'


def _news_search_url_for_ticker(ticker: str | None) -> str | None:
    query = _search_query_from_ticker(ticker)
    return _news_search_url_from_query(query)


def _reddit_search_url(query: str | None) -> str | None:
    if not query:
        return None
    return f'https://www.reddit.com/search/?q={quote_plus(query)}&sort=top&t=all'


def _x_search_url(query: str | None) -> str | None:
    if not query:
        return None
    return f'https://x.com/search?q={quote_plus(query)}'


def _threads_search_url(query: str | None) -> str | None:
    if not query:
        return None
    return f'https://www.threads.net/search?q={quote_plus(query)}'


def _search_query_from_ticker(ticker: str | None) -> str | None:
    if not ticker:
        return None
    text = str(ticker).strip()
    if not text:
        return None
    symbol = text.split('_')[0]
    if not symbol:
        return None
    return symbol


def _tradingview_url_from_ticker(ticker: str | None) -> str | None:
    if not ticker:
        return None
    tv_symbol = tradingview_symbol(ticker)
    if not tv_symbol:
        return None
    safe_symbol = quote(tv_symbol.replace(' ', ''))
    if ':' in tv_symbol:
        return f'https://www.tradingview.com/chart/?symbol={safe_symbol}'
    return f'https://www.tradingview.com/symbols/{safe_symbol}/'


def _html_escape(value: object | None) -> str:
    if value is None:
        return ''
    return html.escape(str(value))


def _link_html(url: str | None, label: str = 'LINK') -> str:
    if not url:
        return _html_escape(label)
    safe_url = html.escape(url, quote=True)
    safe_label = html.escape(label)
    return f'<a href="{safe_url}">{safe_label}</a>'


def _load_yolo_ledger(base_dir: Path) -> list[dict]:
    path = base_dir / 'YOLO_Ledger.json'
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _format_yolo_history_message(base_dir: Path) -> str:
    ledger = _load_yolo_ledger(base_dir)
    if not ledger:
        return 'YOLO history unavailable; no ledger entries found.'
    entries = ledger[-10:]
    total = len(ledger)
    scores = [
        (entry.get('yolo_score'), entry)
        for entry in ledger
        if isinstance(entry.get('yolo_score'), (int, float))
    ]
    best = max(scores, default=(None, None))
    worst = min(scores, default=(None, None))
    lines = [
        'YOLO LEDGER',
        f'Picks logged: {total}',
    ]
    if best[0] is not None and best[1]:
        ticker = best[1].get('ticker', 'unknown')
        lines.append(f'Best score: {best[0]:.2f} ({ticker})')
    if worst[0] is not None and worst[1]:
        ticker = worst[1].get('ticker', 'unknown')
        lines.append(f'Lowest score: {worst[0]:.2f} ({ticker})')
    lines.append('')
    lines.append('Recent picks:')
    for entry in reversed(entries):
        week = entry.get('week_of', 'unknown')
        ticker = entry.get('ticker', 'unknown')
        price = _format_numeric(entry.get('price'))
        score = _format_numeric(entry.get('yolo_score'))
        lines.append(f'{week} → {ticker} @ {price} (score {score})')
    if total > len(entries):
        lines.append(f'...showing {len(entries)} of {total} picks')
    return '\n'.join(lines)

def _build_yolo_link_line(ticker: str | None) -> str | None:
    query = _search_query_from_ticker(ticker)
    if not query:
        return None
    links: list[str] = []
    tv_url = _tradingview_url_from_ticker(ticker)
    if tv_url:
        links.append(f'{_html_escape("📈 ")}{_link_html(tv_url)}')
    news_url = _news_search_url_for_ticker(ticker)
    if news_url:
        links.append(f'{_html_escape("📰 ")}{_link_html(news_url)}')
    reddit_url = _reddit_search_url(query)
    if reddit_url:
        links.append(f'{_html_escape("🧠 ")}{_link_html(reddit_url)}')
    x_url = _x_search_url(query)
    if x_url:
        links.append(f'{_html_escape("🐦 ")}{_link_html(x_url)}')
    threads_url = _threads_search_url(query)
    if threads_url:
        links.append(f'{_html_escape("🧵 ")}{_link_html(threads_url)}')
    if not links:
        return None
    return ' | '.join(links)


def _build_pretrade_link_line(result: dict) -> str | None:
    query = _search_query_from_candidate(result)
    candidate_ticker = result.get('symbol')
    display_ticker = result.get('display_ticker')
    tv_source = result.get('tradingview_url') or _tradingview_url_from_ticker(display_ticker or candidate_ticker)
    links: list[str] = []
    if tv_source:
        links.append(_link_html(tv_source, label='Chart'))
    news_target = display_ticker or candidate_ticker or query
    news_url = _news_search_url_for_ticker(news_target)
    if news_url:
        links.append(_link_html(news_url, label='News'))
    reddit_url = _reddit_search_url(query)
    if reddit_url:
        links.append(_link_html(reddit_url, label='Reddit'))
    x_url = _x_search_url(query)
    if x_url:
        links.append(_link_html(x_url, label='X'))
    threads_url = _threads_search_url(query)
    if threads_url:
        links.append(_link_html(threads_url, label='Threads'))
    if not links:
        return None
    return ' | '.join(links)


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _build_scan_output(base_dir: Path, since: datetime | None) -> str | None:
    path = setup_candidates_path(base_dir)
    if since and path.exists():
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified_at < since:
            return None
    payload = _load_json(path)
    if payload is None:
        return None
    if isinstance(payload, list):
        candidates = [item for item in payload if isinstance(item, dict)]
        meta: dict[str, object] = {}
    elif isinstance(payload, dict):
        candidates = payload.get('candidates') if isinstance(payload.get('candidates'), list) else []
        meta = payload
    else:
        return None

    generated_at = _format_lock_timestamp(str(meta.get('generated_at')) if meta else None)
    data_as_of = _format_lock_timestamp(str(meta.get('data_as_of')) if meta else None)
    mode = str(meta.get('mode', 'unknown')).upper() if meta else 'UNKNOWN'
    lines = [
        'SCAN OUTPUT',
        f'Generated at: {_html_escape(generated_at)}',
        f'Data as of: {_html_escape(data_as_of)}',
        f'Mode: {_html_escape(mode)}',
        f'Candidates: {len(candidates)}',
        '',
    ]
    if not candidates:
        lines.append('No candidates found.')
        return '\n'.join(lines)

    display_limit = min(10, len(candidates))
    for index, candidate in enumerate(candidates[:display_limit], start=1):
        symbol = str(
            candidate.get('display_ticker')
            or candidate.get('symbol')
            or candidate.get('ticker')
            or ''
        ).strip()
        rank = candidate.get('rank') or index
        entry = _format_numeric(candidate.get('planned_entry') or candidate.get('price'))
        stop = _format_numeric(candidate.get('planned_stop') or candidate.get('stop_price'))
        target = _format_numeric(candidate.get('planned_target') or candidate.get('target_price'))
        rr = _format_numeric(candidate.get('rr'))
        currency_symbol = str(candidate.get('currency_symbol') or '').strip()
        currency_code = str(candidate.get('currency_code') or '').strip().upper()
        entry_text = entry if entry == 'n/a' else f'{currency_symbol}{entry}'.strip()

        lines.append(f'{_html_escape(str(rank))}. {_html_escape(symbol)}')
        entry_line = f'   Entry: {_html_escape(entry_text)}'
        if currency_code:
            entry_line = f'{entry_line} {_html_escape(currency_code)}'
        entry_line = (
            f'{entry_line} | Stop: {_html_escape(stop)}'
            f' | Target: {_html_escape(target)} | RR: {_html_escape(rr)}'
        )
        lines.append(entry_line)

        reason = _truncate_text(str(candidate.get('reason') or '').strip())
        volume = _format_numeric(candidate.get('volume_multiple'))
        momentum = _format_ratio_percent(candidate.get('momentum_5d'))
        setup_parts = []
        if reason:
            setup_parts.append(reason)
        if volume != 'n/a':
            setup_parts.append(f'Vol: {volume}x')
        if momentum != 'n/a':
            setup_parts.append(f'Momentum: {momentum}')
        if setup_parts:
            lines.append(f'   Setup: {_html_escape(" | ".join(setup_parts))}')

        chart_url = candidate.get('tradingview_url')
        if chart_url:
            lines.append(f'   Chart: {_link_html(chart_url)}')
        news_url = _news_search_url(candidate)
        if news_url:
            lines.append(f'   News: {_link_html(news_url)}')
        lines.append('')
    if len(candidates) > display_limit:
        lines.append(f'...showing {display_limit} of {len(candidates)} candidates')
    return '\n'.join(lines)


def _pretrade_spread_pct(result: dict) -> float | None:
    spread_pct = result.get('spread_pct')
    if isinstance(spread_pct, (int, float)):
        return float(spread_pct)
    spread = result.get('spread')
    if isinstance(spread, (int, float)):
        return float(spread) * 100
    return None


def _format_rank(value: object | None, fallback: int | None = None) -> str:
    if value is None:
        return str(fallback) if fallback is not None else 'n/a'
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(fallback) if fallback is not None else 'n/a'
    if numeric <= 0:
        return str(fallback) if fallback is not None else 'n/a'
    return str(numeric)


def _result_scan_rank(result: dict) -> int:
    value = result.get('scan_rank')
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 10**9
    if numeric <= 0:
        return 10**9
    return numeric


def _ensure_pretrade_ranks(results: list[dict]) -> None:
    for index, result in enumerate(results, start=1):
        if _result_scan_rank(result) >= 10**9:
            result['scan_rank'] = index

    executables = [item for item in results if item.get('status') == 'EXECUTABLE']
    executables.sort(key=_result_scan_rank)
    for index, result in enumerate(executables, start=1):
        if _format_rank(result.get('exec_rank'), fallback=None) == 'n/a':
            result['exec_rank'] = index


def _build_pretrade_output(base_dir: Path, since: datetime | None) -> str | None:
    outputs_dir = pretrade_viability_path(base_dir)
    latest = _latest_file(outputs_dir, 'pretrade_viability_*.json')
    if latest is None:
        return None
    if since:
        modified_at = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
        if modified_at < since:
            return None
    payload = _load_json(latest)
    if not isinstance(payload, list):
        return None

    results = [item for item in payload if isinstance(item, dict)]
    checked_at = None
    if results:
        checked_at = results[0].get('checked_at')
    checked_at_text = _format_lock_timestamp(str(checked_at)) if checked_at else 'unknown'

    if not results:
        return '\n'.join(
            [
                'PRETRADE OUTPUT',
                f'Checked at: {checked_at_text}',
                'Executable: 0 | Rejected: 0',
                '',
                'No setups evaluated.',
            ]
        )

    _ensure_pretrade_ranks(results)
    executables = [item for item in results if item.get('status') == 'EXECUTABLE']
    rejected = [item for item in results if item.get('status') != 'EXECUTABLE']
    executables.sort(key=_result_scan_rank)
    rejected.sort(key=_result_scan_rank)

    lines = [
        'PRETRADE OUTPUT',
        f'Checked at: {_html_escape(checked_at_text)}',
        f'Executable: {len(executables)} | Rejected: {len(rejected)}',
        '',
        'EXECUTABLES (ranked by scan)',
    ]

    if not executables:
        lines.append('None.')
    else:
        for index, result in enumerate(executables[:15], start=1):
            symbol = str(result.get('symbol') or '').strip()
            scan_rank = _format_rank(result.get('scan_rank'))
            exec_rank = _format_rank(result.get('exec_rank'), fallback=index)
            lines.append(
                f'{_html_escape(exec_rank)}. {_html_escape(symbol)} '
                f'(scan #{_html_escape(scan_rank)}) -> EXECUTABLE'
            )
            spread_pct = _format_percent(_pretrade_spread_pct(result))
            drift = _format_percent(result.get('price_drift_pct'))
            rr = _format_numeric(result.get('real_rr'))
            stop = _format_percent(result.get('real_stop_distance_pct'))
            lines.append(
                f'  Spread: {_html_escape(spread_pct)} | Drift: {_html_escape(drift)} '
                f'| RR: {_html_escape(rr)} | Stop: {_html_escape(stop)}'
            )
            entry = _format_numeric(result.get('planned_entry'))
            stop_px = _format_numeric(result.get('planned_stop'))
            target = _format_numeric(result.get('planned_target'))
            lines.append(
                f'  Entry: {_html_escape(entry)} | Stop: {_html_escape(stop_px)} '
                f'| Target: {_html_escape(target)}'
            )
            links_line = _build_pretrade_link_line(result)
            if links_line:
                lines.append(f'  Links: {links_line}')
            lines.append('')

    lines.append('REJECTED (ranked by scan)')
    if not rejected:
        lines.append('None.')
    else:
        for result in rejected[:15]:
            symbol = str(result.get('symbol') or '').strip()
            scan_rank = _format_rank(result.get('scan_rank'))
            reason = _truncate_text(str(result.get('reject_reason') or 'Unknown reason'))
            lines.append(
                f'{_html_escape(symbol)} (scan #{_html_escape(scan_rank)}) -> REJECTED'
            )
            lines.append(f'  Reason: {_html_escape(reason)}')
            links_line = _build_pretrade_link_line(result)
            if links_line:
                lines.append(f'  Links: {links_line}')
            lines.append('')

    return '\n'.join(lines).strip()


def _build_mooner_output(base_dir: Path, since: datetime | None) -> str | None:
    path = mooner_output_path(base_dir, 'MoonerCallouts.json')
    if not path.exists():
        return None
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    if since and modified_at < since:
        return None

    payload = _load_json(path)
    if not isinstance(payload, list):
        return None

    callouts = [item for item in payload if isinstance(item, dict)]
    lines = [
        'MOONER CALL-OUTS',
        f'Updated: {_format_timestamp(modified_at)}',
        f'Callouts: {len(callouts)}',
        '',
    ]
    if not callouts:
        lines.append('No callouts recorded.')
        return '\n'.join(lines)

    lines.append('Callouts:')
    for line in format_mooner_callout_lines(callouts):
        lines.append(f'- {line}')
    return '\n'.join(lines).strip()


def _short_ticker_label(ticker: str | None) -> str:
    if not ticker:
        return 'unknown'
    parts = str(ticker).split('_')
    return parts[0] if parts else str(ticker)


def _build_yolo_output(base_dir: Path, since: datetime | None) -> str | None:
    path = yolo_output_path(base_dir, 'YOLO_Pick.json')
    if not path.exists():
        return None
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    if since and modified_at < since:
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    week_of = payload.get('week_of', 'unknown')
    ticker_value = str(payload.get('ticker', 'unknown')).strip()
    short_label = _short_ticker_label(ticker_value)
    price = payload.get('price')
    score = payload.get('yolo_score')
    lines = [
        'YOLO PENNY LOTTERY',
        f"Week of: {_html_escape(week_of)}",
        f"Ticker: {_html_escape(short_label)} ({_html_escape(ticker_value)})",
        f"Price: {_html_escape(_format_numeric(price))}",
        f"Score: {_html_escape(_format_numeric(score))}",
    ]
    stake_value = payload.get('stake_gbp')
    if isinstance(stake_value, (int, float)):
        lines.append(f"Stake: £{stake_value:.2f}")
    else:
        lines.append("Stake: n/a")
    lines.extend(['', 'Rationale:'])
    for line in payload.get('rationale', []):
        lines.append(f"- {line}")
    links_line = _build_yolo_link_line(payload.get('ticker'))
    if links_line:
        lines.extend(['', f'Links: {links_line}'])
    return '\n'.join(lines).strip()


def _format_link_badges(links: list[dict[str, str]]) -> str:
    badges: list[str] = []
    for link in links:
        url = link.get('url')
        label = link.get('label') or 'LINK'
        badges.append(_link_html(url, label=label))
    return ' | '.join(badges)


def _format_news_scout_links(links: list[dict[str, str]]) -> str:
    badges: list[str] = []
    for link in links:
        url = link.get('url')
        label = link.get('label') or 'LINK'
        if not url:
            continue
        badges.append(f'{_html_escape(label)}: {_html_escape(url)}')
    return ' | '.join(badges)


def _build_news_scout_output(base_dir: Path, since: datetime | None) -> str | None:
    directory = news_scout_output_path(base_dir)
    latest = _latest_file(directory, 'news_scout_*.json')
    if not latest:
        return None
    if since:
        modified_at = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
        if modified_at < since:
            return None
    payload = _load_json(latest)
    if not isinstance(payload, dict):
        return None
    entries = payload.get('entries') if isinstance(payload.get('entries'), list) else []
    as_of = payload.get('as_of') or latest.stem.split('_', 2)[-1]
    lines = [
        'NEWS SCOUT SUMMARY',
        f'As of: {_html_escape(as_of)}',
        f'Entries: {len(entries)}',
        '',
    ]
    if not entries:
        lines.append('No entries recorded.')
        return '\n'.join(lines)

    for index, entry in enumerate(entries, start=1):
        symbol = _html_escape(str(entry.get('symbol') or 'UNKNOWN').strip())
        display = entry.get('display_ticker')
        scan_rank = _html_escape(_format_rank(entry.get('scan_rank')))
        lines.append(f'{index}. {symbol} (scan #{scan_rank})')
        entry_price = _html_escape(_format_numeric(entry.get('entry')))
        stop = _html_escape(_format_numeric(entry.get('stop')))
        target = _html_escape(_format_numeric(entry.get('target')))
        lines.append(f'   Entry: {entry_price} | Stop: {stop} | Target: {target}')
        reason = str(entry.get('reason') or '').strip()
        if reason:
            lines.append(f'   Setup: {_html_escape(reason)}')
          link_line = _format_news_scout_links(entry.get('links') or [])
          if link_line:
              lines.append(f'   Links: {link_line}')
          insight = entry.get('llm_insight')
          if insight:
              lines.append(f'   AI: {_html_escape(insight)}')
          if display:
              lines.append(f'   Display ticker: {_html_escape(display)}')
          lines.append('')
    return '\n'.join(lines).strip()


def _build_product_output(
    command_name: str,
    base_dir: Path,
    since: datetime | None,
) -> str | None:
    if command_name == 'scan':
        return _build_scan_output(base_dir, since)
    if command_name == 'pretrade':
        return _build_pretrade_output(base_dir, since)
    if command_name == 'mooner':
        return _build_mooner_output(base_dir, since)
    if command_name == 'yolo':
        return _build_yolo_output(base_dir, since)
    if command_name == 'news_scout':
        return _build_news_scout_output(base_dir, since)
    return None


def _format_missing_product_output(command_name: str, return_code: int, base_dir: Path) -> str:
    state_path = base_dir / 'state' / f'{command_name}_state.json'
    state = _load_json(state_path)
    state_line = ''
    if isinstance(state, dict) and state:
        status = str(state.get('status') or 'unknown')
        outcome = str(state.get('last_outcome') or 'unknown')
        finished = _format_lock_timestamp(state.get('run_finished_at'))
        state_line = f' Last state: status={status}, outcome={outcome}, finished {finished}.'
    if return_code == 0:
        return f'{command_name} finished, but no output artifact was found.{state_line}'
    return (
        f'{command_name} failed (exit code {return_code}); '
        f'no output artifact found.{state_line}'
    )


def _format_elapsed(started_at: datetime) -> str:
    delta = datetime.now(timezone.utc) - started_at
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h{minutes:02d}m'
    if minutes:
        return f'{minutes}m{seconds:02d}s'
    return f'{seconds}s'


def _format_running_jobs(
    running_jobs: dict[str, RunningJob],
    exclude_commands: set[str] | None = None,
) -> str:
    exclude_commands = exclude_commands or set()
    filtered = [
        job for job in running_jobs.values() if job.command_name not in exclude_commands
    ]
    if not filtered:
        return 'Ops running jobs: none'
    lines = ['Ops running jobs:']
    for job in sorted(filtered, key=lambda item: item.started_at):
        started = _format_timestamp(job.started_at)
        elapsed = _format_elapsed(job.started_at)
        pid_text = f', pid {job.process_id}' if job.process_id else ''
        lines.append(
            f'- {job.command_name} (job {job.job_id}, started {started}, elapsed {elapsed}{pid_text})'
        )
    return '\n'.join(lines)


def _format_conflict_message(
    command_name: str,
    conflicts: list[RunningJob],
    lock_conflicts: list[LockConflict],
) -> str:
    if not conflicts and not lock_conflicts:
        return f'"{command_name}" already running.'
    details = []
    for job in conflicts:
        started = _format_timestamp(job.started_at)
        elapsed = _format_elapsed(job.started_at)
        details.append(f'{job.command_name} (job {job.job_id}, started {started}, elapsed {elapsed})')
    for lock in lock_conflicts:
        started = _format_lock_timestamp(lock.started_at)
        details.append(f'lock {lock.name} (started {started})')
    joined = '; '.join(details)
    return f'"{command_name}" already running: {joined}'


def _format_start_message(job: RunningJob) -> str:
    started = _format_timestamp(job.started_at)
    return f'Started {job.command_name} (job {job.job_id}, started {started}).'


def _find_conflicts(
    groups: set[str],
    running_jobs: dict[str, RunningJob],
) -> list[RunningJob]:
    if not groups:
        return []
    conflicts: list[RunningJob] = []
    for job in running_jobs.values():
        if job.groups & groups:
            conflicts.append(job)
    return conflicts


def _lock_path(lock_name: str) -> Path:
    return LOCK_DIR / f'{lock_name}.lock'


def _read_lock_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


def _find_lock_conflicts(groups: set[str]) -> list[LockConflict]:
    lock_names: set[str] = set()
    for group in groups:
        lock_names.update(LOCK_GROUPS.get(group, set()))
    conflicts: list[LockConflict] = []
    for name in sorted(lock_names):
        started_at = _read_lock_timestamp(_lock_path(name))
        if started_at:
            conflicts.append(LockConflict(name=name, started_at=started_at))
    return conflicts


def _build_reply_text(
    command_name: str,
    return_code: int,
    output: str,
    job_id: str | None,
    *,
    force_send: bool = False,
    use_html: bool = False,
) -> tuple[str | None, str | None]:
    if OUTPUT_MODE == 'none':
        return None, None
    if force_send:
        parse_mode = 'HTML' if use_html and output else None
        return output, parse_mode
    if command_name in NO_REPLY_COMMANDS:
        return None, None
    if command_name in SILENT_COMMANDS:
        text = command_summary_text(command_name, return_code, job_id)
        parse_mode = 'HTML' if use_html and text else None
        return text, parse_mode
    if OUTPUT_MODE == 'full':
        parse_mode = 'HTML' if use_html and output else None
        return output, parse_mode
    if OUTPUT_MODE == 'errors':
        if return_code != 0:
            parse_mode = 'HTML' if use_html and output else None
            return output, parse_mode
        text = command_summary_text(command_name, return_code, job_id)
        parse_mode = 'HTML' if use_html and text else None
        return text, parse_mode
    text = command_summary_text(command_name, return_code, job_id)
    parse_mode = 'HTML' if use_html and text else None
    return text, parse_mode


def _env_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_time_token(token: str) -> tuple[int, int, int] | None:
    parts = token.strip().split(":")
    if len(parts) not in {2, 3}:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return hour, minute, second


def _parse_days_token(token: str) -> set[int] | None:
    cleaned = token.strip().lower()
    if not cleaned:
        return None
    if cleaned in {"daily", "everyday"}:
        return set(range(7))
    if cleaned in {"weekday", "weekdays"}:
        return {0, 1, 2, 3, 4}
    if cleaned in {"weekend", "weekends"}:
        return {5, 6}
    if "," in cleaned:
        days: set[int] = set()
        for part in cleaned.split(","):
            part = part.strip()
            if not part:
                continue
            day = DAY_ALIASES.get(part)
            if day is None:
                return None
            days.add(day)
        return days or None
    day = DAY_ALIASES.get(cleaned)
    if day is None:
        return None
    return {day}


def _parse_schedule_value(
    command_name: str,
    value: str | None,
    logger: logging.Logger,
) -> list[ScheduleSpec]:
    if not value:
        return []
    entries = [chunk.strip() for chunk in value.split(";") if chunk.strip()]
    specs: list[ScheduleSpec] = []
    for entry in entries:
        tokens = entry.split()
        if not tokens:
            continue
        if len(tokens) == 1:
            days = set(range(7))
            time_token = tokens[0]
        else:
            days = _parse_days_token(tokens[0])
            time_token = tokens[1]
            if days is None:
                logger.warning("Invalid schedule days for %s: %s", command_name, entry)
                continue
        time_parts = _parse_time_token(time_token)
        if not time_parts:
            logger.warning("Invalid schedule time for %s: %s", command_name, entry)
            continue
        hour, minute, second = time_parts
        specs.append(
            ScheduleSpec(
                command_name=command_name,
                days=days,
                hour=hour,
                minute=minute,
                second=second,
                source=entry,
            )
        )
    return specs


def _schedule_now(use_utc: bool) -> datetime:
    if use_utc:
        return datetime.now(timezone.utc)
    return datetime.now().astimezone()


def _next_run_for_spec(spec: ScheduleSpec, now: datetime) -> datetime:
    for offset in range(0, 8):
        candidate_date = (now + timedelta(days=offset)).date()
        if candidate_date.weekday() not in spec.days:
            continue
        candidate = datetime.combine(
            candidate_date,
            time(spec.hour, spec.minute, spec.second),
            tzinfo=now.tzinfo,
        )
        if candidate > now:
            return candidate
    return now + timedelta(days=1)


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _state_path_for_command(command_name: str) -> Path | None:
    if command_name in {"scan", "universe", "pretrade"}:
        return BASE_DIR / "state" / f"{command_name}_state.json"
    if command_name == "market_data":
        return BASE_DIR / "state" / "market_data_state.json"
    if command_name == "mooner":
        return mooner_state_path()
    return None


def _load_command_last_run(command_name: str) -> datetime | None:
    path = _state_path_for_command(command_name)
    if path is None or not path.exists():
        return None
    if command_name == "mooner":
        try:
            stats = path.stat()
        except OSError:
            return None
        return datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    return _parse_iso_timestamp(payload.get("last_run"))


def _previous_run_for_spec(spec: ScheduleSpec, now: datetime) -> datetime | None:
    for offset in range(0, 8):
        candidate_date = (now - timedelta(days=offset)).date()
        if candidate_date.weekday() not in spec.days:
            continue
        candidate = datetime.combine(
            candidate_date,
            time(spec.hour, spec.minute, spec.second),
            tzinfo=now.tzinfo,
        )
        if candidate <= now:
            return candidate
    return None


def _detect_missed_schedule_runs(
    specs: list[ScheduleSpec],
    now: datetime,
    logger: logging.Logger,
) -> list[ScheduleCatchup]:
    catchups: list[ScheduleCatchup] = []
    for spec in specs:
        due = _previous_run_for_spec(spec, now)
        if due is None:
            continue
        overdue = now - due
        if overdue < timedelta(0) or overdue > SCHEDULE_CATCHUP_WINDOW:
            continue
        last_run = _load_command_last_run(spec.command_name)
        if last_run and last_run.astimezone(now.tzinfo) >= due:
            continue
        catchups.append(ScheduleCatchup(spec=spec, expected_at=due))
        logger.info(
            "Scheduled command %s appears missed (expected %s).",
            spec.command_name,
            _format_timestamp(due),
        )
    return catchups


def _describe_schedule_spec(spec: ScheduleSpec) -> str:
    day_names = [DAY_NAMES.get(day, str(day)) for day in sorted(spec.days)]
    day_text = ",".join(day_names) if day_names else "none"
    time_text = f"{spec.hour:02d}:{spec.minute:02d}"
    if spec.second:
        time_text = f"{time_text}:{spec.second:02d}"
    return f"{spec.command_name}: {day_text} {time_text} ({spec.source})"


def _load_schedule_specs(
    logger: logging.Logger,
) -> tuple[list[ScheduleSpec], int | None, bool, list[ScheduleCatchup]]:
    if not _env_truthy(os.getenv("OPS_SCHEDULE_ENABLED")):
        return [], None, False, []
    chat_value = os.getenv("OPS_SCHEDULE_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not chat_value:
        logger.warning("Schedule enabled but no OPS_SCHEDULE_CHAT_ID/TELEGRAM_CHAT_ID set.")
        return [], None, False, []
    try:
        chat_id = int(chat_value)
    except ValueError:
        logger.warning("Invalid schedule chat id: %s", chat_value)
        return [], None, False, []
    use_utc = _env_truthy(os.getenv("OPS_SCHEDULE_USE_UTC"))
    specs: list[ScheduleSpec] = []
    for command_name, env_name in SCHEDULE_ENV_MAP.items():
        specs.extend(
            _parse_schedule_value(
                command_name,
                os.getenv(env_name),
                logger,
            )
        )
    if not specs:
        logger.info("Schedule enabled but no entries configured.")
        return specs, chat_id, use_utc, []
    now = _schedule_now(use_utc)
    catchups = _detect_missed_schedule_runs(specs, now, logger)
    return specs, chat_id, use_utc, catchups


def _build_scheduled_job(
    command_name: str,
    command: list[str],
    chat_id: int,
) -> RunningJob:
    job_id = f'{command_name}-sched-{next(JOB_COUNTER):04d}'
    return RunningJob(
        job_id=job_id,
        command_name=command_name,
        args=command,
        started_at=datetime.now(timezone.utc),
        chat_id=chat_id,
        user_id=None,
        username="scheduler",
        groups=_command_groups(command_name),
    )


async def _trigger_scheduled_job(
    spec: ScheduleSpec,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    logger = context.application.bot_data["logger"]
    running_jobs = context.application.bot_data["running_jobs"]
    running_lock = context.application.bot_data["running_jobs_lock"]
    schedule_chat_id = context.application.bot_data.get("schedule_chat_id")
    notify_skips = _env_truthy(os.getenv("OPS_SCHEDULE_NOTIFY_SKIPS"))
    send_start = _env_truthy(os.getenv("OPS_SCHEDULE_SEND_START"))

    if schedule_chat_id is None:
        logger.warning("Scheduled job skipped; schedule_chat_id missing.")
        return

    command = COMMAND_MAP.get(spec.command_name)
    if not command:
        logger.warning("Scheduled command not configured: %s", spec.command_name)
        return

    job = _build_scheduled_job(spec.command_name, command, schedule_chat_id)
    lock_conflicts = _find_lock_conflicts(job.groups)
    conflicts: list[RunningJob]
    async with running_lock:
        conflicts = _find_conflicts(job.groups, running_jobs)
        if not conflicts and not lock_conflicts:
            running_jobs[job.job_id] = job
    if conflicts or lock_conflicts:
        message = _format_conflict_message(spec.command_name, conflicts, lock_conflicts)
        logger.warning("Scheduled %s blocked: %s", spec.command_name, message)
        if notify_skips:
            try:
                await context.bot.send_message(chat_id=schedule_chat_id, text=message)
            except Exception as exc:  # noqa: BLE001 - log and continue
                logger.error("Failed to send schedule skip message: %s", exc)
        return

    logger.info("Scheduled command started: %s (job %s)", spec.command_name, job.job_id)
    if send_start and OUTPUT_MODE != "none":
        try:
            await context.bot.send_message(chat_id=schedule_chat_id, text=_format_start_message(job))
        except Exception as exc:  # noqa: BLE001 - log and continue
            logger.error("Failed to send schedule start message: %s", exc)

    context.application.create_task(_execute_command(job, context))


async def _schedule_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    schedule_specs: list[ScheduleSpec] = context.application.bot_data.get("schedule_specs", [])
    if not schedule_specs:
        return
    use_utc = bool(context.application.bot_data.get("schedule_use_utc", False))
    now = _schedule_now(use_utc)
    for spec in schedule_specs:
        if spec.next_run is None:
            spec.next_run = _next_run_for_spec(spec, now)
        if spec.next_run and spec.next_run <= now:
            await _trigger_scheduled_job(spec, context)
            spec.next_run = _next_run_for_spec(spec, now + timedelta(seconds=1))


async def _run_startup_schedule_catchups(context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data.get("logger")
    if not logger:
        return
    catchups: list[ScheduleCatchup] = context.application.bot_data.get("startup_schedule_catchups", [])
    if not catchups:
        return
    for catchup in catchups:
        logger.info(
            "Running missed scheduled command %s expected %s",
            catchup.spec.command_name,
            _format_timestamp(catchup.expected_at),
        )
        await _trigger_scheduled_job(catchup.spec, context)
    context.application.bot_data.pop("startup_schedule_catchups", None)


async def _execute_command(
    job: RunningJob,
    context: ContextTypes.DEFAULT_TYPE,
    prefix_text: str | None = None,
    include_running_jobs: bool = False,
) -> None:
    logger = context.application.bot_data["logger"]
    running_jobs = context.application.bot_data["running_jobs"]
    running_lock = context.application.bot_data["running_jobs_lock"]

    env = None
    if job.command_name == 'pretrade' and job.chat_id is not None:
        env = dict(os.environ)
        env['TELEGRAM_CHAT_ID'] = str(job.chat_id)

    try:
        process = await asyncio.create_subprocess_exec(
            *job.args,
            cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 - log and notify without crashing
        logger.error(
            "Failed to start command: job_id=%s command=%s error=%s",
            job.job_id,
            job.command_name,
            exc,
        )
        reply_text = (
            f'Job {job.job_id}: failed to start command "{job.command_name}".'
        )
        if job.command_name not in NO_REPLY_COMMANDS and OUTPUT_MODE != 'none':
            try:
                await context.bot.send_message(chat_id=job.chat_id, text=reply_text)
            except Exception as send_exc:  # noqa: BLE001 - log and continue
                logger.error(
                    'Failed to send Telegram reply to chat %s for job %s: %s',
                    job.chat_id,
                    job.job_id,
                    send_exc,
                )
        async with running_lock:
            running_jobs.pop(job.job_id, None)
        return

    async with running_lock:
        job.process_id = process.pid

    logger.info(
        "Command started: job_id=%s user_id=%s username=%s chat_id=%s command=%s pid=%s",
        job.job_id,
        job.user_id,
        job.username,
        job.chat_id,
        job.command_name,
        job.process_id,
    )

    stdout_bytes, stderr_bytes = await process.communicate()
    stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    result = subprocess.CompletedProcess(
        job.args,
        process.returncode or 0,
        stdout_text,
        stderr_text,
    )
    output = command_output_to_text(result)
    if job.command_name in PRODUCT_OUTPUT_COMMANDS:
        product_output = _build_product_output(job.command_name, BASE_DIR, job.started_at)
        if product_output:
            output = truncate_output(product_output)
        else:
            output = _format_missing_product_output(
                job.command_name,
                process.returncode or 0,
                BASE_DIR,
            )
    if include_running_jobs:
        async with running_lock:
            prefix_text = _format_running_jobs(running_jobs, exclude_commands={'status'})
    if prefix_text and job.command_name not in PRODUCT_OUTPUT_COMMANDS:
        output = f"{prefix_text}\n\n{output}"

    logger.info(
        "Command result: job_id=%s user_id=%s username=%s chat_id=%s command=%s "
        "returncode=%s output=%s",
        job.job_id,
        job.user_id,
        job.username,
        job.chat_id,
        job.command_name,
        process.returncode,
        output.replace("\n", "\\n"),
    )

    reply_text, parse_mode = _build_reply_text(
        job.command_name,
        process.returncode or 0,
        output,
        job.job_id,
        force_send=job.command_name in PRODUCT_OUTPUT_COMMANDS,
        use_html=job.command_name in PRODUCT_OUTPUT_COMMANDS,
    )
    if job.command_name == 'pretrade' and process.returncode == 0:
        reply_text = None
    if reply_text and job.chat_id is not None:
        try:
            await context.bot.send_message(
                chat_id=job.chat_id,
                text=reply_text,
                parse_mode=parse_mode,
            )
        except Exception as send_exc:  # noqa: BLE001 - log and continue
            logger.error(
                'Failed to send Telegram reply to chat %s for job %s: %s',
                job.chat_id,
                job.job_id,
                send_exc,
            )

    async with running_lock:
        running_jobs.pop(job.job_id, None)


async def send_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    authorized_users = context.application.bot_data.get('authorized_users', set())
    if not user_is_authorized(update, logger, authorized_users):
        await update.message.reply_text("Unauthorized user.")
        return
    await update.message.reply_text(command_description_lines(COMMAND_HELP.keys()))


async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if message is None:
        logger.info('Update received: %s', _format_update_context(update))
        return

    logger.info(
        "Message received: update_id=%s user_id=%s username=%s chat_id=%s chat_type=%s chat_title=%s text=%s",
        update.update_id,
        user.id if user else None,
        user.username if user else None,
        chat.id if chat else None,
        chat.type if chat else None,
        chat.title or chat.username if chat else None,
        _truncate_log_text(message.text or message.caption),
    )

    new_members = message.new_chat_members or []
    for member in new_members:
        logger.info(
            "Chat member added: chat_id=%s chat_title=%s member_id=%s member_username=%s member_name=%s",
            chat.id if chat else None,
            chat.title or chat.username if chat else None,
            member.id,
            member.username,
            member.full_name,
        )

    if message.left_chat_member:
        member = message.left_chat_member
        logger.info(
            "Chat member left: chat_id=%s chat_title=%s member_id=%s member_username=%s member_name=%s",
            chat.id if chat else None,
            chat.title or chat.username if chat else None,
            member.id,
            member.username,
            member.full_name,
        )


async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    authorized_users = context.application.bot_data.get('authorized_users', set())
    if not user_is_authorized(update, logger, authorized_users):
        await update.message.reply_text("Unauthorized user.")
        return

    text = (update.message.text or "").lstrip("/")
    parts = text.split()
    if not parts:
        await update.message.reply_text("Please specify a command.")
        return
    command_name = parts[0]
    command_name = COMMAND_ALIASES.get(command_name, command_name)
    command = COMMAND_MAP.get(command_name)
    if not command:
        await update.message.reply_text("Unknown command.")
        logger.info(
            "Unknown command: user_id=%s username=%s chat_id=%s command=%s",
            update.effective_user.id,
            update.effective_user.username,
            update.effective_chat.id if update.effective_chat else None,
            command_name,
        )
        return
    running_jobs = context.application.bot_data["running_jobs"]
    running_lock = context.application.bot_data["running_jobs_lock"]

    command_args = list(command)
    if command_name == 'status' and len(parts) > 1 and parts[1].lower() == 'verbose':
        command_args.append('--verbose')
    job = _build_job(command_name, command_args, update)
    lock_conflicts = _find_lock_conflicts(job.groups)
    conflicts: list[RunningJob]
    async with running_lock:
        conflicts = _find_conflicts(job.groups, running_jobs)
        if not conflicts and not lock_conflicts:
            running_jobs[job.job_id] = job
    if conflicts or lock_conflicts:
        reply_text = _format_conflict_message(command_name, conflicts, lock_conflicts)
        await update.message.reply_text(reply_text)
        logger.info(
            "Command blocked: user_id=%s username=%s chat_id=%s command=%s conflicts=%s lock_conflicts=%s",
            update.effective_user.id,
            update.effective_user.username,
            update.effective_chat.id if update.effective_chat else None,
            command_name,
            ','.join(job_item.job_id for job_item in conflicts),
            ','.join(lock.name for lock in lock_conflicts),
        )
        return

    if command_name not in NO_REPLY_COMMANDS and OUTPUT_MODE != 'none':
        await update.message.reply_text(_format_start_message(job))

    include_running_jobs = command_name == 'status'
    asyncio.create_task(
        _execute_command(job, context, include_running_jobs=include_running_jobs)
    )


async def handle_yolo_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    authorized_users = context.application.bot_data.get('authorized_users', set())
    if not user_is_authorized(update, logger, authorized_users):
        await update.message.reply_text("Unauthorized user.")
        return
    message = _format_yolo_history_message(BASE_DIR)
    try:
        await update.message.reply_text(message)
    except Exception as exc:  # noqa: BLE001 - log but don’t crash
        logger.error("Failed to send YOLO history: %s", exc)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data["logger"]
    authorized_users = context.application.bot_data.get('authorized_users', set())
    if not user_is_authorized(update, logger, authorized_users):
        await update.message.reply_text("Unauthorized user.")
        return
    await update.message.reply_text("Unknown command.")
    logger.info(
        "Unknown command: user_id=%s username=%s chat_id=%s text=%s",
        update.effective_user.id,
        update.effective_user.username,
        update.effective_chat.id if update.effective_chat else None,
        _truncate_log_text(update.message.text),
    )


async def handle_error(update: Update | None, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger = context.application.bot_data.get("logger")
    if not logger:
        return
    logger.error("Telegram handler error: %s | %s", context.error, _format_update_context(update))


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(f"TELEGRAM_BOT_TOKEN is not set (env: {ENV_PATH})")

    logger = setup_logging()
    authorized_users = _parse_authorized_users(os.getenv(AUTHORIZED_USERS_ENV), logger)
    if not authorized_users:
        logger.warning('No authorized users configured via %s.', AUTHORIZED_USERS_ENV)
    logger.info(
        'Ops bot config: authorized_users=%s output_mode=%s silent_commands=%s no_reply_commands=%s',
        sorted(authorized_users),
        OUTPUT_MODE,
        sorted(SILENT_COMMANDS),
        sorted(NO_REPLY_COMMANDS),
    )

    application = ApplicationBuilder().token(token).build()
    application.bot_data["logger"] = logger
    application.bot_data["authorized_users"] = authorized_users
    application.bot_data["running_jobs"] = {}
    application.bot_data["running_jobs_lock"] = asyncio.Lock()

    application.add_handler(CommandHandler("help", send_help))
    application.add_handler(CommandHandler("scan", handle_command))
    application.add_handler(CommandHandler("universe", handle_command))
    application.add_handler(CommandHandler("pretrade", handle_command))
    application.add_handler(CommandHandler("status", handle_command))
    application.add_handler(CommandHandler("market_data", handle_command))
    application.add_handler(CommandHandler("mooner", handle_command))
    application.add_handler(CommandHandler("yolo", handle_command))
    application.add_handler(CommandHandler("news_scout", handle_command))
    application.add_handler(CommandHandler("yolo_history", handle_yolo_history))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(filters.ALL, log_update), group=1)
    application.add_error_handler(handle_error)

    (
        schedule_specs,
        schedule_chat_id,
        schedule_use_utc,
        schedule_catchups,
    ) = _load_schedule_specs(logger)
    if schedule_specs and schedule_chat_id is not None:
        application.bot_data["schedule_specs"] = schedule_specs
        application.bot_data["schedule_chat_id"] = schedule_chat_id
        application.bot_data["schedule_use_utc"] = schedule_use_utc
        if schedule_catchups:
            application.bot_data["startup_schedule_catchups"] = schedule_catchups
        for spec in schedule_specs:
            logger.info("Scheduled job configured: %s", _describe_schedule_spec(spec))
        if application.job_queue:
            application.job_queue.run_repeating(
                _schedule_tick,
                interval=30,
                first=5,
                name="ops_schedule",
            )
            if schedule_catchups:
                application.job_queue.run_once(
                    _run_startup_schedule_catchups,
                    when=5,
                    name="ops_schedule_catchup",
                )

    logger.info("Telegram command client started.")
    try:
        application.run_polling()
    except Exception as exc:  # noqa: BLE001 - ensure crash reason is logged
        logger.error('Telegram command client stopped: %s', exc)
        raise


def _parse_authorized_users(value: str | None, logger: logging.Logger) -> set[int]:
    if not value:
        return set()
    users: set[int] = set()
    for chunk in value.replace(';', ',').split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            users.add(int(chunk))
        except ValueError:
            logger.warning('Skipping invalid user id in %s: %s', AUTHORIZED_USERS_ENV, chunk)
    return users


if __name__ == "__main__":
    main()
