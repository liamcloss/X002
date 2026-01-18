"""
This bot is an operational control surface for Speculation Edge. It enables manual,
remote, auditable execution of trading system commands via Telegram. It is not an
automated trading agent.
"""

import asyncio
import itertools
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Set

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
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
}

COMMAND_HELP = {
    "scan": "Run the scan pipeline.",
    "universe": "Refresh the trading universe.",
    "pretrade": "Run pretrade checks.",
    "status": "Show system status.",
    "help": "Show this help message.",
}

LOG_PATH = BASE_DIR / 'logs' / 'telegram_command_client.log'
MAX_OUTPUT_CHARS = 3500
MAX_MESSAGE_LOG_CHARS = 200
JOB_COUNTER = itertools.count(1)
PRODUCT_OUTPUT_COMMANDS = {'scan', 'pretrade'}

COMMAND_GROUPS = {
    "scan": {"scan", "ideas", "universe", "market_data"},
    "universe": {"universe", "market_data"},
    "pretrade": {"ideas"},
    "status": set(),
}
LOCK_GROUPS = {
    "scan": {"scan", "market_data"},
    "ideas": {"pretrade"},
    "universe": {"universe"},
    "market_data": {"market_data"},
}
LOCK_DIR = BASE_DIR / 'state'


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
    path = base_dir / 'SetupCandidates.json'
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
        f'Generated at: {generated_at}',
        f'Data as of: {data_as_of}',
        f'Mode: {mode}',
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
        currency = str(candidate.get('currency_code') or '').strip().upper()
        entry_text = f'{entry} {currency}'.strip()
        lines.append(
            f'{rank}. {symbol} | Entry {entry_text} | Stop {stop} | Target {target} | RR {rr}'
        )
        reason = _truncate_text(str(candidate.get('reason') or '').strip())
        volume = _format_numeric(candidate.get('volume_multiple'))
        momentum = _format_ratio_percent(candidate.get('momentum_5d'))
        if reason or volume != 'n/a' or momentum != 'n/a':
            lines.append(f'   Reason: {reason}, volume {volume}x, momentum {momentum}')
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


def _build_pretrade_output(base_dir: Path, since: datetime | None) -> str | None:
    outputs_dir = base_dir / 'outputs'
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
    executable_count = sum(1 for item in results if item.get('status') == 'EXECUTABLE')
    rejected_count = sum(1 for item in results if item.get('status') != 'EXECUTABLE')

    lines = [
        'PRETRADE OUTPUT',
        f'Checked at: {checked_at_text}',
        f'Executable: {executable_count} | Rejected: {rejected_count}',
        '',
    ]
    if not results:
        lines.append('No setups evaluated.')
        return '\n'.join(lines)

    display_limit = min(15, len(results))
    for result in results[:display_limit]:
        symbol = str(result.get('symbol') or '').strip()
        status = str(result.get('status') or 'REJECTED')
        lines.append(f'{symbol} -> {status}')
        if status == 'EXECUTABLE':
            spread_pct = _format_percent(_pretrade_spread_pct(result))
            drift = _format_percent(result.get('price_drift_pct'))
            rr = _format_numeric(result.get('real_rr'))
            stop = _format_percent(result.get('real_stop_distance_pct'))
            lines.append(f'  Spread: {spread_pct} | Drift: {drift} | RR: {rr} | Stop: {stop}')
            entry = _format_numeric(result.get('planned_entry'))
            stop_px = _format_numeric(result.get('planned_stop'))
            target = _format_numeric(result.get('planned_target'))
            lines.append(f'  Entry: {entry} | Stop: {stop_px} | Target: {target}')
        else:
            reason = _truncate_text(str(result.get('reject_reason') or 'Unknown reason'))
            lines.append(f'  Reason: {reason}')
        lines.append('')
    if len(results) > display_limit:
        lines.append(f'...showing {display_limit} of {len(results)} results')
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
    return None


def _format_missing_product_output(command_name: str, return_code: int) -> str:
    if return_code == 0:
        return (
            f'{command_name} finished, but no output artifact was found.'
        )
    return f'{command_name} failed (exit code {return_code}); no output artifact found.'


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
) -> str | None:
    if OUTPUT_MODE == 'none':
        return None
    if force_send:
        return output
    if command_name in NO_REPLY_COMMANDS:
        return None
    if command_name in SILENT_COMMANDS:
        return command_summary_text(command_name, return_code, job_id)
    if OUTPUT_MODE == 'full':
        return output
    if OUTPUT_MODE == 'errors':
        if return_code != 0:
            return output
        return command_summary_text(command_name, return_code, job_id)
    return command_summary_text(command_name, return_code, job_id)


async def _execute_command(
    job: RunningJob,
    context: ContextTypes.DEFAULT_TYPE,
    prefix_text: str | None = None,
    include_running_jobs: bool = False,
) -> None:
    logger = context.application.bot_data["logger"]
    running_jobs = context.application.bot_data["running_jobs"]
    running_lock = context.application.bot_data["running_jobs_lock"]

    try:
        process = await asyncio.create_subprocess_exec(
            *job.args,
            cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
                logger.error('Failed to send Telegram reply: %s', send_exc)
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
            output = _format_missing_product_output(job.command_name, process.returncode or 0)
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

    reply_text = _build_reply_text(
        job.command_name,
        process.returncode or 0,
        output,
        job.job_id,
        force_send=job.command_name in PRODUCT_OUTPUT_COMMANDS,
    )
    if reply_text and job.chat_id is not None:
        try:
            await context.bot.send_message(chat_id=job.chat_id, text=reply_text)
        except Exception as send_exc:  # noqa: BLE001 - log and continue
            logger.error('Failed to send Telegram reply: %s', send_exc)

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

    command_name = (update.message.text or "").lstrip("/").split()[0]
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

    job = _build_job(command_name, command, update)
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
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(filters.ALL, log_update), group=1)
    application.add_error_handler(handle_error)

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
