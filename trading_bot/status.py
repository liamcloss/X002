"""System status reporting for Speculation Edge."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_bot import config
from trading_bot.execution.open_trades import get_live_open_trade_count
from trading_bot.paper import get_open_trade_count as get_paper_open_trade_count
from trading_bot.paths import (
    latest_setup_candidates_path,
    pretrade_spread_path,
    pretrade_viability_path,
)

REQUIRED_DIRS = (
    'data',
    'logs',
    'outputs',
    'state',
    'universe',
)


def run_status(base_dir: Path, logger: logging.Logger | None = None, verbose: bool = False) -> None:
    logger = logger or logging.getLogger('trading_bot')
    status = _collect_status(base_dir, logger)
    formatter = _format_status_verbose if verbose else _format_status_summary
    print(formatter(status))


def _format_status_summary(status: dict[str, Any]) -> str:
    timestamp = _format_timestamp(status.get("timestamp"))
    mode = status.get("mode", "unknown")
    directories = status.get("dirs", {})
    total_dirs = len(directories)
    ok_dirs = sum(1 for exists in directories.values() if exists)
    dir_summary = _directory_summary(ok_dirs, total_dirs)

    market_state = status.get('market_data_state', {})
    market_lock = status.get('market_data_lock', {})
    market_status = _market_status_summary(market_state, market_lock)

    command_entries = _command_summary_entries(status)

    pretrade_file = status.get('latest_pretrade')
    pretrade_icon = _artifact_icon(bool(pretrade_file))
    spread_file = status.get('latest_spread_report')
    spread_icon = _artifact_icon(bool(spread_file))

    spread_samples = status.get('spread_samples', {})
    lookback_days = spread_samples.get('lookback_days') or _spread_sampling_lookback()

    open_trades = status.get('open_trades', {})
    live_open = open_trades.get('live')
    paper_open = open_trades.get('paper')

    lines = [
        'SPECULATION EDGE STATUS (SUMMARY)',
        f'🕒 {timestamp} | ⚙️ Mode: {mode}',
        dir_summary,
        f'Market data: {market_status}',
    ]
    if command_entries:
        lines.append('Commands:')
        lines.extend(f'  {entry}' for entry in command_entries)
    lines.extend(
        [
            _format_open_trades(live_open, paper_open),
            f'Latest pretrade: {pretrade_icon} | Latest spread: {spread_icon}',
            f'📊 Spread samples: {spread_samples.get("count", 0)} (lookback {lookback_days}d)',
        ]
    )
    return '\n'.join(lines)


def _collect_status(base_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    status: dict[str, Any] = {
        'timestamp': now.isoformat(),
        'mode': config.CONFIG["mode"] or config.CONFIG.get('mode', 'unknown'),
        'dirs': {name: (base_dir / name).exists() for name in REQUIRED_DIRS},
    }

    status['universe_file'] = _file_info(base_dir / 'universe' / 'clean' / 'universe.parquet')
    status['prices_cache'] = {
        'path': str(base_dir / 'data' / 'prices'),
        'count': _count_files(base_dir / 'data' / 'prices', '*.parquet'),
    }

    setup_path = latest_setup_candidates_path(base_dir)
    if setup_path is None:
        status['setup_candidates'] = {'path': 'n/a', 'exists': False}
    else:
        status['setup_candidates'] = _file_info(setup_path)
        if setup_path.exists():
            status['setup_candidates']['count'] = _count_candidates(setup_path, logger)

    market_state = _safe_load_json(base_dir / 'state' / 'market_data_state.json')
    status['market_data_state'] = {
        'status': market_state.get('status'),
        'last_run': market_state.get('last_run'),
        'run_started_at': market_state.get('run_started_at'),
        'run_finished_at': market_state.get('run_finished_at'),
        'last_duration_seconds': market_state.get('last_duration_seconds'),
    }
    status['market_data_lock'] = _lock_info(base_dir / 'state' / 'market_data.lock')
    status['scan_state'] = _safe_load_json(base_dir / 'state' / 'scan_state.json')
    status['scan_lock'] = _lock_info(base_dir / 'state' / 'scan.lock')
    status['universe_state'] = _safe_load_json(base_dir / 'state' / 'universe_state.json')
    status['universe_lock'] = _lock_info(base_dir / 'state' / 'universe.lock')
    status['pretrade_state'] = _safe_load_json(base_dir / 'state' / 'pretrade_state.json')
    status['pretrade_lock'] = _lock_info(base_dir / 'state' / 'pretrade.lock')

    status['latest_pretrade'] = _latest_file(pretrade_viability_path(base_dir), 'pretrade_viability_*.json')
    status['latest_spread_report'] = _latest_file(pretrade_spread_path(base_dir), 'spread_report_*.json')

    spread_samples = _safe_load_json(base_dir / 'state' / 'spread_samples.json')
    status['spread_samples'] = _count_spread_samples(spread_samples)

    status['open_trades'] = {}
    try:
        status['open_trades']['live'] = get_live_open_trade_count(logger=logger)
    except Exception as exc:  # noqa: BLE001 - status should never crash
        logger.warning('Failed to load live open trades count: %s', exc)
        status['open_trades']['live'] = None
    try:
        status['open_trades']['paper'] = get_paper_open_trade_count()
    except Exception as exc:  # noqa: BLE001 - status should never crash
        logger.warning('Failed to load paper open trades count: %s', exc)
        status['open_trades']['paper'] = None

    return status


def _format_status_verbose(status: dict[str, Any]) -> str:
    show_paths = _show_paths()
    directories = _format_directories(status.get('dirs', {}))
    lines = [
        'SPECULATION EDGE STATUS',
        f'🕒 {_format_timestamp(status.get("timestamp"))} | ⚙️ Mode: {status.get("mode", "unknown")}',
        '',
        'Directory health:',
        *directories,
    ]

    lines.append('')
    lines.append(_format_file('Universe file', status.get('universe_file'), show_paths=show_paths))
    prices_cache = status.get('prices_cache', {})
    prices_path = prices_cache.get('path', '')
    prices_label = f'Prices cache: {prices_cache.get("count", 0)} parquet files'
    if show_paths and prices_path:
        prices_label = f'{prices_label} ({prices_path})'
    lines.append(prices_label)

    setup_info = status.get('setup_candidates')
    if setup_info:
        count = setup_info.get('count')
        count_text = f', candidates={count}' if count is not None else ''
        lines.append(
            _format_file(
                f'SetupCandidates.json{count_text}',
                setup_info,
                show_paths=show_paths,
            )
        )

    market_state = status.get('market_data_state', {})
    lock_info = status.get('market_data_lock', {})
    lines.append('')
    lines.append('Market data:')
    lines.append(_format_market_status(market_state, lock_info))
    lines.append(_format_market_lock(lock_info))

    lines.append('')
    lines.append('Command status:')
    lines.append(_format_command_status('Scan', status.get('scan_state'), status.get('scan_lock')))
    lines.append(_format_command_status('Universe', status.get('universe_state'), status.get('universe_lock')))
    lines.append(_format_command_status('Pretrade', status.get('pretrade_state'), status.get('pretrade_lock')))
    lines.append('')
    lines.append('Command locks:')
    lines.append(_format_command_lock('Scan', status.get('scan_lock')))
    lines.append(_format_command_lock('Universe', status.get('universe_lock')))
    lines.append(_format_command_lock('Pretrade', status.get('pretrade_lock')))

    lines.append('')
    lines.append(_format_file('Latest pretrade report', status.get('latest_pretrade'), show_paths=show_paths))
    lines.append(_format_file('Latest spread report', status.get('latest_spread_report'), show_paths=show_paths))

    spread_samples = status.get('spread_samples', {})
    lookback_days = spread_samples.get('lookback_days')
    if lookback_days is None:
        lookback_days = _spread_sampling_lookback()
    lines.append(
        f'📊 Spread samples: {spread_samples.get("count", 0)} (lookback {lookback_days}d)'
    )

    open_trades = status.get('open_trades', {})
    live_open = open_trades.get('live')
    paper_open = open_trades.get('paper')
    lines.append('')
    lines.append(_format_open_trades(live_open, paper_open))

    return '\n'.join(lines)


def _format_file(label: str, info: dict[str, Any] | None, show_paths: bool = False) -> str:
    if not info:
        return f'{label}: missing'
    if not info.get('exists', False):
        return f'{label}: missing'
    timestamp = _format_timestamp(info.get('modified_at'))
    path_value = str(info.get('path', ''))
    display_path = _format_path(path_value, show_paths)
    return f'{label}: {display_path} (updated {timestamp})'


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'path': str(path), 'exists': False}
    stat = path.stat()
    return {
        'path': str(path),
        'exists': True,
        'modified_at': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        'size_bytes': stat.st_size,
    }


def _lock_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'exists': False}
    try:
        started_at = path.read_text(encoding='utf-8').strip() or None
    except OSError:
        started_at = None
    return {
        'exists': True,
        'started_at': started_at,
    }


def _format_timestamp(value: str | None) -> str:
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
    parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime('%Y-%m-%d %H:%M UTC')


def _format_market_status(
    market_state: dict[str, Any],
    lock_info: dict[str, Any],
) -> str:
    status = str(market_state.get('status') or '').lower()
    if status not in {'running', 'idle', 'failed'}:
        status = 'running' if lock_info.get('exists') else 'idle'
    if status == 'running':
        started = market_state.get('run_started_at') or lock_info.get('started_at')
        return f'Market data status: running (started {_format_timestamp(started)})'
    if status == 'failed':
        finished = market_state.get('run_finished_at') or market_state.get('last_run')
        return f'Market data status: failed (last finish {_format_timestamp(finished)})'
    last_run = _format_timestamp(market_state.get('last_run'))
    duration = market_state.get('last_duration_seconds')
    duration_text = ''
    if isinstance(duration, (int, float)):
        duration_text = f', duration {duration:.1f}s'
    return f'Market data status: idle (last run {last_run}{duration_text})'


def _format_market_lock(lock_info: dict[str, Any]) -> str:
    if not lock_info.get('exists'):
        return 'Market data lock: none'
    started_at = _format_timestamp(lock_info.get('started_at'))
    return f'Market data lock: present (started {started_at})'


def _format_command_status(
    label: str,
    state: dict[str, Any] | None,
    lock_info: dict[str, Any] | None,
) -> str:
    state = state or {}
    lock_info = lock_info or {}
    status = _command_status_value(state, lock_info)
    if status == 'running':
        started = state.get('run_started_at') or lock_info.get('started_at')
        return f'{_command_icon(status)} {label} status: running (started {_format_timestamp(started)})'
    if status == 'failed':
        finished = state.get('run_finished_at') or state.get('last_attempt_at')
        return f'{_command_icon(status)} {label} status: failed (last finish {_format_timestamp(finished)})'

    last_run = _format_timestamp(state.get('last_run'))
    duration = state.get('last_duration_seconds')
    duration_text = f', duration {duration:.1f}s' if isinstance(duration, (int, float)) else ''
    outcome = str(state.get('last_outcome') or '').lower()
    if outcome == 'skipped':
        last_attempt = _format_timestamp(state.get('last_attempt_at'))
        return (
            f'{_command_icon("idle")} {label} status: idle (last run {last_run}, '
            f'last attempt {last_attempt}, outcome skipped)'
        )
    return f'{_command_icon("idle")} {label} status: idle (last run {last_run}{duration_text})'


def _format_command_lock(label: str, lock_info: dict[str, Any] | None) -> str:
    lock_info = lock_info or {}
    if not lock_info.get('exists'):
        return f'{_lock_icon(False)} {label} lock: none'
    started_at = _format_timestamp(lock_info.get('started_at'))
    return f'{_lock_icon(True)} {label} lock: present (started {started_at})'


def _format_directories(dirs: dict[str, bool]) -> list[str]:
    if not dirs:
        return ['  ⚠️ directories unknown']
    entries: list[str] = []
    for name, exists in sorted(dirs.items()):
        entries.append(f'  {_status_icon(exists)} {name}')
    return entries


def _format_open_trades(live: int | None, paper: int | None) -> str:
    live_text = str(live) if live is not None else 'unknown'
    paper_text = str(paper) if paper is not None else 'unknown'
    live_icon = _status_icon(bool(live))
    paper_icon = _status_icon(bool(paper))
    return f'Open trades: live {live_icon} {live_text} | paper {paper_icon} {paper_text}'


def _status_icon(ok: bool) -> str:
    return '✅' if ok else '⚠️'


def _directory_summary(ok_dirs: int, total_dirs: int) -> str:
    if total_dirs == 0:
        return 'Directories: unknown'
    icon = '✅' if ok_dirs == total_dirs else '⚠️'
    return f'Directories: {icon} {ok_dirs}/{total_dirs} OK'


def _command_summary_entries(status: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    items = (
        ('Scan', status.get('scan_state'), status.get('scan_lock')),
        ('Universe', status.get('universe_state'), status.get('universe_lock')),
        ('Pretrade', status.get('pretrade_state'), status.get('pretrade_lock')),
    )
    for label, cmd_state, cmd_lock in items:
        cmd_state = cmd_state or {}
        cmd_lock = cmd_lock or {}
        state_label = _command_status_value(cmd_state, cmd_lock)
        entries.append(f'{label} {_command_icon(state_label)} {state_label}')
    return entries


def _command_status_value(state: dict[str, Any], lock_info: dict[str, Any]) -> str:
    status = str(state.get('status') or '').lower()
    if status not in {'running', 'idle', 'failed'}:
        status = 'running' if lock_info.get('exists') else 'idle'
    return status


def _market_status_summary(market_state: dict[str, Any], lock_info: dict[str, Any]) -> str:
    status = _command_status_value(market_state, lock_info)
    return f'{_command_icon(status)} {status}'


def _artifact_icon(present: bool) -> str:
    return '✅' if present else '⚠️'


def _command_icon(status: str) -> str:
    if status == 'running':
        return '⏳'
    if status == 'failed':
        return '❌'
    return '✅'


def _lock_icon(active: bool) -> str:
    return '🔒' if active else '🟢'


def _format_path(value: str, show_paths: bool) -> str:
    if not value:
        return ''
    if show_paths:
        return value
    return Path(value).name


def _show_paths() -> bool:
    output_mode = os.getenv('OPS_OUTPUT_MODE', '').strip().lower()
    return output_mode == 'full'


def _latest_file(base_dir: Path, pattern: str) -> dict[str, Any] | None:
    if not base_dir.exists():
        return None
    files = list(base_dir.glob(pattern))
    if not files:
        return None
    latest = max(files, key=lambda path: path.stat().st_mtime)
    return _file_info(latest)


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _count_candidates(path: Path, logger: logging.Logger) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning('Failed to read SetupCandidates.json: %s', exc)
        return None
    candidates = _extract_candidates(payload)
    return len(candidates)


def _extract_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in {'candidates', 'setups', 'ideas', 'setupcandidates', 'setup_candidates'}:
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
    return []


def _count_spread_samples(payload: dict[str, Any]) -> dict[str, Any]:
    samples = payload.get('samples', {})
    count = 0
    if isinstance(samples, dict):
        for entries in samples.values():
            if isinstance(entries, list):
                count += len(entries)
    return {
        'count': count,
        'lookback_days': payload.get('lookback_days', config.CONFIG["spread_sampling"]["lookback_days"]),
    }


__all__ = [
    'run_status',
]
