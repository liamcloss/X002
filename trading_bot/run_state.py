"""Run-state tracking for long-running commands."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunHandle:
    name: str
    state_path: Path
    lock_path: Path
    started_at: datetime
    acquired: bool


def start_run(base_dir: Path, name: str, logger: logging.Logger) -> RunHandle:
    """Mark a command run as in-progress and create a lock file."""

    state_path = base_dir / 'state' / f'{name}_state.json'
    lock_path = base_dir / 'state' / f'{name}.lock'
    started_at = datetime.now(timezone.utc)
    existing = _read_lock(lock_path)
    if existing:
        logger.warning('%s already running (lock started %s).', name, existing)
        return RunHandle(
            name=name,
            state_path=state_path,
            lock_path=lock_path,
            started_at=started_at,
            acquired=False,
        )

    _write_lock(lock_path, started_at)
    state = _load_state(state_path)
    state['status'] = 'running'
    state['run_started_at'] = started_at.isoformat()
    state['run_finished_at'] = None
    state['last_attempt_at'] = started_at.isoformat()
    state['last_outcome'] = None
    _save_state(state_path, state)
    return RunHandle(
        name=name,
        state_path=state_path,
        lock_path=lock_path,
        started_at=started_at,
        acquired=True,
    )


def finish_run(
    handle: RunHandle,
    logger: logging.Logger,
    *,
    failed: bool,
    completed: bool,
) -> None:
    """Finalize run state and remove the lock file."""

    if not handle.acquired:
        return
    finished_at = datetime.now(timezone.utc)
    duration = (finished_at - handle.started_at).total_seconds()
    outcome = _resolve_outcome(failed, completed)

    state = _load_state(handle.state_path)
    state['status'] = 'failed' if failed else 'idle'
    state['run_started_at'] = handle.started_at.isoformat()
    state['run_finished_at'] = finished_at.isoformat()
    state['last_attempt_at'] = handle.started_at.isoformat()
    state['last_outcome'] = outcome
    if completed and not failed:
        state['last_run'] = finished_at.isoformat()
        state['last_duration_seconds'] = round(duration, 1)
    _save_state(handle.state_path, state)
    _clear_lock(handle.lock_path, logger)


def _resolve_outcome(failed: bool, completed: bool) -> str:
    if failed:
        return 'failed'
    if completed:
        return 'completed'
    return 'skipped'


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _read_lock(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


def _write_lock(path: Path, started_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(started_at.isoformat(), encoding='utf-8')


def _clear_lock(path: Path, logger: logging.Logger) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError as exc:
        logger.warning('Failed to clear %s lock: %s', path.name, exc)


__all__ = [
    'RunHandle',
    'start_run',
    'finish_run',
]
