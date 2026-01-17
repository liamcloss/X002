"""Spread sampling and reporting for pre-trade checks."""

from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trading_bot import config
from trading_bot.symbols import t212_market_code

SPREAD_SAMPLE_FILE = 'spread_samples.json'
REPORT_PREFIX = 'spread_report_'

MARKET_OPEN_BY_CODE: dict[str, tuple[str, time]] = {
    '_US': ('America/New_York', time(9, 30)),
    '_CA': ('America/Toronto', time(9, 30)),
    '_BE': ('Europe/Brussels', time(9, 0)),
    '_BB': ('Europe/Brussels', time(9, 0)),
    '_AT': ('Europe/Vienna', time(9, 0)),
    '_PT': ('Europe/Lisbon', time(8, 0)),
    '_FR': ('Europe/Paris', time(9, 0)),
    'l': ('Europe/London', time(8, 0)),
    'd': ('Europe/Berlin', time(9, 0)),
    'p': ('Europe/Paris', time(9, 0)),
    's': ('Europe/Zurich', time(9, 0)),
    'm': ('Europe/Rome', time(9, 0)),
    'a': ('Europe/Amsterdam', time(9, 0)),
    'e': ('Europe/Madrid', time(9, 0)),
}


def collect_spread_sample(
    *,
    symbol: str,
    spread: float | None,
    last: float | None,
    quote_timestamp: datetime | None,
    checked_at: datetime,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    if spread is None or not isinstance(spread, (int, float)):
        return None

    market_code = t212_market_code(symbol)
    if _is_in_open_cooldown(market_code, checked_at):
        logger.info('Skipping spread sample for %s: within open cooldown.', symbol)
        return None

    timestamp = _coerce_timestamp(quote_timestamp or checked_at)
    return {
        'symbol': symbol,
        'market_code': market_code,
        'spread': float(spread),
        'last': float(last) if isinstance(last, (int, float)) else None,
        'timestamp': timestamp.isoformat(),
    }


def write_spread_report(
    base_dir: Path,
    samples: list[dict[str, Any]],
    checked_at: datetime,
    logger: logging.Logger,
) -> Path | None:
    if not samples:
        return None

    state_path = base_dir / 'state' / SPREAD_SAMPLE_FILE
    state = _load_state(state_path)
    samples_by_symbol = state.get('samples', {})

    for sample in samples:
        symbol = str(sample.get('symbol', '')).strip()
        if not symbol:
            continue
        samples_by_symbol.setdefault(symbol, []).append(sample)

    cutoff = checked_at - timedelta(days=config.SPREAD_SAMPLE_LOOKBACK_DAYS)
    samples_by_symbol = _prune_samples(samples_by_symbol, cutoff)

    _save_state(
        state_path,
        {
            'updated_at': _ensure_utc(checked_at).isoformat(),
            'samples': samples_by_symbol,
        },
    )

    report = _build_report(samples_by_symbol, checked_at)
    report_path = _write_report(base_dir, report, checked_at, logger)
    return report_path


def _build_report(samples_by_symbol: dict[str, list[dict[str, Any]]], checked_at: datetime) -> dict[str, Any]:
    report: dict[str, Any] = {
        'generated_at': _ensure_utc(checked_at).isoformat(),
        'source': 'yfinance',
        'lookback_days': config.SPREAD_SAMPLE_LOOKBACK_DAYS,
        'open_cooldown_minutes': config.SPREAD_SAMPLE_OPEN_COOLDOWN_MINUTES,
        'symbols': {},
    }

    for symbol, entries in samples_by_symbol.items():
        spreads = _extract_spreads(entries)
        if not spreads:
            continue
        spreads_sorted = sorted(spreads)
        latest = _latest_entry(entries)
        report['symbols'][symbol] = {
            'count': len(spreads_sorted),
            'min_pct': _to_pct(spreads_sorted[0]),
            'p50_pct': _to_pct(_percentile(spreads_sorted, 0.50)),
            'p90_pct': _to_pct(_percentile(spreads_sorted, 0.90)),
            'p95_pct': _to_pct(_percentile(spreads_sorted, 0.95)),
            'max_pct': _to_pct(spreads_sorted[-1]),
            'latest_pct': _to_pct(latest['spread']) if latest else None,
            'latest_at': latest.get('timestamp') if latest else None,
        }

    return report


def _write_report(
    base_dir: Path,
    report: dict[str, Any],
    checked_at: datetime,
    logger: logging.Logger,
) -> Path:
    outputs_dir = base_dir / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{REPORT_PREFIX}{checked_at.strftime("%Y%m%d_%H%M%S")}.json'
    path = outputs_dir / filename
    try:
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    except OSError as exc:
        logger.error('Failed to write spread report to %s: %s', path, exc)
    return path


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'samples': {}}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'samples': {}}
    if not isinstance(payload, dict):
        return {'samples': {}}
    return payload


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding='utf-8')


def _prune_samples(
    samples_by_symbol: dict[str, list[dict[str, Any]]],
    cutoff: datetime,
) -> dict[str, list[dict[str, Any]]]:
    pruned: dict[str, list[dict[str, Any]]] = {}
    cutoff_utc = _ensure_utc(cutoff)
    for symbol, entries in samples_by_symbol.items():
        valid_entries: list[dict[str, Any]] = []
        for entry in entries:
            timestamp = _parse_iso(entry.get('timestamp'))
            if timestamp is None:
                continue
            if _ensure_utc(timestamp) >= cutoff_utc:
                valid_entries.append(entry)
        if valid_entries:
            valid_entries.sort(key=lambda item: _parse_iso(item.get('timestamp')) or cutoff_utc)
            pruned[symbol] = valid_entries
    return pruned


def _extract_spreads(entries: list[dict[str, Any]]) -> list[float]:
    spreads: list[float] = []
    for entry in entries:
        value = entry.get('spread')
        if isinstance(value, (int, float)):
            spreads.append(float(value))
    return spreads


def _latest_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(entries, key=lambda item: _parse_iso(item.get('timestamp')) or datetime.min.replace(tzinfo=timezone.utc))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = int(round((len(values) - 1) * percentile))
    index = max(0, min(index, len(values) - 1))
    return values[index]


def _to_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 2)


def _is_in_open_cooldown(market_code: str | None, checked_at: datetime) -> bool:
    if not market_code:
        return False
    mapping = MARKET_OPEN_BY_CODE.get(market_code)
    if not mapping:
        return False
    zone_name, open_time = mapping
    zone = ZoneInfo(zone_name)
    checked_at_utc = _ensure_utc(checked_at)
    local_time = checked_at_utc.astimezone(zone)
    open_dt = datetime.combine(local_time.date(), open_time, tzinfo=zone)
    if local_time < open_dt:
        return True
    return local_time < (open_dt + timedelta(minutes=config.SPREAD_SAMPLE_OPEN_COOLDOWN_MINUTES))


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _coerce_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _parse_iso(value: object | None) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


__all__ = [
    'collect_spread_sample',
    'write_spread_report',
]
