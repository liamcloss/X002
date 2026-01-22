"""SetupCandidates.json output helpers."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

from trading_bot.paths import setup_candidates_snapshot_path

def build_setup_candidates_payload(
    candidates: Iterable[dict[str, Any]],
    *,
    mode: str,
    data_as_of: str,
    generated_at: str,
    scanner_version: str,
) -> dict[str, Any]:
    payload_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        symbol = _extract_symbol(candidate)
        display_ticker = candidate.get('display_ticker') or candidate.get('ticker') or symbol
        detected_at_utc = candidate.get('detected_at_utc') or generated_at
        detected_close_date = candidate.get('detected_close_date') or data_as_of
        detected_price = candidate.get('detected_price') or candidate.get('price')
        payload_candidates.append(
            {
                'symbol': symbol,
                'display_ticker': display_ticker,
                'planned_entry': candidate.get('price'),
                'planned_stop': candidate.get('stop_price'),
                'planned_target': candidate.get('target_price'),
                'rank': index,
                'reason': candidate.get('reason'),
                'rr': candidate.get('rr'),
                'stop_pct': candidate.get('stop_pct'),
                'target_pct': candidate.get('target_pct'),
                'currency_code': candidate.get('currency_code'),
                'price': candidate.get('price'),
                'volume_multiple': candidate.get('volume_multiple'),
                'momentum_5d': candidate.get('momentum_5d'),
                'pct_from_20d_high': candidate.get('pct_from_20d_high'),
                'tradingview_url': candidate.get('tradingview_url'),
                'detected_at_utc': detected_at_utc,
                'detected_price': detected_price,
                'detected_close_date': detected_close_date,
                'scanner_version': candidate.get('scanner_version') or scanner_version,
            }
        )

    return {
        'generated_at': generated_at,
        'mode': mode,
        'data_as_of': data_as_of,
        'candidates': payload_candidates,
    }


def write_setup_candidates(
    base_dir: Path,
    candidates: Iterable[dict[str, Any]],
    *,
    mode: str,
    data_as_of: str,
    generated_at: str,
    logger: Any | None = None,
) -> Path:
    scanner_version = _scanner_version(base_dir)
    payload = build_setup_candidates_payload(
        candidates,
        mode=mode,
        data_as_of=data_as_of,
        generated_at=generated_at,
        scanner_version=scanner_version,
    )
    path = setup_candidates_snapshot_path(base_dir, _normalize_date(data_as_of))
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    _emit_self_test(payload, logger=logger)
    return path


def _extract_symbol(candidate: dict[str, Any]) -> str:
    for key in ('raw_ticker', 'base_ticker', 'symbol', 'ticker'):
        value = candidate.get(key)
        if value:
            return str(value).strip()
    return ''


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).strip()[:10]


def _scanner_version(base_dir: Path) -> str:
    git_dir = base_dir / '.git'
    head_path = git_dir / 'HEAD'
    if head_path.exists():
        head = head_path.read_text(encoding='utf-8').strip()
        if head.startswith('ref:'):
            ref_path = git_dir / head.split(' ', maxsplit=1)[-1].strip()
            if ref_path.exists():
                return ref_path.read_text(encoding='utf-8').strip()
        if head:
            return head
    digest = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    return digest[:12]


def _emit_self_test(payload: dict[str, Any], logger: Any | None) -> None:
    candidates = payload.get('candidates')
    if not isinstance(candidates, list):
        return
    if not candidates:
        message = 'SetupCandidates self-test: no candidates.'
        if logger:
            logger.info(message)
        else:
            print(message)
        return
    sample = candidates[0]
    schema = sorted(sample.keys())
    na_rates = _compute_na_rates(candidates, schema)
    message_lines = [
        f'SetupCandidates schema: {schema}',
        f'SetupCandidates sample: {json.dumps(sample, sort_keys=True)}',
        f'SetupCandidates NA rates: {json.dumps(na_rates, sort_keys=True)}',
    ]
    message = '\n'.join(message_lines)
    if logger:
        logger.info(message)
    else:
        print(message)


def _compute_na_rates(candidates: list[dict[str, Any]], schema: list[str]) -> dict[str, float]:
    counts = {key: 0 for key in schema}
    total = len(candidates)
    for candidate in candidates:
        for key in schema:
            value = candidate.get(key)
            if value is None or value == '':
                counts[key] += 1
    return {key: counts[key] / total for key in schema}

__all__ = [
    'build_setup_candidates_payload',
    'write_setup_candidates',
]
