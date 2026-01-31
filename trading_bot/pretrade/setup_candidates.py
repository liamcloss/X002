"""SetupCandidates.json output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from trading_bot.paths import setup_candidates_path
def build_setup_candidates_payload(
    candidates: Iterable[dict[str, Any]],
    *,
    mode: str,
    data_as_of: str,
    generated_at: str,
) -> dict[str, Any]:
    payload_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        symbol = _extract_symbol(candidate)
        display_ticker = candidate.get('display_ticker') or candidate.get('ticker') or symbol
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
                'mooner_state': candidate.get('mooner_state'),
                'mooner_context': candidate.get('mooner_context'),
                'mooner_as_of': candidate.get('mooner_as_of'),
                'currency_code': candidate.get('currency_code'),
                'price': candidate.get('price'),
                'volume_multiple': candidate.get('volume_multiple'),
                'momentum_5d': candidate.get('momentum_5d'),
                'pct_from_20d_high': candidate.get('pct_from_20d_high'),
                'tradingview_url': candidate.get('tradingview_url'),
                'region': candidate.get('region'),
                'market_code': candidate.get('market_code'),
                'market_label': candidate.get('market_label'),
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
) -> Path:
    payload = build_setup_candidates_payload(
        candidates,
        mode=mode,
        data_as_of=data_as_of,
        generated_at=generated_at,
    )
    path = setup_candidates_path(base_dir)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def _extract_symbol(candidate: dict[str, Any]) -> str:
    for key in ('raw_ticker', 'base_ticker', 'symbol', 'ticker'):
        value = candidate.get(key)
        if value:
            return str(value).strip()
    return ''


__all__ = [
    'build_setup_candidates_payload',
    'write_setup_candidates',
]
