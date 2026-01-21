"""Helpers for summarizing the weekly YOLO lottery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_bot.paths import yolo_output_path

YOLO_PICK_FILENAME = "YOLO_Pick.json"
YOLO_LEDGER_FILENAME = "YOLO_Ledger.json"


@dataclass(frozen=True)
class AlternativePick:
    ticker: str | None
    price: float | None
    yolo_score: float | None
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class YOLOSummary:
    week_of: str | None
    ticker: str | None
    price: float | None
    yolo_score: float | None
    rationale: tuple[str, ...]
    alternatives: tuple[AlternativePick, ...]
    repeat_weeks: int
    previous_ticker: str | None


def load_yolo_summary(base_dir: Path | None = None) -> YOLOSummary | None:
    """Return the latest weekly pick summary, if the pick file exists."""

    pick_path = yolo_output_path(base_dir, YOLO_PICK_FILENAME)
    pick_payload = _read_json_payload(pick_path)
    if not isinstance(pick_payload, dict):
        return None

    ledger = _load_ledger(base_dir)
    summary = YOLOSummary(
        week_of=_normalize_str(pick_payload.get("week_of")),
        ticker=_normalize_str(pick_payload.get("ticker")),
        price=_to_float(pick_payload.get("price")),
        yolo_score=_to_float(pick_payload.get("yolo_score")),
        rationale=_normalize_str_list(pick_payload.get("rationale")),
        alternatives=tuple(
            _build_alternative(candidate)
            for candidate in pick_payload.get("alternatives") or []
            if isinstance(candidate, dict)
        ),
        repeat_weeks=_count_consecutive_repeats(ledger, _normalize_str(pick_payload.get("ticker"))),
        previous_ticker=_previous_ledger_ticker(ledger),
    )
    return summary


def format_yolo_summary(summary: YOLOSummary) -> list[str]:
    """Render a human-friendly summary kit for the weekly YOLO pick."""

    lines: list[str] = ["YOLO Weekly Lottery"]
    lines.append(f"Week of {summary.week_of or 'Unknown'}")
    winner = summary.ticker or "UNKNOWN"
    price_line = _format_price(summary.price)
    score_line = _format_score(summary.yolo_score)
    lines.append(f"Winner: {winner} | Price: {price_line} | Score: {score_line}")
    if summary.repeat_weeks > 1:
        lines.append(f"Same ticker for {summary.repeat_weeks} consecutive week(s).")
    elif summary.previous_ticker:
        lines.append(f"Previous winner: {summary.previous_ticker}")
    if summary.rationale:
        lines.append(f"Rationale: {', '.join(summary.rationale)}")

    if summary.alternatives:
        lines.append("")
        lines.append("Alternative contenders:")
        for index, alternative in enumerate(summary.alternatives, start=1):
            alt_ticker = alternative.ticker or "UNKNOWN"
            alt_price = _format_price(alternative.price)
            alt_score = _format_score(alternative.yolo_score)
            lines.append(f"{index}. {alt_ticker} | Price: {alt_price} | Score: {alt_score}")
            if alternative.rationale:
                lines.append(f"   Rationale: {', '.join(alternative.rationale)}")

    return lines


def _read_json_payload(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_ledger(base_dir: Path | None) -> list[dict[str, Any]]:
    ledger_path = yolo_output_path(base_dir, YOLO_LEDGER_FILENAME)
    payload = _read_json_payload(ledger_path)
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _build_alternative(payload: dict[str, Any]) -> AlternativePick:
    return AlternativePick(
        ticker=_normalize_str(payload.get("ticker")),
        price=_to_float(payload.get("price")),
        yolo_score=_to_float(payload.get("yolo_score")),
        rationale=_normalize_str_list(payload.get("rationale")),
    )


def _normalize_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return tuple()
    normalized = []
    for item in value:
        text = _normalize_str(item)
        if text:
            normalized.append(text)
    return tuple(normalized)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _count_consecutive_repeats(
    ledger: list[dict[str, Any]], ticker: str | None
) -> int:
    if not ticker:
        return 0
    count = 0
    for entry in reversed(ledger):
        if _normalize_str(entry.get("ticker")) == ticker:
            count += 1
        else:
            break
    return count


def _previous_ledger_ticker(ledger: list[dict[str, Any]]) -> str | None:
    if len(ledger) < 2:
        return None
    for entry in reversed(ledger[:-1]):
        ticker = _normalize_str(entry.get("ticker"))
        if ticker:
            return ticker
    return None


def _format_price(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}"


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


__all__ = [
    "AlternativePick",
    "YOLOSummary",
    "format_yolo_summary",
    "load_yolo_summary",
]
