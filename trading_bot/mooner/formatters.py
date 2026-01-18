"""Formatting helpers for Mooner callouts."""

from __future__ import annotations

from typing import Iterable, List


def format_mooner_callout_lines(callouts: Iterable[dict]) -> List[str]:
    """Format a list of Mooner callouts into human-readable lines."""

    lines: List[str] = []
    for callout in callouts:
        if not isinstance(callout, dict):
            continue
        ticker = str(callout.get("ticker") or "").strip()
        state = str(callout.get("state") or "").strip()
        detected_on = str(callout.get("detected_on") or "").strip()
        context = str(callout.get("context") or "").strip()
        severity = str(callout.get("severity") or "").strip()

        parts: list[str] = []
        if ticker:
            parts.append(ticker)
        if state:
            parts.append(state)

        detail = " — ".join(parts) if parts else "Unknown callout"

        extras: list[str] = []
        if detected_on:
            extras.append(f"detected on {detected_on}")
        if context:
            extras.append(context)
        if extras:
            detail = f"{detail} ({'; '.join(extras)})"
        if severity:
            detail = f"{detail} [{severity}]"
        lines.append(detail)
    return lines
