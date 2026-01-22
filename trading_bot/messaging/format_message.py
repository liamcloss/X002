"""Message formatting utilities for Telegram."""

from __future__ import annotations

import math
from collections import defaultdict
from urllib.parse import quote_plus

_RANK_EMOJIS = ("🥇", "🥈", "🥉")
_SEPARATOR = "--------------------------------"
_REGION_DISPLAY_ORDER = ("UK", "EU", "US", "Canada", "Other", "Global")


def _scale_to_percent(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value) * 100


def _format_percentage(value: float | None) -> str:
    percent = _scale_to_percent(value)
    if percent != percent:  # NaN
        return "N/A"
    sign = "+" if percent > 0 else ""
    return f"{sign}{percent:.2f}%"


def _format_volume_multiple(value: float | None) -> str:
    if value is None:
        return 'N/A'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'N/A'
    if numeric != numeric:  # NaN
        return 'N/A'
    if numeric == 0:
        return '0'

    abs_value = abs(numeric)
    order = math.floor(math.log10(abs_value))
    scale = 10 ** (2 - order)
    floored = math.floor(abs_value * scale) / scale
    if numeric < 0:
        floored = -floored
    decimals = max(0, 2 - order)
    return f'{floored:.{decimals}f}'


def _news_search_url(candidate: dict) -> str | None:
    display_ticker = (
        candidate.get('display_ticker')
        or candidate.get('raw_ticker')
        or candidate.get('ticker')
        or ''
    )
    query = str(display_ticker).strip()
    if not query:
        return None
    return f'https://news.google.com/search?q={quote_plus(f"{query} stock")}'


def _format_candidate(candidate: dict, rank: int) -> str:
    emoji = _RANK_EMOJIS[rank] if rank < len(_RANK_EMOJIS) else "⭐"
    symbol = candidate["currency_symbol"]
    momentum = _format_percentage(candidate["momentum_5d"])
    reason = candidate["reason"]
    volume_multiple = _format_volume_multiple(candidate.get("volume_multiple"))

    display_ticker = candidate.get('display_ticker') or candidate['ticker']
    price = candidate['price']
    stop_pct = _scale_to_percent(candidate['stop_pct'])
    target_pct = _scale_to_percent(candidate['target_pct'])
    lines = [
        f"{emoji} {display_ticker}",
        f"Setup: {reason} | Vol: {volume_multiple}x | Momentum: {momentum}",
        (
            f"Price: {symbol}{price:.2f} {candidate['currency_code']} | "
            f"20D High Dist: {_scale_to_percent(candidate['pct_from_20d_high']):.2f}%"
        ),
        (
            f"Plan: Entry {symbol}{price:.2f} | "
            f"Stop {symbol}{candidate['stop_price']:.2f} (-{stop_pct:.2f}%) | "
            f"Target {symbol}{candidate['target_price']:.2f} (+{target_pct:.2f}%) | "
            f"RR {candidate['rr']:.2f}"
        ),
        f"Size: £{candidate['position_size']} | Risk: £{candidate['risk_gbp']} | Reward: £{candidate['reward_gbp']}",
    ]
    market_label = candidate.get('market_label') or candidate.get('market_code')
    if market_label:
        lines.append(f"Market: {market_label}")
    chart_url = candidate.get('tradingview_url')
    if chart_url:
        lines.append(f"Chart: {chart_url}")
    news_url = _news_search_url(candidate)
    if news_url:
        lines.append(f"News: {news_url}")
    return "\n".join(lines)


def _format_header(date: str, mode: str, dry_run: bool) -> list[str]:
    lines = [f"📊 Trade Candidates – {date}", f"MODE: {mode.upper()}"]
    if dry_run:
        lines.append("⚠️ DRY RUN – No Telegram actions or state updates")
    lines.append("")
    return lines


def format_daily_scan_message(
    date: str,
    mode: str,
    candidates: list[dict],
    scanned_count: int,
    valid_count: int,
    data_as_of: str,
    generated_at: str,
    dry_run: bool,
    mooner_callouts: list[dict] | None = None,
) -> str:
    """Build the daily scan message in Markdown."""

    mooner_section = _format_mooner_section(mooner_callouts or [])

    if not candidates:
        lines = _format_header(date, mode, dry_run)
        lines.extend(
            [
                "No valid trades today. Do nothing.",
                "",
            ]
        )
        if mooner_section:
            lines.extend(mooner_section)
            lines.append("")
        lines.extend(
            [
                f"Scanned: {scanned_count} instruments",
                "Valid setups: 0",
                "",
                f"Data as of: {data_as_of}",
                f"Generated: {generated_at}",
            ]
        )
        return "\n".join(lines)

    lines: list[str] = _format_header(date, mode, dry_run)
    region_sections = _format_region_sections(candidates)
    lines.append("\n\n".join(region_sections))
    lines.append("")
    if mooner_section:
        lines.extend(mooner_section)
        lines.append("")
    lines.append(_SEPARATOR)
    lines.extend(
        [
            f"Scanned: {scanned_count} instruments",
            f"Valid setups: {valid_count}",
            f"Top {len(candidates)} delivered",
            "",
            f"Data as of: {data_as_of}",
            f"Generated: {generated_at}",
            "",
            "Remember:",
            "Max risk per trade £1–£2 (TEST MODE) or £10–£20 (LIVE MODE)",
            "No averaging down. Stop losses are mandatory.",
        ]
    )
    return "\n".join(lines)


def _format_mooner_section(callouts: list[dict]) -> list[str]:
    if not callouts:
        return []
    lines = ["⚠️ Mooner Watchlist (Informational Only)"]
    for callout in callouts:
        ticker = str(callout.get("ticker", "")).strip()
        state = str(callout.get("state", "")).strip()
        detected_on = str(callout.get("detected_on", "")).strip()
        context = str(callout.get("context", "")).strip()
        parts = [ticker, state]
        detail = " — ".join(part for part in parts if part)
        if detected_on:
            detail = f"{detail} ({detected_on})"
        if context:
            detail = f"{detail}: {context}"
        lines.append(f"- {detail}".strip())
    return lines


def _format_region_sections(candidates: list[dict]) -> list[str]:
    sections: list[str] = []
    for region, entries in _group_candidates_by_region(candidates):
        lines = [f'📍 {region} • {len(entries)} setup{"s" if len(entries) != 1 else ""}']
        for index, candidate in entries:
            lines.append(_format_candidate(candidate, index))
            lines.append("")
        sections.append("\n".join(lines).strip())
    return sections


def _group_candidates_by_region(candidates: list[dict]) -> list[tuple[str, list[tuple[int, dict]]]]:
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        region = candidate.get("region") or "Global"
        groups[region].append((index, candidate))
    return sorted(groups.items(), key=lambda item: _region_sort_key(item[0]))


def _region_sort_key(region: str) -> int:
    try:
        return _REGION_DISPLAY_ORDER.index(region)
    except ValueError:
        return len(_REGION_DISPLAY_ORDER)


def format_error_message(error_text: str) -> str:
    """Format an error notification message."""

    return f"⚠️ Trading system error\n{error_text}"


def format_universe_success(summary: dict) -> str:
    """Format a universe refresh success message."""

    lines = [
        "✅ Trading212 Universe Updated",
        f"Active instruments: {summary['active_instruments']}",
        (
            f"Stocks: {summary['stocks']} | ETFs: {summary['etfs']}"
        ),
    ]
    return "\n".join(lines)


def format_universe_failure(error: str, attempt: int, max_attempts: int) -> str:
    """Format a universe refresh failure message."""

    lines = [
        "⚠️ Trading212 Universe Refresh Failed",
        f"Attempt: {attempt}/{max_attempts}",
        f"Error: {error}",
    ]
    return "\n".join(lines)
