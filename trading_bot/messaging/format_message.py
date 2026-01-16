"""Message formatting utilities for Telegram."""

from __future__ import annotations

import math

_RANK_EMOJIS = ("🥇", "🥈", "🥉")
_SEPARATOR = "--------------------------------"


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


def _format_candidate(candidate: dict, rank: int) -> str:
    emoji = _RANK_EMOJIS[rank] if rank < len(_RANK_EMOJIS) else "⭐"
    symbol = candidate["currency_symbol"]
    momentum = _format_percentage(candidate["momentum_5d"])
    reason = candidate["reason"]
    volume_multiple = _format_volume_multiple(candidate.get("volume_multiple"))

    display_ticker = candidate.get('display_ticker') or candidate['ticker']
    lines = [
        f"{emoji} {display_ticker}",
        _SEPARATOR,
        f"Reason: {reason}, {volume_multiple}× volume, {momentum}",
        "",
        f"Price: {symbol}{candidate['price']:.2f} {candidate['currency_code']}",
        f"20D High Distance: {_scale_to_percent(candidate['pct_from_20d_high']):.2f}%",
        "",
        (
            f"Stop: −{_scale_to_percent(candidate['stop_pct']):.2f}% → "
            f"{symbol}{candidate['stop_price']:.2f}"
        ),
        (
            f"Target: +{_scale_to_percent(candidate['target_pct']):.2f}% → "
            f"{symbol}{candidate['target_price']:.2f}"
        ),
        f"R:R = {candidate['rr']:.2f}",
        "",
        f"Position: £{candidate['position_size']}",
        f"Risk: £{candidate['risk_gbp']}",
        f"Reward: £{candidate['reward_gbp']}",
        "",
        f"Chart: {candidate['tradingview_url']}",
    ]
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
) -> str:
    """Build the daily scan message in Markdown."""

    if not candidates:
        lines = _format_header(date, mode, dry_run)
        lines.extend(
            [
                "No valid trades today. Do nothing.",
                "",
                f"Scanned: {scanned_count} instruments",
                "Valid setups: 0",
                "",
                f"Data as of: {data_as_of}",
                f"Generated: {generated_at}",
            ]
        )
        return "\n".join(lines)

    lines: list[str] = _format_header(date, mode, dry_run)
    formatted_candidates = [
        _format_candidate(candidate, idx) for idx, candidate in enumerate(candidates)
    ]
    lines.extend(formatted_candidates)
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
