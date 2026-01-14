"""Trade idea construction and formatting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Iterable

import pandas as pd

from trading_bot.config import Config
from trading_bot.constants import MAX_IDEAS, MIN_RISK_REWARD, STOP_PCTS, TARGET_PCTS


@dataclass(frozen=True)
class TradeIdea:
    """Structured trade idea for messaging and future execution."""

    symbol: str
    entry_price: float
    stop_pct: float
    target_pct: float
    risk_reward: float
    suggested_shares: int
    position_gbp: float
    risk_gbp: float
    score: float


def _viable_stop_target_pairs() -> list[tuple[float, float, float]]:
    pairs: list[tuple[float, float, float]] = []
    for stop_pct in STOP_PCTS:
        for target_pct in TARGET_PCTS:
            risk_reward = target_pct / stop_pct
            if risk_reward >= MIN_RISK_REWARD:
                pairs.append((stop_pct, target_pct, risk_reward))
    return pairs


def _select_preferred_pair(pairs: Iterable[tuple[float, float, float]]) -> tuple[float, float, float]:
    sorted_pairs = sorted(pairs, key=lambda pair: (-pair[2], pair[0], pair[1]))
    if not sorted_pairs:
        raise ValueError("No viable stop/target pairs meet risk/reward requirements.")
    return sorted_pairs[0]


def _suggest_position(
    entry_price: float,
    stop_pct: float,
    config: Config,
) -> tuple[int, float, float]:
    risk_per_share = entry_price * stop_pct
    max_risk = config.mode_settings.max_risk_gbp_max
    max_position_value = config.mode_settings.max_position_gbp
    if risk_per_share <= 0:
        return 0, 0.0, 0.0

    shares_by_risk = int(max_risk // risk_per_share)
    shares_by_position = int(max_position_value // entry_price)
    shares = max(0, min(shares_by_risk, shares_by_position))
    position_value = shares * entry_price
    risk_value = shares * risk_per_share
    return shares, position_value, risk_value


def build_trade_ideas(candidates: pd.DataFrame, config: Config) -> list[TradeIdea]:
    """Build up to MAX_IDEAS trade ideas with sizing suggestions."""

    if candidates.empty:
        return []

    pairs = _viable_stop_target_pairs()
    stop_pct, target_pct, risk_reward = _select_preferred_pair(pairs)

    ideas: list[TradeIdea] = []
    for _, row in candidates.head(MAX_IDEAS).iterrows():
        entry_price = float(row["entry_price"])
        shares, position_value, risk_value = _suggest_position(entry_price, stop_pct, config)
        ideas.append(
            TradeIdea(
                symbol=str(row["symbol"]),
                entry_price=entry_price,
                stop_pct=stop_pct,
                target_pct=target_pct,
                risk_reward=risk_reward,
                suggested_shares=shares,
                position_gbp=position_value,
                risk_gbp=risk_value,
                score=float(row["score"]),
            )
        )
    return ideas


def format_ideas_message(ideas: list[TradeIdea], config: Config) -> str:
    """Format trade ideas for Telegram output."""

    uk_time = datetime.now(tz=ZoneInfo("Europe/London"))
    today = uk_time.strftime("%Y-%m-%d %H:%M %Z")
    header = (
        f"Swing-trade ideas for {today} (mode: {config.mode}, "
        f"bankroll: £{config.bankroll_gbp:,.0f})"
    )

    if not ideas:
        return f"{header}\n\nNo valid trades today based on current scans."

    lines = [header, ""]
    for idx, idea in enumerate(ideas, start=1):
        stop_price = idea.entry_price * (1 - idea.stop_pct)
        target_price = idea.entry_price * (1 + idea.target_pct)
        lines.append(
            (
                f"{idx}) {idea.symbol} | Entry £{idea.entry_price:.2f} | "
                f"Stop £{stop_price:.2f} ({idea.stop_pct:.0%}) | "
                f"Target £{target_price:.2f} ({idea.target_pct:.0%}) | "
                f"R:R {idea.risk_reward:.1f} | "
                f"Size {idea.suggested_shares} shares (~£{idea.position_gbp:.2f}, "
                f"risk £{idea.risk_gbp:.2f})"
            )
        )
    return "\n".join(lines)
