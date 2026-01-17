"""Strict pre-trade validation firewall for breakout entries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reasons: list[str]
    checks: list[dict[str, str]]


def validate_trade(trade: dict, open_trades: int) -> ValidationResult:
    """
    Validate a trade candidate against strict breakout rules.

    Returns a ValidationResult containing overall pass/fail, failure reasons,
    and a compact check table. This function does not mutate state.
    """

    reasons: list[str] = []
    checks: list[dict[str, str]] = []

    def record(rule: str, value: str, passed: bool, reason: str | None = None) -> None:
        checks.append({"rule": rule, "value": value, "status": "PASS" if passed else "FAIL"})
        if not passed and reason:
            reasons.append(reason)

    def range_check(
        rule: str,
        raw_value: Any,
        min_value: float,
        max_value: float,
        formatter: callable,
    ) -> float | None:
        value = _coerce_float(raw_value)
        if value is None:
            record(rule, "N/A", False, f"{rule} is missing")
            return None
        if value < min_value:
            record(
                rule,
                formatter(value),
                False,
                f"{rule} {formatter(value)} is below minimum {formatter(min_value)}",
            )
        elif value > max_value:
            record(
                rule,
                formatter(value),
                False,
                f"{rule} {formatter(value)} exceeds maximum {formatter(max_value)}",
            )
        else:
            record(rule, formatter(value), True)
        return value

    def minimum_check(
        rule: str,
        raw_value: Any,
        min_value: float,
        formatter: callable,
    ) -> float | None:
        value = _coerce_float(raw_value)
        if value is None:
            record(rule, "N/A", False, f"{rule} is missing")
            return None
        if value < min_value:
            record(
                rule,
                formatter(value),
                False,
                f"{rule} {formatter(value)} is below minimum {formatter(min_value)}",
            )
        else:
            record(rule, formatter(value), True)
        return value

    def maximum_check(
        rule: str,
        raw_value: Any,
        max_value: float,
        formatter: callable,
    ) -> float | None:
        value = _coerce_float(raw_value)
        if value is None:
            record(rule, "N/A", False, f"{rule} is missing")
            return None
        if value > max_value:
            record(
                rule,
                formatter(value),
                False,
                f"{rule} {formatter(value)} exceeds maximum {formatter(max_value)}",
            )
        else:
            record(rule, formatter(value), True)
        return value

    bankroll_value = _coerce_float(trade.get("bankroll_gbp", 1000.0))
    bankroll_ok = bankroll_value == 1000.0
    record(
        "Bankroll baseline",
        _format_gbp(bankroll_value) if bankroll_value is not None else "N/A",
        bankroll_ok,
        "Bankroll baseline must be £1,000" if not bankroll_ok else None,
    )

    range_check(
        "Position size",
        _first_value(trade, "position_gbp", "position_size", "position"),
        100.0,
        200.0,
        _format_gbp,
    )
    range_check(
        "Risk per trade",
        _first_value(trade, "risk_gbp", "risk"),
        10.0,
        20.0,
        _format_gbp,
    )

    if open_trades is None:
        record("Open trades", "N/A", False, "Open trades count is missing")
    else:
        open_trades_value = int(open_trades)
        if open_trades_value > 2:
            record(
                "Open trades",
                str(open_trades_value),
                False,
                f"Open trades {open_trades_value} exceeds maximum 2",
            )
        else:
            record("Open trades", str(open_trades_value), True)

    distance_value = _normalize_percent(
        _first_value(
            trade,
            "pct_from_20d_high",
            "high_20d_distance",
            "high_20d_distance_percent",
            "20d_high_distance",
            "20d_high_distance_percent",
        )
    )
    if distance_value is None:
        record("20D High Distance", "N/A", False, "20D High Distance is missing")
    elif distance_value < 0.0:
        record(
            "20D High Distance",
            _format_percent(distance_value),
            False,
            f"20D High Distance {_format_percent(distance_value)} is below minimum 0.00%",
        )
    elif distance_value > 1.5:
        record(
            "20D High Distance",
            _format_percent(distance_value),
            False,
            f"20D High Distance {_format_percent(distance_value)} exceeds maximum 1.50%",
        )
    else:
        record("20D High Distance", _format_percent(distance_value), True)

    price_above_ema = _coerce_bool(_first_value(trade, "price_above_20ema", "price_above_ema20"))
    if price_above_ema is True:
        record("Price above 20EMA", "True", True)
    else:
        record("Price above 20EMA", _format_bool(price_above_ema), False, "Price must be above 20EMA")

    ema_alignment = _coerce_bool(
        _first_value(trade, "ema20_above_ema50", "ema20_above_ema_50", "ema_20_above_ema_50")
    )
    if ema_alignment is True:
        record("EMA20 above EMA50", "True", True)
    else:
        record("EMA20 above EMA50", _format_bool(ema_alignment), False, "EMA20 must be above EMA50")

    minimum_check(
        "Volume multiple",
        _first_value(trade, "volume_multiple"),
        2.0,
        _format_number,
    )

    stop_percent = _normalize_percent(_first_value(trade, "stop_percent", "stop_pct", "stop_percentage"))
    atr_percent = _normalize_percent(_first_value(trade, "atr_percent", "atr_pct", "atr_percentage"))
    if stop_percent is None:
        record("Stop percent", "N/A", False, "Stop percent is missing")
    else:
        record("Stop percent", _format_percent(stop_percent), True)
    if atr_percent is None:
        record("ATR percent", "N/A", False, "ATR percent is missing")
    else:
        record("ATR percent", _format_percent(atr_percent), True)

    if stop_percent is not None and atr_percent is not None:
        if stop_percent >= atr_percent:
            record(
                "Stop ≥ ATR",
                f"{_format_percent(stop_percent)} / {_format_percent(atr_percent)}",
                True,
            )
        else:
            record(
                "Stop ≥ ATR",
                f"{_format_percent(stop_percent)} / {_format_percent(atr_percent)}",
                False,
                f"Stop percent {_format_percent(stop_percent)} is below ATR {_format_percent(atr_percent)}",
            )
    else:
        record("Stop ≥ ATR", "N/A", False, "Stop vs ATR comparison unavailable")

    minimum_check(
        "Avg daily volume",
        _first_value(trade, "avg_daily_volume", "average_daily_volume", "adv"),
        1_000_000.0,
        _format_volume,
    )

    maximum_check(
        "Spread percent",
        _normalize_percent(
            _first_value(trade, "spread_percent", "spread_pct", "spread_percentage")
        ),
        0.5,
        _format_percent,
    )

    if stop_percent is not None:
        if stop_percent < 5.0:
            record(
                "Stop range",
                _format_percent(stop_percent),
                False,
                f"Stop percent {_format_percent(stop_percent)} is below minimum 5.00%",
            )
        elif stop_percent > 7.0:
            record(
                "Stop range",
                _format_percent(stop_percent),
                False,
                f"Stop percent {_format_percent(stop_percent)} exceeds maximum 7.00%",
            )
        else:
            record("Stop range", _format_percent(stop_percent), True)
    else:
        record("Stop range", "N/A", False, "Stop percent is missing")

    target_percent = _normalize_percent(
        _first_value(trade, "target_percent", "target_pct", "target_percentage")
    )
    if target_percent is None:
        record("Target percent", "N/A", False, "Target percent is missing")
    else:
        record("Target percent", _format_percent(target_percent), True)
        if target_percent < 10.0:
            record(
                "Target minimum",
                _format_percent(target_percent),
                False,
                f"Target percent {_format_percent(target_percent)} is below minimum 10.00%",
            )
        else:
            record("Target minimum", _format_percent(target_percent), True)

    if stop_percent is not None and target_percent is not None and stop_percent > 0:
        rr_value = target_percent / stop_percent
        if rr_value >= 2.0:
            record("Risk/Reward", f"{rr_value:.2f}", True)
        else:
            record(
                "Risk/Reward",
                f"{rr_value:.2f}",
                False,
                f"R:R {rr_value:.2f} is below minimum 2.00",
            )
    else:
        record("Risk/Reward", "N/A", False, "R:R unavailable (missing stop/target percent)")

    passed = not reasons
    return ValidationResult(passed=passed, reasons=reasons, checks=checks)


def _first_value(source: dict, *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _normalize_percent(value: Any) -> float | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    if abs(numeric) <= 1:
        return numeric * 100
    return numeric


def _format_gbp(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"£{value:,.2f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _format_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _format_volume(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "True" if value else "False"
