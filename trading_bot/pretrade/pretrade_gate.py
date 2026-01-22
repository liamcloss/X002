"""Pure pre-trade viability rules."""

from __future__ import annotations

from typing import Any

from trading_bot import config

TRADE_CAP = 2


def evaluate_pretrade(
    symbol: str,
    planned_entry: float,
    planned_stop: float,
    planned_target: float,
    live_bid: float,
    live_ask: float,
    open_trades_count: int,
    checked_at: str,
    *,
    detected_price: float | None = None,
    current_price: float | None = None,
    mode: str | None = None,
    max_move_since_detection: float | None = None,
) -> dict[str, Any]:
    """Evaluate a single setup against the pre-trade gate rules."""

    mid_price = (live_bid + live_ask) / 2
    invalid_reason = _validate_inputs(
        planned_entry=planned_entry,
        planned_stop=planned_stop,
        live_bid=live_bid,
        live_ask=live_ask,
        mid_price=mid_price,
    )
    if invalid_reason:
        return _reject_result(
            symbol=symbol,
            bid=live_bid,
            ask=live_ask,
            open_trades_count=open_trades_count,
            checked_at=checked_at,
            reason=invalid_reason,
        )

    effective_price = current_price if current_price is not None else mid_price
    spread_pct = (live_ask - live_bid) / mid_price * 100
    price_drift_pct = abs(mid_price - planned_entry) / planned_entry * 100
    real_stop_distance_pct = abs(planned_entry - planned_stop) / planned_entry * 100
    real_rr = (planned_target - planned_entry) / (planned_entry - planned_stop)

    status, reject_reason = _apply_rules(
        spread_pct=spread_pct,
        price_drift_pct=price_drift_pct,
        real_stop_distance_pct=real_stop_distance_pct,
        real_rr=real_rr,
        open_trades_count=open_trades_count,
        detected_price=detected_price,
        current_price=effective_price,
        mode=mode,
        max_move_since_detection=max_move_since_detection,
    )

    return {
        'symbol': symbol,
        'bid': live_bid,
        'ask': live_ask,
        'mid_price': mid_price,
        'current_price': effective_price,
        'detected_price': detected_price,
        'pct_move_since_detection': _pct_move_since_detection(
            detected_price,
            effective_price,
        ),
        'spread_pct': spread_pct,
        'price_drift_pct': price_drift_pct,
        'real_stop_distance_pct': real_stop_distance_pct,
        'real_rr': real_rr,
        'open_trades_count': open_trades_count,
        'status': status,
        'reject_reason': reject_reason,
        'checked_at': checked_at,
    }


def _apply_rules(
    spread_pct: float,
    price_drift_pct: float,
    real_stop_distance_pct: float,
    real_rr: float,
    open_trades_count: int,
    detected_price: float | None,
    current_price: float,
    mode: str | None,
    max_move_since_detection: float | None,
) -> tuple[str, str | None]:
    max_spread_pct = config.CONFIG["max_spread_pct"] * 100
    if spread_pct > max_spread_pct:
        return 'REJECTED', _format_spread_reason(spread_pct)
    if price_drift_pct > 1.0:
        return 'REJECTED', _format_drift_reason(price_drift_pct)
    if real_stop_distance_pct < 5.0 or real_stop_distance_pct > 7.0:
        return 'REJECTED', _format_stop_reason(real_stop_distance_pct)
    if real_rr < 1.8:
        return 'REJECTED', _format_rr_reason(real_rr)
    gate_reason = _late_move_reason(
        detected_price,
        current_price,
        mode=mode,
        max_move_since_detection=max_move_since_detection,
    )
    if gate_reason:
        return 'REJECTED', gate_reason
    if open_trades_count >= TRADE_CAP:
        return 'REJECTED', _format_cap_reason(open_trades_count)
    return 'EXECUTABLE', None


def _validate_inputs(
    planned_entry: float,
    planned_stop: float,
    live_bid: float,
    live_ask: float,
    mid_price: float,
) -> str | None:
    if planned_entry <= 0:
        return 'Invalid planned entry (<= 0)'
    if planned_stop <= 0:
        return 'Invalid planned stop (<= 0)'
    if planned_entry == planned_stop:
        return 'Invalid planned stop (entry equals stop)'
    if live_bid <= 0 or live_ask <= 0:
        return 'Invalid live price data (bid/ask <= 0)'
    if mid_price <= 0:
        return 'Invalid live price data (mid price <= 0)'
    return None


def _reject_result(
    symbol: str,
    bid: float,
    ask: float,
    open_trades_count: int,
    checked_at: str,
    reason: str,
) -> dict[str, Any]:
    return {
        'symbol': symbol,
        'bid': bid,
        'ask': ask,
        'mid_price': None,
        'current_price': None,
        'detected_price': None,
        'pct_move_since_detection': None,
        'spread_pct': None,
        'price_drift_pct': None,
        'real_stop_distance_pct': None,
        'real_rr': None,
        'open_trades_count': open_trades_count,
        'status': 'REJECTED',
        'reject_reason': reason,
        'checked_at': checked_at,
    }


def _format_spread_reason(spread_pct: float) -> str:
    max_spread_pct = config.CONFIG["max_spread_pct"] * 100
    return f'Spread too wide ({_fmt(spread_pct)}% > {max_spread_pct:.2f}%)'


def _format_drift_reason(price_drift_pct: float) -> str:
    return f'Price drift too high ({_fmt(price_drift_pct)}% > 1.00%)'


def _format_stop_reason(real_stop_distance_pct: float) -> str:
    return (
        'Stop distance out of bounds '
        f'({_fmt(real_stop_distance_pct)}% not in 5.00%-7.00%)'
    )


def _format_rr_reason(real_rr: float) -> str:
    return f'RR too low ({real_rr:.2f} < 1.80)'


def _format_cap_reason(open_trades_count: int) -> str:
    return f'Trade cap reached (open_trades_count={open_trades_count}, cap={TRADE_CAP})'


def _late_move_reason(
    detected_price: float | None,
    current_price: float,
    *,
    mode: str | None,
    max_move_since_detection: float | None,
) -> str | None:
    if detected_price is None or detected_price <= 0:
        return None
    threshold = (
        float(max_move_since_detection)
        if max_move_since_detection is not None
        else float(config.LATE_MOVE_PCT_MAX)
    )
    if mode == "YOLO":
        return None
    pct_move = abs(current_price - detected_price) / detected_price
    if pct_move > threshold:
        return _format_late_move_reason(pct_move, threshold)
    return None


def _format_late_move_reason(pct_move: float, threshold: float) -> str:
    return (
        'Late move gate '
        f'({_fmt(pct_move * 100)}% > {_fmt(threshold * 100)}%)'
    )


def _pct_move_since_detection(
    detected_price: float | None,
    current_price: float,
) -> float | None:
    if detected_price is None or detected_price <= 0:
        return None
    return (current_price - detected_price) / detected_price * 100


def _fmt(value: float) -> str:
    return f'{value:.2f}'


__all__ = [
    'evaluate_pretrade',
]
