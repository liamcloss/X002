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
    )

    return {
        'symbol': symbol,
        'bid': live_bid,
        'ask': live_ask,
        'mid_price': mid_price,
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
) -> tuple[str, str | None]:
    max_spread_pct = config.MAX_SPREAD_PCT * 100
    if spread_pct > max_spread_pct:
        return 'REJECTED', _format_spread_reason(spread_pct)
    if price_drift_pct > 1.0:
        return 'REJECTED', _format_drift_reason(price_drift_pct)
    if real_stop_distance_pct < 5.0 or real_stop_distance_pct > 7.0:
        return 'REJECTED', _format_stop_reason(real_stop_distance_pct)
    if real_rr < 1.8:
        return 'REJECTED', _format_rr_reason(real_rr)
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
    max_spread_pct = config.MAX_SPREAD_PCT * 100
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


def _fmt(value: float) -> str:
    return f'{value:.2f}'


__all__ = [
    'evaluate_pretrade',
]
