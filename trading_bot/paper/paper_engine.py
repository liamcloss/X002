"""Paper trading engine (simulation only)."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from trading_bot import config
from trading_bot.config import LIVE_MODE_MAX_RISK, LIVE_MODE_POSITION_MAX, LIVE_MODE_POSITION_MIN
from trading_bot.market_data import cache
from trading_bot.messaging.telegram_client import send_paper_message
from trading_bot.signals.risk_geometry import find_risk_geometry

logger = logging.getLogger("trading_bot")

PAPER_COLUMNS = (
    "trade_id",
    "ticker",
    "entry_date",
    "entry_price",
    "stop_pct",
    "stop_price",
    "target_pct",
    "target_price",
    "position_size",
    "risk_gbp",
    "reward_gbp",
    "status",
    "close_date",
    "close_price",
    "close_reason",
)


def process_open_trades(today_date: str) -> None:
    """
    Checks all open paper trades against latest prices.
    Closes trades if stop/target hit.
    """

    store = _load_store()
    if store.empty:
        logger.info("No paper trades to process.")
        return

    open_trades = store[store["status"] == "OPEN"].copy()
    if open_trades.empty:
        logger.info("No open paper trades to process.")
        return

    prices_dir = cache.ensure_prices_dir(_base_dir())
    updated = False

    for _, trade in open_trades.iterrows():
        ticker = str(trade["ticker"])
        last_price = _latest_cached_price(prices_dir, ticker)
        if last_price is None:
            logger.warning("No cached price available for %s; skipping.", ticker)
            continue

        stop_price = float(trade["stop_price"])
        target_price = float(trade["target_price"])

        close_reason = None
        if last_price <= stop_price:
            close_reason = "STOP"
        elif last_price >= target_price:
            close_reason = "TARGET"

        if close_reason:
            entry_price = _safe_float(trade.get('entry_price'))
            position_size = _safe_float(trade.get('position_size'))
            _close_trade(
                store,
                trade_id=str(trade['trade_id']),
                close_reason=close_reason,
                close_price=last_price,
                close_date=today_date,
            )
            try:
                _notify_close(
                    ticker=ticker,
                    price=last_price,
                    reason=close_reason,
                    entry_price=entry_price,
                    position_size=position_size,
                )
            except Exception as exc:  # noqa: BLE001 - persist close even if messaging fails
                logger.warning('Paper trade close notification failed for %s: %s', ticker, exc)
            updated = True

    if updated:
        _save_store(store)


def open_paper_trade(candidate: dict, today_date: str) -> None:
    """
    Attempts to open a new paper trade from the #1 ranked candidate.
    Enforces max 2 open trades.
    """

    store = _load_store()
    open_trades = store[store["status"] == "OPEN"]
    if open_trades.shape[0] >= 2:
        logger.info("Paper trade skipped: capacity full (2 open trades)")
        return

    ticker = str(candidate.get("ticker", "")).strip().upper()
    if not ticker:
        raise ValueError("Candidate missing ticker")

    if not open_trades.empty and ticker in open_trades["ticker"].astype(str).str.upper().tolist():
        logger.info("Paper trade skipped: %s already open", ticker)
        return

    entry_price = _candidate_price(candidate)
    geometry = _candidate_geometry(candidate, entry_price)
    stop_pct = float(geometry["stop_pct"])
    target_pct = float(geometry["target_pct"])

    stop_price = float(candidate.get("stop_price", entry_price * (1 - stop_pct)))
    target_price = float(candidate.get("target_price", entry_price * (1 + target_pct)))

    position_size = _candidate_position_size(candidate, stop_pct)
    risk_gbp = _candidate_amount(candidate.get('risk_gbp'))
    reward_gbp = _candidate_amount(candidate.get('reward_gbp'))
    if risk_gbp is None:
        risk_gbp = round(position_size * stop_pct, 2)
    if reward_gbp is None:
        reward_gbp = round(position_size * target_pct, 2)

    trade = {
        "trade_id": uuid4().hex,
        "ticker": ticker,
        "entry_date": today_date,
        "entry_price": entry_price,
        "stop_pct": stop_pct,
        "stop_price": stop_price,
        "target_pct": target_pct,
        "target_price": target_price,
        "position_size": position_size,
        "risk_gbp": risk_gbp,
        "reward_gbp": reward_gbp,
        "status": "OPEN",
        "close_date": "",
        "close_price": "",
        "close_reason": "",
    }

    store = pd.concat([store, pd.DataFrame([trade])], ignore_index=True)
    _save_store(store)
    logger.info("Opened paper trade for %s at %.2f", ticker, entry_price)
    try:
        _notify_open(
            ticker=ticker,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            position_size=position_size,
            risk_gbp=risk_gbp,
            reward_gbp=reward_gbp,
        )
    except Exception as exc:  # noqa: BLE001 - keep trade even if messaging fails
        logger.warning('Paper trade open notification failed for %s: %s', ticker, exc)


def get_open_trade_count() -> int:
    """
    Returns number of OPEN paper trades.
    """

    store = _load_store()
    if store.empty:
        return 0
    return int((store["status"] == "OPEN").sum())


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _store_path() -> Path:
    return Path(__file__).resolve().parent / "paper_store.csv"


def _load_store() -> pd.DataFrame:
    path = _store_path()
    if not path.exists():
        df = pd.DataFrame(columns=PAPER_COLUMNS)
        df.to_csv(path, index=False)
        return df

    df = pd.read_csv(path)
    missing = [col for col in PAPER_COLUMNS if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = ""
        df = df[PAPER_COLUMNS]
    return df


def _save_store(store: pd.DataFrame) -> None:
    store.to_csv(_store_path(), index=False)


def _latest_cached_price(prices_dir: Path, ticker: str) -> float | None:
    df = cache.load_cache(prices_dir, ticker, logger)
    if df.empty:
        return None
    df = df.sort_index()
    last_row = df.iloc[-1]
    for column in ("close", "Close", "close_price"):
        if column in last_row:
            price = float(last_row[column])
            if price > 0:
                return price
    return None


def _candidate_price(candidate: dict) -> float:
    for key in ("price", "entry_price", "close_price"):
        if key in candidate and candidate[key] is not None:
            price = float(candidate[key])
            if price > 0:
                return price
    raise ValueError("Candidate missing valid price")


def _candidate_geometry(candidate: dict, entry_price: float) -> dict:
    stop_pct = candidate.get("stop_pct")
    target_pct = candidate.get("target_pct")
    if stop_pct is not None and target_pct is not None:
        return {"stop_pct": float(stop_pct), "target_pct": float(target_pct)}

    geometry = find_risk_geometry(entry_price)
    if not geometry:
        raise ValueError("No risk geometry available for candidate price")
    return geometry


def _position_size_for_risk(stop_pct: float) -> float:
    if config.MODE == 'TEST':
        return float(config.TEST_MODE_POSITION_SIZE)
    if stop_pct <= 0:
        return float(LIVE_MODE_POSITION_MIN)
    target_size = LIVE_MODE_MAX_RISK / stop_pct
    bounded = max(LIVE_MODE_POSITION_MIN, min(LIVE_MODE_POSITION_MAX, target_size))
    return float(round(bounded))


def _candidate_position_size(candidate: dict, stop_pct: float) -> float:
    raw_size = candidate.get('position_size')
    if raw_size is not None:
        try:
            numeric = float(raw_size)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and numeric > 0:
            return numeric
    return _position_size_for_risk(stop_pct)


def _candidate_amount(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return numeric


def _safe_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _parse_date(value: object | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _calculate_pnl(
    entry_price: float | None,
    close_price: float | None,
    position_size: float | None,
) -> float | None:
    if entry_price is None or close_price is None or position_size is None:
        return None
    if entry_price <= 0:
        return None
    return position_size * ((close_price - entry_price) / entry_price)


def _format_price(value: float | None) -> str:
    if value is None:
        return 'N/A'
    return f'${value:.2f}'


def _format_gbp(value: float | None) -> str:
    if value is None:
        return 'N/A'
    return f'£{value:.2f}'


def _format_signed_gbp(value: float) -> str:
    sign = '+' if value > 0 else ''
    return f'{sign}£{value:.2f}'


def _format_signed_percent(value: float) -> str:
    sign = '+' if value > 0 else ''
    return f'{sign}{value:.2f}%'


def _format_pnl(
    entry_price: float | None,
    close_price: float | None,
    position_size: float | None,
) -> str:
    pnl_gbp = _calculate_pnl(entry_price, close_price, position_size)
    if pnl_gbp is None or entry_price is None or close_price is None:
        return 'N/A'
    pnl_pct = (close_price - entry_price) / entry_price * 100
    return f'{_format_signed_gbp(pnl_gbp)} ({_format_signed_percent(pnl_pct)})'


def _close_trade(
    store: pd.DataFrame,
    trade_id: str,
    close_reason: str,
    close_price: float,
    close_date: str,
) -> None:
    idx = store.index[store["trade_id"].astype(str) == trade_id]
    if idx.empty:
        logger.warning("Paper trade %s not found for close", trade_id)
        return
    store.loc[idx, "status"] = "CLOSED"
    store.loc[idx, "close_reason"] = close_reason
    store.loc[idx, "close_price"] = round(float(close_price), 2)
    store.loc[idx, "close_date"] = close_date
    logger.info("Closed paper trade %s (%s)", trade_id, close_reason)


def _notify_open(
    ticker: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    position_size: float,
    risk_gbp: float,
    reward_gbp: float,
) -> None:
    entry_str = _format_price(entry_price)
    stop_str = _format_price(stop_price)
    target_str = _format_price(target_price)
    size_str = _format_gbp(position_size)
    risk_str = _format_gbp(risk_gbp)
    reward_str = _format_gbp(reward_gbp)
    text = (
        f'🟦 PAPER TRADE OPENED – {ticker} at {entry_str}\n'
        f'Stop: {stop_str} | Target: {target_str}\n'
        f'Size: {size_str} | Risk: {risk_str} | Reward: {reward_str}'
    )
    send_paper_message(text)


def _notify_close(
    ticker: str,
    price: float,
    reason: str,
    entry_price: float | None = None,
    position_size: float | None = None,
) -> None:
    price_str = _format_price(price)
    entry_str = _format_price(entry_price)
    pnl_str = _format_pnl(entry_price, price, position_size)
    if reason == "STOP":
        header = f"🟥 PAPER TRADE CLOSED – Stop hit on {ticker}"
    else:
        header = f"🟩 PAPER TRADE CLOSED – Target hit on {ticker}"
    lines = [header, f'Entry: {entry_str} | Exit: {price_str}']
    if pnl_str != 'N/A':
        lines.append(f'P&L: {pnl_str}')
    send_paper_message('\n'.join(lines))


__all__ = [
    "process_open_trades",
    "open_paper_trade",
    "get_open_trade_count",
]
