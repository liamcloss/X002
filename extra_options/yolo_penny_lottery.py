"""Standalone Penny Stock YOLO lottery module."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from trading_bot.market_data import cache
from trading_bot.paths import yolo_blocked_path, yolo_output_path

from extra_options.yolo_positions import (
    DEFAULT_STAKE_GBP,
    append_position,
    close_position_entry,
    entry_open_after_week,
    find_open_position,
    latest_close,
    load_positions,
    write_positions,
)

YOLO_PICK_FILENAME = "YOLO_Pick.json"
YOLO_LEDGER_FILENAME = "YOLO_Ledger.json"
MIN_PRICE = 0.20
MAX_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5_000_000
MIN_SHARE_VOLUME = 1_000_000
RISK_CLOSE_MOVE = 1.0
RISK_INTRADAY_RANGE = 0.5
RISK_HIGH_VOLUME_SPIKE = 25.0
RISK_LOW_ATR_RATIO = 0.08
RISK_MICRO_PRICE = 0.5
ALTERNATIVE_CANDIDATE_COUNT = 3


@dataclass(frozen=True)
class CandidateMetrics:
    ticker: str
    close: float
    volume_spike_ratio: float
    max_intraday_range: float
    max_close_move: float
    atr_ratio: float
    recent_high_breakout: bool
    rationale: List[str]
    score: float


def run_yolo_penny_lottery(
    base_dir: Path | None = None,
    logger: logging.Logger | None = None,
    reroll: bool = False,
    close_ticker: str | None = None,
) -> dict | None:
    """Run the weekly YOLO pick and persist the result."""

    root = base_dir or Path(__file__).resolve().parents[1]
    logger = logger or logging.getLogger("yolo_penny")
    if close_ticker:
        return _close_position(root, close_ticker, logger)
    week_of = _week_start(date.today())
    existing = _load_last_pick(root)
    if existing and existing.get("week_of") == week_of.isoformat():
        if reroll:
            logger.info("YOLO reroll requested for week %s; generating a replacement pick.", week_of)
        else:
            logger.info("YOLO pick already logged for week %s (%s); refreshing timestamp.", week_of, existing["ticker"])
            _write_pick(root, existing)
            return existing

    candidates = _gather_candidates(root, logger)
    if not candidates:
        logger.warning("No YOLO candidates found for week %s.", week_of)
        return None

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    blocked_tickers: list[str] = []
    selected_index: int | None = None
    selected_candidate: CandidateMetrics | None = None
    for index, candidate in enumerate(ranked):
        risk_reasons = _assess_pick_risk(candidate)
        if risk_reasons:
            blocked_payload = _build_blocked_payload(candidate, week_of, risk_reasons)
            _record_blocked_pick(root, blocked_payload)
            logger.warning(
                "YOLO pick blocked for week %s (%s); reasons: %s",
                week_of,
                candidate.ticker,
                "; ".join(risk_reasons),
            )
            blocked_tickers.append(candidate.ticker)
            continue
        selected_candidate = candidate
        selected_index = index
        break

    if not selected_candidate:
        logger.warning(
            "No YOLO candidate passed risk checks for week %s; %s blocked entries logged.",
            week_of,
            len(blocked_tickers),
        )
        return None

    alternatives = []
    for index, candidate in enumerate(ranked):
        if index == selected_index:
            continue
        if len(alternatives) >= ALTERNATIVE_CANDIDATE_COUNT:
            break
        if candidate.ticker in blocked_tickers:
            continue
        alternatives.append(_serialize_candidate(candidate))
    entry_tuple = entry_open_after_week(root, week_of, selected_candidate.ticker)
    fallback_date = week_of + timedelta(days=1)
    entry_date, entry_price = entry_tuple if entry_tuple else (fallback_date, selected_candidate.close)
    entry_date_iso = entry_date.isoformat()
    payload = {
        "week_of": week_of.isoformat(),
        "ticker": selected_candidate.ticker,
        "price": round(selected_candidate.close, 4),
        "yolo_score": round(selected_candidate.score, 4),
        "rationale": selected_candidate.rationale,
        "stake_gbp": 2.00,
        "intent": "LOTTERY_TICKET",
        "reroll": reroll,
        "alternatives": alternatives,
    }
    _write_pick(root, payload)
    _append_ledger(root, payload, overwrite_week=reroll)
    _record_open_position(
        root,
        week_of.isoformat(),
        selected_candidate.ticker,
        entry_date_iso,
        entry_price,
    )
    logger.info(
        "YOLO penny pick for week %s: %s (score %.2f)",
        week_of,
        selected_candidate.ticker,
        selected_candidate.score,
    )
    return payload


def _week_start(today: date) -> date:
    return today - timedelta(days=today.weekday())


def _make_pick_path(base_dir: Path) -> Path:
    return yolo_output_path(base_dir, YOLO_PICK_FILENAME)


def _make_ledger_path(base_dir: Path) -> Path:
    return yolo_output_path(base_dir, YOLO_LEDGER_FILENAME)


def _load_last_pick(base_dir: Path) -> dict | None:
    path = _make_pick_path(base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _gather_candidates(base_dir: Path, logger: logging.Logger) -> list[CandidateMetrics]:
    universe_path = base_dir / "universe" / "clean" / "universe.parquet"
    if not universe_path.exists():
        logger.warning("Universe file missing; YOLO lottery aborted.")
        return []

    df = pd.read_parquet(universe_path)
    if "ticker" not in df.columns:
        logger.warning("Universe missing ticker column; YOLO lottery aborted.")
        return []

    tickers = [
        str(value)
        for value in df["ticker"]
        if isinstance(value, str) and value.endswith("_US_EQ") and "OTC" not in value
    ]

    prices_dir = base_dir / "data" / "prices"
    metrics: list[CandidateMetrics] = []

    for ticker in tickers:
        candidate = _evaluate_ticker(ticker, prices_dir, logger)
        if candidate:
            metrics.append(candidate)
    return metrics


def _evaluate_ticker(ticker: str, prices_dir: Path, logger: logging.Logger) -> CandidateMetrics | None:
    df = cache.load_cache(prices_dir, ticker, logger)
    if df.empty:
        return None
    df = df.sort_index()
    if len(df) < 30:
        return None

    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()
    volume = df["Volume"].dropna()
    if len(close) < 30 or len(volume) < 30:
        return None

    price = float(close.iloc[-1])
    if price < MIN_PRICE or price > MAX_PRICE:
        return None

    recent_volume = volume.tail(30)
    avg_volume = recent_volume.mean()
    avg_dollar_volume = (close.tail(30) * recent_volume).mean()
    if avg_volume < MIN_SHARE_VOLUME or avg_dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    volume_ratio = recent_volume / avg_volume
    max_spike = float(volume_ratio.max())
    if max_spike < 5:
        return None

    max_range = float(((high - low) / low).replace([pd.NA, pd.NaT, float("nan")], 0.0).tail(30).max())
    max_close_move = float(close.pct_change().abs().tail(30).max())
    if max_range < 0.25 and max_close_move < 0.2:
        return None

    top_decile = volume_ratio.quantile(0.9)
    top_count = int((volume_ratio >= top_decile).sum())
    doubled = _price_doubled(close)
    atr14 = _compute_atr(df, 14)
    atr14_latest = _last_valid(atr14)
    atr_ratio = float(atr14_latest / price) if atr14_latest and price else 0.0
    narrative = top_count >= 2 or doubled or atr_ratio >= 0.15
    if not narrative:
        return None

    if _reverse_splited(close.tail(30)):
        return None

    if _flatline_after_spike(recent_volume, avg_volume):
        return None

    recent_high = high.tail(30).max()
    breakout = price > recent_high
    score = (
        0.4 * max_spike
        + 0.3 * max_range
        + 0.2 * atr_ratio
        + 0.1 * (1.0 if breakout else 0.0)
    )

    rationale = []
    rationale.append(f"{max_spike:.1f}x volume spike")
    if max_range >= 0.25:
        rationale.append(f"{max_range*100:.0f}% intraday range")
    if max_close_move >= 0.2:
        rationale.append(f"{max_close_move*100:.0f}% close move")
    rationale.append(f"ATR {atr_ratio*100:.0f}% of price")
    if breakout:
        rationale.append("Recent high breakout")

    return CandidateMetrics(
        ticker=ticker,
        close=price,
        volume_spike_ratio=max_spike,
        max_intraday_range=max_range,
        max_close_move=max_close_move,
        atr_ratio=atr_ratio,
        recent_high_breakout=breakout,
        rationale=rationale,
        score=score,
    )


def _price_doubled(close: pd.Series) -> bool:
    if len(close) < 90:
        return False
    rolling_min = close.rolling(window=90, min_periods=90).min()
    comparison = close / rolling_min
    return (comparison.tail(90) >= 2.0).any()


def _reverse_splited(series: pd.Series) -> bool:
    pct_change = series.pct_change()
    return (pct_change >= 8.0).any()


def _flatline_after_spike(volume: pd.Series, avg_volume: float) -> bool:
    if volume.empty or avg_volume == 0:
        return False
    spike_idx = volume.idxmax()
    idx_pos = volume.index.get_loc(spike_idx)
    following = volume.iloc[idx_pos + 1 : idx_pos + 4]
    if following.empty:
        return False
    return (following <= avg_volume * 0.5).all()


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _last_valid(series: pd.Series) -> float | None:
    if series.empty:
        return None
    for value in reversed(series.tolist()):
        if pd.notna(value):
            return float(value)
    return None


def _write_pick(base_dir: Path, payload: dict) -> None:
    path = _make_pick_path(base_dir)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_ledger(base_dir: Path, entry: dict, overwrite_week: bool = False) -> None:
    history = _load_ledger_entries(base_dir)
    if overwrite_week:
        week = entry.get("week_of")
        if week:
            history = [item for item in history if item.get("week_of") != week]
    history.append(entry)
    path = _make_ledger_path(base_dir)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _load_ledger_entries(base_dir: Path) -> list[dict]:
    path = yolo_output_path(base_dir, YOLO_LEDGER_FILENAME)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _blocked_path(base_dir: Path) -> Path:
    return yolo_blocked_path(base_dir)


def _load_blocked_history(base_dir: Path) -> list[dict]:
    path = _blocked_path(base_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return payload


def _record_blocked_pick(base_dir: Path, entry: dict) -> None:
    history = _load_blocked_history(base_dir)
    history.append(entry)
    path = _blocked_path(base_dir)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _build_blocked_payload(
    candidate: CandidateMetrics,
    week_of: date,
    reasons: list[str],
) -> dict:
    return {
        "week_of": week_of.isoformat(),
        "ticker": candidate.ticker,
        "price": round(candidate.close, 4),
        "yolo_score": round(candidate.score, 4),
        "rationale": candidate.rationale,
        "stake_gbp": 2.00,
        "intent": "LOTTERY_TICKET",
        "blocked_reasons": reasons,
        "blocked_at": datetime.now(timezone.utc).isoformat(),
    }


def _assess_pick_risk(candidate: CandidateMetrics) -> list[str]:
    reasons: list[str] = []
    if candidate.max_close_move >= RISK_CLOSE_MOVE:
        reasons.append("Massive close move (>100%)")
    if candidate.max_intraday_range >= RISK_INTRADAY_RANGE and not candidate.recent_high_breakout:
        reasons.append("Extreme intraday range without breakout")
    if candidate.volume_spike_ratio >= RISK_HIGH_VOLUME_SPIKE and candidate.atr_ratio < RISK_LOW_ATR_RATIO:
        reasons.append("Huge spike but ATR still muted")
    if candidate.close <= RISK_MICRO_PRICE:
        reasons.append("Micro-cap price subject to manipulation")
    if candidate.volume_spike_ratio >= RISK_HIGH_VOLUME_SPIKE and candidate.max_close_move >= 1.5:
        reasons.append("Explosive spike coupled with outsized close move")
    return list(dict.fromkeys(reasons))


def _serialize_candidate(candidate: CandidateMetrics) -> dict[str, Any]:
    return {
        "ticker": candidate.ticker,
        "price": round(candidate.close, 4),
        "yolo_score": round(candidate.score, 4),
        "rationale": candidate.rationale,
    }


def _record_open_position(
    base_dir: Path,
    week_of: str,
    ticker: str,
    entry_date: str,
    entry_price: float,
) -> None:
    if not entry_date or not entry_price:
        return
    if entry_price <= 0:
        return
    entries = load_positions(base_dir)
    filtered = [
        entry
        for entry in entries
        if not (
            entry.get("week_of") == week_of
            and str(entry.get("ticker", "")).strip().upper() == ticker.strip().upper()
            and entry.get("status") == "OPEN"
        )
    ]
    shares = DEFAULT_STAKE_GBP / entry_price
    append_position(filtered, week_of, ticker, entry_date, entry_price, shares)
    write_positions(base_dir, filtered)


def _close_position(
    base_dir: Path,
    ticker: str,
    logger: logging.Logger,
) -> dict | None:
    entries = load_positions(base_dir)
    entry = find_open_position(entries, ticker)
    if not entry:
        logger.info("No open YOLO position recorded for %s.", ticker)
        return None
    close_info = latest_close(base_dir, ticker)
    if not close_info:
        logger.warning("Cannot close YOLO position %s; price data unavailable.", ticker)
        return None
    close_date, close_price = close_info
    close_position_entry(entry, close_date, close_price)
    write_positions(base_dir, entries)
    logger.info(
        "Closed YOLO position for %s at %.4f on %s (value %.2f).",
        ticker,
        close_price,
        close_date,
        entry.get("close_value"),
    )
    return entry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_yolo_penny_lottery()
