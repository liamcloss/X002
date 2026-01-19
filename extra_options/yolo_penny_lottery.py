"""Standalone Penny Stock YOLO lottery module."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from trading_bot.market_data import cache
from trading_bot.paths import yolo_blocked_path, yolo_output_path

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


def run_yolo_penny_lottery(base_dir: Path | None = None, logger: logging.Logger | None = None) -> dict | None:
    """Run the weekly YOLO pick and persist the result."""

    root = base_dir or Path(__file__).resolve().parents[1]
    logger = logger or logging.getLogger("yolo_penny")
    week_of = _week_start(date.today())
    existing = _load_last_pick(root)
    if existing and existing.get("week_of") == week_of.isoformat():
        logger.info("YOLO pick already logged for week %s (%s); refreshing timestamp.", week_of, existing["ticker"])
        _write_pick(root, existing)
        return existing

    candidates = _gather_candidates(root, logger)
    if not candidates:
        logger.warning("No YOLO candidates found for week %s.", week_of)
        return None

    best = sorted(candidates, key=lambda item: item.score, reverse=True)[0]
    risk_reasons = _assess_pick_risk(best)
    if risk_reasons:
        blocked_payload = _build_blocked_payload(best, week_of, risk_reasons)
        _record_blocked_pick(root, blocked_payload)
        logger.warning(
            "YOLO pick blocked for week %s (%s); reasons: %s",
            week_of,
            best.ticker,
            "; ".join(risk_reasons),
        )
        return None
    payload = {
        "week_of": week_of.isoformat(),
        "ticker": best.ticker,
        "price": round(best.close, 4),
        "yolo_score": round(best.score, 4),
        "rationale": best.rationale,
        "stake_gbp": 2.00,
        "intent": "LOTTERY_TICKET",
    }
    _write_pick(root, payload)
    _append_ledger(root, payload)
    logger.info("YOLO penny pick for week %s: %s (score %.2f)", week_of, best.ticker, best.score)
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


def _append_ledger(base_dir: Path, entry: dict) -> None:
    path = _make_ledger_path(base_dir)
    history: list = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history = []
    history.append(entry)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_yolo_penny_lottery()
