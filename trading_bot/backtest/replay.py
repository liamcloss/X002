"""Historical replay/backtest engine."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, TypedDict
from urllib.parse import quote

import pandas as pd

from trading_bot import config
from trading_bot.logging_setup import setup_logging
from trading_bot.market_data import cache
from trading_bot.messaging.format_message import format_daily_scan_message
from trading_bot.signals import apply_filters, detect_pullback, find_risk_geometry, rank_candidates
from trading_bot.symbols import tradingview_symbol
from trading_bot.state import (
    add_pullback,
    cleanup_expired_cooldowns,
    increment_pullback_day,
    invalidate_pullback,
    in_cooldown,
    is_alerted,
    mark_alerted,
    reset_pullback,
)
from trading_bot.state.schema import default_state


class ReplayCandidate(TypedDict):
    ticker: str
    currency_symbol: str
    momentum_5d: float
    reason: str
    volume_multiple: float
    price: float
    currency_code: str
    pct_from_20d_high: float
    stop_pct: float
    stop_price: float
    target_pct: float
    target_price: float
    rr: float
    position_size: int
    risk_gbp: float
    reward_gbp: float
    isa_eligible: bool
    tradingview_url: str


@dataclass(frozen=True)
class ReplayUniverseItem:
    ticker: str
    currency_code: str
    isa_eligible: bool


@dataclass(frozen=True)
class ReplayDiagnostics:
    scanned: int
    passed_filters: int
    valid_pullbacks: int
    final_ranked: int


@dataclass(frozen=True)
class RankedCandidate:
    ticker: str
    score: float
    stop_pct: float
    target_pct: float


_CURRENCY_SYMBOLS = {
    "GBP": "£",
    "GBX": "£",
    "USD": "$",
    "EUR": "€",
}


def run_replay(days: int, start_date: Optional[str] = None) -> None:
    """
    Runs historical replay for N trading days.
    Writes outputs to backtest/outputs/.
    """

    base_dir = Path(__file__).resolve().parents[2]
    logger = _ensure_logger(base_dir)

    universe_path = base_dir / "universe" / "clean" / "universe.parquet"
    prices_dir = base_dir / "data" / "prices"

    _ensure_prerequisites(universe_path, prices_dir)

    replay_dates = _build_replay_dates(days, start_date)
    universe = _load_universe(universe_path)
    ticker_map = {item.ticker: item for item in universe}
    prices_cache: dict[str, pd.DataFrame] = {}

    output_root = Path(__file__).resolve().parent / "outputs"
    daily_dir = output_root / "daily"
    summary_dir = output_root / "summaries"
    daily_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    state = default_state()
    daily_files: list[Path] = []

    for replay_date in replay_dates:
        logger.info("Replay scan for %s", replay_date.isoformat())
        cleanup_expired_cooldowns(state, replay_date)

        scan_result = _scan_day(
            replay_date=replay_date,
            universe=universe,
            ticker_map=ticker_map,
            prices_dir=prices_dir,
            prices_cache=prices_cache,
            state=state,
            logger=logger,
        )

        message = _format_replay_message(
            replay_date=replay_date,
            candidates=scan_result.candidates,
            diagnostics=scan_result.diagnostics,
            ranked_candidates=scan_result.ranked_candidates,
        )
        output_path = daily_dir / f"{replay_date.isoformat()}.txt"
        output_path.write_text(message, encoding="utf-8")
        daily_files.append(output_path)

        for candidate in scan_result.candidates:
            mark_alerted(state, candidate["ticker"], replay_date)

    _write_summary(summary_dir, replay_dates, daily_files)
    _write_state_snapshot(output_root, state)


@dataclass(frozen=True)
class ScanResult:
    candidates: list[ReplayCandidate]
    diagnostics: ReplayDiagnostics
    ranked_candidates: list[RankedCandidate]


def _ensure_logger(base_dir: Path) -> logging.Logger:
    logger = logging.getLogger("trading_bot")
    if not logger.handlers:
        return setup_logging(base_dir / "logs")
    return logger


def _ensure_prerequisites(universe_path: Path, prices_dir: Path) -> None:
    if not universe_path.exists():
        raise RuntimeError(f"Universe file missing: {universe_path}")
    if not prices_dir.exists():
        raise RuntimeError(f"Market data cache missing: {prices_dir}")
    if not any(prices_dir.glob("*.parquet")):
        raise RuntimeError(f"Market data cache is empty: {prices_dir}")


def _load_universe(universe_path: Path) -> list[ReplayUniverseItem]:
    df = pd.read_parquet(universe_path)
    if df.empty or "ticker" not in df.columns:
        raise RuntimeError("Universe file is empty or missing ticker column.")

    if "currency_code" not in df.columns:
        df["currency_code"] = "GBP"

    if "isa_eligible" not in df.columns:
        df["isa_eligible"] = False

    df = df.sort_values(by="ticker")
    return [
        ReplayUniverseItem(
            ticker=str(row["ticker"]).strip().upper(),
            currency_code=str(row["currency_code"]).strip().upper(),
            isa_eligible=bool(row["isa_eligible"]),
        )
        for _, row in df.iterrows()
    ]


def _build_replay_dates(days: int, start_date: Optional[str]) -> list[date]:
    if days <= 0:
        raise ValueError("Days must be a positive integer.")

    if start_date:
        start = _parse_date(start_date)
        return _forward_trading_days(start, days)

    end = date.today() - timedelta(days=1)
    return _backward_trading_days(end, days)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("start_date must be in YYYY-MM-DD format") from exc


def _forward_trading_days(start: date, days: int) -> list[date]:
    dates: list[date] = []
    current = start
    while len(dates) < days:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _backward_trading_days(end: date, days: int) -> list[date]:
    dates: list[date] = []
    current = end
    while len(dates) < days:
        if current.weekday() < 5:
            dates.append(current)
        current -= timedelta(days=1)
    return sorted(dates)


def _scan_day(
    replay_date: date,
    universe: list[ReplayUniverseItem],
    ticker_map: dict[str, ReplayUniverseItem],
    prices_dir: Path,
    prices_cache: dict[str, pd.DataFrame],
    state: dict,
    logger: logging.Logger,
) -> ScanResult:
    scanned = len(universe)
    passed_filters = 0
    valid_pullbacks = 0

    candidate_rows: list[dict[str, float | str]] = []
    candidate_metadata: dict[str, dict[str, float | str]] = {}

    for item in universe:
        ticker = item.ticker
        price_df = _load_price_history(prices_dir, ticker, prices_cache, logger)
        if price_df.empty:
            continue
        history = _slice_history(price_df, replay_date)
        if history.empty:
            continue

        pullback = detect_pullback(history)
        _update_pullback_state(state, ticker, pullback, replay_date)

        filtered = apply_filters(history, max_pct_from_20d_high=0.05)
        if filtered.empty:
            continue

        passed_filters += 1
        if not pullback.get("in_pullback"):
            continue

        valid_pullbacks += 1
        if in_cooldown(state, ticker, replay_date) or is_alerted(state, ticker):
            continue

        last_row = filtered.iloc[-1]
        price = float(last_row.get("close_price", last_row.get("close", 0.0)))
        geometry = find_risk_geometry(price)
        if not geometry:
            continue

        candidate_rows.append(
            {
                "ticker": ticker,
                "volume_multiple": float(last_row.get("volume_multiple", 0.0)),
                "pct_from_20d_high": float(last_row.get("pct_from_20d_high", 0.0)),
                "momentum_5d": float(last_row.get("momentum_5d", 0.0)),
            }
        )
        candidate_metadata[ticker] = {
            "price": price,
            "stop_pct": float(geometry["stop_pct"]),
            "target_pct": float(geometry["target_pct"]),
            "rr": float(geometry["rr"]),
        }

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        diagnostics = ReplayDiagnostics(
            scanned=scanned,
            passed_filters=passed_filters,
            valid_pullbacks=valid_pullbacks,
            final_ranked=0,
        )
        return ScanResult(candidates=[], diagnostics=diagnostics, ranked_candidates=[])

    ranked = rank_candidates(candidate_df)
    ranked = ranked.sort_values(by=["score", "ticker"], ascending=[False, True])
    final_ranked = int(ranked.shape[0])

    top_ranked = ranked.head(3)
    candidates = [
        _build_candidate(
            row=row,
            meta=candidate_metadata[row["ticker"]],
            universe_item=ticker_map[row["ticker"]],
        )
        for _, row in top_ranked.iterrows()
    ]

    ranked_candidates = [
        RankedCandidate(
            ticker=str(row["ticker"]),
            score=float(row["score"]),
            stop_pct=float(candidate_metadata[row["ticker"]]["stop_pct"]),
            target_pct=float(candidate_metadata[row["ticker"]]["target_pct"]),
        )
        for _, row in top_ranked.iterrows()
    ]

    diagnostics = ReplayDiagnostics(
        scanned=scanned,
        passed_filters=passed_filters,
        valid_pullbacks=valid_pullbacks,
        final_ranked=final_ranked,
    )
    return ScanResult(
        candidates=candidates,
        diagnostics=diagnostics,
        ranked_candidates=ranked_candidates,
    )


def _load_price_history(
    prices_dir: Path,
    ticker: str,
    cache_map: dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> pd.DataFrame:
    if ticker in cache_map:
        return cache_map[ticker]
    df = cache.load_cache(prices_dir, ticker, logger)
    if not df.empty:
        df = df.sort_index()
    cache_map[ticker] = df
    return df


def _slice_history(df: pd.DataFrame, replay_date: date) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp(replay_date)
    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    return normalized[normalized.index.normalize() <= cutoff]


def _update_pullback_state(
    state: dict,
    ticker: str,
    pullback: dict,
    replay_date: date,
) -> None:
    pullback_pct = float(pullback.get("pullback_pct", 0.0))
    in_pullback = bool(pullback.get("in_pullback"))

    if in_pullback:
        if ticker in state.get("pullbacks", {}):
            increment_pullback_day(state, ticker)
        else:
            add_pullback(state, ticker, float(pullback.get("breakout_high", 0.0)), replay_date)
        return

    if ticker not in state.get("pullbacks", {}):
        return

    if pullback_pct > 0.05:
        invalidate_pullback(state, ticker, replay_date)
    elif pullback_pct < 0.01:
        reset_pullback(state, ticker, replay_date)
    else:
        increment_pullback_day(state, ticker)


def _build_candidate(
    row: pd.Series,
    meta: dict[str, float | str],
    universe_item: ReplayUniverseItem,
) -> ReplayCandidate:
    price = float(meta["price"])
    stop_pct = float(meta["stop_pct"])
    target_pct = float(meta["target_pct"])
    position_size = _position_size_for_risk(stop_pct)
    currency_code = universe_item.currency_code

    stop_price = price * (1 - stop_pct)
    target_price = price * (1 + target_pct)
    risk_gbp = round(position_size * stop_pct, 2)
    reward_gbp = round(position_size * target_pct, 2)

    return {
        "ticker": str(row["ticker"]),
        "currency_symbol": _currency_symbol(currency_code),
        "momentum_5d": float(row["momentum_5d"]),
        "reason": "Pullback",
        "volume_multiple": round(float(row["volume_multiple"]), 2),
        "price": price,
        "currency_code": currency_code,
        "pct_from_20d_high": float(row["pct_from_20d_high"]),
        "stop_pct": stop_pct,
        "stop_price": stop_price,
        "target_pct": target_pct,
        "target_price": target_price,
        "rr": float(meta["rr"]),
        "position_size": position_size,
        "risk_gbp": risk_gbp,
        "reward_gbp": reward_gbp,
        "isa_eligible": universe_item.isa_eligible,
        "tradingview_url": _tradingview_url(str(row["ticker"])),
    }


def _position_size_for_risk(stop_pct: float) -> int:
    live_mode_config = config.CONFIG["live_mode"]
    if stop_pct <= 0:
        return int(live_mode_config["position_min"])
    target_size = live_mode_config["max_risk"] / stop_pct
    bounded = max(
        live_mode_config["position_min"],
        min(live_mode_config["position_max"], target_size),
    )
    return int(round(bounded))


def _currency_symbol(currency_code: str) -> str:
    return _CURRENCY_SYMBOLS.get(currency_code.upper(), "")


def _tradingview_url(ticker: str) -> str:
    symbol = tradingview_symbol(ticker)
    if not symbol:
        return ''
    safe_symbol = quote(symbol.replace(' ', ''))
    if ':' in symbol:
        return f'https://www.tradingview.com/chart/?symbol={safe_symbol}'
    return f'https://www.tradingview.com/symbols/{safe_symbol}/'


def _format_replay_message(
    replay_date: date,
    candidates: list[ReplayCandidate],
    diagnostics: ReplayDiagnostics,
    ranked_candidates: list[RankedCandidate],
) -> str:
    date_str = replay_date.isoformat()
    message = format_daily_scan_message(
        date=date_str,
        mode=config.CONFIG["mode"],
        candidates=candidates,
        scanned_count=diagnostics.scanned,
        valid_count=diagnostics.final_ranked,
        data_as_of=date_str,
        generated_at=date_str,
        dry_run=False,
    )

    lines = message.splitlines()
    if lines:
        lines[0] = lines[0].replace("📊 Trade Candidates", "📊 REPLAY – Trade Candidates")
    if len(lines) > 1:
        lines[1] = f"MODE: {config.CONFIG['mode']} (Replay)"

    lines.append("")
    lines.append("--- Diagnostics ---")
    lines.append(f"Scanned instruments: {diagnostics.scanned}")
    lines.append(f"Passed hard filters: {diagnostics.passed_filters}")
    lines.append(f"Valid pullbacks: {diagnostics.valid_pullbacks}")
    lines.append(f"Final ranked candidates: {diagnostics.final_ranked}")

    if ranked_candidates:
        lines.append("")
        lines.append("Top 3 scores:")
        for ranked in ranked_candidates:
            lines.append(
                "- "
                f"{ranked.ticker}: {ranked.score:.4f} "
                f"(Stop {ranked.stop_pct * 100:.2f}%, Target {ranked.target_pct * 100:.2f}%)"
            )

    return "\n".join(lines)


def _write_summary(summary_dir: Path, replay_dates: list[date], daily_files: list[Path]) -> None:
    if not replay_dates:
        return
    start = replay_dates[0].isoformat()
    end = replay_dates[-1].isoformat()
    header = ["REPLAY SUMMARY", f"Period: {start} → {end}", ""]

    contents: list[str] = []
    for path in daily_files:
        contents.append(path.read_text(encoding="utf-8"))

    summary_path = summary_dir / f"replay_{start}_to_{end}.txt"
    summary_path.write_text("\n\n".join(header + contents), encoding="utf-8")


def _write_state_snapshot(output_root: Path, state: dict) -> None:
    state_path = output_root / "state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
