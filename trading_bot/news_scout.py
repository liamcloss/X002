"""News scout sidecar for capturing link-rich summaries."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

from trading_bot.paths import news_scout_output_path
from trading_bot.symbols import tradingview_symbol

SETUP_FILENAME = "SetupCandidates.json"
NEWS_SCOUT_LIMIT = 10


def run_news_scout(base_dir: Path, logger: logging.Logger, limit: int = NEWS_SCOUT_LIMIT) -> list[dict[str, Any]]:
    """Build a news link summary for the latest setup candidates."""

    setup_path = _find_latest_setup_file(base_dir)
    candidates = _load_setup_candidates(setup_path)
    if not candidates:
        logger.warning("News scout found no setup candidates (scan may not have run).")
        return []

    ranked = sorted(candidates, key=lambda item: _extract_scan_rank(item, 10**9))
    entries: list[dict[str, Any]] = []
    for candidate in ranked[:limit]:
        symbol = _extract_symbol(candidate)
        display = str(candidate.get("display_ticker") or "").strip() or None
        entry_price = _extract_value(candidate, ("planned_entry", "entry", "price"))
        stop_price = _extract_value(candidate, ("planned_stop", "stop"))
        target_price = _extract_value(candidate, ("planned_target", "target"))
        tradingview_url = str(
            candidate.get("tradingview_url") or _build_tradingview_url(symbol, display) or ""
        ).strip() or None
        links = _build_links(symbol, display, tradingview_url)
        entries.append(
            {
                "symbol": symbol or "UNKNOWN",
                "display_ticker": display,
                "scan_rank": _extract_scan_rank(candidate, 10**9),
                "reason": str(candidate.get("reason") or "").strip() or None,
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "tradingview_url": tradingview_url,
                "links": links,
            }
        )
    payload = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entries": entries,
    }
    output_dir = news_scout_output_path(base_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"news_scout_{timestamp}.json"
    path = output_dir / filename
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("News scout output written to %s", path)
    except OSError as exc:
        logger.error("Failed to persist news scout summary: %s", exc)
    return entries


def _find_latest_setup_file(base_dir: Path) -> Path | None:
    candidates = [
        base_dir / SETUP_FILENAME,
        base_dir / "outputs" / SETUP_FILENAME,
        base_dir / "data" / SETUP_FILENAME,
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        existing = list(base_dir.rglob(SETUP_FILENAME))
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _load_setup_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            return [item for item in candidates if isinstance(item, dict)]
    return []


def _extract_symbol(payload: dict[str, Any]) -> str | None:
    for key in ("symbol", "ticker", "base_ticker"):
        value = payload.get(key)
        if value:
            return str(value).strip()
    return None


def _extract_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric != numeric:
                continue
            return numeric
    return None


def _extract_scan_rank(payload: dict[str, Any], default: int) -> int:
    for key in ("scan_rank", "rank"):
        if key in payload:
            try:
                value = int(payload.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return default


def _build_tradingview_url(symbol: str | None, display: str | None) -> str | None:
    tv_symbol = None
    if display:
        tv_symbol = tradingview_symbol(display)
    if not tv_symbol and symbol:
        tv_symbol = tradingview_symbol(symbol)
    if not tv_symbol:
        return None
    safe_symbol = quote(tv_symbol.replace(" ", ""))
    if ":" in tv_symbol:
        return f"https://www.tradingview.com/chart/?symbol={safe_symbol}"
    return f"https://www.tradingview.com/symbols/{safe_symbol}/"


def _search_query(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace(" ", "+")


def _build_links(symbol: str | None, display: str | None, tv_url: str | None) -> list[dict[str, str]]:
    query = _search_query(display or symbol)
    links: list[dict[str, str]] = []
    if tv_url:
        links.append({"label": "Chart", "url": tv_url})
    news_url = _news_search_url(query)
    if news_url:
        links.append({"label": "News", "url": news_url})
    reddit_url = _reddit_search_url(query)
    if reddit_url:
        links.append({"label": "Reddit", "url": reddit_url})
    x_url = _x_search_url(query)
    if x_url:
        links.append({"label": "X", "url": x_url})
    threads_url = _threads_search_url(query)
    if threads_url:
        links.append({"label": "Threads", "url": threads_url})
    return links


def _news_search_url(query: str | None) -> str | None:
    if not query:
        return None
    return f"https://news.google.com/search?q={quote_plus(query + ' stock')}"


def _reddit_search_url(query: str | None) -> str | None:
    if not query:
        return None
    return f"https://www.reddit.com/search/?q={quote_plus(query)}&sort=top&t=week"


def _x_search_url(query: str | None) -> str | None:
    if not query:
        return None
    return f"https://x.com/search?q={quote_plus(query)}"


def _threads_search_url(query: str | None) -> str | None:
    if not query:
        return None
    return f"https://www.threads.net/search?q={quote_plus(query)}"
