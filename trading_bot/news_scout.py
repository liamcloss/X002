"""News scout sidecar for capturing link-rich summaries."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

import os
from contextlib import suppress
import requests
import textwrap

from src.market_data import yfinance_client

from trading_bot import config as bot_config
from trading_bot.news_links import news_links_for_query
from trading_bot.paths import news_scout_output_path
from trading_bot.pretrade.spread_gate import SpreadGate
from trading_bot.symbols import tradingview_symbol, yfinance_symbol

SETUP_FILENAME = "SetupCandidates.json"
NEWS_SCOUT_LIMIT = 10
LLM_MAX_TOKENS = 200
LLM_TEMPERATURE = 0.2
AI_HEADER = "News scout AI sentiment"

NEWS_SCOUT_LLM_CONFIG = bot_config.CONFIG.get("news_scout", {})
NEWS_SCOUT_LLM_ENABLED = NEWS_SCOUT_LLM_CONFIG.get("llm_enabled", False)
NEWS_SCOUT_LLM_MODEL = NEWS_SCOUT_LLM_CONFIG.get("llm_model", "gpt-3.5-turbo")
NEWS_SCOUT_API_KEY_ENV = NEWS_SCOUT_LLM_CONFIG.get("api_key_env", "OPENAI_API_KEY")

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
        entry_price = _round_price(_extract_value(candidate, ("planned_entry", "entry", "price")))
        stop_price = _round_price(_extract_value(candidate, ("planned_stop", "stop")))
        target_price = _round_price(_extract_value(candidate, ("planned_target", "target")))
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
                "currency_code": candidate.get("currency_code"),
                "currency_symbol": candidate.get("currency_symbol"),
            }
        )
    entries = _enrich_with_llm(entries, base_dir, logger)
    entries, filtered_out, rejection_summary = _filter_entries_by_spread(entries, logger)
    if filtered_out:
        logger.info(
            'News scout filtered %d entries due to spread gate (%s).',
            len(filtered_out),
            "; ".join(rejection_summary),
        )
    if not entries:
        logger.warning('News scout produced no entries after spread filter.')
    payload = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entries": entries,
        "filtered_out_count": len(filtered_out),
        "filtered_reasons": rejection_summary,
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
    news_links = news_links_for_query(query)
    links.extend(news_links)
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


def _enrich_with_llm(
    entries: list[dict[str, Any]],
    base_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    if not NEWS_SCOUT_LLM_ENABLED:
        return entries
    api_key = os.getenv(NEWS_SCOUT_API_KEY_ENV)
    if not api_key:
        logger.info(
            'News scout LLM disabled (missing API key in env %s).',
            NEWS_SCOUT_API_KEY_ENV,
        )
        return entries
    logger.info(
        'News scout LLM enrichment requested (model=%s, entries=%s).',
        NEWS_SCOUT_LLM_MODEL,
        len(entries),
    )
    prompt = _build_llm_prompt(entries)
    if not prompt:
        logger.info('News scout LLM prompt empty, skipping LLM enrichment.')
        return entries
    logger.info('News scout LLM call starting (model=%s).', NEWS_SCOUT_LLM_MODEL)
    try:
        response_text = _call_openai(prompt, api_key)
    except Exception as exc:
        logger.warning('News scout LLM call failed: %s', exc)
        _log_troubleshoot_issue(
            base_dir,
            f'LLM call failed: {exc}',
            'Check outbound access to https://api.openai.com:443 and validate OPENAI_API_KEY.',
        )
        return entries
    try:
        insights = _parse_llm_response(response_text, logger)
    except Exception as exc:
        logger.warning('News scout LLM response parse failed: %s', exc)
        _log_troubleshoot_issue(
            base_dir,
            f'LLM response parse failed: {exc}',
            'Inspect the raw LLM reply in logs or retry the prompt; ensure JSON-only responses.',
        )
        _store_llm_response_failure(base_dir, prompt, response_text)
        return entries
    if not insights:
        logger.info('News scout LLM returned no insights.')
        return entries
    for entry in entries:
        symbol = entry.get('symbol')
        if symbol and symbol in insights:
            entry['llm_insight'] = insights[symbol]
    logger.info(
        'News scout LLM applied insights to %d entries: %s',
        sum(1 for entry in entries if entry.get('llm_insight')),
        ', '.join(sorted(insights.keys())),
    )
    return entries


def _build_llm_prompt(entries: list[dict[str, Any]]) -> str | None:
    if not entries:
        return None
    lines = []
    for entry in entries[:5]:
        symbol = entry.get('symbol') or 'UNKNOWN'
        reason = entry.get('reason') or 'No explicit reason provided'
        links = [f"{link.get('label')}: {link.get('url')}" for link in entry.get('links') or [] if link.get('url')]
        link_block = "; ".join(links[:3])
        lines.append(f"{symbol} — {reason}. Links: {link_block}")
    body = "\n".join(lines)
    prompt = (
        f"{AI_HEADER}\n"
        "You receive a series of market setup summaries. "
        "For each symbol, reply with a JSON array of objects, "
        "each containing 'symbol', 'sentiment' (positive/neutral/negative), "
        "'highlight' (<=60 chars), and 'callout' (short note). "
        "If you cannot provide insight, set sentiment to 'neutral'. "
        "JSON only, no extra text.\n"
        f"Setups:\n{body}"
    )
    return prompt


def _call_openai(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": NEWS_SCOUT_LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You summarize speculative setups carefully."},
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved_env: dict[str, str] = {}
    for key in proxy_keys:
        saved_env[key] = os.environ.pop(key, None)
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    finally:
        for key, value in saved_env.items():
            if value is None:
                with suppress(KeyError):
                    del os.environ[key]
            else:
                os.environ[key] = value


def _parse_llm_response(text: str, logger: logging.Logger) -> dict[str, str]:
    snippet = _extract_json_array(text)
    if snippet is None:
        raise ValueError("LLM response missing JSON array")
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:
        logger.warning('News scout LLM response invalid JSON: %s', exc)
        return {}
    insights: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        sentiment = str(item.get("sentiment") or "").strip()
        highlight = str(item.get("highlight") or "").strip()
        callout = str(item.get("callout") or "").strip()
        if symbol:
            parts = [sentiment] if sentiment else []
            if highlight:
                parts.append(highlight)
            if callout:
                parts.append(callout)
            insights[symbol] = " | ".join(part for part in parts if part)
    return insights


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _filter_entries_by_spread(
    entries: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    gate = SpreadGate(logger=logger)
    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for entry in entries:
        symbol = entry.get("symbol")
        display = entry.get("display_ticker")
        quote_symbol = _resolve_quote_symbol(symbol, display)
        if not quote_symbol:
            reason = "Symbol mapping failed"
            entry["spread_reason"] = reason
            rejected.append(entry)
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        quote = yfinance_client.get_quote(quote_symbol)
        entry["quote_symbol"] = quote_symbol
        entry["spread_pct"] = quote.get("spread")
        passes, reason = gate.evaluate(quote)
        entry["spread_reason"] = reason
        if passes:
            passed.append(entry)
        else:
            rejected.append(entry)
            key = reason or "Spread gate rejected"
            reasons[key] = reasons.get(key, 0) + 1
    summary = [
        f"{count}x {text}" for text, count in sorted(reasons.items(), key=lambda item: -item[1])
    ]
    return passed, rejected, summary


def _resolve_quote_symbol(symbol: str | None, display: str | None) -> str | None:
    if not symbol:
        return None
    short_name = _short_display(display, symbol)
    return yfinance_symbol(symbol, short_name=short_name)


def _short_display(display: str | None, symbol: str) -> str | None:
    if not display:
        return None
    candidate = display.strip()
    if not candidate or " " in candidate:
        return None
    if candidate.upper() == symbol.upper():
        return None
    return candidate
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


def _log_troubleshoot_issue(base_dir: Path, issue: str, suggestion: str) -> None:
    path = base_dir / "logs" / "news_scout_troubleshoot.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"{timestamp} | {issue} | Suggestion: {suggestion}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def _store_llm_response_failure(base_dir: Path, prompt: str, response: str) -> None:
    path = base_dir / "logs" / "news_scout_llm_failures.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    shortened_prompt = textwrap.shorten(prompt, width=400, placeholder="…")
    shortened_response = textwrap.shorten(response, width=400, placeholder="…")
    entry = (
        f"{timestamp} | Prompt: {shortened_prompt} | Response: {shortened_response}\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def _round_price(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)
