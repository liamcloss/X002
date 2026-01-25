"""Shared helpers for assembling curated news URLs."""

from __future__ import annotations

from urllib.parse import quote_plus


def _normalize_symbol(query: str | None) -> str | None:
    if not query:
        return None
    candidate = query.strip()
    if not candidate:
        return None
    return candidate.replace(" ", "-").upper()


def _encode_query(query: str | None) -> str | None:
    if not query:
        return None
    candidate = query.strip()
    if not candidate:
        return None
    return quote_plus(candidate)


def news_links_for_query(query: str | None) -> list[dict[str, str]]:
    """Return curated news links for a ticker or search term."""

    encoded = _encode_query(query)
    symbol = _normalize_symbol(query)

    links: list[dict[str, str]] = []
    if symbol:
        links.append(
            {
                "label": "Yahoo Finance",
                "url": f"https://finance.yahoo.com/quote/{symbol}",
            }
        )
    if encoded:
        links.append(
            {
                "label": "Investing.com",
                "url": f"https://www.investing.com/search/?q={encoded}",
            }
        )
        links.append(
            {
                "label": "MarketBeat",
                "url": f"https://www.marketbeat.com/search/?s={encoded}",
            }
        )
    return links


def first_news_link(query: str | None) -> str | None:
    """Return a single fallback news URL (used where only one link is wanted)."""

    candidates = news_links_for_query(query)
    if not candidates:
        return None
    return candidates[0]["url"]
