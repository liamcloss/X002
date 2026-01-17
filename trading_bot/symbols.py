"""Symbol mapping helpers for external data providers."""

from __future__ import annotations

import re


_T212_MARKET_PATTERN = re.compile(
    r'^(?P<symbol>.+)_(?P<market>[A-Z]{2})_(?P<kind>[A-Z]+)$'
)
_T212_SUFFIX_PATTERN = re.compile(
    r'^(?P<symbol>.+)(?P<market>[a-z])_(?P<kind>[A-Z]+)$'
)
_T212_SIMPLE_PATTERN = re.compile(r'^(?P<symbol>.+)_(?P<kind>[A-Z]+)$')

_YFINANCE_SUFFIX_BY_CODE = {
    '_US': '',
    '_CA': '.TO',
    '_BE': '.BR',
    '_BB': '.BR',
    '_AT': '.VI',
    '_PT': '.LS',
    '_FR': '.PA',
    'l': '.L',
    'd': '.DE',
    'p': '.PA',
    's': '.SW',
    'm': '.MI',
    'a': '.AS',
    'e': '.MC',
}

_TRADINGVIEW_EXCHANGE_BY_CODE = {
    '_CA': 'TSX',
    '_BE': 'EBR',
    '_BB': 'EBR',
    '_AT': 'VIE',
    '_PT': 'ELI',
    '_FR': 'EPA',
    'l': 'LSE',
    'd': 'XETR',
    'p': 'EPA',
    's': 'SWX',
    'm': 'MIL',
    'a': 'AMS',
    'e': 'BME',
}


def _normalize_symbol(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None


def _parse_t212_ticker(ticker: str) -> tuple[str, str | None]:
    text = _normalize_symbol(ticker) or ''
    if not text:
        return '', None

    match = _T212_MARKET_PATTERN.match(text)
    if match:
        return match.group('symbol'), '_' + match.group('market')

    match = _T212_SUFFIX_PATTERN.match(text)
    if match:
        return match.group('symbol'), match.group('market')

    match = _T212_SIMPLE_PATTERN.match(text)
    if match:
        return match.group('symbol'), None

    return text, None


def _normalize_yfinance_symbol(symbol: str, suffix: str) -> str:
    if suffix and symbol.upper().endswith(suffix.upper()):
        return symbol
    if '.' in symbol:
        return symbol.replace('.', '-')
    return symbol


def yfinance_symbol(ticker: str, short_name: str | None = None) -> str:
    """Build the Yahoo Finance symbol for a Trading212 ticker."""

    base_symbol, market_code = _parse_t212_ticker(ticker)
    candidate = _normalize_symbol(short_name) or base_symbol or ticker
    candidate = candidate.strip().upper()
    suffix = _YFINANCE_SUFFIX_BY_CODE.get(market_code, '')
    normalized = _normalize_yfinance_symbol(candidate, suffix)
    if suffix and not normalized.upper().endswith(suffix.upper()):
        normalized = f'{normalized}{suffix}'
    return normalized


def tradingview_symbol(ticker: str, short_name: str | None = None) -> str:
    """Build the TradingView symbol for a Trading212 ticker."""

    base_symbol, market_code = _parse_t212_ticker(ticker)
    candidate = _normalize_symbol(short_name) or base_symbol or ticker
    candidate = candidate.strip().upper()
    exchange = _TRADINGVIEW_EXCHANGE_BY_CODE.get(market_code)
    if exchange:
        return f'{exchange}:{candidate}'
    return candidate


__all__ = [
    'tradingview_symbol',
    'yfinance_symbol',
]
