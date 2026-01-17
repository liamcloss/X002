"""Shared constants and enums for the trading bot."""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    TEST = "TEST"
    LIVE = "LIVE"


class RunType(str, Enum):
    SCAN = "SCAN"
    UNIVERSE = "UNIVERSE"
    REPLAY = "REPLAY"
    PRETRADE = "PRETRADE"
    STATUS = "STATUS"


class AlertState(str, Enum):
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"
    INVALIDATED = "INVALIDATED"


class InstrumentType(str, Enum):
    STOCK = "STOCK"
    ETF = "ETF"
    CFD = "CFD"


class PaperTradeState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


REQUIRED_SECRET_KEYS = (
    "T212_API_KEY",
    "T212_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)
