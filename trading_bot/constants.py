"""Shared constants for the trading bot."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
STATE_DIR = BASE_DIR / "state"

BANKROLL_GBP = 1_000.0
STOP_PCTS = (0.05, 0.06, 0.07)
TARGET_PCTS = (0.10, 0.12, 0.15, 0.20)
MIN_RISK_REWARD = 2.0
MAX_IDEAS = 3

REQUIRED_ENV_VARS = (
    "T212_API_KEY",
    "T212_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)
