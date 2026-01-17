"""Non-secret configuration for the trading bot."""

from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv

from trading_bot.constants import Mode, REQUIRED_SECRET_KEYS

MODE = "LIVE"  # or "TEST"

TEST_MODE_POSITION_SIZE = 25
TEST_MODE_MAX_RISK = 2

LIVE_MODE_POSITION_MIN = 100
LIVE_MODE_POSITION_MAX = 200
LIVE_MODE_MAX_RISK = 20

MAX_SPREAD_PCT = 0.005
PRETRADE_CANDIDATE_LIMIT = 10

STOP_PERCENT_RANGE = (None, None)
TARGET_PERCENT_RANGE = (None, None)
COOLDOWN_DAYS = None
PULLBACK_LIMITS = None


def load_secrets(env_file: str | None = None) -> dict[str, str]:
    """Load required secrets from .env and return them without logging values."""

    load_dotenv(dotenv_path=env_file)
    missing = _missing_env_vars(REQUIRED_SECRET_KEYS)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required environment variables: "
            f"{joined}. Ensure .env is present and loaded."
        )

    return {key: os.environ[key] for key in REQUIRED_SECRET_KEYS}


def validate_mode() -> None:
    """Ensure MODE is a valid configured mode."""

    if MODE not in {Mode.TEST.value, Mode.LIVE.value}:
        raise RuntimeError(f"Invalid MODE '{MODE}'. Must be TEST or LIVE.")


def _missing_env_vars(required: Iterable[str]) -> list[str]:
    missing = []
    for key in required:
        if not os.environ.get(key):
            missing.append(key)
    return missing
