"""Configuration loading for the trading bot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from trading_bot.constants import BANKROLL_GBP, REQUIRED_ENV_VARS

MODE_ENV_VAR = "MODE"
VALID_MODES = ("TEST", "LIVE")


@dataclass(frozen=True)
class ModeSettings:
    """Risk and sizing settings for the selected mode."""

    name: str
    max_position_gbp: float
    max_risk_gbp_min: float
    max_risk_gbp_max: float


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    t212_api_key: str
    t212_api_secret: str
    telegram_bot_token: str
    telegram_chat_id: str
    mode: str
    bankroll_gbp: float
    mode_settings: ModeSettings


def load_environment(env_file: Path | None = None) -> None:
    """Load environment variables from a .env file."""

    load_dotenv(dotenv_path=env_file)


def _missing_env_vars(required: Iterable[str]) -> list[str]:
    missing = []
    for key in required:
        if not _get_env_value(key):
            missing.append(key)
    return missing


def _get_env_value(key: str) -> str | None:
    return os.environ.get(key)


def _load_mode() -> str:
    raw_mode = _get_env_value(MODE_ENV_VAR) or "TEST"
    mode = raw_mode.upper()
    if mode not in VALID_MODES:
        raise RuntimeError(f"Invalid MODE '{raw_mode}'. Must be one of: {VALID_MODES}.")
    return mode


def _mode_settings(mode: str) -> ModeSettings:
    if mode == "TEST":
        return ModeSettings(
            name=mode,
            max_position_gbp=25.0,
            max_risk_gbp_min=1.0,
            max_risk_gbp_max=2.0,
        )
    return ModeSettings(
        name=mode,
        max_position_gbp=200.0,
        max_risk_gbp_min=5.0,
        max_risk_gbp_max=10.0,
    )


def load_config() -> Config:
    """Load config from environment variables, raising if any required are missing."""

    missing = _missing_env_vars(REQUIRED_ENV_VARS)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required environment variables: "
            f"{joined}. Ensure .env is present and loaded."
        )

    mode = _load_mode()

    return Config(
        t212_api_key=_get_env_value("T212_API_KEY") or "",
        t212_api_secret=_get_env_value("T212_API_SECRET") or "",
        telegram_bot_token=_get_env_value("TELEGRAM_BOT_TOKEN") or "",
        telegram_chat_id=_get_env_value("TELEGRAM_CHAT_ID") or "",
        mode=mode,
        bankroll_gbp=BANKROLL_GBP,
        mode_settings=_mode_settings(mode),
    )
