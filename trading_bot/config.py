"""Configuration loader for the trading bot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv

from trading_bot.constants import Mode, REQUIRED_SECRET_KEYS

# This will be populated by the load_config function
CONFIG: dict[str, Any] = {}


def load_config(config_file: str | Path = "config.yaml") -> None:
    """Load settings from the YAML config file and populate the global CONFIG dict."""
    global CONFIG
    with open(config_file, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    CONFIG = loaded if isinstance(loaded, dict) else {}


def load_secrets(env_file: str | None = None) -> dict[str, str]:
    """Load required secrets from .env and return them without logging values."""
    load_dotenv(dotenv_path=env_file)
    missing = _missing_env_vars(REQUIRED_SECRET_KEYS)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables: {joined}. Ensure .env is present and loaded."
        )
    return {key: os.environ[key] for key in REQUIRED_SECRET_KEYS}


def validate_mode() -> None:
    """Ensure the configured mode is valid."""
    if MODE not in {Mode.TEST.value, Mode.LIVE.value}:
        raise RuntimeError(f"Invalid mode '{MODE}' in config.yaml. Must be TEST or LIVE.")


def _missing_env_vars(required: Iterable[str]) -> list[str]:
    missing = []
    for key in required:
        if not os.environ.get(key):
            missing.append(key)
    return missing

# Load the config automatically when this module is imported
load_config()


def _extract_mode() -> str:
    raw = CONFIG.get("mode")
    if isinstance(raw, str):
        normalized = raw.strip().upper()
        if normalized in {Mode.TEST.value, Mode.LIVE.value}:
            return normalized
    return Mode.LIVE.value


MODE = _extract_mode()


def _live_mode_value(key: str, default: float) -> float:
    live_mode = CONFIG.get("live_mode")
    if isinstance(live_mode, dict):
        value = live_mode.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(default)


LIVE_MODE_POSITION_MIN = _live_mode_value("position_min", 100.0)
LIVE_MODE_POSITION_MAX = _live_mode_value("position_max", 200.0)
LIVE_MODE_MAX_RISK = _live_mode_value("max_risk", 20.0)


def _get_number(key: str, default: float) -> float:
    value = CONFIG.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _get_int(key: str, default: int) -> int:
    value = CONFIG.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return int(default)


def _get_str(key: str, default: str) -> str:
    value = CONFIG.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _get_tuple(key: str, default: tuple[Any, ...]) -> tuple[Any, ...]:
    value = CONFIG.get(key)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return default


def _section(key: str) -> dict[str, Any]:
    value = CONFIG.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _section_int(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return int(default)


def _section_float(section: dict[str, Any], key: str, default: float) -> float:
    value = section.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


TEST_MODE_CONFIG = _section("test_mode")
LIVE_MODE_CONFIG = _section("live_mode")

TEST_MODE_POSITION_SIZE = _section_int(TEST_MODE_CONFIG, "position_size", 25)
TEST_MODE_MAX_RISK = _section_float(TEST_MODE_CONFIG, "max_risk", 2.0)

MAX_SPREAD_PCT = _get_number("max_spread_pct", 0.005)
PRETRADE_CANDIDATE_LIMIT = _get_int("pretrade_candidate_limit", 10)
DAILY_SCAN_DISPLAY_LIMIT = _get_int("daily_scan_display_limit", 10)
SCAN_REFRESH_MODE = _get_str("scan_refresh_mode", "skip")
MARKET_DATA_REFRESH_MAX_AGE_HOURS = _get_number("market_data_refresh_max_age_hours", 24.0)

SPREAD_SAMPLING_LOOKBACK_DAYS = _section_int(_section("spread_sampling"), "lookback_days", 20)
SPREAD_SAMPLING_OPEN_COOLDOWN_MINUTES = _section_int(_section("spread_sampling"), "open_cooldown_minutes", 30)

SIGNAL_CONFIG = _section("signals")
CANDIDATE_MAX_SPREAD_PCT = _section_float(SIGNAL_CONFIG, "max_spread_pct", 0.035)
CANDIDATE_MAX_ATR_PCT = _section_float(SIGNAL_CONFIG, "max_atr_pct", 0.02)
SIGNAL_MAX_PCT_FROM_20D_HIGH = _section_float(SIGNAL_CONFIG, "max_pct_from_20d_high", 0.05)
SIGNAL_VOLUME_MULTIPLE_THRESHOLD = _section_float(SIGNAL_CONFIG, "volume_multiple_threshold", 1.5)
SIGNAL_MOMENTUM_THRESHOLD = _section_float(SIGNAL_CONFIG, "momentum_threshold", 0.03)
SIGNAL_EXTENSION_THRESHOLD = _section_float(SIGNAL_CONFIG, "extension_from_ma50_threshold", 0.2)
SIGNAL_MAX_DAYS_SINCE_HIGH = _section_int(SIGNAL_CONFIG, "max_days_since_20d_high", 4)

MARKET_DATA_CONFIG = _section("market_data")
MARKET_DATA_BATCH_SIZE = _section_int(MARKET_DATA_CONFIG, "batch_size", 20)
MARKET_DATA_RATE_LIMIT_MIN = _section_float(MARKET_DATA_CONFIG, "rate_limit_delay_min", 0.6)
MARKET_DATA_RATE_LIMIT_MAX = _section_float(MARKET_DATA_CONFIG, "rate_limit_delay_max", 1.2)
MARKET_DATA_BURST_BATCHES = _section_int(MARKET_DATA_CONFIG, "burst_batches", 4)
MARKET_DATA_BURST_COOLDOWN_SECONDS = _section_float(MARKET_DATA_CONFIG, "burst_cooldown_seconds", 3.0)

STOP_PERCENT_RANGE = _get_tuple("stop_percent_range", (None, None))
TARGET_PERCENT_RANGE = _get_tuple("target_percent_range", (None, None))
COOLDOWN_DAYS = CONFIG.get("cooldown_days")
PULLBACK_LIMITS = CONFIG.get("pullback_limits")

MOONER_CONFIG = _section("mooner")
MOONER_SUBSET_MAX = _section_int(MOONER_CONFIG, "subset_max", 10)
MOONER_CANDIDATE_VOLUME_THRESHOLD = _section_int(MOONER_CONFIG, "candidate_volume_threshold", 1_000_000)
MOONER_CANDIDATE_DOLLAR_VOLUME_GBP = _section_float(MOONER_CONFIG, "candidate_dollar_volume_gbp", 20_000_000)
MOONER_CANDIDATE_SURGE_MULTIPLIER = _section_float(MOONER_CONFIG, "candidate_surge_multiplier", 2.5)
MOONER_ATR_PERCENTILE = _section_float(MOONER_CONFIG, "atr_percentile", 0.8)
MOONER_ATR_COMPRESSION_RATIO = _section_float(MOONER_CONFIG, "atr_compression_ratio", 0.75)
MOONER_PRICE_MIN_GBP = _section_float(MOONER_CONFIG, "price_min_gbp", 2.0)
MOONER_DRAWDOWN_MAX = _section_float(MOONER_CONFIG, "drawdown_max", 0.65)
MOONER_RESISTANCE_BUFFER = _section_float(MOONER_CONFIG, "resistance_buffer", 0.1)
MOONER_STATE_VOLUME_MULTIPLIER = _section_float(MOONER_CONFIG, "state_volume_multiplier", 1.5)
MOONER_STATE_ATR_RISE_DAYS = _section_int(MOONER_CONFIG, "state_atr_rise_days", 3)
MOONER_REL_STRENGTH_LOOKBACK = _section_int(MOONER_CONFIG, "rel_strength_lookback", 90)
