"""Orchestrate Trading212 universe refresh."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from trading_bot.universe.normalise_universe import normalise_universe
from trading_bot.universe.t212_client import Trading212Client

MAX_ATTEMPTS = 3
ATTEMPT_COOLDOWN = timedelta(hours=24)


def run_universe_refresh() -> None:
    """Refresh Trading212 universe data."""

    logger = logging.getLogger("trading_bot")
    base_dir = Path(__file__).resolve().parents[2]
    universe_dir = base_dir / "universe"
    raw_dir = universe_dir / "raw"
    clean_dir = universe_dir / "clean"
    summary_dir = universe_dir / "summaries"
    for path in (raw_dir, clean_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    state_path = summary_dir / "universe_refresh_state.json"
    now = datetime.now(timezone.utc)
    state = _load_state(state_path)

    if state:
        last_attempt = _parse_timestamp(state.get("last_attempt"))
        failure_count = int(state.get("failure_count", 0))
        if last_attempt and now - last_attempt >= ATTEMPT_COOLDOWN:
            failure_count = 0
            _save_state(
                state_path,
                {
                    "last_attempt": state.get("last_attempt"),
                    "failure_count": failure_count,
                    "last_success": state.get("last_success"),
                },
            )
        if failure_count >= MAX_ATTEMPTS:
            _send_telegram(
                "❌ Trading212 universe refresh failed after 3 attempts\n"
                "Last successful universe retained",
                logger,
            )
            return
        if last_attempt and now - last_attempt < ATTEMPT_COOLDOWN:
            logger.info("Universe refresh skipped; last attempt within 24 hours.")
            return

    api_key = os.environ.get("T212_API_KEY", "")
    api_secret = os.environ.get("T212_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("Trading212 API credentials are missing.")

    try:
        client = Trading212Client(api_key=api_key, api_secret=api_secret, logger=logger)
        response = client.fetch_instruments()
        raw_path = raw_dir / f"t212_raw_{now.strftime('%Y%m%d_%H%M')}.json"
        raw_path.write_text(response.raw_text, encoding="utf-8")

        df, summary = normalise_universe(response.json_data, logger)
        clean_path = clean_dir / "universe.parquet"
        df.to_parquet(clean_path, index=False)

        summary_path = summary_dir / "universe_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _save_state(
            state_path,
            {
                "last_attempt": now.isoformat(),
                "failure_count": 0,
                "last_success": now.isoformat(),
            },
        )

        message = (
            "✅ Trading212 Universe Updated\n"
            f"Active instruments: {summary['active_instruments']}\n"
            f"Stocks: {summary['stocks']} | ETFs: {summary['etfs']}\n"
            f"ISA eligible: {summary['isa_eligible']}"
        )
        _send_telegram(message, logger)
        logger.info("Trading212 universe refresh completed successfully.")
    except Exception as exc:  # noqa: BLE001 - report failure details
        failure_count = int(state.get("failure_count", 0)) + 1 if state else 1
        _save_state(
            state_path,
            {
                "last_attempt": now.isoformat(),
                "failure_count": failure_count,
                "last_success": state.get("last_success") if state else None,
            },
        )
        if failure_count >= MAX_ATTEMPTS:
            _send_telegram(
                "❌ Trading212 universe refresh failed after 3 attempts\n"
                "Last successful universe retained",
                logger,
            )
        else:
            _send_telegram(
                "⚠️ Trading212 universe refresh failed\n"
                f"Attempt {failure_count} of {MAX_ATTEMPTS}\n"
                f"Error: {exc}",
                logger,
            )
        logger.exception("Trading212 universe refresh failed: %s", exc)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    cleaned = {key: value for key, value in payload.items() if value is not None}
    path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _send_telegram(message: str, logger: logging.Logger) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram credentials missing; message not sent.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, data=payload, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Telegram message failed: %s", exc)
