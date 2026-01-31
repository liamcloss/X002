"""Orchestrate Trading212 universe refresh."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_bot.messaging import telegram_client
from trading_bot.universe.normalise_universe import normalise_universe
from trading_bot.universe.t212_client import Trading212Client
from trading_bot.run_state import finish_run, start_run

MAX_ATTEMPTS = 3
ATTEMPT_COOLDOWN = timedelta(hours=24)


def run_universe_refresh() -> None:
    """Refresh Trading212 universe data."""

    logger = logging.getLogger("trading_bot")
    base_dir = Path(__file__).resolve().parents[2]
    run_handle = start_run(base_dir, 'universe', logger)
    if not run_handle.acquired:
        return
    failed = False
    completed = False
    scan_lock = base_dir / 'state' / 'scan.lock'
    if scan_lock.exists():
        logger.warning('Scan in progress; universe refresh aborted to avoid contention.')
        finish_run(run_handle, logger, failed=failed, completed=completed)
        return
    market_data_lock = base_dir / 'state' / 'market_data.lock'
    if market_data_lock.exists():
        logger.warning('Market data refresh in progress; universe refresh aborted.')
        finish_run(run_handle, logger, failed=failed, completed=completed)
        return
    universe_dir = base_dir / "universe"
    raw_dir = universe_dir / "raw"
    clean_dir = universe_dir / "clean"
    summary_dir = universe_dir / "summaries"
    for path in (raw_dir, clean_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    state_path = summary_dir / "universe_refresh_state.json"
    now = datetime.now(timezone.utc)
    state = _load_state(state_path)
    last_attempt_raw = state.get("last_attempt") if state else None
    last_success_raw = state.get("last_success") if state else None
    last_attempt = _parse_timestamp(last_attempt_raw)
    last_success = _parse_timestamp(last_success_raw)
    failure_count = int(state.get("failure_count", 0)) if state else 0

    try:
        if failure_count >= MAX_ATTEMPTS:
            if last_attempt and now - last_attempt >= ATTEMPT_COOLDOWN:
                logger.info(
                    "Resetting universe refresh failure counter after %s cooldown.",
                    ATTEMPT_COOLDOWN,
                )
                failure_count = 0
                _save_state(
                    state_path,
                    {
                        "last_attempt": last_attempt_raw,
                        "failure_count": failure_count,
                        "last_success": last_success_raw,
                    },
                )
            else:
                remaining = (
                    ATTEMPT_COOLDOWN - (now - last_attempt)
                    if last_attempt
                    else ATTEMPT_COOLDOWN
                )
                logger.info(
                    "Universe refresh skipped; waiting %.1fh after repeated failures.",
                    remaining.total_seconds() / 3600,
                )
                return

        if failure_count == 0 and last_success and now - last_success < ATTEMPT_COOLDOWN:
            logger.info("Universe refresh skipped; last success within 24 hours.")
            return
        if failure_count > 0 and last_attempt and now - last_attempt < timedelta(minutes=1):
            logger.info("Universe refresh skipped; last attempt was just moments ago.")
            return

        api_key = os.environ.get("T212_API_KEY", "")
        api_secret = os.environ.get("T212_API_SECRET", "")
        if not api_key or not api_secret:
            raise RuntimeError("Trading212 API credentials are missing.")

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

        type_breakdown = summary.get("type_breakdown", {})
        type_summary = ", ".join(
            f"{type_name}: {count}" for type_name, count in sorted(type_breakdown.items())
        )
        if not type_summary:
            type_summary = "none"

        message = (
            "✅ Trading212 Universe Updated\n"
            f"Total instruments: {summary['total_instruments']}\n"
            f"Types: {type_summary}\n"
            f"Extended hours count: {summary['extended_hours_count']}"
        )
        _send_telegram(message, logger, context='universe-refresh-success')
        logger.info("Trading212 universe refresh completed successfully.")
        completed = True
    except Exception as exc:  # noqa: BLE001 - report failure details
        failed = True
        failure_count += 1
        _save_state(
            state_path,
            {
                "last_attempt": now.isoformat(),
                "failure_count": failure_count,
                "last_success": last_success_raw,
            },
        )
        if failure_count >= MAX_ATTEMPTS:
            _send_telegram(
                "❌ Trading212 universe refresh failed after 3 attempts\n"
                "Last successful universe retained",
                logger,
                context='universe-refresh-error',
            )
        else:
            _send_telegram(
                "⚠️ Trading212 universe refresh failed\n"
                f"Attempt {failure_count} of {MAX_ATTEMPTS}\n"
                f"Error: {exc}",
                logger,
                context='universe-refresh-error',
            )
        logger.exception("Trading212 universe refresh failed: %s", exc)
    finally:
        finish_run(run_handle, logger, failed=failed, completed=completed)


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


def _send_telegram(
    message: str,
    logger: logging.Logger,
    *,
    context: str = 'universe-refresh',
) -> None:
    sent = telegram_client.send_message(message, context=context)
    if not sent:
        logger.warning('Telegram notification %s failed or skipped.', context)
