"""CLI entrypoint for the trading bot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading_bot.config import load_secrets, validate_mode
from trading_bot.constants import RunType
from trading_bot.logging_setup import setup_logging
from trading_bot.backtest.replay import run_replay
from trading_bot.pipeline.daily_scan import run_daily_scan
from trading_bot.universe.refresh_universe import run_universe_refresh
from trading_bot.pretrade.pipeline import run_pretrade
from trading_bot.mooner import format_mooner_callout_lines, run_mooner_sidecar
from trading_bot.status import run_status

REQUIRED_DIRS = (
    "data",
    "logs",
    "universe",
    "state",
)


def _ensure_directories(base_dir: Path) -> None:
    for name in REQUIRED_DIRS:
        (base_dir / name).mkdir(parents=True, exist_ok=True)


def preflight(base_dir: Path):
    """Run pre-flight checks and initialize logging."""

    validate_mode()
    _ensure_directories(base_dir)
    logger = setup_logging(base_dir / "logs")
    load_secrets()
    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading bot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Run market scan")
    scan_parser.add_argument("--dry-run", action="store_true", help="Skip external outputs")

    subparsers.add_parser("universe", help="Refresh trading universe")
    subparsers.add_parser("pretrade", help="Run pre-trade viability check")
    subparsers.add_parser("status", help="Show system status")
    subparsers.add_parser("mooner", help="Run the Mooner regime sidecar")

    replay_parser = subparsers.add_parser("replay", help="Run replay mode")
    replay_parser.add_argument("--days", type=int, default=90, help="Days to replay")
    replay_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Replay start date in YYYY-MM-DD (optional)",
    )

    return parser


def _handle_scan(logger, dry_run: bool) -> None:
    logger.info('Starting daily scan')
    if dry_run:
        logger.info('DRY RUN – no Telegram, no state updates')
    run_daily_scan(dry_run=dry_run)


def _handle_universe(logger) -> None:
    logger.info("Starting Trading212 universe refresh")
    run_universe_refresh()


def _handle_replay(logger, days: int, start_date: str | None) -> None:
    logger.info("Replay mode starting for %s trading days", days)
    run_replay(days=days, start_date=start_date)


def _handle_pretrade(logger) -> None:
    logger.info('Pre-trade viability check starting')
    run_pretrade()


def _handle_mooner(logger, base_dir: Path) -> None:
    logger.info('Mooner sidecar starting')
    try:
        callouts = run_mooner_sidecar(base_dir, logger)
    except Exception as exc:
        logger.exception('Mooner sidecar failed: %s', exc)
        print(f"Mooner sidecar failed: {exc}")
        return
    logger.info('Mooner sidecar completed with %s callouts', len(callouts))
    _print_mooner_callouts(callouts)


def _print_mooner_callouts(callouts: list[dict]) -> None:
    lines = format_mooner_callout_lines(callouts)
    if not lines:
        print('Mooner sidecar completed: no active regimes detected.')
        return
    print(f'Mooner sidecar emitted {len(lines)} callout(s):')
    for line in lines:
        print(f'- {line}')


def _handle_status(logger) -> None:
    logger.info('Status report requested')
    run_status(Path(__file__).resolve().parent, logger=logger)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    try:
        logger = preflight(base_dir)
    except RuntimeError as exc:
        print(f"Pre-flight checks failed: {exc}")
        return 1

    if args.command == RunType.SCAN.value.lower():
        _handle_scan(logger, dry_run=args.dry_run)
    elif args.command == RunType.UNIVERSE.value.lower():
        _handle_universe(logger)
    elif args.command == RunType.REPLAY.value.lower():
        _handle_replay(logger, days=args.days, start_date=args.start_date)
    elif args.command == RunType.PRETRADE.value.lower():
        _handle_pretrade(logger)
    elif args.command == RunType.MOONER.value.lower():
        _handle_mooner(logger, base_dir)
    elif args.command == RunType.STATUS.value.lower():
        _handle_status(logger)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
