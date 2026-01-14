"""CLI entrypoint for the trading bot."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from time import sleep

from trading_bot.config import Config, load_config, load_environment
from trading_bot.constants import DATA_DIR, LOG_DIR, STATE_DIR
from trading_bot.ideas import build_trade_ideas, format_ideas_message
from trading_bot.logging_setup import setup_logging
from trading_bot.market_data import get_market_data
from trading_bot.messaging import send_message
from trading_bot.paper_trading import execute_paper_trades
from trading_bot.replay import run_replay
from trading_bot.signals import generate_signals
from trading_bot.state import load_state, save_state
from trading_bot.universe import refresh_universe, universe_to_frame


def _ensure_python_version() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or higher is required.")


def _ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def preflight() -> Config:
    """Run pre-flight checks."""

    _ensure_python_version()
    _ensure_directories()
    load_environment()
    return load_config()


def _handle_universe(logger: logging.Logger) -> None:
    universe = refresh_universe()
    frame = universe_to_frame(universe.symbols)
    output_path = DATA_DIR / "universe.parquet"
    frame.to_parquet(output_path, index=False)
    logger.info("Universe saved to %s", output_path)


def _handle_scan(
    logger: logging.Logger,
    config: Config,
    dry_run: bool,
    iterations: int,
    pause_seconds: int,
) -> None:
    logger.info(
        "Starting scan (dry_run=%s, iterations=%d, pause_seconds=%d)",
        dry_run,
        iterations,
        pause_seconds,
    )
    state = load_state(STATE_DIR)
    logger.info("Loaded state last_run=%s", state.last_run or "<none>")

    for cycle in range(1, iterations + 1):
        logger.info("Scan cycle %d of %d", cycle, iterations)
        universe = refresh_universe()
        market_data = get_market_data(universe.symbols)
        signal_result = generate_signals(market_data.frame)
        ideas = build_trade_ideas(signal_result.signals, config)
        message = format_ideas_message(ideas, config)

        if dry_run:
            logger.info("Dry run enabled; skipping messaging and paper trades.")
            logger.info("Dry run message preview:\n%s", message)
        else:
            if ideas:
                execute_paper_trades(logger, ideas)
            send_message(logger, config, message)

        now = datetime.now(timezone.utc).isoformat()
        save_state(STATE_DIR, last_run=now)
        logger.info("State saved with last_run=%s", now)

        if cycle < iterations and pause_seconds > 0:
            logger.info("Pausing for %d seconds before next cycle.", pause_seconds)
            sleep(pause_seconds)


def _handle_replay(logger: logging.Logger, days: int) -> None:
    logger.info("Starting replay for %d days", days)
    run_replay(logger, days)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading bot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Run market scan")
    scan_parser.add_argument("--dry-run", action="store_true", help="Do not trade or send messages")
    scan_parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of scan cycles to run (for iterative testing).",
    )
    scan_parser.add_argument(
        "--pause-seconds",
        type=int,
        default=0,
        help="Pause in seconds between scan cycles.",
    )

    subparsers.add_parser("universe", help="Refresh trading universe")

    replay_parser = subparsers.add_parser("replay", help="Run replay")
    replay_parser.add_argument("--days", type=int, default=90, help="Days to replay")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = preflight()
    except RuntimeError as exc:
        print(f"Pre-flight checks failed: {exc}")
        return 1

    logger = setup_logging(LOG_DIR)
    logger.info("Trading bot started")

    if args.command == "universe":
        _handle_universe(logger)
    elif args.command == "scan":
        _handle_scan(
            logger,
            config,
            dry_run=args.dry_run,
            iterations=args.iterations,
            pause_seconds=args.pause_seconds,
        )
    elif args.command == "replay":
        _handle_replay(logger, days=args.days)
    else:
        parser.print_help()
        return 1

    logger.info("Trading bot finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
