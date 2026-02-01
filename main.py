"""CLI entrypoint for the trading bot."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
from pathlib import Path

from extra_options.yolo_penny_lottery import run_yolo_penny_lottery
from trading_bot.config import load_secrets, validate_mode
from trading_bot.constants import RunType
from trading_bot.logging_setup import setup_logging
from trading_bot.backtest.replay import run_replay
from trading_bot.pipeline.daily_scan import run_daily_scan
from trading_bot.pretrade.win_history import (
    load_win_history,
    record_outcome,
    score_from_entry,
)
from trading_bot.yolo import format_yolo_summary, load_yolo_summary
from trading_bot.universe.refresh_universe import run_universe_refresh
from trading_bot.pretrade.pipeline import run_pretrade
from trading_bot.mooner import format_mooner_callout_lines, run_mooner_sidecar
from trading_bot.news_scout import run_news_scout
from trading_bot.status import run_status
from trading_bot.messaging import send_message

REQUIRED_DIRS = (
    "data",
    "logs",
    "universe",
    "state",
)

PRETRADE_HISTORY_COMMAND = RunType.PRETRADE_HISTORY.value.lower().replace("_", "-")


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
    scan_parser.add_argument(
        "--force-market-data",
        action="store_true",
        help="Re-run the market data refresh even if the cache is not stale.",
    )

    subparsers.add_parser("universe", help="Refresh trading universe")
    subparsers.add_parser("pretrade", help="Run pre-trade viability check")
    history_parser = subparsers.add_parser(
        PRETRADE_HISTORY_COMMAND, help="Record or inspect pre-trade win history"
    )
    history_parser.add_argument(
        "--symbol",
        type=str,
        help="Symbol to update or inspect in win history (e.g., TSLA, AAPL)",
    )
    history_parser.add_argument(
        "--outcome",
        type=str.lower,
        choices=("target-hit", "stopped"),
        help="Record the outcome for the provided symbol (target-hit = win, stopped = loss)",
    )
    history_parser.add_argument(
        "--show",
        action="store_true",
        help="Print the recorded win/loss history (filtered by --symbol when provided)",
    )
    yolo_parser = subparsers.add_parser("yolo", help="Run the weekly YOLO lottery and summarize the pick")
    yolo_parser.add_argument(
        "--reroll",
        action="store_true",
        help="Force rerun of the current week's YOLO draw (overwrites the stored pick and ledger)",
    )
    yolo_parser.add_argument(
        "--close",
        type=str,
        help="Close an open YOLO position for the provided ticker.",
    )
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--verbose", action="store_true", help="Emit full status report")
    subparsers.add_parser("mooner", help="Run the Mooner regime sidecar")
    subparsers.add_parser("news-scout", help="Produce the news scout summary")
    messaging_parser = subparsers.add_parser(
        "telegram-test", help="Send a short test alert through the configured Telegram bot"
    )
    messaging_parser.add_argument(
        "--text", type=str, help="Optional headline text to send instead of the canned message"
    )
    messaging_parser.add_argument(
        "--context",
        type=str,
        default="telegram-test",
        help="Context label recorded in the Telegram client logs",
    )

    replay_parser = subparsers.add_parser("replay", help="Run replay mode")
    replay_parser.add_argument("--days", type=int, default=90, help="Days to replay")
    replay_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Replay start date in YYYY-MM-DD (optional)",
    )

    return parser


def _handle_scan(logger, dry_run: bool, force_market_data: bool) -> None:
    logger.info('Starting daily scan')
    if dry_run:
        logger.info('DRY RUN – no Telegram, no state updates')
    if force_market_data:
        logger.info('Force-refresh requested for market data via CLI flag.')
    run_daily_scan(dry_run=dry_run, force_market_data=force_market_data)


def _handle_universe(logger) -> None:
    logger.info("Starting Trading212 universe refresh")
    run_universe_refresh()


def _handle_replay(logger, days: int, start_date: str | None) -> None:
    logger.info("Replay mode starting for %s trading days", days)
    run_replay(days=days, start_date=start_date)


def _handle_pretrade(logger) -> None:
    logger.info('Pre-trade viability check starting')
    run_pretrade()


def _handle_pretrade_history(logger, base_dir: Path, args) -> None:
    """Update or inspect the pre-trade win history."""
    performed = False
    if args.outcome:
        symbol = _normalize_cli_symbol(args.symbol)
        if not symbol:
            print('The --symbol argument is required when recording an outcome.')
            return
        success = args.outcome == 'target-hit'
        entry = record_outcome(symbol, success, base_dir=base_dir)
        score = score_from_entry(entry)
        logger.info(
            'Pretrade win history updated via CLI for %s: %s wins / %s losses (score=%.2f)',
            symbol,
            entry.get('wins', 0),
            entry.get('losses', 0),
            score,
        )
        print(
            f'Win history updated for {symbol}: '
            f'wins={entry.get("wins", 0)} losses={entry.get("losses", 0)} '
            f'(score={score:.2f})'
        )
        performed = True
    if args.show:
        history = load_win_history(base_dir)
        _print_history_summary(history, args.symbol)
        performed = True
    if not performed:
        print('Add --outcome or --show to pretrade-history to record or inspect results.')


def _print_history_summary(history: dict[str, dict[str, int]], symbol: str | None) -> None:
    if not history:
        print('No win/loss history recorded yet.')
        return
    filter_symbol = _normalize_cli_symbol(symbol)
    entries = []
    if filter_symbol:
        entry = history.get(filter_symbol)
        if not entry:
            print(f'No history recorded for {filter_symbol}.')
            return
        entries = [(filter_symbol, entry)]
    else:
        entries = sorted(history.items())
    print('Pre-trade win history:')
    for target_symbol, entry in entries:
        wins = entry.get('wins', 0)
        losses = entry.get('losses', 0)
        total = wins + losses
        score = wins / (total + 1)
        print(f'- {target_symbol}: wins={wins} losses={losses} score={score:.2f}')


def _normalize_cli_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    normalized = symbol.strip().upper()
    return normalized if normalized else None


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


def _handle_yolo(logger, base_dir: Path, reroll: bool, close_ticker: str | None = None) -> None:
    logger.info('YOLO lottery requested')
    try:
        result = run_yolo_penny_lottery(base_dir, logger, reroll=reroll, close_ticker=close_ticker)
    except Exception as exc:
        logger.exception('YOLO lottery failed: %s', exc)
        print(f"YOLO lottery failed: {exc}")
        return
    if close_ticker:
        if result:
            close_price = result.get("close_price")
            close_value = result.get("close_value")
            close_date = result.get("close_date")
            ticker = result.get("ticker", close_ticker)
            close_price_display = f"{close_price:.4f}" if isinstance(close_price, (int, float)) else "n/a"
            close_value_display = f"£{close_value:.2f}" if isinstance(close_value, (int, float)) else "n/a"
            print(
                f'Closed YOLO position for {ticker} at {close_price_display} on {close_date}; value={close_value_display}'
            )
        else:
            print(f'No open YOLO position recorded for {close_ticker}.')
        return
    summary = load_yolo_summary(base_dir)
    if summary is None:
        print('YOLO lottery has no recorded pick yet.')
        return
    for line in format_yolo_summary(summary):
        print(line)


def _handle_news_scout(logger, base_dir: Path) -> None:
    logger.info('News scout sidecar starting')
    entries = run_news_scout(base_dir, logger)
    logger.info('News scout completed with %s entries', len(entries))
    _print_news_scout_summary(entries)


def _handle_telegram_test(logger, text: str | None, context: str) -> None:
    payload = text or (
        f'Telegram connectivity check at {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}'
    )
    logger.info('Telegram test requested (context=%s)', context)
    sent = send_message(payload, context=context)
    if sent:
        print('Telegram test message sent; check your chat for arrival.')
        return
    print('Telegram test message not sent; inspect logs for diagnostics.')


def _print_news_scout_summary(entries: list[dict]) -> None:
    if not entries:
        print('News scout produced no setup candidates.')
        return
    print('NEWS SCOUT SUMMARY')
    for entry in entries:
        symbol = entry.get('symbol') or 'UNKNOWN'
        display = entry.get('display_ticker') or ''
        rank = entry.get('scan_rank')
        print(f'{rank}. {symbol} {f"({display})" if display else ""}')
        entry_price = entry.get('entry')
        stop = entry.get('stop')
        target = entry.get('target')
        print(f'  Entry: {entry_price} | Stop: {stop} | Target: {target}')
        reason = entry.get('reason')
        if reason:
            print(f'  Setup: {reason}')
        llm_insight = entry.get('llm_insight')
        if llm_insight:
            print(f'  AI: {llm_insight}')
        links = entry.get('links') or []
        if links:
            formatted = ' | '.join(
                f"{link['label']}: {link['url']}" for link in links if link.get('url')
            )
            print(f'  Links: {formatted}')
        spread = entry.get('spread_pct')
        if isinstance(spread, (int, float)):
            print(f'  Spread: {spread * 100:.2f}%')
        print('')


def _print_mooner_callouts(callouts: list[dict]) -> None:
    lines = format_mooner_callout_lines(callouts)
    if not lines:
        print('Mooner sidecar completed: no active regimes detected.')
        return
    print(f'Mooner sidecar emitted {len(lines)} callout(s):')
    for line in lines:
        print(f'- {line}')


def _handle_status(logger, base_dir: Path, verbose: bool) -> None:
    logger.info('Status report requested')
    run_status(base_dir, logger=logger, verbose=verbose)


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
        _handle_scan(logger, dry_run=args.dry_run, force_market_data=args.force_market_data)
    elif args.command == RunType.UNIVERSE.value.lower():
        _handle_universe(logger)
    elif args.command == RunType.REPLAY.value.lower():
        _handle_replay(logger, days=args.days, start_date=args.start_date)
    elif args.command == RunType.PRETRADE.value.lower():
        _handle_pretrade(logger)
    elif args.command == PRETRADE_HISTORY_COMMAND:
        _handle_pretrade_history(logger, base_dir, args)
    elif args.command == RunType.MOONER.value.lower():
        _handle_mooner(logger, base_dir)
    elif args.command == RunType.STATUS.value.lower():
        _handle_status(logger, base_dir, verbose=args.verbose)
    elif args.command == RunType.YOLO.value.lower():
        _handle_yolo(logger, base_dir, reroll=args.reroll, close_ticker=args.close)
    elif args.command in {RunType.NEWS_SCOUT.value.lower(), "news-scout"}:
        _handle_news_scout(logger, base_dir)
    elif args.command == "telegram-test":
        _handle_telegram_test(logger, text=args.text, context=args.context)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
