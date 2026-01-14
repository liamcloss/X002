"""Normalise Trading212 universe responses into a trimmed schema."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

FIELD_MAPPING: tuple[tuple[str, str], ...] = (
    ("ticker", "ticker"),
    ("name", "name"),
    ("shortName", "short_name"),
    ("type", "type"),
    ("currencyCode", "currency_code"),
    ("isin", "isin"),
    ("extendedHours", "extended_hours"),
    ("maxOpenQuantity", "max_open_quantity"),
    ("addedOn", "added_on"),
    ("workingScheduleId", "working_schedule_id"),
)


def normalise_universe(
    raw_json: Any, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform the Trading212 payload into a normalized DataFrame and summary."""

    instruments = _extract_instruments(raw_json, logger)
    if not instruments:
        raise RuntimeError("Trading212 payload contained no instruments.")

    rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for instrument in instruments:
        if not isinstance(instrument, dict):
            logger.warning("Skipping non-dict instrument entry: %s", type(instrument))
            continue

        rows.append(
            {column_name: instrument.get(field_name) for field_name, column_name in FIELD_MAPPING}
        )

    if not rows:
        raise RuntimeError("Trading212 payload contained no valid instruments.")

    df = pd.DataFrame(rows, columns=_schema_columns())
    df = _filter_equity_universe(df, logger)
    summary = _build_summary(df, timestamp)
    return df, summary


def _extract_instruments(raw_json: Any, logger: logging.Logger) -> list[Any]:
    if isinstance(raw_json, list):
        return raw_json
    if isinstance(raw_json, dict):
        top_keys = sorted(raw_json.keys())
        logger.info("Trading212 payload keys: %s", top_keys)
        for value in raw_json.values():
            if isinstance(value, list):
                return value
    raise RuntimeError("Unexpected Trading212 payload structure; unable to locate instruments list.")


def _schema_columns() -> list[str]:
    return [column_name for _, column_name in FIELD_MAPPING]


def _filter_equity_universe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    allowed = {"ETF", "STOCK"}
    type_series = df["type"].fillna("UNKNOWN").astype(str).str.upper()
    mask = type_series.isin(allowed)
    if not mask.any():
        raise RuntimeError("No ETF or STOCK instruments found after filtering.")
    if not mask.all():
        excluded = type_series[~mask].value_counts().to_dict()
        logger.info("Filtered non-ETF/STOCK instruments: %s", excluded)
    return df.loc[mask].reset_index(drop=True)


def _build_summary(df: pd.DataFrame, timestamp: str) -> dict[str, Any]:
    total = int(df.shape[0])
    type_series = df["type"].fillna("UNKNOWN").astype(str)
    type_breakdown = type_series.value_counts().to_dict()
    currency_series = df["currency_code"].dropna().astype(str)
    currency_breakdown = currency_series.value_counts().to_dict()

    extended_hours_count = int(
        df["extended_hours"].fillna(False).astype(bool).sum()
        if "extended_hours" in df
        else 0
    )

    return {
        "timestamp": timestamp,
        "total_instruments": total,
        "type_breakdown": type_breakdown,
        "currency_breakdown": currency_breakdown,
        "extended_hours_count": extended_hours_count,
    }
