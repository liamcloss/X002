"""Normalise Trading212 universe responses into a clean schema."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

KNOWN_FIELDS = {
    "id",
    "instrumentId",
    "instrument_id",
    "ticker",
    "symbol",
    "instrumentCode",
    "code",
    "name",
    "instrumentName",
    "description",
    "exchange",
    "exchangeCode",
    "exchangeId",
    "market",
    "country",
    "countryCode",
    "country_name",
    "currency",
    "currencyCode",
    "currencySymbol",
    "type",
    "instrumentType",
    "assetClass",
    "tradable",
    "isTradable",
    "tradingEnabled",
    "canTrade",
    "isaEligible",
    "isIsaEligible",
    "isaEligibleIndicator",
    "isCfd",
    "cfd",
    "isCFD",
    "isLeveraged",
    "leveraged",
    "isLeveragedEtf",
    "isInverse",
    "inverse",
    "isInverseEtf",
}


def normalise_universe(
    raw_json: Any, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform raw Trading212 JSON into a normalized DataFrame and summary."""

    instruments = _extract_instruments(raw_json, logger)
    if not instruments:
        raise RuntimeError("Trading212 payload contained no instruments.")

    unknown_fields = set()
    rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for instrument in instruments:
        if not isinstance(instrument, dict):
            logger.warning("Skipping non-dict instrument entry: %s", type(instrument))
            continue

        unknown_fields.update(set(instrument.keys()) - KNOWN_FIELDS)

        tradable = _coerce_bool(_first(instrument, ["tradable", "isTradable", "tradingEnabled", "canTrade"]))
        is_cfd = _coerce_bool(_first(instrument, ["isCfd", "cfd", "isCFD"]))
        is_leveraged = _coerce_bool(
            _first(instrument, ["isLeveraged", "leveraged", "isLeveragedEtf"])
        )
        is_inverse = _coerce_bool(_first(instrument, ["isInverse", "inverse", "isInverseEtf"]))

        instrument_type = _coerce_type(_first(instrument, ["type", "instrumentType", "assetClass"]))

        rows.append(
            {
                "instrument_id": _first(
                    instrument, ["id", "instrumentId", "instrument_id"]
                ),
                "ticker": _first(instrument, ["ticker", "symbol", "instrumentCode", "code"]),
                "name": _first(instrument, ["name", "instrumentName", "description"]),
                "exchange": _first(instrument, ["exchange", "exchangeCode", "exchangeId", "market"]),
                "country": _first(instrument, ["country", "countryCode", "country_name"]),
                "currency": _first(instrument, ["currency", "currencyCode", "currencySymbol"]),
                "type": instrument_type,
                "tradable": tradable,
                "isa_eligible": _coerce_bool(
                    _first(instrument, ["isaEligible", "isIsaEligible", "isaEligibleIndicator"])
                ),
                "is_cfd": is_cfd,
                "is_leveraged": is_leveraged,
                "is_inverse": is_inverse,
                "active": tradable and not is_cfd and not is_leveraged and not is_inverse,
                "source": "Trading212",
                "last_updated": timestamp,
            }
        )

    if unknown_fields:
        logger.info("Unknown Trading212 instrument fields detected: %s", sorted(unknown_fields))

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
    return [
        "instrument_id",
        "ticker",
        "name",
        "exchange",
        "country",
        "currency",
        "type",
        "tradable",
        "isa_eligible",
        "is_cfd",
        "is_leveraged",
        "is_inverse",
        "active",
        "source",
        "last_updated",
    ]


def _filter_equity_universe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    allowed = {"ETF", "STOCK"}
    type_series = df["type"].astype(str).str.upper()
    mask = type_series.isin(allowed)
    if not mask.all():
        excluded = (
            type_series[~mask]
            .value_counts()
            .sort_values(ascending=False)
            .to_dict()
        )
        logger.info("Filtered non-ETF/STOCK instruments: %s", excluded)
    filtered = df.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise RuntimeError("No ETF or STOCK instruments found after filtering.")
    return filtered


def _first(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_type(value: Any) -> str:
    if not value:
        return "UNKNOWN"
    text = str(value).upper()
    if "ETF" in text:
        return "ETF"
    if "STOCK" in text or "SHARE" in text:
        return "STOCK"
    return text


def _build_summary(df: pd.DataFrame, timestamp: str) -> dict[str, Any]:
    if df.empty:
        return {
            "timestamp": timestamp,
            "total_instruments": 0,
            "active_instruments": 0,
            "stocks": 0,
            "etfs": 0,
            "isa_eligible": 0,
            "excluded": {"cfd": 0, "leveraged": 0, "inverse": 0, "non_tradable": 0},
        }

    total = int(df.shape[0])
    active = int(df["active"].sum())
    stocks = int((df["type"].str.upper() == "STOCK").sum())
    etfs = int((df["type"].str.upper() == "ETF").sum())
    isa = int(df["isa_eligible"].sum())
    excluded = {
        "cfd": int(df["is_cfd"].sum()),
        "leveraged": int(df["is_leveraged"].sum()),
        "inverse": int(df["is_inverse"].sum()),
        "non_tradable": int((~df["tradable"]).sum()),
    }
    return {
        "timestamp": timestamp,
        "total_instruments": total,
        "active_instruments": active,
        "stocks": stocks,
        "etfs": etfs,
        "isa_eligible": isa,
        "excluded": excluded,
    }
