"""Risk geometry selection."""

from __future__ import annotations

from typing import Optional


STOP_PCTS = (0.05, 0.06, 0.07)
TARGET_PCTS = (0.10, 0.12, 0.15, 0.20)
STOP_PREFERENCE = (0.06, 0.05, 0.07)


def find_risk_geometry(price: float) -> Optional[dict]:
    """
    Returns:
    {
        "stop_pct": float,
        "target_pct": float,
        "rr": float
    }
    or None if no valid geometry exists.
    """

    if price <= 0:
        return None

    for target in TARGET_PCTS:
        valid_stops = [stop for stop in STOP_PCTS if (target / stop) >= 2.0]
        if not valid_stops:
            continue
        for preferred_stop in STOP_PREFERENCE:
            if preferred_stop in valid_stops:
                rr = target / preferred_stop
                return {
                    "stop_pct": float(preferred_stop),
                    "target_pct": float(target),
                    "rr": float(rr),
                }

    return None
