"""Runtime phase tracking for scanner/execution guardrails."""

from __future__ import annotations

from typing import Literal

Phase = Literal["scanner", "execution", "unknown"]

_PHASE: Phase = "unknown"


def set_phase(phase: Phase) -> None:
    """Set the current runtime phase for guardrails."""

    global _PHASE
    _PHASE = phase


def get_phase() -> Phase:
    """Return the current runtime phase."""

    return _PHASE


__all__ = [
    "get_phase",
    "set_phase",
]
