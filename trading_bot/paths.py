"""Shared helpers for locating generated artifacts."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or BASE_DIR
    return _ensure_dir(root / "outputs")


def outputs_subdir(base_dir: Path | None = None, *segments: str) -> Path:
    current = outputs_dir(base_dir)
    for segment in segments:
        current = current / segment
    return _ensure_dir(current)


def setup_candidates_dir(base_dir: Path | None = None) -> Path:
    return outputs_subdir(base_dir, "setup_candidates")


def setup_candidates_snapshot_path(
    base_dir: Path | None = None,
    snapshot_date: str | None = None,
) -> Path:
    directory = setup_candidates_dir(base_dir)
    if snapshot_date:
        return directory / f"SetupCandidates_{snapshot_date}.json"
    return directory / "SetupCandidates.json"


def setup_candidates_path(base_dir: Path | None = None) -> Path:
    return setup_candidates_snapshot_path(base_dir)


def latest_setup_candidates_path(base_dir: Path | None = None) -> Path | None:
    directory = setup_candidates_dir(base_dir)
    candidates = list(directory.glob("SetupCandidates_*.json"))
    if not candidates:
        default_path = setup_candidates_snapshot_path(base_dir)
        return default_path if default_path.exists() else None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def mooner_output_path(base_dir: Path | None = None, filename: str = "") -> Path:
    output = outputs_subdir(base_dir, "mooner")
    return output / filename if filename else output


def yolo_output_path(base_dir: Path | None = None, filename: str = "") -> Path:
    output = outputs_subdir(base_dir, "yolo")
    return output / filename if filename else output


def news_scout_output_path(base_dir: Path | None = None, filename: str = "") -> Path:
    output = outputs_subdir(base_dir, "news_scout")
    return output / filename if filename else output


def mooner_state_path(base_dir: Path | None = None) -> Path:
    root = base_dir or BASE_DIR
    state_dir = _ensure_dir(root / "state" / "mooner")
    return state_dir / "MoonerState.json"


def yolo_blocked_path(base_dir: Path | None = None) -> Path:
    return yolo_output_path(base_dir, "blocked_picks.json")


def pretrade_outputs_dir(base_dir: Path | None = None) -> Path:
    return outputs_subdir(base_dir, "pretrade")


def pretrade_viability_path(base_dir: Path | None = None, filename: str = "") -> Path:
    output = outputs_subdir(base_dir, "pretrade", "viability")
    return output / filename if filename else output


def pretrade_spread_path(base_dir: Path | None = None, filename: str = "") -> Path:
    output = outputs_subdir(base_dir, "pretrade", "spread_reports")
    return output / filename if filename else output
