"""Utility helpers for the risk prediction module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
RISK_DIR = BASE_DIR / "data" / "risk_prediction"
GRAPHML_PATH = BASE_DIR / "data" / "graph" / "supply_chain_graph.graphml"
BUSINESS_IMPACT_EVENTS_CSV = BASE_DIR / "data" / "processed" / "business_impacts" / "business_impact_events.csv"


def create_directory(path: Path) -> None:
    """Create a directory and its parents if needed."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    """Clean and normalize a text value for deterministic comparison."""
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning a safe default for invalid inputs."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default


def save_json(path: Path, payload: Any) -> None:
    """Persist a JSON payload to disk."""
    create_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def log_info(message: str, *args: Any) -> None:
    """Log information in a consistent format."""
    logging.info(message, *args)
