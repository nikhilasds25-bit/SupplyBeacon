"""Load and validate isolated current-state artifacts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from src.event_extraction.temporal_validity import apply_runtime_temporal_status, is_currently_eligible

BASE_DIR = Path(__file__).resolve().parents[2]
CURRENT_DIR = BASE_DIR / "data" / "current"


def _json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _apply_runtime_graph_event_validity(graph: Any, events: list[dict[str, Any]]) -> Any:
    if graph is None:
        return graph
    by_title = {str(event.get("Title", "")).strip(): event for event in events}
    for node_id, attrs in list(graph.nodes(data=True)):
        if attrs.get("node_type") != "Event":
            continue
        event = by_title.get(str(attrs.get("name", "")).strip())
        if event is None:
            attrs["event_status"] = "UNKNOWN"
            attrs["operational_current"] = "false"
            continue
        attrs["event_status"] = event.get("event_status", "UNKNOWN")
        attrs["event_start"] = event.get("event_start") or ""
        attrs["event_end"] = event.get("event_end") or ""
        attrs["temporal_basis"] = event.get("temporal_basis", "")
        attrs["operational_current"] = str(is_currently_eligible(event)).lower()
        if attrs["operational_current"] != "true":
            graph.remove_edges_from(list(graph.in_edges(node_id)) + list(graph.out_edges(node_id)))
    return graph


def _temporal_roll_forward(
    graph: Any,
    risk: pd.DataFrame,
    impacts: pd.DataFrame,
    events: list[dict[str, Any]],
) -> tuple[Any, pd.DataFrame]:
    """Rebuild local current exposure when persisted event status is stale."""
    persisted_statuses = {
        str(attrs.get("name", "")): str(attrs.get("event_status", ""))
        for _, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "Event"
    } if graph is not None else {}
    runtime_statuses = {
        str(event.get("Title", "")): str(event.get("event_status", "UNKNOWN"))
        for event in events
    }
    if persisted_statuses == runtime_statuses:
        return graph, risk
    from src.dynamic_risk.refresh_current_state import _build_current_graph, _current_risk

    refreshed_graph = _build_current_graph(events)
    refreshed_risk = _current_risk(refreshed_graph, impacts, baseline_override=risk)
    return refreshed_graph, refreshed_risk


def load_current_events() -> dict[str, Any]:
    """Load only event artifacts for read-only event queries."""
    events = _json(CURRENT_DIR / "current_events.json", [])
    if isinstance(events, list):
        events = [apply_runtime_temporal_status(event) for event in events]
    return {
        "metadata": _json(CURRENT_DIR / "current_run_metadata.json", {}),
        "weather": _json(CURRENT_DIR / "current_weather.json", {}),
        "news": _json(CURRENT_DIR / "current_news.json", {}),
        "events": events,
        "available": (CURRENT_DIR / "current_events.json").exists(),
    }


def load_current_metadata() -> dict[str, Any]:
    """Load metadata without loading graph, risk, or business-impact artifacts."""
    return _json(CURRENT_DIR / "current_run_metadata.json", {})


def load_current_state() -> dict[str, Any]:
    """Return current artifacts without reading historical external files."""
    metadata = _json(CURRENT_DIR / "current_run_metadata.json", {})
    weather = _json(CURRENT_DIR / "current_weather.json", {})
    news = _json(CURRENT_DIR / "current_news.json", {})
    events = _json(CURRENT_DIR / "current_events.json", [])
    if isinstance(events, list):
        events = [apply_runtime_temporal_status(event) for event in events]
    impacts_path = CURRENT_DIR / "current_business_impacts.csv"
    risk_path = CURRENT_DIR / "current_risk_scores.csv"
    graph_path = CURRENT_DIR / "current_supply_chain_graph.graphml"
    impacts = _csv(impacts_path)
    risk = _csv(risk_path)
    graph = nx.read_graphml(str(graph_path)) if graph_path.exists() else None
    graph, risk = _temporal_roll_forward(graph, risk, impacts, events)
    graph = _apply_runtime_graph_event_validity(graph, events)
    overlay_path = Path(os.getenv("SUPPLYBEACON_MASTER_GENERATED_DIR", str(BASE_DIR / "data" / "master" / "generated"))) / "company_overlay.graphml"
    if graph is not None and overlay_path.exists():
        overlay = nx.read_graphml(str(overlay_path))
        for node_id, attrs in overlay.nodes(data=True):
            if node_id not in graph:
                graph.add_node(node_id, **attrs)
        for source, target, attrs in overlay.edges(data=True):
            if source in graph and target in graph:
                graph.add_edge(source, target, **attrs)
    return {
        "metadata": metadata,
        "weather": weather,
        "news": news,
        "events": events,
        "business_impacts": impacts,
        "risk": risk,
        "graph": graph,
        "available": bool(metadata),
    }


def freshness_label(metadata: dict[str, Any], now: datetime | None = None) -> tuple[str, bool]:
    """Return display timestamp and stale flag; missing data is stale."""
    status = freshness_status(metadata, now=now)
    timestamps = [
        value for value in (
            metadata.get("weather_fetch_timestamp"),
            metadata.get("news_fetch_timestamp"),
        ) if value
    ]
    newest = max(timestamps, default=None)
    return format_fetch_timestamp(newest), status["overall_stale"]


def freshness_status(
    metadata: dict[str, Any],
    now: datetime | None = None,
    threshold_hours: float = 24,
) -> dict[str, Any]:
    """Evaluate weather and news freshness independently."""
    now = now or datetime.now(timezone.utc)

    def source_status(value: Any) -> tuple[bool, float | None]:
        if not value:
            return False, None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age = max(0.0, (now - parsed).total_seconds() / 3600)
            return age <= threshold_hours, age
        except ValueError:
            return False, None

    weather_fresh, weather_age = source_status(metadata.get("weather_fetch_timestamp"))
    news_fresh, news_age = source_status(metadata.get("news_fetch_timestamp"))
    return {
        "weather_fresh": weather_fresh,
        "news_fresh": news_fresh,
        "weather_age_hours": weather_age,
        "news_age_hours": news_age,
        "overall_stale": not (weather_fresh and news_fresh),
    }


def format_fetch_timestamp(value: Any) -> str:
    if not value:
        return "Unavailable"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M IST")
    except ValueError:
        return "Unknown"
