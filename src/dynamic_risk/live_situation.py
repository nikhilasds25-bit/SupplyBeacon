"""Deterministic current-intelligence view over existing current artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.event_extraction.temporal_validity import apply_runtime_temporal_status, is_currently_eligible
from src.external_data.weather_collector import warehouse_weather_locations


def _iso_from_epoch(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(value), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _edge_relationship(graph: Any, source: str, target: str) -> str:
    return str((graph.get_edge_data(source, target) or {}).get("relationship", ""))


def _event_node(graph: Any, event: dict[str, Any]) -> str | None:
    artifact_id = str(event.get("Event_ID", ""))
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "Event" and (
            str(node_id) in {artifact_id, f"CURRENT_EVENT:{artifact_id}"}
            or str(attrs.get("event_id", "")) == artifact_id
        ):
            return str(node_id)
    return None


def _graph_resolution(graph: Any, event: dict[str, Any]) -> dict[str, Any]:
    node_id = _event_node(graph, event) if graph is not None else None
    if node_id is None:
        return {"resolved": False, "warehouse_ids": [], "location_ids": []}
    warehouses: list[str] = []
    locations: list[str] = []
    for target in graph.successors(node_id):
        if _edge_relationship(graph, node_id, target) != "AFFECTS":
            continue
        node_type = graph.nodes[target].get("node_type")
        if node_type == "Warehouse":
            warehouses.append(str(target))
        elif node_type == "Location":
            locations.append(str(target))
    return {
        "resolved": bool(warehouses or locations),
        "warehouse_ids": warehouses,
        "location_ids": locations,
    }


def exclusion_reason(event: dict[str, Any]) -> str:
    """Explain eligibility from persisted rules and temporal fields only."""
    current = apply_runtime_temporal_status(event)
    status = current.get("event_status", "UNKNOWN")
    if current.get("Operationally_Relevant") is True and status == "ACTIVE":
        return "Active event satisfies the existing operational eligibility rules."
    if current.get("Operationally_Relevant") is not True:
        rule = str(current.get("Rule_Triggered") or "").strip()
        if rule:
            return rule
        if str(current.get("Category", "")).casefold() == "weather":
            return "The observed condition does not meet the configured operational disruption criteria."
        return "No validated active operational disruption was identified by the existing event rules."
    if status == "UPCOMING":
        return "The event is operationally relevant but its validity window has not started."
    if status == "EXPIRED":
        return "The event is operationally relevant but its validity window has expired."
    return "Operational validity could not be established from the available source timestamps."


def _signal_payload(event: dict[str, Any], graph: Any) -> dict[str, Any]:
    current = apply_runtime_temporal_status(event)
    eligible = is_currently_eligible(current)
    resolution = _graph_resolution(graph, current)
    return {
        "event_id": current.get("Event_ID"),
        "title": current.get("Title", ""),
        "event_type": current.get("Event_Type", ""),
        "category": current.get("Category", ""),
        "source_channel": current.get("source_channel", "OTHER"),
        "location": current.get("Location", "Unknown"),
        "timestamp": current.get("Date"),
        "event_status": current.get("event_status", "UNKNOWN"),
        "severity": current.get("Severity", ""),
        "event_class": current.get("Event_Class", ""),
        "operational_eligible": eligible,
        "operational_reason": exclusion_reason(current),
        "temporal_basis": current.get("temporal_basis", "No event-validity date available"),
        "event_start": current.get("event_start"),
        "event_end": current.get("event_end"),
        "graph_resolution": resolution,
        "risk_contribution": "Applied through validated graph exposure" if eligible and resolution["resolved"] else "None",
    }


def _weather_record(city: str, key: str, record: dict[str, Any], signal_by_location: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    provider = str(record.get("source_channel") or ("WEATHERAPI" if key.endswith(":WEATHERAPI") else "OPENWEATHER"))
    alerts = record.get("alerts") or []
    kind = str(record.get("record_kind") or ("ALERT" if alerts else "DAILY_FORECAST" if provider == "WEATHERAPI" else "OBSERVATION"))
    condition = (record.get("weather") or [{}])[0]
    relevant_signals = signal_by_location.get(city.casefold(), [])
    active = any(item["event_status"] == "ACTIVE" for item in relevant_signals)
    operational = any(item["operational_eligible"] for item in relevant_signals)
    return {
        "city": city,
        "provider": provider,
        "record_kind": kind,
        "condition": condition.get("description") or condition.get("main") or "Unavailable",
        "temperature_c": (record.get("main") or {}).get("temp") if kind == "OBSERVATION" else None,
        "rainfall_mm": record.get("rain") or ({"daily": record.get("daily_precipitation_mm")} if record.get("daily_precipitation_mm") is not None else {}),
        "wind": record.get("wind") or ({"max_kph": record.get("max_wind_kph")} if record.get("max_wind_kph") is not None else {}),
        "timestamp": _iso_from_epoch(record.get("dt")),
        "forecast_date": record.get("forecast_date"),
        "alerts": alerts,
        "active_signal": active,
        "operational_impact": operational,
        "operational_significance": "Eligible operational event" if operational else "No operational disruption detected",
    }


def build_live_situation(state: dict[str, Any], freshness: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    signals = [_signal_payload(event, graph) for event in state.get("events", [])]
    active = [item for item in signals if item["event_status"] == "ACTIVE"]
    operational = [item for item in signals if item["operational_eligible"]]
    resolved = [item for item in operational if item["graph_resolution"]["resolved"]]
    precise_locations = warehouse_weather_locations(graph) if graph is not None else []

    signal_by_location: dict[str, list[dict[str, Any]]] = {}
    for item in signals:
        signal_by_location.setdefault(str(item["location"]).casefold(), []).append(item)
    raw_weather = state.get("weather", {}).get("cities", {}) or {}
    represented = {str(key).split(":", 1)[0].casefold() for key in raw_weather}
    weather_rows = []
    for city in precise_locations:
        records = [
            _weather_record(city, key, record, signal_by_location)
            for key, record in raw_weather.items()
            if str(key).split(":", 1)[0].casefold() == city.casefold() and isinstance(record, dict)
        ]
        observations = [row for row in records if row["record_kind"] == "OBSERVATION"]
        forecasts = [row for row in records if row["record_kind"] == "DAILY_FORECAST"]
        alerts = [row for row in records if row["record_kind"] == "ALERT"]
        weather_rows.append({
            "city": city,
            "available": bool(records),
            "observations": observations,
            "forecasts": forecasts,
            "alerts": alerts,
        })

    news = [item for item in signals if str(item["category"]).casefold() == "news"]
    relevant_news = [item for item in news if item["event_class"] in {"RISK_WARNING", "ACTUAL_DISRUPTION"}]
    filtered_news = [item for item in news if item not in relevant_news]
    risk = state.get("risk")
    levels = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    overall = max(
        (str(row.get("Risk_Level", "Low")) for row in risk.to_dict(orient="records")),
        key=lambda value: levels.get(value, 0),
        default="Unavailable",
    ) if risk is not None and not risk.empty else "Unavailable"
    affected_warehouses = sorted({warehouse for item in resolved for warehouse in item["graph_resolution"]["warehouse_ids"]})
    metadata = state.get("metadata", {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_refreshed": metadata.get("pipeline_run_timestamp") or metadata.get("run_timestamp"),
        "freshness": freshness,
        "summary": {
            "candidate_events": len(signals),
            "active_signals": len(active),
            "active_weather_signals": sum(1 for item in active if str(item["category"]).casefold() == "weather"),
            "recent_news_signals": len(news),
            "operational_events": len(operational),
            "graph_resolved_events": len(resolved),
            "affected_warehouses": len(affected_warehouses),
            "network_risk": overall,
        },
        "signals": signals,
        "active_signals": active,
        "weather": weather_rows,
        "coverage": {
            "required": len(precise_locations),
            "represented": sum(1 for city in precise_locations if city.casefold() in represented),
            "missing": [city for city in precise_locations if city.casefold() not in represented],
            "locations": precise_locations,
        },
        "news": {
            "relevant_candidates": relevant_news,
            "filtered_non_operational": filtered_news,
        },
        "why_risk": {
            "headline": f"Current network risk remains {overall}",
            "explanation": (
                f"{len(active)} active signal(s) are being monitored, but {len(operational)} currently satisfy "
                "the operational disruption criteria required to propagate exposure through the supply-chain graph."
            ),
            "pipeline": [
                {"label": "Candidate Events", "count": len(signals)},
                {"label": "Operational Events", "count": len(operational)},
                {"label": "Graph Resolved Events", "count": len(resolved)},
                {"label": "Affected Warehouses", "count": len(affected_warehouses)},
                {"label": "Network Risk", "value": overall},
            ],
        },
    }
