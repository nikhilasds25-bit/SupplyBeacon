"""Read-only, evidence-aware current risk and watchlist views."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.event_extraction.temporal_validity import apply_runtime_temporal_status
from .location_resolution import resolve_event_location

ENTITY_TYPES = {"Warehouse", "Product", "Supplier"}
COMPONENTS = (
    "Structural_Baseline", "Weather_Event_Component", "News_Event_Component",
    "Business_Impact_Component", "Event_Exposure_Component", "Final_Scenario_Risk_Score",
)
SEVERITY = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
LEVELS = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def numeric(value: Any) -> float | None:
    try:
        if value is None or str(value).strip().casefold() in {"", "nan", "none", "null"}:
            return None
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None


def risk_rows(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    risk = state.get("risk")
    if risk is None or risk.empty or "Node_ID" not in risk:
        return {}
    return {str(row["Node_ID"]): row.to_dict() for _, row in risk.iterrows()}


def related(graph: Any, source: str, relationship: str, node_type: str) -> list[str]:
    if graph is None or source not in graph:
        return []
    return [
        str(target) for target in graph.successors(source)
        if (graph.get_edge_data(source, target) or {}).get("relationship") == relationship
        and graph.nodes[target].get("node_type") == node_type
    ]


def operational_event_ids(graph: Any, node_id: str) -> list[str]:
    """Only direct AFFECTS or approved inventory-backed paths are admissible."""
    if graph is None or node_id not in graph:
        return []
    node_type = graph.nodes[node_id].get("node_type")
    if node_type == "Supplier":
        return sorted({
            event_id
            for brand_id in related(graph, node_id, "SUPPLIES", "Brand")
            for product_id in related(graph, brand_id, "PRODUCES", "Product")
            for event_id in operational_event_ids(graph, product_id)
        })
    if node_type == "Product":
        return sorted({
            event_id
            for warehouse_id in related(graph, node_id, "STORED_AT", "Warehouse")
            for event_id in operational_event_ids(graph, warehouse_id)
        })
    if node_type == "Warehouse":
        direct = [event_id for event_id in graph.predecessors(node_id)
                  if (graph.get_edge_data(event_id, node_id) or {}).get("relationship") == "AFFECTS"]
        locations = related(graph, node_id, "LOCATED_IN", "Location")
        indirect = [event_id for location_id in locations for event_id in graph.predecessors(location_id)
                    if (graph.get_edge_data(event_id, location_id) or {}).get("relationship") == "AFFECTS"]
        candidates = direct + indirect
    else:
        return []
    return sorted({
        str(event_id) for event_id in candidates
        if graph.nodes[event_id].get("node_type") == "Event"
        and graph.nodes[event_id].get("event_status") == "ACTIVE"
        and graph.nodes[event_id].get("operational_current") == "true"
        and graph.nodes[event_id].get("operationally_relevant", "true") == "true"
    })


def risk_availability(row: dict[str, Any] | None) -> str:
    if not row or numeric(row.get("Risk_Score")) is None:
        return "UNAVAILABLE"
    value = row.get("Risk_Availability")
    return str(value) if value is not None and pd.notna(value) and str(value).strip() else "AVAILABLE"


def network_summary(state: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    rows = risk_rows(state)
    entities = [(str(node_id), attrs) for node_id, attrs in graph.nodes(data=True)
                if attrs.get("node_type") in ENTITY_TYPES] if graph is not None else []
    available = [rows.get(node_id) for node_id, _ in entities
                 if risk_availability(rows.get(node_id)) != "UNAVAILABLE"]
    levels = [str(row.get("Risk_Level")) for row in available if row is not None]
    overall = max(levels, key=lambda level: LEVELS.get(level, 0), default="Unavailable")
    active_signals = sum(apply_runtime_temporal_status(event)["event_status"] == "ACTIVE"
                         for event in state.get("events", []))
    operational_events = sum(
        apply_runtime_temporal_status(event)["event_status"] == "ACTIVE"
        and event.get("Operationally_Relevant") is True
        for event in state.get("events", [])
    )
    return {
        "overall_risk_level": overall,
        "monitored_entity_count": len(entities),
        "available_risk_entity_count": len(available),
        "unavailable_risk_entity_count": len(entities) - len(available),
        "operationally_exposed_entity_count": sum(bool(operational_event_ids(graph, node_id)) for node_id, _ in entities),
        "active_signal_count": active_signals,
        "operational_event_count": operational_events,
    }


def watchlist(state: dict[str, Any], entity_type: str = "Warehouse") -> list[dict[str, Any]]:
    if entity_type not in ENTITY_TYPES:
        raise ValueError("Unsupported watchlist entity type")
    graph = state.get("graph")
    if graph is None:
        return []
    rows = risk_rows(state)
    updated_at = state.get("metadata", {}).get("pipeline_run_timestamp") or state.get("metadata", {}).get("run_timestamp")
    events = state.get("events", [])
    signal_counts: dict[str, int] = {}
    if entity_type == "Warehouse":
        for event in events:
            if apply_runtime_temporal_status(event)["event_status"] != "ACTIVE":
                continue
            for warehouse_id in resolve_event_location(graph, event.get("Location"))["warehouse_ids"]:
                signal_counts[warehouse_id] = signal_counts.get(warehouse_id, 0) + 1
    result = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != entity_type:
            continue
        node_id = str(node_id)
        row = rows.get(node_id, {})
        ids = operational_event_ids(graph, node_id)
        operational_events = [graph.nodes[event_id] for event_id in ids]
        top = sorted(operational_events, key=lambda item: (-SEVERITY.get(str(item.get("severity")), 0), str(item.get("name", ""))))
        active_signal_count = signal_counts.get(node_id, 0) if entity_type == "Warehouse" else len(ids)
        score = numeric(row.get("Risk_Score"))
        availability = risk_availability(row)
        result.append({
            "entity_id": node_id,
            "entity_name": attrs.get("name", node_id),
            "entity_type": entity_type,
            "risk_score": score,
            "risk_level": row.get("Risk_Level", "Unavailable") if score is not None else "Unavailable",
            "risk_availability": availability,
            "operational_event_count": len(ids),
            "active_signal_count": active_signal_count,
            "top_event": top[0].get("name") if top else None,
            "top_event_severity": top[0].get("severity") if top else None,
            "location": ", ".join(filter(None, [str(attrs.get("city") or ""), str(attrs.get("state") or "")])) or None,
            "inventory_relationship_available": any(
                graph.nodes[product_id].get("node_type") == "Product"
                and (graph.get_edge_data(product_id, node_id) or {}).get("relationship") == "STORED_AT"
                for product_id in graph.predecessors(node_id)
            ) if entity_type == "Warehouse" else None,
            "catalog_only": attrs.get("catalog_only", False) if entity_type == "Warehouse" else None,
            "risk_components": {field: numeric(row.get(field)) for field in COMPONENTS},
            "reason": row.get("Risk_Reason") or "No applicable operational risk evidence is available.",
            "updated_at": updated_at,
        })
    return sorted(result, key=lambda item: (
        -int(item["operational_event_count"] > 0),
        -(item["risk_score"] if item["risk_score"] is not None else -1),
        -SEVERITY.get(str(item["top_event_severity"]), 0),
        item["entity_id"],
    ))
