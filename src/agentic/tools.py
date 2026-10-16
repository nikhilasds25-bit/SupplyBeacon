"""LangChain tool wrappers around existing validated SupplyBeacon functions."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import re
import threading
from typing import Any

from langchain_core.tools import StructuredTool
import pandas as pd

from src.agent.tools import get_shipping_model_info
from src.graphrag.current_retriever import CurrentEvidenceRetriever
from src.graphrag.explanation_builder import build_explanation
from src.dynamic_risk import current_state
from src.dynamic_risk.current_state import (
    freshness_status,
    load_current_events,
    load_current_metadata,
    load_current_state,
)
from src.event_extraction.temporal_validity import apply_runtime_temporal_status, is_currently_eligible
from src.dynamic_risk.delivery_risk import estimate_delivery_risk
from src.dynamic_risk.current_state import BASE_DIR
from .weather import current_weather_tool

_cache_lock = threading.RLock()


def _retriever_signature() -> tuple[Any, ...]:
    paths = (
        current_state.CURRENT_DIR / "current_supply_chain_graph.graphml",
        current_state.CURRENT_DIR / "current_risk_scores.csv",
        current_state.CURRENT_DIR / "current_events.json",
        current_state.CURRENT_DIR / "current_run_metadata.json",
        current_state.CURRENT_DIR / "current_business_impacts.csv",
        Path(os.getenv("SUPPLYBEACON_MASTER_GENERATED_DIR", str(BASE_DIR / "data" / "master" / "generated"))) / "company_overlay.graphml",
    )
    files = tuple((str(path), path.stat().st_mtime_ns if path.exists() else 0) for path in paths)
    # Event validity changes with the clock even when no artifact was rewritten.
    events = load_current_events()["events"]
    statuses = tuple((str(event.get("Event_ID")), event.get("event_status")) for event in events)
    return files, statuses


@lru_cache(maxsize=1)
def _cached_retriever(signature: tuple[Any, ...]) -> CurrentEvidenceRetriever:
    return CurrentEvidenceRetriever()


def current_retriever() -> CurrentEvidenceRetriever:
    with _cache_lock:
        return _cached_retriever(_retriever_signature())


def invalidate_retriever_cache() -> None:
    with _cache_lock:
        _cached_retriever.cache_clear()


def graph_context_tool(query: str) -> dict[str, Any]:
    result = current_retriever().retrieve(query)
    explanation = build_explanation(result)
    return {
        **result,
        "answer": explanation["answer"],
        "provenance_summary": explanation["provenance_summary"],
    }


def current_risk_tool(entity_id: str) -> dict[str, Any]:
    retriever = current_retriever()
    if entity_id not in retriever.graph:
        return {"entity_id": entity_id, "risk_evidence": [], "limitations": ["Entity does not exist in the current graph."]}
    facts = retriever.node_facts(entity_id)
    return {"entity_id": entity_id, "risk_evidence": [fact for fact in facts if fact.get("provenance") == "MODEL_OUTPUT"]}


def _format_event_record(event: dict[str, Any]) -> dict[str, Any]:
    operational_eligible = bool(is_currently_eligible(event))
    event_status = str(event.get("event_status", "UNKNOWN")).upper()
    is_active = event_status == "ACTIVE"
    event_type = str(event.get("Event_Type") or event.get("event_type") or "event")
    location = event.get("Location") or event.get("location") or "Unknown"
    event_id = event.get("Event_ID") if event.get("Event_ID") is not None else event.get("event_id")
    source_channel = event.get("source_channel") or event.get("Source") or "EXTERNAL_API"

    if operational_eligible:
        reason = (
            event.get("Classification_Reason")
            or event.get("Rule_Triggered")
            or "Event satisfies operational disruption criteria and temporal validity."
        )
    else:
        if "light rain" in event_type.lower():
            reason = "Light rain is monitored but does not by itself meet operational disruption criteria and does not contribute to supply-chain risk."
        elif event.get("Classification_Reason"):
            reason = str(event["Classification_Reason"])
        elif event.get("Rule_Triggered"):
            reason = str(event.get("Rule_Triggered"))
        elif event.get("Operationally_Relevant") is False:
            reason = "Event does not meet operational disruption criteria."
        elif not is_active:
            reason = f"Event temporal status is {event_status}, not ACTIVE."
        else:
            reason = "Event does not contribute to current supply-chain risk."

    formatted = dict(event)
    formatted.update({
        "Event_ID": event_id,
        "event_id": event_id,
        "event_type": event_type,
        "Event_Type": event_type,
        "location": location,
        "Location": location,
        "temporal_status": event_status,
        "event_status": event_status,
        "operational_eligible": operational_eligible,
        "operationally_relevant": operational_eligible,
        "active": "YES" if is_active else "NO",
        "active_flag": "YES" if is_active else "NO",
        "operational": "YES" if operational_eligible else "NO",
        "operational_flag": "YES" if operational_eligible else "NO",
        "source_channel": source_channel,
        "operational_reason": reason,
        "reason": reason,
    })
    return formatted


def current_events_tool(operational_only: bool = True) -> dict[str, Any]:
    state = load_current_events()
    active = [apply_runtime_temporal_status(event) for event in state["events"]]
    active = [_format_event_record(event) for event in active if event["event_status"] == "ACTIVE"]
    events = [event for event in active if event["operational_eligible"]] if operational_only else active
    return {
        "events": events,
        "active_event_count": len(active),
        "operational_event_count": sum(1 for event in active if event["operational_eligible"]),
        "event_scope": "OPERATIONAL" if operational_only else "ACTIVE",
        "retrieval_ok": state["available"],
        "provenance": [{"provenance": "EXTERNAL_API", "source": "data/current/current_events.json"}],
    }


def current_risk_ranking_tool(
    entity_type: str,
    minimum_risk_level: str = "Medium",
    limit: int = 10,
) -> dict[str, Any]:
    """Return sorted rows from the authoritative current risk artifact."""
    state = load_current_state()
    risk = state["risk"]
    if risk.empty:
        return {"entities": [], "retrieval_ok": False, "limitations": ["Current risk artifact is unavailable."]}
    levels = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    threshold = levels.get(minimum_risk_level, 1)
    rows = risk[
        (risk["Node_Type"].astype(str) == entity_type)
        & risk["Risk_Level"].map(lambda value: levels.get(str(value), -1) >= threshold)
    ].copy()
    rows["score_numeric"] = pd.to_numeric(rows["Risk_Score"], errors="coerce")
    rows = rows[rows["score_numeric"].notna()]
    rows = rows.sort_values(["score_numeric", "Node_Name"], ascending=[False, True]).head(max(1, min(limit, 100)))
    entities = [
        {
            "node_id": str(row["Node_ID"]),
            "node_type": str(row["Node_Type"]),
            "name": row.get("Node_Name", row["Node_ID"]),
            "risk_score": float(row["score_numeric"]),
            "risk_level": row.get("Risk_Level", ""),
            "connected_events": row.get("Connected_Events", "0"),
            "risk_reason": row.get("Risk_Reason", ""),
            "risk_availability": row.get("Risk_Availability", "AVAILABLE"),
            "risk_components": {
                component: (float(row[component]) if component in row and pd.notna(pd.to_numeric(row[component], errors="coerce")) else None)
                for component in (
                    "Structural_Baseline", "Weather_Event_Component", "News_Event_Component",
                    "Business_Impact_Component", "Event_Exposure_Component", "Final_Scenario_Risk_Score",
                )
            },
            "provenance": "MODEL_OUTPUT",
            "source_artifact": "data/current/current_risk_scores.csv",
        }
        for _, row in rows.iterrows()
    ]
    return {
        "entities": entities,
        "retrieval_ok": True,
        "source_artifact": "data/current/current_risk_scores.csv",
        "provenance": ["MODEL_OUTPUT"],
    }


def delivery_scenario_tool(entity_id: str) -> dict[str, Any]:
    return estimate_delivery_risk(entity_id)


def historical_shipping_tool() -> dict[str, Any]:
    return get_shipping_model_info()


def freshness_tool() -> dict[str, Any]:
    metadata = load_current_metadata()
    return {**metadata, **freshness_status(metadata)}


def entity_lookup_tool(query: str) -> dict[str, Any]:
    retriever = current_retriever()
    product_bases: dict[str, list[str]] = {}
    for node_id, attrs in retriever.graph.nodes(data=True):
        if attrs.get("node_type") == "Product" and " (" in str(attrs.get("name", "")):
            base = str(attrs["name"]).split(" (", 1)[0]
            product_bases.setdefault(base.casefold(), []).append(str(node_id))
    entity_ids = [
        str(node_id) for node_id, attrs in retriever.graph.nodes(data=True)
        if any(value and re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", query, re.IGNORECASE)
               for value in (node_id, attrs.get("name")))
    ]
    entity_ids.extend(
        node_ids[0] for base, node_ids in product_bases.items()
        if len(node_ids) == 1 and re.search(r"(?<!\w)" + re.escape(base) + r"(?!\w)", query, re.IGNORECASE)
    )
    if not entity_ids:
        entity_ids = retriever.match_entities(query, limit=10)
    entity_ids = sorted(set(entity_ids), key=lambda node_id: (-len(str(retriever.graph.nodes[node_id].get("name", ""))), node_id))[:10]
    return {"query": query, "entities": [
        {"node_id": node_id, "node_type": retriever.graph.nodes[node_id].get("node_type", ""),
         "name": retriever.graph.nodes[node_id].get("name", ""), "provenance": "OBSERVED"}
        for node_id in entity_ids
    ]}


graph_context = StructuredTool.from_function(graph_context_tool, name="graph_context_tool", description="Retrieve bounded graph-grounded evidence.")
current_risk = StructuredTool.from_function(current_risk_tool, name="current_risk_tool", description="Read existing current risk artifact values.")
current_risk_ranking = StructuredTool.from_function(
    current_risk_ranking_tool,
    name="current_risk_ranking_tool",
    description="Rank current entities directly from the current risk CSV without graph traversal.",
)
current_events = StructuredTool.from_function(current_events_tool, name="current_events_tool", description="Read active events, optionally restricted to operationally eligible events. Does not alter risk.")
current_weather = StructuredTool.from_function(current_weather_tool, name="current_weather_tool", description="Read provider-separated weather evidence from current_weather.json; no live collection or risk calculation.")
delivery_scenario = StructuredTool.from_function(delivery_scenario_tool, name="delivery_scenario_tool", description="Read an existing rule-based delivery scenario.")
historical_shipping = StructuredTool.from_function(historical_shipping_tool, name="historical_shipping_tool", description="Read frozen DataCo historical shipping metrics.")
freshness = StructuredTool.from_function(freshness_tool, name="freshness_tool", description="Read current artifact freshness metadata.")
entity_lookup = StructuredTool.from_function(entity_lookup_tool, name="entity_lookup_tool", description="Resolve entities using existing graph matching.")
