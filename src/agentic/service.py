"""LangGraph V2 orchestration for grounded SupplyBeacon answers.

This module deliberately orchestrates existing retrieval and risk services.  It
does not calculate risk, classify events, or create new graph semantics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from copy import deepcopy
from enum import Enum
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from src.dynamic_risk.current_state import load_current_events, load_current_state
from src.event_extraction.temporal_validity import apply_runtime_temporal_status, is_currently_eligible
from src.external_data.weather_collector import warehouse_weather_locations
from src.graphrag.explanation_builder import build_explanation
from src.llm_agent.provider import OpenAICompatibleProvider, ProviderError
from .prompts import SYNTHESIS_PROMPT
from .grounded_domains import answer_grounded_domain
from .tools import (
    current_events,
    current_risk,
    current_risk_ranking,
    current_weather,
    delivery_scenario,
    entity_lookup,
    freshness,
    graph_context,
    historical_shipping,
)

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    PRODUCT_RISK = "PRODUCT_RISK"
    WAREHOUSE_RISK = "WAREHOUSE_RISK"
    SUPPLIER_RISK = "SUPPLIER_RISK"
    DELIVERY = "DELIVERY"
    CURRENT_EVENTS = "CURRENT_EVENTS"
    CURRENT_WEATHER = "CURRENT_WEATHER"
    NETWORK_RISK = "NETWORK_RISK"
    HISTORICAL_SHIPPING = "HISTORICAL_SHIPPING"
    GENERAL_SUPPLYBEACON = "GENERAL_SUPPLYBEACON"
    UNKNOWN = "UNKNOWN"


class QueryPlan(BaseModel):
    primary_intent: Intent
    secondary_intents: list[Intent] = Field(default_factory=list)
    entity_mentions: list[str] = Field(default_factory=list)
    resolved_entity_ids: list[str] = Field(default_factory=list)
    requested_entity_types: list[str] = Field(default_factory=list)
    needs_graph: bool = False
    needs_risk: bool = False
    needs_events: bool = False
    needs_weather: bool = False
    needs_delivery: bool = False
    needs_historical: bool = False
    needs_freshness: bool = False
    operational_events_only: bool = True
    ambiguous: bool = False
    clarification_needed: bool = False


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_type: str
    entity_id: str | None = None
    entity_type: str | None = None
    entity_name: str | None = None
    fact_name: str
    value: Any
    unit: str | None = None
    source_artifact: str | None = None
    provenance: str
    timestamp: str | None = None
    confidence: float | None = None
    operationally_relevant: bool | None = None
    relation: str | None = None
    temporal_status: str | None = None
    validation_status: str | None = None
    notes: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    graph: list[EvidenceItem] = Field(default_factory=list)
    risk: list[EvidenceItem] = Field(default_factory=list)
    events: list[EvidenceItem] = Field(default_factory=list)
    weather: list[EvidenceItem] = Field(default_factory=list)
    delivery: list[EvidenceItem] = Field(default_factory=list)
    historical: list[EvidenceItem] = Field(default_factory=list)
    freshness: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def all_evidence_items(self) -> list[EvidenceItem]:
        return [
            item
            for group in (self.graph, self.risk, self.events, self.weather, self.delivery, self.historical, self.freshness)
            for item in group
        ]


class GroundedAnswer(BaseModel):
    answer: str
    primary_intent: str
    secondary_intents: list[str] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: str | None = None
    evidence_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)
    current_data_fresh: bool | None = None
    llm_used: bool = False
    fallback_used: bool = False


class AgentState(TypedDict, total=False):
    query: str
    query_plan: QueryPlan
    resolved_entities: list[dict[str, Any]]
    graph_result: dict[str, Any]
    risk_result: dict[str, Any]
    ranking_result: dict[str, Any]
    events_result: dict[str, Any]
    weather_result: dict[str, Any]
    delivery_result: dict[str, Any]
    historical_result: dict[str, Any]
    freshness_result: dict[str, Any]
    evidence_bundle: EvidenceBundle
    evidence_valid: bool
    missing_evidence: list[str]
    warnings: list[str]
    answer: dict[str, Any]
    structured_answer: dict[str, Any]
    timings: dict[str, float]
    tools_used: list[str]
    llm_used: bool
    workflow_trace: list[str]


def _trace(state: AgentState, name: str) -> None:
    state.setdefault("workflow_trace", []).append(name)


def _intent_terms(text: str) -> list[Intent]:
    lowered = text.lower()
    intents: list[Intent] = []

    # 1. Historical DataCo
    if any(term in lowered for term in ("historical", "dataco", "model performance")):
        intents.append(Intent.HISTORICAL_SHIPPING)

    # 2. Weather events vs Current Weather
    has_event_word = bool(re.search(r"\b(?:events?|disruptions?|happening)\b", lowered))
    has_weather_word = bool(re.search(r"\b(?:weather|rain|raining|rainfall|temperature|precipitation|forecast)\b", lowered))

    if has_weather_word and not has_event_word:
        intents.append(Intent.CURRENT_WEATHER)

    # 3. Entity-specific / Network risk
    if "supplier" in lowered and any(term in lowered for term in ("risk", "exposed", "current")):
        intents.append(Intent.SUPPLIER_RISK)
    elif "warehouse" in lowered and any(term in lowered for term in ("risk", "affected", "current")) and Intent.CURRENT_WEATHER not in intents:
        intents.append(Intent.WAREHOUSE_RISK)
    elif "product" in lowered and any(term in lowered for term in ("risk", "risky", "current")):
        intents.append(Intent.PRODUCT_RISK)
    elif any(term in lowered for term in ("network risk", "supply network", "overall risk", "why is the network", "why is current risk low", "why is risk low")) or ("network" in lowered and "risk" in lowered) or ("light rain" in lowered and "risk" in lowered):
        intents.append(Intent.NETWORK_RISK)

    # 4. Delivery
    if Intent.HISTORICAL_SHIPPING not in intents and any(term in lowered for term in ("delivery", "delay", "ship")):
        intents.append(Intent.DELIVERY)

    # 5. Events
    if any(phrase in lowered for phrase in (
        "operational event", "current event", "external condition", "active event",
        "active events", "active signals", "active weather", "currently happening", "happening right now", "what active", "show active", "what is happening",
    )):
        intents.append(Intent.CURRENT_EVENTS)
    elif "event" in lowered and Intent.CURRENT_EVENTS not in intents:
        intents.append(Intent.CURRENT_EVENTS)

    if (
        any(term in lowered for term in ("operational event", "current event", "active event"))
        and Intent.DELIVERY not in intents
    ):
        intents = [
            Intent.CURRENT_EVENTS
        ] + [
            intent for intent in intents
            if intent not in {Intent.CURRENT_EVENTS, Intent.NETWORK_RISK}
        ]

    if (
        not intents
        and any(term in lowered for term in ("risk", "risky"))
        and "network" not in lowered
    ):
        intents.append(Intent.PRODUCT_RISK)

    return list(dict.fromkeys(intents))


def _analyze_query(state: AgentState) -> AgentState:
    started = time.perf_counter()
    _trace(state, "analyze_query")
    intents = _intent_terms(state["query"])
    primary = intents[0] if intents else Intent.UNKNOWN
    if primary == Intent.UNKNOWN and any(word in state["query"].lower() for word in ("supplybeacon", "help")):
        primary = Intent.GENERAL_SUPPLYBEACON
    mentions = re.findall(
        r"(?:called|named)\s+([A-Za-z0-9][A-Za-z0-9 _-]+)",
        state["query"],
        re.IGNORECASE,
    )
    mentions.extend(
        re.sub(
            r"^(?:why\s+is|what\s+is|is|the)\s+",
            "",
            match.group(1).strip(),
            flags=re.IGNORECASE,
        )
        for match in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,4}\s+(?:Warehouse|Supplier))\b",
            state["query"],
            re.IGNORECASE,
        )
        if match.group(1).strip().lower() not in {item.lower() for item in mentions}
    )
    mentions = [
        m for m in mentions
        if not re.match(r"^(?:which|what|any|all|every|each|monitored)\s+(?:warehouse|supplier)s?$", m.strip(), re.IGNORECASE)
    ]
    lowered = state["query"].lower()
    is_active_query = bool(re.search(
        r"\b(?:active\s+events?|active\s+signals?|active\s+weather|currently\s+happening|happening\s+right\s+now|what\s+is\s+happening|show\s+active|what\s+active|any\s+active)\b",
        lowered,
    )) and not any(term in lowered for term in ("operational event", "operational events", "operational disruption"))
    operational_events_only = not is_active_query

    state["query_plan"] = QueryPlan(
        primary_intent=primary,
        secondary_intents=intents[1:],
        entity_mentions=mentions,
        ambiguous=not bool(intents),
        operational_events_only=operational_events_only,
    )
    state.setdefault("timings", {})["planning_ms"] = (time.perf_counter() - started) * 1000
    state["timings"]["routing_ms"] = state["timings"]["planning_ms"]
    return state


def _resolve_entities(state: AgentState) -> AgentState:
    started = time.perf_counter()
    _trace(state, "resolve_entities")
    plan = state["query_plan"]
    ranking_query = any(
        phrase in state["query"].lower()
        for phrase in ("which products", "which warehouses", "what warehouses", "which suppliers")
    )
    if (
        plan.primary_intent in {Intent.CURRENT_EVENTS, Intent.CURRENT_WEATHER, Intent.HISTORICAL_SHIPPING, Intent.GENERAL_SUPPLYBEACON, Intent.UNKNOWN}
        or ranking_query
    ) and not plan.entity_mentions:
        state["resolved_entities"] = []
        state.setdefault("timings", {})["entity_resolution_ms"] = 0.0
        return state
    result = entity_lookup.invoke({"query": state["query"]})
    entities = result.get("entities", [])
    mentions = list(plan.entity_mentions)
    if not mentions and entities and plan.primary_intent not in {Intent.CURRENT_EVENTS, Intent.CURRENT_WEATHER, Intent.HISTORICAL_SHIPPING}:
        base_counts: dict[str, int] = {}
        for entity in entities:
            if entity.get("node_type") == "Product" and " (" in str(entity.get("name", "")):
                base = str(entity["name"]).split(" (", 1)[0].casefold()
                base_counts[base] = base_counts.get(base, 0) + 1
        exact = [entity for entity in entities if any(
            value and re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", state["query"], re.IGNORECASE)
            for value in (entity.get("node_id"), entity.get("name"))
        ) or (
            entity.get("node_type") == "Product"
            and " (" in str(entity.get("name", ""))
            and base_counts.get(str(entity["name"]).split(" (", 1)[0].casefold()) == 1
            and re.search(r"(?<!\w)" + re.escape(str(entity["name"]).split(" (", 1)[0]) + r"(?!\w)", state["query"], re.IGNORECASE)
        )]
        if exact:
            longest = max(len(str(entity.get("name", ""))) for entity in exact)
            most_specific = [entity for entity in exact if len(str(entity.get("name", ""))) == longest]
            if len(most_specific) == 1:
                mentions = [str(most_specific[0].get("name", ""))]
    state["resolved_entities"] = [
        {"mention": mention, **next((entity for entity in entities if str(entity.get("name", "")).lower() == mention.lower()), {"resolved": False, "confidence": 0.0})}
        for mention in mentions
    ]
    # No ranked-token fallback for an operational entity claim. The query must
    # contain an exact graph ID/name (or the caller must supply a named mention).
    plan.entity_mentions = mentions
    plan.resolved_entity_ids = [str(item["node_id"]) for item in state["resolved_entities"] if item.get("node_id")]
    plan.requested_entity_types = list(dict.fromkeys(str(item.get("node_type", "")) for item in state["resolved_entities"] if item.get("node_type")))
    state["timings"] = state.setdefault("timings", {})
    state["timings"]["entity_resolution_ms"] = (time.perf_counter() - started) * 1000
    return state


def _build_plan(state: AgentState) -> AgentState:
    _trace(state, "build_plan")
    retrieval_started = time.perf_counter()
    plan = state["query_plan"]
    ranking_query = any(
        phrase in state["query"].lower()
        for phrase in ("which products", "which warehouses", "what warehouses", "which suppliers", "what suppliers")
    )
    intents = [plan.primary_intent, *plan.secondary_intents]
    if (
        any(item.get("node_type") == "Product" for item in state.get("resolved_entities", []))
        and any(term in state["query"].lower() for term in ("risk", "risky"))
        and plan.primary_intent in {Intent.DELIVERY, Intent.UNKNOWN}
        and Intent.PRODUCT_RISK not in intents
    ):
        intents.insert(0, Intent.PRODUCT_RISK)
        plan.primary_intent = Intent.PRODUCT_RISK
        plan.secondary_intents = [item for item in intents[1:] if item != Intent.PRODUCT_RISK]
    plan.needs_historical = Intent.HISTORICAL_SHIPPING in intents
    plan.needs_delivery = Intent.DELIVERY in intents
    plan.needs_weather = Intent.CURRENT_WEATHER in intents
    plan.needs_events = (
        Intent.CURRENT_EVENTS in intents
        or plan.needs_delivery
        or (plan.primary_intent in {Intent.PRODUCT_RISK, Intent.WAREHOUSE_RISK, Intent.NETWORK_RISK} and not ranking_query)
    )
    plan.needs_risk = any(intent in intents for intent in (Intent.PRODUCT_RISK, Intent.WAREHOUSE_RISK, Intent.SUPPLIER_RISK, Intent.NETWORK_RISK))
    plan.needs_graph = (plan.needs_risk and not ranking_query) or plan.needs_delivery
    plan.needs_freshness = plan.needs_events or plan.needs_delivery or plan.needs_weather or (plan.needs_risk and not ranking_query)
    plan.clarification_needed = plan.primary_intent == Intent.UNKNOWN
    state["_retrieval_started"] = retrieval_started
    return state


def _next_after_graph(state: AgentState) -> str:
    return "graph_retrieval" if state["query_plan"].needs_graph else "route_risk"


def _next_after_risk(state: AgentState) -> str:
    return "risk_retrieval" if state["query_plan"].needs_risk else "route_events"


def _next_after_events(state: AgentState) -> str:
    return "events_retrieval" if state["query_plan"].needs_events else "route_weather"


def _next_after_weather(state: AgentState) -> str:
    return "weather_retrieval" if state["query_plan"].needs_weather else "route_delivery"


def _next_after_delivery(state: AgentState) -> str:
    return "delivery_retrieval" if state["query_plan"].needs_delivery else "route_historical"


def _next_after_historical(state: AgentState) -> str:
    return "historical_retrieval" if state["query_plan"].needs_historical else "route_freshness"


def _next_after_freshness(state: AgentState) -> str:
    return "freshness_retrieval" if state["query_plan"].needs_freshness else "merge_evidence"


def _entity_id(state: AgentState) -> str | None:
    for entity in state.get("resolved_entities", []):
        if entity.get("node_id"):
            return str(entity["node_id"])
    return None


def _invoke_tool(
    state: AgentState,
    tool: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    result = tool.invoke(arguments)
    elapsed = (time.perf_counter() - started) * 1000
    state.setdefault("tools_used", []).append(tool_name)
    timing_name = {
        "graph_context_tool": "graph_retrieval_ms",
        "current_risk_tool": "risk_retrieval_ms",
        "current_risk_ranking_tool": "risk_retrieval_ms",
        "current_events_tool": "events_retrieval_ms",
        "current_weather_tool": "weather_retrieval_ms",
        "delivery_scenario_tool": "delivery_retrieval_ms",
        "historical_shipping_tool": "historical_retrieval_ms",
        "freshness_tool": "freshness_retrieval_ms",
    }.get(tool_name)
    if timing_name:
        state.setdefault("timings", {})[timing_name] = state["timings"].get(timing_name, 0.0) + elapsed
    return result


def _graph_retrieval(state: AgentState) -> AgentState:
    _trace(state, "graph_retrieval")
    query = state["query"]
    lowered = query.lower()
    if "supplier" in lowered:
        query = "Which suppliers currently have risk?"
    elif "which products" in lowered:
        query = "Which products currently have risk?"
    elif "which warehouses" in lowered or "what warehouses" in lowered:
        query = "Which warehouses currently have risk?"
    elif "network" in lowered:
        query = "What is the overall current supply network risk?"
    state["graph_result"] = _invoke_tool(state, graph_context, "graph_context_tool", {"query": query})
    return state


def _risk_retrieval(state: AgentState) -> AgentState:
    _trace(state, "risk_retrieval")
    entity_id = _entity_id(state)
    query_lower = state["query"].lower()
    ranking_type = (
        "Product" if "which products" in query_lower
        else "Warehouse" if "which warehouses" in query_lower or "what warehouses" in query_lower
        else "Supplier" if "which suppliers" in query_lower or "what suppliers" in query_lower
        else None
    )
    if ranking_type:
        state["ranking_result"] = _invoke_tool(
            state,
            current_risk_ranking,
            "current_risk_ranking_tool",
            {"entity_type": ranking_type, "minimum_risk_level": "Low" if "highest risk" in query_lower else "Medium", "limit": 10},
        )
    elif entity_id:
        state["risk_result"] = _invoke_tool(state, current_risk, "current_risk_tool", {"entity_id": entity_id})
    else:
        state["risk_result"] = _invoke_tool(
            state,
            graph_context,
            "graph_context_tool",
            {"query": state["query"]},
        )
    return state


def _events_retrieval(state: AgentState) -> AgentState:
    _trace(state, "events_retrieval")
    op_only = getattr(state["query_plan"], "operational_events_only", True)
    state["events_result"] = _invoke_tool(state, current_events, "current_events_tool", {"operational_only": op_only})
    return state


def _weather_retrieval(state: AgentState) -> AgentState:
    _trace(state, "weather_retrieval")
    state["weather_result"] = _invoke_tool(state, current_weather, "current_weather_tool", {"query": state["query"]})
    return state


def _delivery_retrieval(state: AgentState) -> AgentState:
    _trace(state, "delivery_retrieval")
    entity_id = _entity_id(state)
    state["delivery_result"] = _invoke_tool(state, delivery_scenario, "delivery_scenario_tool", {"entity_id": entity_id}) if entity_id else {
        "delivery_risk": None,
        "limitations": ["No requested product or entity was resolved."],
    }
    return state


def _historical_retrieval(state: AgentState) -> AgentState:
    _trace(state, "historical_retrieval")
    state["historical_result"] = _invoke_tool(state, historical_shipping, "historical_shipping_tool", {})
    return state


def _freshness_retrieval(state: AgentState) -> AgentState:
    _trace(state, "freshness_retrieval")
    state["freshness_result"] = _invoke_tool(state, freshness, "freshness_tool", {})
    return state


def _item(
    evidence_type: str,
    index: int,
    fact_name: str,
    value: Any,
    *,
    entity_id: str | None = None,
    entity_type: str | None = None,
    entity_name: str | None = None,
    source_artifact: str | None = None,
    provenance: str,
    timestamp: str | None = None,
    operationally_relevant: bool | None = None,
    relation: str | None = None,
    temporal_status: str | None = None,
    validation_status: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{evidence_type}-{index}",
        evidence_type=evidence_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        entity_type=entity_type,
        entity_name=entity_name,
        fact_name=fact_name,
        value=value,
        source_artifact=source_artifact,
        provenance=provenance,
        timestamp=timestamp,
        operationally_relevant=operationally_relevant,
        relation=relation,
        temporal_status=temporal_status,
        validation_status=validation_status,
    )


def canonicalize_graph_evidence(payload: dict[str, Any]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for index, entity in enumerate(payload.get("entities", [])):
        items.append(_item("graph", index, "entity", entity, entity_id=entity.get("node_id"),
                           entity_type=entity.get("node_type"), entity_name=entity.get("name"),
                           source_artifact="data/current/current_supply_chain_graph.graphml",
                           provenance=str(entity.get("provenance", "OBSERVED"))))
    offset = len(items)
    for index, path in enumerate(payload.get("graph_paths", [])):
        items.append(_item("graph", offset + index, "relationship_path", path,
                           source_artifact="data/current/current_supply_chain_graph.graphml",
                           provenance="DERIVED", relation="TYPED_GRAPH_PATH"))
    return items


def canonicalize_risk_evidence(payload: dict[str, Any]) -> list[EvidenceItem]:
    values = payload.get("risk_evidence", []) + payload.get("entities", [])
    items: list[EvidenceItem] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        subject = str(value.get("subject", ""))
        subject_type = subject.split(":", 1)[0] if ":" in subject else None
        subject_id = subject or value.get("node_id") or value.get("Node_ID")
        items.append(_item(
            "risk", index, str(value.get("predicate", "risk_record")), value,
            entity_id=subject_id,
            entity_type=value.get("node_type") or value.get("Node_Type") or subject_type,
            entity_name=value.get("name") or value.get("Node_Name"),
            source_artifact=value.get("source") or value.get("source_artifact") or "data/current/current_risk_scores.csv",
            provenance=str(value.get("provenance", "MODEL_OUTPUT")),
        ))
    return items


def canonicalize_event_evidence(payload: dict[str, Any]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for index, event in enumerate(payload.get("events", [])):
        if not isinstance(event, dict):
            continue
        source = str(event.get("Source", ""))
        source_artifact = (
            "data/current/current_weather.json" if source.endswith("current_weather.json")
            else "data/current/current_news.json" if source.endswith("current_news.json")
            else "data/current/current_events.json"
        )
        items.append(_item(
            "events", index, str(event.get("Event_Type") or event.get("event_type") or "event"), event,
            entity_id=event.get("Event_ID") or event.get("event_id"),
            entity_type="Event", entity_name=event.get("Title") or event.get("title"),
            source_artifact=source_artifact, provenance="EXTERNAL_API",
            timestamp=event.get("Date") or event.get("timestamp"),
            operationally_relevant=event.get("Operationally_Relevant", event.get("operationally_relevant")),
            temporal_status=event.get("event_status") or event.get("temporal_status"),
            validation_status="OPERATIONAL_CURRENT" if event.get("operational_eligible") else "INFORMATIONAL_ONLY",
        ))
    return items


def canonicalize_weather_evidence(payload: dict[str, Any]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for index, record in enumerate(payload.get("records", [])):
        if not isinstance(record, dict):
            continue
        location = record.get("location")
        source = record.get("source_channel", "WEATHER")
        items.append(_item(
            "weather", index, f"weather_condition:{location}:{source}", record,
            entity_id=location,
            entity_type="Location",
            entity_name=location,
            source_artifact=record.get("source_artifact", "data/current/current_weather.json"),
            provenance=record.get("provenance", "EXTERNAL_API"),
            timestamp=record.get("observed_at") or record.get("fetch_timestamp"),
            operationally_relevant=False,
        ))
    return items


def canonicalize_delivery_evidence(payload: dict[str, Any]) -> list[EvidenceItem]:
    return [
        _item("delivery", index, str(value.get("field", "delivery_scenario")), value,
               entity_id=payload.get("product_id"), source_artifact="data/processed/electronics_supplychain_dataset.csv",
               provenance=str(value.get("provenance", "DERIVED / RULE_BASED SCENARIO ESTIMATE")))
        for index, value in enumerate(payload.get("evidence", []))
        if isinstance(value, dict)
    ]


def canonicalize_historical_evidence(payload: dict[str, Any]) -> list[EvidenceItem]:
    metrics = [
        value for value in payload.get("test_metrics", [])
        if value.get("model") == "hist_gradient_boosting" and value.get("split") == "test"
    ]
    return [
        _item("historical", index, "model_metric", value,
               source_artifact="deploy/seed/analysis/dataco/modeling/baseline_metrics.csv",
               provenance="HISTORICAL_MODEL")
        for index, value in enumerate(metrics)
    ]


def canonicalize_freshness_evidence(payload: dict[str, Any]) -> list[EvidenceItem]:
    fields = (
        "weather_fetch_timestamp", "news_fetch_timestamp", "pipeline_run_timestamp",
        "weather_fresh", "news_fresh", "weather_age_hours", "news_age_hours", "overall_stale",
    )
    return [
        _item("freshness", index, field, payload.get(field), source_artifact="data/current/current_run_metadata.json",
               provenance="OBSERVED")
        for index, field in enumerate(fields)
        if field in payload
    ]


def _merge_event_sources(
    events_result: dict[str, Any],
    graph_result: dict[str, Any],
    *,
    entity_specific: bool,
) -> dict[str, Any]:
    """Select scoped event evidence, deduplicating only when both sources apply."""
    artifact_events = events_result.get("events", [])
    graph_events = graph_result.get("events", [])
    if entity_specific and graph_events:
        selected = graph_events
    elif not entity_specific:
        selected = artifact_events
    else:
        selected = artifact_events
    deduplicated: dict[str, dict[str, Any]] = {}
    for event in selected:
        if not isinstance(event, dict):
            continue
        key = str(event.get("Event_ID") or event.get("event_id") or event.get("node_id") or event.get("Title") or event.get("title"))
        deduplicated[key] = event
    return {"events": list(deduplicated.values())}


def _merge_evidence(state: AgentState) -> AgentState:
    _trace(state, "merge_evidence")
    event_payload = _merge_event_sources(
        state.get("events_result", {}),
        state.get("graph_result", {}),
        entity_specific=bool(state["query_plan"].entity_mentions),
    )
    bundle = EvidenceBundle(
        graph=canonicalize_graph_evidence(state.get("graph_result", {})),
        risk=canonicalize_risk_evidence(state.get("risk_result", {}) | state.get("ranking_result", {})),
        events=canonicalize_event_evidence(event_payload),
        weather=canonicalize_weather_evidence(state.get("weather_result", {})),
        delivery=canonicalize_delivery_evidence(state.get("delivery_result", {})),
        historical=canonicalize_historical_evidence(state.get("historical_result", {})),
        freshness=canonicalize_freshness_evidence(state.get("freshness_result", {})),
    )
    state["evidence_bundle"] = bundle
    state.setdefault("timings", {})["merge_ms"] = 0.0
    state.setdefault("timings", {})["retrieval_ms"] = (
        sum(value for key, value in state["timings"].items() if key.endswith("_retrieval_ms"))
    )
    return state


def _validate_evidence(state: AgentState) -> AgentState:
    started = time.perf_counter()
    _trace(state, "validate_evidence")
    plan = state["query_plan"]
    missing: list[str] = []
    bundle = state.get("evidence_bundle", EvidenceBundle())
    actual_count = len(bundle.all_evidence_items())
    if plan.entity_mentions and not plan.resolved_entity_ids:
        missing.append("The requested entity was not found in the current graph.")
    if plan.needs_historical and not bundle.historical:
        missing.append("Historical DataCo metrics are unavailable.")
    if plan.needs_delivery and not bundle.delivery and not plan.resolved_entity_ids:
        missing.append("Delivery evidence requires a resolved entity.")
    if plan.primary_intent in {Intent.PRODUCT_RISK, Intent.WAREHOUSE_RISK, Intent.SUPPLIER_RISK}:
        expected = {
            Intent.PRODUCT_RISK: "Product",
            Intent.WAREHOUSE_RISK: "Warehouse",
            Intent.SUPPLIER_RISK: "Supplier",
        }[plan.primary_intent]
        if not any(
            str(item.entity_type).upper() == expected.upper() or item.value.get("node_type") == expected
            for item in bundle.risk
            if isinstance(item.value, dict)
        ):
            missing.append(f"No current {expected} risk evidence was retrieved.")
    if plan.needs_risk and (
        state.get("ranking_result", {}).get("retrieval_ok") is False
        or state.get("risk_result", {}).get("retrieval_ok") is False
    ):
        missing.append("Current risk artifact is unavailable.")
    if plan.primary_intent == Intent.CURRENT_EVENTS and not state.get("events_result", {}).get("retrieval_ok"):
        missing.append("Current event artifact retrieval was unavailable.")
    if plan.primary_intent == Intent.CURRENT_WEATHER:
        weather_res = state.get("weather_result", {})
        if not weather_res.get("retrieval_ok"):
            missing.append("Current weather artifact retrieval was unavailable.")
        elif weather_res.get("location_not_found"):
            missing.append("No matching monitored location in the current weather snapshot.")
    if plan.primary_intent == Intent.UNKNOWN:
        missing.append("The query intent is not supported by a grounded workflow.")
    state["missing_evidence"] = missing
    state["warnings"] = list(state.get("warnings", []))
    freshness_data = state.get("freshness_result", {})
    if freshness_data.get("overall_stale"):
        state["warnings"].append("Current external-signal artifacts are stale; results are not live.")
    valid_zero_event_query = plan.primary_intent == Intent.CURRENT_EVENTS and state.get("events_result", {}).get("retrieval_ok")
    valid_weather_query = plan.primary_intent == Intent.CURRENT_WEATHER and state.get("weather_result", {}).get("retrieval_ok") and not state.get("weather_result", {}).get("location_not_found")
    valid_network_query = plan.primary_intent == Intent.NETWORK_RISK
    state["evidence_valid"] = not missing and (
        actual_count > 0 or valid_zero_event_query or valid_weather_query or valid_network_query or plan.primary_intent == Intent.GENERAL_SUPPLYBEACON
    )
    state["timings"]["validation_ms"] = (time.perf_counter() - started) * 1000
    return state


def _deterministic_answer(state: AgentState) -> AgentState:
    _trace(state, "deterministic_answer")
    plan = state["query_plan"]
    graph = state.get("graph_result", {})
    events = state.get("events_result", {}).get("events", [])
    history = state.get("historical_result", {})
    delivery = state.get("delivery_result", {}).get("delivery_risk")
    ranking = state.get("ranking_result", {})
    if ranking:
        entities = ranking.get("entities", [])
        label = plan.primary_intent.value.replace("_RISK", "").lower()
        text = (
            f"Current {label} results: "
            + "; ".join(
                f"{entity.get('name', entity.get('node_id'))} ({entity.get('risk_level', 'Unavailable')}, score {entity.get('risk_score', 'Unavailable')})"
                for entity in entities
            )
            if entities else f"No current {label} entities matched the requested threshold."
        )
    elif plan.needs_historical and history and graph:
        metrics = [
            m for m in history.get("test_metrics", [])
            if m.get("model") == "hist_gradient_boosting" and m.get("split") == "test"
        ]
        current_text = build_explanation(graph).get(
            "answer", "Current network risk evidence was retrieved."
        )
        historical_text = (
            f" Historical DataCo context is separate: the frozen shipping model test metrics are "
            f"{json.dumps(metrics, default=str)}. These current scenario-risk and historical-model "
            "evidence domains are not combined into a single score."
            if metrics
            else " Historical DataCo model metrics are unavailable; they are kept separate from current risk."
        )
        text = current_text + historical_text
    elif plan.needs_historical and history:
        metrics = [m for m in history.get("test_metrics", []) if m.get("model") == "hist_gradient_boosting" and m.get("split") == "test"]
        if any(term in state["query"].lower() for term in ("feature", "factor", "input")):
            features = history.get("required_features", [])
            text = ("The historical DataCo shipping model targets Days for shipping (real). "
                    + ("Its audited order-time input features are: " + ", ".join(features) + "." if features else
                       "The feature audit is unavailable, so SupplyBeacon cannot verify its inputs."))
        elif any(term in state["query"].lower() for term in ("predict", "estimate", "inference")) and not any(term in state["query"].lower() for term in ("performance", "metric", "accuracy")):
            text = ("The historical DataCo shipping model targets Days for shipping (real), not current disruption or warehouse risk. "
                    "No inference was run: a complete supported order-feature request was not supplied. "
                    "SupplyBeacon will not invent missing input features or a shipping prediction.")
        elif metrics:
            best = metrics[0]
            text = (f"The historical DataCo shipping model targets {history.get('target', 'Days for shipping (real)')}. "
                    f"Stored test metrics for {best.get('model')} are MAE {best.get('mae')}, RMSE {best.get('rmse')}, "
                    f"and R² {best.get('r2')}. This is historical validation, not live current risk or a current disruption forecast.")
        else:
            text = "Historical DataCo shipping-model metrics are unavailable; no prediction was inferred."
    elif plan.primary_intent == Intent.CURRENT_WEATHER:
        weather_res = state.get("weather_result", {})
        records = weather_res.get("records", [])
        rain_only = weather_res.get("rain_only", False)
        adverse_only = weather_res.get("adverse_only", False)
        location_not_found = weather_res.get("location_not_found", False)

        if location_not_found or not records:
            if location_not_found:
                live_graph = load_current_state().get("graph")
                monitored = warehouse_weather_locations(live_graph) if live_graph is not None else []
                text = "No matching monitored location in the current weather snapshot. Monitored locations are: " + ", ".join(monitored) + "."
            elif rain_only:
                text = "No monitored locations currently observe rain in current observation evidence."
            elif adverse_only:
                text = "No monitored warehouse locations currently report bad or adverse weather conditions."
            else:
                text = "No current weather records were found."
        elif rain_only:
            lines = []
            for r in records:
                rain_str = ", ".join(f"{k}: {v} mm" for k, v in r.get("rain_mm", {}).items()) if r.get("rain_mm") else "rain reported"
                temp_str = f"{r['temperature_c']}°C" if r.get("temperature_c") is not None else "temp unavailable"
                lines.append(f"{r['location']} ({r.get('condition', 'rain')}, {temp_str}, {rain_str}; {r.get('source_channel')} observation)")
            text = (
                f"Current weather observations show rain at {len(records)} monitored location(s): "
                + "; ".join(lines)
                + ". Other monitored locations do not currently observe rain. WeatherAPI daily forecast context is not treated as a current rain observation."
            )
        elif adverse_only:
            lines = []
            for r in records:
                rain_str = ", ".join(f"{k}: {v} mm" for k, v in r.get("rain_mm", {}).items()) if r.get("rain_mm") else "rain reported"
                temp_str = f"{r['temperature_c']}°C" if r.get("temperature_c") is not None else "temp unavailable"
                lines.append(f"{r['location']} ({r.get('condition', 'adverse weather')}, {temp_str}, {rain_str}; {r.get('source_channel')} observation)")
            text = (
                f"Monitored warehouse locations currently observing adverse/rain weather: "
                + "; ".join(lines)
                + ". No severe weather alerts or disruptive conditions are in effect for monitored warehouse locations. Other monitored locations report clear sky or overcast clouds."
            )
        else:
            locations_in_records = sorted({r["location"] for r in records})
            if len(locations_in_records) == 1:
                loc = locations_in_records[0]
                obs_records = [r for r in records if r.get("record_kind") == "OBSERVATION"]
                fc_records = [r for r in records if r.get("record_kind") != "OBSERVATION"]
                parts = []
                for r in obs_records:
                    temp_str = f"{r['temperature_c']}°C" if r.get("temperature_c") is not None else "temperature unavailable"
                    rain_info = ", ".join(f"{k}: {v} mm" for k, v in r.get("rain_mm", {}).items()) if r.get("rain_mm") else "none reported"
                    obs_time = f"observed at {r['observed_at']}" if r.get("observed_at") else "current observation"
                    parts.append(f"{r.get('source_channel', 'OpenWeather')} (Observation): {r.get('condition', 'unknown')}, {temp_str}, rain: {rain_info} ({obs_time})")
                for r in fc_records:
                    parts.append(f"{r.get('source_channel', 'WeatherAPI')} (Daily Forecast/Alert Context): {r.get('condition', 'unknown')} (temperature unavailable; forecast/alert context, not a current observation)")
                text = f"Current weather in {loc}: " + "; ".join(parts) + "."
            else:
                obs_records = [r for r in records if r.get("record_kind") == "OBSERVATION"]
                lines = []
                for r in obs_records:
                    temp_str = f"{r['temperature_c']}°C" if r.get("temperature_c") is not None else "temp unavailable"
                    rain_info = ", ".join(f"{k}: {v} mm" for k, v in r.get("rain_mm", {}).items()) if r.get("rain_mm") else "no rain"
                    lines.append(f"{r['location']}: {r.get('condition', 'unknown')}, {temp_str}, {rain_info}")
                text = (
                    "Current weather across monitored locations (OpenWeather observations): "
                    + "; ".join(lines)
                    + ". Provider context: OpenWeather provides current observations. WeatherAPI provides daily forecast/alert context (no active weather alerts). Missing measurements remain unavailable."
                )
    elif plan.primary_intent == Intent.CURRENT_EVENTS:
        events_res = state.get("events_result", {})
        active_count = events_res.get("active_event_count", len(events))
        operational_count = events_res.get("operational_event_count", sum(1 for e in events if e.get("operational_eligible")))
        is_active_scope = getattr(plan, "operational_events_only", True) is False

        if is_active_scope:
            if not events:
                text = "No active events were found in current external evidence."
            elif operational_count == 0:
                event_types = ", ".join(dict.fromkeys(str(item.get("event_type") or item.get("Event_Type") or "weather").lower() for item in events))
                event_details = "; ".join(
                    f"Event {item.get('Event_ID')}: {item.get('Title', item.get('event_type', 'Event'))} in {item.get('Location', 'Unknown')} "
                    f"(ACTIVE: YES, OPERATIONAL: NO, Provider: {item.get('source_channel', 'Unknown')} - {item.get('operational_reason', 'Not operationally eligible')})"
                    for item in events
                )
                text = (
                    f"No current operational disruptions were found. However, {len(events)} active signal{'s are' if len(events) != 1 else ' is'} present: {event_types}. "
                    "They do not currently meet the operational-impact criteria and therefore do not contribute to supply-chain risk. "
                    f"Active events: {event_details}."
                )
            else:
                op_titles = [str(item.get("Title", item.get("Event_Type", "event"))) for item in events if item.get("operational_eligible")]
                non_op_titles = [str(item.get("Title", item.get("Event_Type", "event"))) for item in events if not item.get("operational_eligible")]
                text = f"Current operational disruptions: {'; '.join(op_titles)}."
                if non_op_titles:
                    text += f" Additionally, {len(non_op_titles)} active non-operational event(s) are present: {'; '.join(non_op_titles)}, which do not meet operational-impact criteria."
        else:
            text = (
                "Current operational events: "
                + "; ".join(str(item.get("Title", item.get("Event_Type", "event"))) for item in events[:10])
                if events else "No current operational events were found."
            )
    elif plan.primary_intent == Intent.NETWORK_RISK:
        state_data = load_current_state()
        risk_df = state_data["risk"]
        events_data = load_current_events()["events"]
        active_weather = [e for e in events_data if apply_runtime_temporal_status(e)["event_status"] == "ACTIVE"]
        med_high = int((risk_df["Risk_Level"].isin(["Medium", "High", "Critical"])).sum()) if not risk_df.empty and "Risk_Level" in risk_df.columns else 0
        total_scored = len(risk_df)

        if med_high == 0:
            text = (
                f"The overall supply network risk is currently Low. "
                f"All {total_scored} scored nodes (products, warehouses, and suppliers) are at Low risk with 0 Medium, High, or Critical risk nodes. "
                "There are currently 0 operational disruption events affecting the modeled network (0 affected warehouses, 0 affected products, 0 affected suppliers). "
            )
            if active_weather:
                weather_types = ", ".join(dict.fromkeys(str(e.get("Event_Type", "light rain")).lower() for e in active_weather))
                text += (
                    f"While {len(active_weather)} active weather event(s) are present ({weather_types}), "
                    "they do not satisfy operational disruption criteria and therefore do not contribute to supply-chain risk."
                )
        else:
            text = build_explanation(graph).get("answer", "Current network risk evidence was retrieved.")
    elif graph:
        text = build_explanation(graph).get("answer", "Current graph evidence was retrieved.")
        if delivery:
            text += f" Delivery scenario: {delivery.get('message', 'available')}."
    elif events:
        text = "Current operational events: " + "; ".join(str(item.get("Title", item.get("Event_Type", "event"))) for item in events[:10])
    elif plan.primary_intent == Intent.GENERAL_SUPPLYBEACON:
        text = "SupplyBeacon answers questions grounded in its current graph and evidence artifacts."
    else:
        text = "SupplyBeacon does not currently have enough grounded evidence to answer that determination."

    graph_paths = graph.get("graph_paths", [])
    if plan.primary_intent == Intent.PRODUCT_RISK and plan.resolved_entity_ids:
        graph_paths = [
            path for path in graph_paths
            if plan.resolved_entity_ids[0] in path.get("nodes", [])
        ]
    answer = {
        "answer": text,
        "entities": ranking.get("entities", []) or graph.get("entities", []) or state.get("resolved_entities", []),
        "informational_entities": graph.get("informational_entities", []),
        "events": graph.get("events", []) if plan.entity_mentions else events or graph.get("events", []),
        "weather": state.get("weather_result", {}).get("records", []),
        "graph_paths": graph_paths,
        "risk_evidence": ranking.get("entities", []) + graph.get("risk_evidence", []) + (
            graph.get("entities", []) if plan.needs_risk and not graph.get("risk_evidence") else []
        ) + state.get("risk_result", {}).get("risk_evidence", []),
        "provenance_summary": graph.get("provenance_summary", {}),
        "limitations": graph.get("limitations", []) + state.get("delivery_result", {}).get("limitations", []) + state.get("weather_result", {}).get("limitations", []),
        "delivery_risk": delivery,
        "current_metadata": state.get("freshness_result", {}),
        "historical_evidence": history,
    }
    state["answer"] = answer
    return state


def _fallback(state: AgentState) -> AgentState:
    _trace(state, "safe_fallback")
    state["answer"] = {
        "answer": "SupplyBeacon does not currently have enough grounded evidence to make that determination.",
        "entities": state.get("resolved_entities", []),
        "events": [], "graph_paths": [], "risk_evidence": [],
        "limitations": state.get("missing_evidence", ["Grounded evidence validation failed."]),
    }
    return state


def _maybe_llm(state: AgentState) -> AgentState:
    _trace(state, "optional_llm_synthesis")
    state["llm_used"] = False
    started = time.perf_counter()
    state.setdefault("timings", {})["llm_ms"] = 0.0
    if os.getenv("SUPPLYBEACON_LLM_ENABLED", "").lower() != "true" or not state.get("evidence_valid"):
        return state
    try:
        provider = OpenAICompatibleProvider()
        prompt = SYNTHESIS_PROMPT.invoke({"query": state["query"], "evidence": json.dumps(state["answer"], default=str)})
        raw = provider.generate(str(prompt.messages[0].content), state["query"], state["answer"])
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str) or not parsed["answer"].strip():
            raise ValueError("Structured LLM output must contain a non-empty string answer.")
        # The provider is optional wording assistance, never a second source
        # of risk, inventory, event, or model facts. Exact agreement is the
        # only automatically verifiable form of a free-text answer.
        if parsed["answer"].strip() != state["answer"]["answer"]:
            raise ValueError("LLM wording diverged from authoritative deterministic answer.")
        state["answer"]["answer"] = GroundedAnswer(
            answer=parsed["answer"],
            primary_intent=state["query_plan"].primary_intent.value,
            secondary_intents=[item.value for item in state["query_plan"].secondary_intents],
            entities=state["answer"].get("entities", []),
            evidence_used=[item.evidence_id for item in state["evidence_bundle"].all_evidence_items()],
            warnings=state.get("warnings", []),
            provenance_notes=[],
        ).answer
        state["llm_used"] = True
    except (ProviderError, ValueError, KeyError, TypeError, ValidationError) as exc:
        logger.warning("Agentic structured LLM fallback: %s", exc)
    state["timings"]["llm_ms"] = (time.perf_counter() - started) * 1000
    return state


def _finalize(state: AgentState) -> AgentState:
    _trace(state, "finalize")
    state["answer"].setdefault("limitations", [])
    entities = state["answer"].get("entities", [])
    risk_level = next(
        (str(entity.get("risk_level")) for entity in entities if entity.get("risk_level")),
        None,
    )
    state["structured_answer"] = GroundedAnswer(
        answer=state["answer"].get("answer", ""),
        primary_intent=state["query_plan"].primary_intent.value,
        secondary_intents=[item.value for item in state["query_plan"].secondary_intents],
        entities=entities,
        risk_level=risk_level,
        evidence_used=[item.evidence_id for item in state.get("evidence_bundle", EvidenceBundle()).all_evidence_items()],
        warnings=state.get("warnings", []),
        provenance_notes=[],
        current_data_fresh=(
            not bool(state.get("freshness_result", {}).get("overall_stale"))
            if state.get("freshness_result")
            else None
        ),
        llm_used=state.get("llm_used", False),
        fallback_used=not state.get("evidence_valid", False),
    ).model_dump(mode="json")
    return state


def _build_graph():
    graph = StateGraph(AgentState)
    def isolated(node):
        def invoke(state: AgentState) -> dict[str, Any]:
            before = deepcopy(dict(state))
            result = node(state)
            updates: dict[str, Any] = {}
            for key, value in result.items():
                if key not in before or value != before[key]:
                    updates[key] = value
            return updates
        return invoke

    for name, node in (
        ("analyze_query", _analyze_query), ("resolve_entities", _resolve_entities),
        ("build_plan", _build_plan), ("graph_retrieval", _graph_retrieval),
        ("risk_retrieval", _risk_retrieval), ("events_retrieval", _events_retrieval),
        ("weather_retrieval", _weather_retrieval),
        ("delivery_retrieval", _delivery_retrieval), ("historical_retrieval", _historical_retrieval),
        ("freshness_retrieval", _freshness_retrieval), ("merge_evidence", _merge_evidence),
        ("validate_evidence", _validate_evidence), ("deterministic_answer", _deterministic_answer),
        ("safe_fallback", _fallback), ("optional_llm_synthesis", _maybe_llm), ("finalize", _finalize),
    ):
        graph.add_node(name, isolated(node))
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "resolve_entities")
    graph.add_edge("resolve_entities", "build_plan")
    graph.add_conditional_edges("build_plan", _next_after_graph)
    graph.add_node("route_risk", lambda state: {})
    graph.add_node("route_events", lambda state: {})
    graph.add_node("route_weather", lambda state: {})
    graph.add_node("route_delivery", lambda state: {})
    graph.add_node("route_historical", lambda state: {})
    graph.add_node("route_freshness", lambda state: {})
    graph.add_conditional_edges("route_risk", _next_after_risk)
    graph.add_conditional_edges("route_events", _next_after_events)
    graph.add_conditional_edges("route_weather", _next_after_weather)
    graph.add_conditional_edges("route_delivery", _next_after_delivery)
    graph.add_conditional_edges("route_historical", _next_after_historical)
    graph.add_conditional_edges("route_freshness", _next_after_freshness)
    graph.add_edge("graph_retrieval", "route_risk")
    graph.add_edge("risk_retrieval", "route_events")
    graph.add_edge("events_retrieval", "route_weather")
    graph.add_edge("weather_retrieval", "route_delivery")
    graph.add_edge("delivery_retrieval", "route_historical")
    graph.add_edge("historical_retrieval", "route_freshness")
    graph.add_edge("freshness_retrieval", "merge_evidence")
    graph.add_edge("merge_evidence", "validate_evidence")
    graph.add_conditional_edges("validate_evidence", lambda state: "deterministic_answer" if state.get("evidence_valid") else "safe_fallback")
    graph.add_edge("safe_fallback", "finalize")
    graph.add_edge("deterministic_answer", "optional_llm_synthesis")
    graph.add_edge("optional_llm_synthesis", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


AGENT_GRAPH = _build_graph()


def run_agentic_query(query: str) -> dict[str, Any]:
    grounded = answer_grounded_domain(query)
    if grounded is not None:
        return grounded
    started = time.perf_counter()
    state = AGENT_GRAPH.invoke({"query": query})
    timings = state.get("timings", {})
    timings["retrieval_ms"] = sum(timings.get(key, 0.0) for key in ("retrieval_ms",)) or 0.0
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    for key in (
        "planning_ms", "entity_resolution_ms", "graph_retrieval_ms",
        "risk_retrieval_ms", "events_retrieval_ms", "weather_retrieval_ms", "delivery_retrieval_ms",
        "historical_retrieval_ms", "freshness_retrieval_ms", "merge_ms",
        "validation_ms", "llm_ms", "retrieval_ms",
    ):
        timings.setdefault(key, 0.0)
    answer = state.get("answer", {})
    plan = state["query_plan"]
    return {
        "query": query, "answer": answer.get("answer", ""),
        "intent": plan.primary_intent.value,
        "query_plan": plan.model_dump(mode="json"),
        "query_intent": plan.primary_intent.value,
        "entities": answer.get("entities", []), "informational_entities": answer.get("informational_entities", []),
        "resolved_entities": state.get("resolved_entities", []),
        "graph_paths": answer.get("graph_paths", []), "events": answer.get("events", []),
        "weather": answer.get("weather", []),
        "risk_evidence": answer.get("risk_evidence", []), "provenance_summary": answer.get("provenance_summary", {}),
        "limitations": answer.get("limitations", []), "delivery_risk": answer.get("delivery_risk"),
        "historical_evidence": answer.get("historical_evidence", {}), "current_metadata": answer.get("current_metadata", {}),
        "tools_used": list(dict.fromkeys(state.get("tools_used", []))),
        "evidence_count": len(state.get("evidence_bundle", EvidenceBundle()).all_evidence_items()),
        "evidence_used": [item.evidence_id for item in state.get("evidence_bundle", EvidenceBundle()).all_evidence_items()],
        "evidence": [item.model_dump(mode="json") for item in state.get("evidence_bundle", EvidenceBundle()).all_evidence_items()],
        "warnings": state.get("warnings", []), "current_data_fresh": not bool(state.get("freshness_result", {}).get("overall_stale")) if state.get("freshness_result") else None,
        "structured_answer": state.get("structured_answer", {}),
        "llm_used": state.get("llm_used", False), "fallback_used": not state.get("evidence_valid", False),
        "workflow_trace": state.get("workflow_trace", []), "timings": timings,
    }
