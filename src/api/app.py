"""Minimal HTTP API for the frozen deterministic GraphRAG backend."""
from __future__ import annotations

import threading
import os
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, AsyncIterator

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parents[2]
from src.external_data.utils import load_env

# Load local defaults before importing modules that resolve filesystem paths.
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    for _name, _value in load_env(_env_path).items():
        os.environ.setdefault(_name, _value)

from src.graphrag.explanation_builder import build_explanation
from src.dynamic_risk.current_state import CURRENT_DIR, bootstrap_current_data, freshness_status, load_current_state
from src.dynamic_risk.live_situation import build_live_situation
from src.dynamic_risk.refresh_current_state import refresh_current_state
from src.dynamic_risk.risk_views import COMPONENTS, network_summary, numeric as risk_numeric, risk_availability, watchlist
from src.event_extraction.temporal_validity import apply_runtime_temporal_status, is_currently_eligible
from src.agentic.service import run_agentic_query
from src.agentic.grounded_domains import answer_grounded_domain
from src.agentic.tools import current_retriever, invalidate_retriever_cache
from src.llm_agent.provider import OpenAICompatibleProvider, ProviderError
from src.master_data.router import router as master_data_router
from src.master_data.repository import connect as connect_master_database, generated_dir as master_generated_dir


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    bootstrap_current_data()
    master_generated_dir().mkdir(parents=True, exist_ok=True)
    connection = connect_master_database()
    connection.close()
    yield


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class ScenarioRequest(BaseModel):
    scenario_type: str
    target_node_id: str
    duration_hours: int
    scenario_severity: str


class ScenarioResponse(BaseModel):
    scenario_id: str
    generated_at: str
    scenario: dict[str, Any]
    current_state: dict[str, Any]
    scenario_state: dict[str, Any]
    delta: dict[str, Any]
    newly_exposed_product_ids: list[str]
    overlap_product_ids: list[str]
    product_ids: list[str]
    supplier_ids: list[str]
    warehouse_ids: list[str]
    location_ids: list[str]
    alternative_warehouse_candidates: list[dict[str, Any]]
    alternative_supplier_candidates: list[dict[str, Any]]
    assumptions: list[str]
    provenance: str
    inventory_relationship_available: bool
    catalog_only: bool
    catalog_association_count: int
    affected_product_ids: list[str]
    affected_product_count: int
    affected_supplier_ids: list[str]
    affected_supplier_count: int
    current_risk_score: float | None
    current_risk_level: str
    impact_status: str
    limitations: list[str]
    evidence_basis: list[str]
    explanation: str


class ScenarioExplanationRequest(BaseModel):
    candidate_type: str
    evidence: dict[str, Any]


class ScenarioExplanationResponse(BaseModel):
    answer: str
    provenance: str
    limitations: list[str]
    explanation_mode: str


app = FastAPI(title="ElectroGraph-AI GraphRAG API", lifespan=lifespan)
app.include_router(master_data_router)
_configured_origins = os.getenv(
    "SUPPLYBEACON_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
_allowed_origins = [origin.strip() for origin in _configured_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_refresh_lock = threading.Lock()


def _risk_rows(state: dict[str, Any], node_type: str | None = None) -> pd.DataFrame:
    risk = state["risk"]
    if risk.empty:
        return risk
    if node_type is not None:
        return risk[risk["Node_Type"].astype(str) == node_type].copy()
    return risk.copy()


def _risk_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["Node_ID"]): row.to_dict()
        for _, row in state["risk"].iterrows()
        if str(row.get("Node_ID", ""))
    }


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _neighbors(graph: Any, node_id: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if graph is None or node_id not in graph:
        return []
    results = []
    for neighbor in graph.successors(node_id):
        results.append((neighbor, graph.nodes[neighbor], graph.get_edge_data(node_id, neighbor) or {}))
    for neighbor in graph.predecessors(node_id):
        results.append((neighbor, graph.nodes[neighbor], graph.get_edge_data(neighbor, node_id) or {}))
    return results


def _risk_payload(row: dict[str, Any], fields: dict[str, Any] | None = None) -> dict[str, Any]:
    score = risk_numeric(row.get("Risk_Score"))
    payload = {
        "node_id": str(row.get("Node_ID", "")),
        "name": row.get("Node_Name", "Unavailable"),
        "risk_score": score,
        "risk_level": row.get("Risk_Level", "Unavailable") if score is not None else "Unavailable",
        "risk_availability": risk_availability(row),
        "connected_events": int(_numeric(row.get("Connected_Events"))),
        "risk_components": {field: risk_numeric(row.get(field)) for field in COMPONENTS},
        "risk_reason": row.get("Risk_Reason") or "No applicable operational risk evidence is available.",
    }
    if fields:
        payload.update(fields)
    return payload


def _product_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    risk_map = _risk_map(state)
    graph = state["graph"]
    rows = []
    for node_id, attrs in (graph.nodes(data=True) if graph is not None else []):
        if attrs.get("node_type") != "Product":
            continue
        row = risk_map.get(str(node_id), {})
        warehouse = next(
            (neighbor_attrs.get("name", neighbor_id) for neighbor_id, neighbor_attrs, edge in _neighbors(graph, str(node_id))
             if neighbor_attrs.get("node_type") == "Warehouse"),
            "Unavailable",
        )
        rows.append(_risk_payload(row, {
            "node_id": str(node_id),
            "name": attrs.get("name", node_id),
            "brand": attrs.get("brand", "Unavailable"),
            "category": attrs.get("category", "Unavailable"),
            "warehouse": warehouse,
            "risk_reason": row.get("Risk_Reason", "Unavailable"),
        }))
    return rows


def _warehouse_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    risk_map = _risk_map(state)
    graph = state["graph"]
    watch_by_id = {item["entity_id"]: item for item in watchlist(state)}
    rows = []
    for node_id, attrs in (graph.nodes(data=True) if graph is not None else []):
        if attrs.get("node_type") != "Warehouse":
            continue
        row = risk_map.get(str(node_id), {})
        products = {
            neighbor_id for neighbor_id, neighbor_attrs, _ in _neighbors(graph, str(node_id))
            if neighbor_attrs.get("node_type") == "Product"
        }
        event_edges = [
            edge for neighbor_id, neighbor_attrs, edge in _neighbors(graph, str(node_id))
            if neighbor_attrs.get("node_type") == "Event" and edge.get("relationship") == "AFFECTS"
        ]
        catalog_models = {
            neighbor_id for neighbor_id, neighbor_attrs, edge in _neighbors(graph, str(node_id))
            if neighbor_attrs.get("node_type") == "ProductModel"
            and edge.get("relationship") == "CATALOG_ASSOCIATED_WITH"
        }
        watch = watch_by_id.get(str(node_id), {})
        rows.append(_risk_payload(row, {
            "node_id": str(node_id),
            "name": attrs.get("name", node_id),
            "product_count": len(products),
            "operational_event_count": watch.get("operational_event_count", 0),
            "active_signal_count": watch.get("active_signal_count", 0),
            "exposure_scope": event_edges[0].get("exposure_scope", "Unavailable") if event_edges else "Unavailable",
            "exposure_location": event_edges[0].get("exposure_location", "Unavailable") if event_edges else "Unavailable",
            "resolution_provenance": event_edges[0].get("resolution_provenance", "Unavailable") if event_edges else "Unavailable",
            "warehouse_id": attrs.get("warehouse_id", node_id),
            "warehouse_name": attrs.get("warehouse_name", attrs.get("name", node_id)),
            "city": attrs.get("city", "Unavailable"),
            "state": attrs.get("state", "Unavailable"),
            "operator": attrs.get("warehouse_operator", "Unavailable"),
            "evidence_grade": attrs.get("warehouse_evidence_grade", "Unavailable"),
            "verification_status": attrs.get("warehouse_verification_status", "Unavailable"),
            "source_url": attrs.get("warehouse_source_url", "Unavailable"),
            "catalog_only": attrs.get("catalog_only", False),
            "inventory_relationship_available": watch.get("inventory_relationship_available", False),
            "synthetic": attrs.get("synthetic", "Unavailable"),
            "catalog_association_count": len(catalog_models),
        }))
    return rows


def _supplier_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    risk_map = _risk_map(state)
    graph = state["graph"]
    rows = []
    for node_id, attrs in (graph.nodes(data=True) if graph is not None else []):
        if attrs.get("node_type") != "Supplier":
            continue
        rows.append(_risk_payload(risk_map.get(str(node_id), {
            "Node_ID": node_id,
            "Node_Name": attrs.get("name", node_id),
            "Risk_Level": "Unavailable",
        }), {"name": attrs.get("name", node_id), "provenance": "SIMULATED"}))
    return rows


def _metadata_payload(state: dict[str, Any]) -> dict[str, Any]:
    metadata = state["metadata"]
    freshness = freshness_status(metadata)
    eligible = [event for event in state["events"] if is_currently_eligible(event)]
    graph = state["graph"]
    resolved = 0
    if graph is not None:
        for event in eligible:
            node = _resolve_event_node(graph, event)
            if node is not None and any(
                edge.get("relationship") == "AFFECTS"
                and graph.nodes[target].get("node_type") in {"Warehouse", "Location"}
                for _, target, edge in graph.out_edges(node, data=True)
            ):
                resolved += 1
    return {
        "weather_fetch_timestamp": metadata.get("weather_fetch_timestamp"),
        "news_fetch_timestamp": metadata.get("news_fetch_timestamp"),
        "pipeline_run_timestamp": metadata.get("pipeline_run_timestamp", metadata.get("run_timestamp")),
        **freshness,
        "raw_event_count": metadata.get("raw_event_count", len(state["events"])),
        "operational_event_count": len(eligible),
        "excluded_event_count": len(state["events"]) - len(eligible),
        "graph_entity_resolved_count": resolved,
        "graph_entity_unresolved_count": len(eligible) - resolved,
    }


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    event = apply_runtime_temporal_status(event)
    source = str(event.get("Source", ""))
    if event.get("source_channel"):
        source_channel = event["source_channel"]
    elif source.endswith("current_weather.json"):
        source_channel = "OPENWEATHER"
    elif source.endswith("current_news.json"):
        source_channel = "NEWS_API"
    else:
        source_channel = event.get("source_channel", "OTHER")
    return {
        "event_id": event.get("Event_ID"),
        "title": event.get("Title", ""),
        "event_type": event.get("Event_Type", ""),
        "event_domain": event.get("event_domain", "OTHER"),
        "source_channel": source_channel,
        "location": event.get("Location", ""),
        "severity": event.get("Severity", ""),
        "risk_score": event.get("Risk_Score"),
        "impact_type": event.get("Impact_Type", ""),
        "impact_level": event.get("Impact_Level", ""),
        "possible_supply_chain_effect": event.get("Possible_Supply_Chain_Effect", ""),
        "operationally_relevant": event.get("Operationally_Relevant", False),
        "event_class": event.get("Event_Class", ""),
        "evidence_level": event.get("Evidence_Level", ""),
        "event_family": event.get("Event_Family", ""),
        "evidence_directness": event.get("Evidence_Directness", ""),
        "classification_reason": event.get("Classification_Reason", ""),
        "matched_operational_evidence": event.get("matched_operational_evidence", ""),
        "event_start": event.get("event_start"),
        "event_end": event.get("event_end"),
        "event_status": event.get("event_status", "UNKNOWN"),
        "temporal_confidence": event.get("temporal_confidence", "LOW"),
        "temporal_basis": event.get("temporal_basis", "No event-validity date available"),
        "location_resolution_basis": event.get("location_resolution_basis", "UNRESOLVED"),
        "location_resolution_confidence": event.get("location_resolution_confidence", "NONE"),
        "matched_location_node": event.get("matched_location_node", ""),
        "matched_warehouse_ids": event.get("matched_warehouse_ids", ""),
        "location_scope": event.get("Location_Scope", event.get("location_scope", "UNRESOLVED")),
        "location_mentions": event.get("Location_Mentions", []),
        "operational_relevance_basis": event.get("operational_relevance_basis", ""),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "system": "ElectroGraph-AI"}


@app.get("/current/summary")
def current_summary() -> dict[str, Any]:
    state = load_current_state()
    risk = state["risk"]
    risky_levels = {"Medium", "High", "Critical"}
    monitored_ids = {
        str(node_id) for node_id, attrs in state["graph"].nodes(data=True)
        if attrs.get("node_type") in {"Warehouse", "Product", "Supplier"}
    } if state["graph"] is not None else set()
    high_critical = risk[
        risk["Risk_Level"].isin({"High", "Critical"}) & risk["Node_ID"].astype(str).isin(monitored_ids)
    ] if not risk.empty else risk
    warehouse_rows = _warehouse_rows(state)
    product_rows = _product_rows(state)
    supplier_rows = _supplier_rows(state)
    network = network_summary(state)
    risk_counts = {
        node_type: {
            level: int(((risk["Node_Type"] == node_type) & (risk["Risk_Level"] == level)).sum())
            for level in ("Low", "Medium", "High", "Critical")
        }
        for node_type in sorted(risk["Node_Type"].dropna().astype(str).unique())
    }
    return {
        "operational_events": sum(1 for event in state["events"] if is_currently_eligible(event)),
        "affected_warehouses": sum(1 for row in warehouse_rows if row["connected_events"] > 0),
        "risky_products": sum(1 for row in product_rows if row["risk_level"] in risky_levels),
        "risky_suppliers": sum(1 for row in supplier_rows if row["risk_level"] in risky_levels),
        "high_critical_nodes": len(high_critical),
        "graph_nodes": state["graph"].number_of_nodes() if state["graph"] is not None else 0,
        "graph_edges": state["graph"].number_of_edges() if state["graph"] is not None else 0,
        "overall_risk_level": network["overall_risk_level"],
        **{key: value for key, value in network.items() if key != "overall_risk_level"},
        "risk_counts": risk_counts,
    }


@app.get("/current/risk-watchlist")
def current_risk_watchlist(entity_type: str = Query("Warehouse")) -> list[dict[str, Any]]:
    if entity_type not in {"Warehouse", "Product", "Supplier"}:
        raise HTTPException(status_code=400, detail="entity_type must be Warehouse, Product, or Supplier")
    return watchlist(load_current_state(), entity_type)


@app.get("/current/events")
def current_events(operational_only: bool = Query(False)) -> list[dict[str, Any]]:
    state = load_current_state()
    events = (
        state["events"]
        if not operational_only
        else [event for event in state["events"] if is_currently_eligible(event)]
    )
    return [_event_payload(event) for event in events]


@app.get("/current/weather")
def current_weather() -> dict[str, Any]:
    path = CURRENT_DIR / "current_weather.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Current weather artifact not found")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@app.get("/current/live-situation")
def current_live_situation() -> dict[str, Any]:
    """Aggregate current artifacts without changing their risk semantics."""
    state = load_current_state()
    return build_live_situation(state, freshness_status(state["metadata"]))


def _related_ids(graph: Any, node_id: str, relationship: str, node_type: str) -> list[str]:
    if graph is None or node_id not in graph:
        return []
    related = []
    for neighbor_id, edge in list(graph[node_id].items()):
        if edge.get("relationship") == relationship and graph.nodes[neighbor_id].get("node_type") == node_type:
            related.append(str(neighbor_id))
    for neighbor_id in graph.predecessors(node_id):
        edge = graph.get_edge_data(neighbor_id, node_id) or {}
        if edge.get("relationship") == relationship and graph.nodes[neighbor_id].get("node_type") == node_type:
            related.append(str(neighbor_id))
    return list(dict.fromkeys(related))


def _resolve_event_node(graph: Any, event: dict[str, Any]) -> str | None:
    artifact_id = str(event.get("Event_ID", ""))
    candidates = {artifact_id, f"CURRENT_EVENT:{artifact_id}"}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "Event" and (
            str(node_id) in candidates or str(attrs.get("event_id", "")) == artifact_id
        ):
            return str(node_id)
    return None


@app.get("/current/impact")
def current_impact() -> list[dict[str, Any]]:
    state = load_current_state()
    graph = state["graph"]
    risk_map = _risk_map(state)
    traces = []
    for raw_event in state["events"]:
        event = apply_runtime_temporal_status(raw_event)
        if event.get("event_status") != "ACTIVE" or event.get("Operationally_Relevant") is not True:
            continue
        event_id = _resolve_event_node(graph, event)
        if event_id is None:
            continue
        warehouses = []
        affected_warehouse_ids = _related_ids(graph, event_id, "AFFECTS", "Warehouse")
        affected_location_ids = _related_ids(graph, event_id, "AFFECTS", "Location")
        for warehouse_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "Warehouse":
                continue
            warehouse_id = str(warehouse_id)
            own_locations = set(_related_ids(graph, warehouse_id, "LOCATED_IN", "Location"))
            if warehouse_id not in affected_warehouse_ids and not own_locations.intersection(affected_location_ids):
                continue
            product_ids = [
                str(product_id) for product_id, product_attrs in graph.nodes(data=True)
                if product_attrs.get("node_type") == "Product"
                and warehouse_id in _related_ids(graph, str(product_id), "STORED_AT", "Warehouse")
            ]
            warehouses.append({
                "node_id": warehouse_id,
                "name": attrs.get("name", warehouse_id),
                "product_ids": product_ids,
                "location_ids": sorted(own_locations.intersection(affected_location_ids)),
            })
        product_ids = {product_id for warehouse in warehouses for product_id in warehouse["product_ids"]}
        warehouse_ids = {warehouse["node_id"] for warehouse in warehouses}
        location_ids = {
            location_id
            for warehouse in warehouses
            for location_id in warehouse["location_ids"]
        }
        product_paths = []
        for warehouse in warehouses:
            warehouse_id = warehouse["node_id"]
            path_location_ids = _related_ids(graph, warehouse_id, "LOCATED_IN", "Location")
            for product_id in warehouse["product_ids"]:
                product_paths.append({
                    "product_id": product_id,
                    "warehouse_id": warehouse_id,
                    "location_ids": path_location_ids,
                    "event_id": event_id,
                })
        supplier_ids = set()
        supplier_paths = []
        for supplier_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "Supplier":
                continue
            for brand_id in _related_ids(graph, str(supplier_id), "SUPPLIES", "Brand"):
                supplier_products = {
                    product_id for product_id, product_attrs in graph.nodes(data=True)
                    if product_attrs.get("node_type") == "Product"
                    and product_id in _related_ids(graph, brand_id, "PRODUCES", "Product")
                }
                if supplier_products.intersection(product_ids):
                    supplier_ids.add(str(supplier_id))
                    product_id = next(iter(supplier_products.intersection(product_ids)))
                    warehouse = next(
                        warehouse for warehouse in warehouses
                        if product_id in warehouse["product_ids"]
                    )
                    supplier_paths.append({
                        "supplier_id": str(supplier_id),
                        "brand_id": brand_id,
                        "product_id": product_id,
                        "warehouse_id": warehouse["node_id"],
                        "location_ids": _related_ids(graph, warehouse["node_id"], "LOCATED_IN", "Location"),
                        "event_id": event_id,
                    })
                    break
        traces.append({
            "event": _event_payload(event),
            "graph_event_node_id": event_id,
            "warehouses": warehouses,
            "product_ids": sorted(product_ids),
            "supplier_ids": sorted(supplier_ids),
            "product_paths": product_paths,
            "supplier_paths": supplier_paths,
            "product_risk_levels": {
                level: sum(1 for product_id in product_ids if risk_map.get(product_id, {}).get("Risk_Level") == level)
                for level in ("Medium", "High", "Critical")
            },
            "risk_levels": {
                level: sum(1 for product_id in product_ids if risk_map.get(product_id, {}).get("Risk_Level") == level)
                for level in ("Medium", "High", "Critical")
            },
            "network_risk_levels": {
                level: sum(
                    1
                    for node_id in product_ids | warehouse_ids | supplier_ids | location_ids
                    if risk_map.get(node_id, {}).get("Risk_Level") == level
                )
                for level in ("Medium", "High", "Critical")
            },
            "provenance": "DERIVED / RULE_BASED; network links are MODELED unless explicitly observed",
        })
    return traces


@app.post("/scenario/simulate", response_model=ScenarioResponse)
def simulate_scenario(request: ScenarioRequest) -> ScenarioResponse:
    if request.scenario_type != "WAREHOUSE_UNAVAILABLE":
        raise HTTPException(status_code=400, detail="Only WAREHOUSE_UNAVAILABLE is supported")
    if request.duration_hours not in {12, 24, 48, 72}:
        raise HTTPException(status_code=400, detail="duration_hours must be 12, 24, 48, or 72")
    if request.scenario_severity not in {"Medium", "High", "Critical"}:
        raise HTTPException(status_code=400, detail="scenario_severity must be Medium, High, or Critical")
    state = load_current_state()
    graph = state["graph"]
    target = request.target_node_id
    if graph is None or target not in graph:
        raise HTTPException(status_code=400, detail="Target warehouse was not found in the current merged graph")
    target_attrs = graph.nodes[target]
    if target_attrs.get("node_type") != "Warehouse":
        raise HTTPException(status_code=400, detail="Scenario target must be a Warehouse")
    if str(target_attrs.get("status", "ACTIVE")).upper() in {"INACTIVE", "DEACTIVATED"} or target_attrs.get("active") in {False, "false", "False", "NO"}:
        raise HTTPException(status_code=400, detail="Target warehouse is inactive or deactivated")
    scenario_products = {
        str(product_id)
        for product_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "Product"
        and target in _related_ids(graph, str(product_id), "STORED_AT", "Warehouse")
    }
    scenario_warehouses = {target}
    scenario_locations = set(_related_ids(graph, target, "LOCATED_IN", "Location"))
    catalog_association_count = len(_related_ids(graph, target, "CATALOG_ASSOCIATED_WITH", "ProductModel"))
    scenario_suppliers = set()
    for supplier_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Supplier":
            continue
        for brand_id in _related_ids(graph, str(supplier_id), "SUPPLIES", "Brand"):
            brand_products = {
                str(product_id)
                for product_id, product_attrs in graph.nodes(data=True)
                if product_attrs.get("node_type") == "Product"
                and str(product_id) in _related_ids(graph, brand_id, "PRODUCES", "Product")
            }
            if brand_products.intersection(scenario_products):
                scenario_suppliers.add(str(supplier_id))
                break
    current_traces = current_impact()
    current_products = {product_id for trace in current_traces for product_id in trace["product_ids"]}
    current_suppliers = {supplier_id for trace in current_traces for supplier_id in trace["supplier_ids"]}
    risk_map = _risk_map(state)
    current_levels = {
        level: sum(1 for row in state["risk"].to_dict(orient="records") if row.get("Risk_Level") == level)
        for level in ("Medium", "High", "Critical")
    }
    alternatives = []
    for warehouse_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Warehouse" or str(warehouse_id) == target or str(attrs.get("status", "ACTIVE")).upper() in {"INACTIVE", "DEACTIVATED"} or attrs.get("active") in {False, "false", "False", "NO"}:
            continue
        matching = [
            product_id for product_id in scenario_products
            if str(warehouse_id) in _related_ids(graph, product_id, "STORED_AT", "Warehouse")
        ]
        if matching:
            alternatives.append({
                "warehouse_id": str(warehouse_id),
                "warehouse_name": attrs.get("name", warehouse_id),
                "matching_product_count": len(matching),
                "matching_product_ids": sorted(matching),
                "provenance": "MODELED",
                "current_risk": risk_map.get(str(warehouse_id), {}).get("Risk_Level"),
                "current_event_exposure": sum(
                    1 for trace in current_traces
                    if any(item["node_id"] == str(warehouse_id) for item in trace["warehouses"])
                ),
                "limitation": "Candidate indicates an existing modeled STORED_AT relationship. Operational capacity and inventory availability are not verified.",
            })
    alternatives.sort(key=lambda item: (-item["matching_product_count"], item["warehouse_id"]))
    supplier_alternatives = []
    for supplier_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Supplier" or str(supplier_id) in scenario_suppliers:
            continue
        matching = []
        for brand_id in _related_ids(graph, str(supplier_id), "SUPPLIES", "Brand"):
            matching.extend(
                product_id for product_id in scenario_products
                if product_id in _related_ids(graph, brand_id, "PRODUCES", "Product")
            )
        matching = sorted(set(matching))
        if matching:
            supplier_alternatives.append({
                "supplier_id": str(supplier_id),
                "supplier_name": attrs.get("name", supplier_id),
                "matching_product_count": len(matching),
                "matching_product_ids": matching,
                "provenance": "MODELED",
                "current_risk": risk_map.get(str(supplier_id), {}).get("Risk_Level"),
                "current_event_exposure": sum(1 for trace in current_traces if str(supplier_id) in trace["supplier_ids"]),
            })
    supplier_alternatives.sort(key=lambda item: (-item["matching_product_count"], item["supplier_id"]))
    inventory_available = bool(scenario_products)
    target_risk = risk_map.get(target, {})
    current_risk_score = risk_numeric(target_risk.get("Risk_Score"))
    current_risk_level = str(target_risk.get("Risk_Level", "Unavailable")) if current_risk_score is not None else "Unavailable"
    limitations = [
        "Warehouse stock quantities and free capacity are unavailable.",
        "Transport routes, lead times, contracts, and rerouting feasibility are not modeled.",
        "Duration and severity are user-selected assumptions; they do not change graph exposure counts or inventory.",
    ]
    if not inventory_available:
        limitations.insert(0, "No verified STORED_AT inventory relation is available for this warehouse; operational product impact is unavailable from verified inventory evidence.")
    explanation = (
        f"Under this modeled WAREHOUSE_UNAVAILABLE scenario for {target_attrs.get('name', target)} "
        f"({request.duration_hours} hours; {request.scenario_severity} severity assumption), "
        + (
            f"graph STORED_AT evidence identifies {len(scenario_products)} directly affected verified products. "
            f"SUPPLIES → PRODUCES paths connect {len(scenario_suppliers)} suppliers to those products. "
            f"There are {len(alternatives)} alternative warehouse candidates based on the same products' STORED_AT relations. "
            if inventory_available else
            "this warehouse is present in the curated reference network, but SupplyBeacon has no verified operational STORED_AT inventory relationship for it. Therefore the simulation cannot claim specific affected products or suppliers. "
        )
        + "Alternative warehouse candidates are graph candidates, not confirmed logistics options. "
        + f"Current warehouse risk is {current_risk_level} and is separate from this scenario; historical DataCo prediction is not used. "
        + " ".join(limitations)
    )
    return ScenarioResponse(
        scenario_id=f"SCENARIO:{uuid4()}",
        generated_at=datetime.now(timezone.utc).isoformat(),
        scenario={"scenario_type": request.scenario_type, "target_node_id": target, "target_name": graph.nodes[target].get("name", target), "duration_hours": request.duration_hours, "severity_assumption": request.scenario_severity},
        current_state={"active_event_count": len(current_traces), "exposed_products": len(current_products), "exposed_suppliers": len(current_suppliers), "exposed_warehouses": len({w["node_id"] for t in current_traces for w in t["warehouses"]}), **current_levels},
        scenario_state={"disrupted_warehouses": 1, "exposed_products": len(scenario_products), "exposed_suppliers": len(scenario_suppliers), "affected_locations": len(scenario_locations), "severity_assumption": request.scenario_severity},
        delta={"newly_exposed_products": sorted(scenario_products - current_products), "newly_exposed_suppliers": sorted(scenario_suppliers - current_suppliers), "overlap_products": sorted(scenario_products & current_products), "overlap_suppliers": sorted(scenario_suppliers & current_suppliers)},
        newly_exposed_product_ids=sorted(scenario_products - current_products),
        overlap_product_ids=sorted(scenario_products & current_products),
        product_ids=sorted(scenario_products), supplier_ids=sorted(scenario_suppliers), warehouse_ids=sorted(scenario_warehouses), location_ids=sorted(scenario_locations),
        alternative_warehouse_candidates=alternatives, alternative_supplier_candidates=supplier_alternatives,
        assumptions=[
            "Warehouse availability is user-selected scenario input.",
            "No external APIs were called.",
            "Capacity, inventory, transport feasibility, and commercial eligibility are not verified.",
            *(["No verified operational product inventory relationships are available for this warehouse."] if not scenario_products else []),
        ],
        provenance="DERIVED / RULE_BASED scenario calculations using MODELED graph relationships.",
        inventory_relationship_available=inventory_available,
        catalog_only=target_attrs.get("catalog_only") in {True, "true", "True", 1, "1"},
        catalog_association_count=catalog_association_count,
        affected_product_ids=sorted(scenario_products),
        affected_product_count=len(scenario_products),
        affected_supplier_ids=sorted(scenario_suppliers),
        affected_supplier_count=len(scenario_suppliers),
        current_risk_score=current_risk_score,
        current_risk_level=current_risk_level,
        impact_status="DERIVED_FROM_VERIFIED_STORED_AT" if inventory_available else "UNAVAILABLE_VERIFIED_INVENTORY",
        limitations=limitations,
        evidence_basis=["Product → STORED_AT → target Warehouse", "Supplier → SUPPLIES → Brand → PRODUCES → affected Product", "Affected Product → STORED_AT → alternative Warehouse", "CATALOG_ASSOCIATED_WITH is contextual only"],
        explanation=explanation,
    )


@app.post("/scenario/explain", response_model=ScenarioExplanationResponse)
def explain_scenario_candidate(request: ScenarioExplanationRequest) -> ScenarioExplanationResponse:
    if request.candidate_type not in {"warehouse", "supplier"}:
        raise HTTPException(status_code=400, detail="candidate_type must be warehouse or supplier")
    allowed_fields = {
        "candidate_id", "candidate_name", "disrupted_warehouse",
        "target_node_id",
        "matching_product_count", "sample_matching_product_ids",
        "matching_product_ids", "explicit_relationship", "stored_at_evidence",
        "supplies_produces_evidence", "current_risk", "current_event_exposure",
    }
    evidence = {key: value for key, value in request.evidence.items() if key in allowed_fields}
    target_id = str(evidence.get("target_node_id") or evidence.get("disrupted_warehouse") or "")
    verified = False
    if target_id:
        result = simulate_scenario(ScenarioRequest(
            scenario_type="WAREHOUSE_UNAVAILABLE", target_node_id=target_id,
            duration_hours=48, scenario_severity="Medium",
        ))
        candidates = result.alternative_warehouse_candidates if request.candidate_type == "warehouse" else result.alternative_supplier_candidates
        id_field = "warehouse_id" if request.candidate_type == "warehouse" else "supplier_id"
        name_field = "warehouse_name" if request.candidate_type == "warehouse" else "supplier_name"
        authoritative = next((item for item in candidates if item[id_field] == evidence.get("candidate_id")), None)
        if authoritative is None:
            raise HTTPException(status_code=400, detail="Candidate is not supported by this scenario's graph evidence")
        evidence = {
            "candidate_id": authoritative[id_field],
            "candidate_name": authoritative[name_field],
            "matching_product_count": authoritative["matching_product_count"],
            "matching_product_ids": authoritative["matching_product_ids"],
            "target_node_id": target_id,
        }
        verified = True
    candidate = evidence.get("candidate_name", evidence.get("candidate_id", "candidate"))
    matching = evidence.get("matching_product_count", 0)
    relationship = "STORED_AT" if request.candidate_type == "warehouse" else "SUPPLIES / PRODUCES"
    limitations = [
        "inventory availability not verified",
        "capacity not verified",
        "logistics feasibility not verified",
        "commercial eligibility not verified",
        "pricing not verified",
    ]
    deterministic_answer = (
        f"POTENTIAL MODELED ALTERNATIVE: {candidate} is linked to {matching} scenario product(s) "
        f"through explicit {relationship} graph evidence. This candidate was selected by deterministic "
        "graph logic, not by an LLM."
    ) if verified else "POTENTIAL MODELED ALTERNATIVE: The supplied candidate evidence has not been verified against a target warehouse scenario."
    fallback_answer = deterministic_answer + " " + "; ".join(limitations) + "."
    if not verified or os.getenv("SUPPLYBEACON_LLM_ENABLED", "").lower() != "true":
        return ScenarioExplanationResponse(
            answer=fallback_answer,
            provenance="MODELED / DERIVED",
            limitations=limitations,
            explanation_mode="DETERMINISTIC_FALLBACK",
        )
    system_prompt = (
        "You are SupplyBeacon's optional grounded scenario explanation layer. "
        "Explain only the supplied structured evidence. Python already selected the candidate; "
        "you must not choose, rank, recalculate, or add relationships. Never invent facts. "
        "Return one JSON object with an answer string only. The answer must include the exact phrase "
        "'POTENTIAL MODELED ALTERNATIVE', distinguish evidence from assumptions, and mention every "
        "limitation supplied in the evidence."
    )
    llm_evidence = {
        "candidate_type": request.candidate_type,
        "evidence": evidence,
        "limitations": limitations,
    }
    try:
        raw = OpenAICompatibleProvider().generate(
            system_prompt,
            "WHY THIS ALTERNATIVE?",
            llm_evidence,
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
            raise ValueError("Structured explanation response must contain an answer string.")
        answer = parsed["answer"].strip()
        # Only an exact restatement can be trusted automatically. Any novel
        # LLM wording could introduce an unsupported operational claim.
        if answer != fallback_answer:
            raise ValueError("LLM explanation diverged from authoritative evidence.")
        return ScenarioExplanationResponse(
            answer=answer,
            provenance="MODELED / DERIVED",
            limitations=limitations,
            explanation_mode="LLM_GROUNDED",
        )
    except (ProviderError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return ScenarioExplanationResponse(
            answer=fallback_answer,
            provenance="MODELED / DERIVED",
            limitations=limitations,
            explanation_mode="DETERMINISTIC_FALLBACK",
        )


@app.get("/current/products")
def current_products() -> list[dict[str, Any]]:
    return _product_rows(load_current_state())


@app.get("/current/warehouses")
def current_warehouses() -> list[dict[str, Any]]:
    return _warehouse_rows(load_current_state())


@app.get("/current/suppliers")
def current_suppliers() -> list[dict[str, Any]]:
    return _supplier_rows(load_current_state())


@app.get("/current/metadata")
def current_metadata() -> dict[str, Any]:
    return _metadata_payload(load_current_state())


@app.get("/current/products/{node_id:path}")
def current_product(node_id: str) -> dict[str, Any]:
    state = load_current_state()
    row = next((item for item in _product_rows(state) if item["node_id"] == node_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Current product not found")
    return row


@app.get("/current/warehouses/{node_id:path}")
def current_warehouse(node_id: str) -> dict[str, Any]:
    state = load_current_state()
    row = next((item for item in _warehouse_rows(state) if item["node_id"] == node_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Current warehouse not found")
    return row


@app.get("/current/warehouse-catalog/{node_id:path}")
def current_warehouse_catalog(node_id: str) -> list[dict[str, Any]]:
    state = load_current_state()
    graph = state["graph"]
    if graph is None or node_id not in graph or graph.nodes[node_id].get("node_type") != "Warehouse":
        raise HTTPException(status_code=404, detail="Current warehouse not found")
    rows = []
    for neighbor_id, attrs, edge in _neighbors(graph, node_id):
        if attrs.get("node_type") != "ProductModel" or edge.get("relationship") != "CATALOG_ASSOCIATED_WITH":
            continue
        rows.append({
            "node_id": str(neighbor_id),
            "product_model": attrs.get("product_model", attrs.get("name")),
            "brand": attrs.get("brand"),
            "category": attrs.get("category"),
            "subcategory": attrs.get("subcategory"),
            "product_source_url": attrs.get("product_source_url"),
            "catalog_as_of": attrs.get("catalog_as_of"),
            "association_basis": edge.get("association_basis"),
            "inventory_verified": False,
            "operational_inventory_relation": False,
            "provenance": edge.get("provenance", "MANUALLY_CURATED_PUBLIC_SOURCE"),
        })
    return sorted(rows, key=lambda item: (str(item.get("brand")), str(item.get("product_model"))))


@app.get("/current/graph/entity/{node_id:path}")
def current_graph_entity(node_id: str) -> dict[str, Any]:
    state = load_current_state()
    graph = state["graph"]
    if graph is None or node_id not in graph:
        raise HTTPException(status_code=404, detail="Current graph entity not found")
    attrs = graph.nodes[node_id]
    neighbors = []
    for neighbor_id, neighbor_attrs, edge in _neighbors(graph, node_id):
        neighbors.append({
            "node_id": str(neighbor_id),
            "node_type": neighbor_attrs.get("node_type", "Unavailable"),
            "name": neighbor_attrs.get("name", neighbor_id),
            "relationship": edge.get("relationship", "Unavailable"),
            "provenance": edge.get("provenance", "Unavailable"),
            "exposure_scope": edge.get("exposure_scope", "Unavailable"),
            "exposure_location": edge.get("exposure_location", "Unavailable"),
            "resolution_provenance": edge.get("resolution_provenance", "Unavailable"),
        })
    return {
        "node_id": node_id,
        "node_type": attrs.get("node_type", "Unavailable"),
        "name": attrs.get("name", node_id),
        "attributes": dict(attrs),
        "neighbors": neighbors,
    }


@app.get("/historical/model-metrics")
def historical_model_metrics() -> dict[str, Any]:
    path = BASE_DIR / "deploy" / "seed" / "analysis" / "dataco" / "modeling" / "baseline_metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Historical metrics artifact not found")
    metrics = pd.read_csv(path)
    selected = metrics[
        (metrics["model"] == "hist_gradient_boosting") & (metrics["split"] == "test")
    ]
    if selected.empty:
        raise HTTPException(status_code=404, detail="HistGradientBoosting test metrics not found")
    return selected.iloc[0].to_dict()


@app.post("/current/refresh")
def current_refresh(x_supplybeacon_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    expected_key = os.getenv("SUPPLYBEACON_ADMIN_API_KEY")
    if not expected_key or x_supplybeacon_admin_key != expected_key:
        raise HTTPException(status_code=401, detail="Admin refresh authorization required")
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A current-data refresh is already running")
    try:
        refresh_current_state()
        invalidate_retriever_cache()
        return current_metadata()
    finally:
        _refresh_lock.release()


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    query_lower = request.query.casefold()
    current_events_question = (
        "event" in query_lower
        and any(term in query_lower for term in ("current", "currently", "active"))
        and any(term in query_lower for term in ("affect", "operational", "impact"))
        and any(term in query_lower for term in ("warehouse", "supply network", "supply chain"))
    )
    warehouse_ranking_question = (
        "warehouse" in query_lower
        and any(term in query_lower for term in ("highest risk", "highest-risk", "most risky", "most at risk"))
    )
    historical_question = any(
        term in query_lower for term in ("dataco", "historical shipping model", "historical model")
    )
    if current_events_question or warehouse_ranking_question or historical_question:
        return run_agentic_query(request.query)
    grounded_answer = answer_grounded_domain(request.query)
    if grounded_answer is not None:
        return grounded_answer
    evidence = current_retriever().retrieve(request.query)
    explanation = build_explanation(evidence)
    delivery = evidence.get("delivery_risk")
    query_type = evidence.get("query_type", "")
    if query_type == "weather_ranking":
        names = [event.get("Title", event.get("Event_Type", "current event")) for event in evidence["events"]]
        explanation["answer"] = (
            "Current weather risks: " + "; ".join(names) + "."
            if names
            else "No current weather risks were found in the current artifacts."
        )
    elif query_type.endswith("_ranking"):
        label = query_type.replace("_ranking", "").capitalize()
        if evidence["entities"]:
            explanation["answer"] = (
                f"Current {label} results: "
                + "; ".join(
                    f"{entity.get('name')} ({entity.get('risk_level')}, score {entity.get('risk_score')})"
                    for entity in evidence["entities"]
                )
                + "."
            )
        elif label == "Product":
            explanation["answer"] = (
                "No products currently meet the Medium/High/Critical risk threshold. "
                "The highest current product scores are available as informational evidence."
            )
        elif label == "Supplier":
            explanation["answer"] = (
                "No suppliers currently meet the Medium/High/Critical risk threshold. "
                "The highest current supplier scores are available as informational evidence."
            )
        else:
            explanation["answer"] = "No current affected warehouses were found."
    if delivery:
        explanation["answer"] = (
            f"{evidence['entities'][0]['name']} has current risk level "
            f"{delivery.get('current_risk_level', 'Unavailable')}. "
            f"Baseline Lead_Time_Days is {delivery.get('baseline_delivery_days', 'Unavailable')} days (SIMULATED). "
            f"Scenario delay range is {delivery.get('scenario_delay_days_min', 'Unavailable')}–"
            f"{delivery.get('scenario_delay_days_max', 'Unavailable')} days, with estimated delivery of "
            f"{delivery.get('estimated_delivery_days_min', 'Unavailable')}–"
            f"{delivery.get('estimated_delivery_days_max', 'Unavailable')} days. "
            f"{delivery.get('message', '')}"
        )
    return {
        "query": request.query,
        "answer": explanation["answer"],
        "intent": query_type or "CURRENT_GRAPH",
        "resolved_entities": evidence["entities"],
        "evidence": evidence.get("provenance", []),
        "warnings": [],
        "entities": evidence["entities"],
        "informational_entities": evidence.get("informational_entities", []),
        "graph_paths": evidence["graph_paths"],
        "events": evidence["events"],
        "risk_evidence": evidence["risk_evidence"],
        "provenance_summary": explanation["provenance_summary"],
        "limitations": evidence["limitations"],
        "delivery_risk": delivery,
        "current_metadata": evidence.get("current_metadata", {}),
    }


@app.post("/query/agentic")
def agentic_query(request: QueryRequest) -> dict[str, Any]:
    """Run the deterministic-first LangGraph workflow without replacing /query."""
    return run_agentic_query(request.query)
