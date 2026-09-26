"""Refresh isolated current weather/news/event/risk artifacts.

This command never reads historical external JSON as a current-data fallback
and never overwrites the baseline graph or risk outputs.
"""
from __future__ import annotations

import json
import logging
import sys
import importlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .current_state import BASE_DIR, CURRENT_DIR
from src.event_extraction.temporal_validity import apply_runtime_temporal_status, is_currently_eligible
from src.external_data.unified_sources import collect_unified_news, collect_unified_weather
from src.external_data.weather_collector import warehouse_weather_locations

EXTERNAL_DIR = BASE_DIR / "data" / "external"
BASE_GRAPH = BASE_DIR / "data" / "graph" / "supply_chain_graph.graphml"
SUPPLY_PATH = BASE_DIR / "data" / "processed" / "electronics_supplychain_dataset.csv"
MAPPING_PATH = BASE_DIR / "data" / "processed" / "brand_supplier_mapping.csv"
LOCATION_TO_WAREHOUSE = {
    "Delhi": "Delhi Warehouse",
    "Karnataka": "Bengaluru Warehouse",
    "Maharashtra": "Mumbai Warehouse",
    "Gujarat": "Ahmedabad Warehouse",
    "Kochi": "Kochi Warehouse",
}


def _event_semantics(event: dict[str, Any]) -> tuple[str, str]:
    event_type = str(event.get("Event_Type", "")).lower()
    category = str(event.get("Category", "")).lower()
    source = str(event.get("Source", "")).lower()
    if category == "weather" or event_type in {"heavy rain", "light rain", "moderate rain", "thunderstorm", "storm", "flood", "cyclone", "extreme heat"}:
        domain = "WEATHER"
    elif any(term in event_type for term in ("logistics", "transport", "shipping", "port", "road", "delivery")):
        domain = "LOGISTICS"
    elif any(term in event_type for term in ("factory", "production", "manufacturing", "warehouse")):
        domain = "MANUFACTURING"
    elif any(term in event_type for term in ("shortage", "semiconductor", "supply", "tariff", "trade")):
        domain = "MARKET_SUPPLY"
    else:
        domain = "OTHER"
    channel = str(event.get("source_channel") or "")
    if not channel:
        if "news" in source:
            channel = "NEWS_API"
        elif "weather" in source:
            channel = "OPENWEATHER"
        else:
            channel = "OTHER"
    return domain, channel


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _collect_external() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from src.external_data.unified_sources import _configured_key
    news_key = _configured_key("NEWS_API_KEY")
    weather_key = _configured_key("OPENWEATHER_API_KEY")
    from .current_state import load_current_state

    state = load_current_state()
    graph = state.get("graph")
    cities = warehouse_weather_locations(graph) if graph is not None else []
    if not cities:
        raise RuntimeError("No precise Warehouse -> LOCATED_IN -> Location paths are available for weather collection")
    news_articles, news_sources = collect_unified_news(news_key)
    weather, weather_sources = collect_unified_weather(weather_key, cities)
    news = {"collected_at": datetime.now(timezone.utc).isoformat(), "articles": news_articles}
    return weather, news, {"news": news_sources, "weather": weather_sources}


def _extract_events(weather: dict[str, Any], news: dict[str, Any]) -> list[dict[str, Any]]:
    extraction_dir = BASE_DIR / "src" / "event_extraction"
    if str(extraction_dir) not in sys.path:
        sys.path.insert(0, str(extraction_dir))
    sys.modules.pop("utils", None)
    importlib.invalidate_caches()
    from src.event_extraction.event_extractor import (
        article_to_events,
        assign_event_ids,
        deduplicate_events,
        extract_articles_from_news_payload,
        weather_to_events,
    )

    source = CURRENT_DIR
    extracted: list[dict[str, Any]] = []
    for article in extract_articles_from_news_payload(news):
        extracted.extend(article_to_events(article, source / "current_news.json"))
    for city, record in (weather.get("cities") or {}).items():
        if ":WEATHERAPI" in city and not (record.get("alerts") or []):
            continue
        extracted.extend(weather_to_events(city.split(":")[0], record, source / "current_weather.json"))
    return assign_event_ids(deduplicate_events([event for event in extracted if event.get("Description")]))


def _apply_business_rules(events: list[dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    rules_dir = BASE_DIR / "src" / "business_rules"
    if str(rules_dir) not in sys.path:
        sys.path.insert(0, str(rules_dir))
    sys.modules.pop("utils", None)
    importlib.invalidate_caches()
    from rule_engine import extract_event_row

    mapping = pd.read_csv(MAPPING_PATH)
    supply = pd.read_csv(SUPPLY_PATH)
    rows = [extract_event_row(pd.Series(event), mapping, supply) for event in events]
    return pd.DataFrame(rows)


def _event_locations(graph: nx.DiGraph, event_id: str, event: dict[str, Any]) -> None:
    if not is_currently_eligible(event):
        return
    location_text = str(event.get("Location", ""))
    requested = [token.strip() for token in location_text.split(",") if token.strip()]
    warehouse_names = {
        str(attrs.get("name", "")).strip().lower(): node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "Warehouse"
    }
    for location in requested:
        warehouse_name = LOCATION_TO_WAREHOUSE.get(location, f"{location} Warehouse")
        node_id = warehouse_names.get(warehouse_name.lower())
        if node_id is None:
            node_id = warehouse_names.get(f"{location} warehouse".lower())
        if node_id is None:
            continue
        exact_location = location.lower() == warehouse_name.replace(" Warehouse", "").lower()
        exposure_scope = "EXACT_LOCATION" if exact_location else "REGIONAL_RULE"
        edge_metadata = {
            "event_type": event.get("Event_Type", ""),
            "exposure_scope": exposure_scope,
            "resolution_provenance": "DERIVED",
            "exposure_location": location,
        }
        graph.add_edge(event_id, node_id, relationship="AFFECTS", **edge_metadata)
        for location_id in graph.successors(node_id):
            location_name = str(graph.nodes[location_id].get("name", "")).strip().lower()
            if (
                graph.nodes[location_id].get("node_type") == "Location"
                and location_name in {location.lower(), warehouse_name.replace(" Warehouse", "").lower()}
            ):
                graph.add_edge(event_id, location_id, relationship="AFFECTS", **edge_metadata)


def _build_current_graph(events: list[dict[str, Any]]) -> nx.DiGraph:
    graph = nx.read_graphml(str(BASE_GRAPH)).to_directed()
    overlay_path = Path(os.getenv(
        "SUPPLYBEACON_MASTER_GENERATED_DIR",
        str(BASE_DIR / "data" / "master" / "generated"),
    )) / "company_overlay.graphml"
    if overlay_path.exists():
        overlay = nx.read_graphml(str(overlay_path))
        for node_id, attrs in overlay.nodes(data=True):
            if node_id not in graph:
                graph.add_node(node_id, **attrs)
        for source, target, attrs in overlay.edges(data=True):
            if source in graph and target in graph:
                graph.add_edge(source, target, **attrs)
    for node_id in list(graph.nodes):
        if graph.nodes[node_id].get("node_type") == "Event":
            graph.remove_node(node_id)
    for index, event in enumerate(events, start=1):
        event = apply_runtime_temporal_status(event)
        event_id = f"CURRENT_EVENT:{index}"
        event_domain, source_channel = _event_semantics(event)
        graph.add_node(
            event_id,
            node_type="Event",
            name=event.get("Title", f"{event.get('Event_Type', 'Event')}"),
            event_type=event.get("Event_Type", ""),
            severity=event.get("Severity", ""),
            confidence=str(event.get("Confidence", "")),
            source=event.get("Source", ""),
            date=event.get("Date", ""),
            current="true",
            event_class=event.get("Event_Class", ""),
            operationally_relevant=str(event.get("Operationally_Relevant", True)).lower(),
            risk_score=str(event.get("Risk_Score", 0) or 0),
            impact_level=event.get("Impact_Level", ""),
            impact_type=event.get("Impact_Type", ""),
            category=event.get("Category", ""),
            event_confidence=str(event.get("Confidence", 0) or 0),
            event_domain=event_domain,
            source_channel=source_channel,
            event_id=str(event.get("Event_ID", "")),
            event_start=event.get("event_start") or "",
            event_end=event.get("event_end") or "",
            event_status=event.get("event_status", "UNKNOWN"),
            temporal_confidence=event.get("temporal_confidence", "LOW"),
            temporal_basis=event.get("temporal_basis", "No event-validity date available"),
            operational_current=str(is_currently_eligible(event)).lower(),
        )
        _event_locations(graph, event_id, event)
    return graph


def _current_risk(
    graph: nx.DiGraph,
    impacts: pd.DataFrame,
    baseline_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    baseline_path = BASE_DIR / "data" / "risk_prediction" / "risk_scores.csv"
    baseline = baseline_override.copy() if baseline_override is not None else pd.read_csv(baseline_path, dtype=str)
    from src.risk_prediction.risk_models import risk_level

    def relationship_between(source: str, target: str) -> str:
        edge = graph.get_edge_data(source, target) or graph.get_edge_data(target, source) or {}
        return str(edge.get("relationship", ""))

    def related_nodes(node_id: str, relationship: str, node_type: str | None = None) -> list[str]:
        related = []
        for neighbor in graph.neighbors(node_id):
            if relationship_between(node_id, neighbor) != relationship:
                continue
            if node_type is None or graph.nodes[neighbor].get("node_type") == node_type:
                related.append(neighbor)
        for predecessor in graph.predecessors(node_id):
            if relationship_between(node_id, predecessor) != relationship:
                continue
            if node_type is None or graph.nodes[predecessor].get("node_type") == node_type:
                related.append(predecessor)
        return list(dict.fromkeys(related))

    def event_affects_warehouse_or_location(warehouse_id: str, event_id: str) -> bool:
        if relationship_between(event_id, warehouse_id) == "AFFECTS":
            return True
        for location_id in related_nodes(warehouse_id, "LOCATED_IN", "Location"):
            if relationship_between(event_id, location_id) == "AFFECTS":
                return True
        return False

    def product_event_path(product_id: str, event_id: str) -> list[str] | None:
        for warehouse_id in related_nodes(product_id, "STORED_AT", "Warehouse"):
            if event_affects_warehouse_or_location(warehouse_id, event_id):
                return [product_id, warehouse_id, event_id]
        return None

    def node_event_path(node_id: str, event_id: str) -> list[str] | None:
        node_type = graph.nodes[node_id].get("node_type")
        if node_type == "Product":
            return product_event_path(node_id, event_id)
        if node_type == "Warehouse":
            return [node_id, event_id] if relationship_between(event_id, node_id) == "AFFECTS" else None
        if node_type == "Location":
            return [node_id, event_id] if relationship_between(event_id, node_id) == "AFFECTS" else None
        if node_type == "Supplier":
            for brand_id in related_nodes(node_id, "SUPPLIES", "Brand"):
                for product_id in related_nodes(brand_id, "PRODUCES", "Product"):
                    if product_event_path(product_id, event_id):
                        return [node_id, brand_id, product_id, *product_event_path(product_id, event_id)[1:]]
        return None

    rows = []
    for _, base in baseline.iterrows():
        node_id = str(base["Node_ID"])
        event_ids = []
        for event_id, attrs in graph.nodes(data=True):
            if (
                attrs.get("node_type") != "Event"
                or attrs.get("event_status") != "ACTIVE"
                or attrs.get("operational_current") != "true"
            ):
                continue
            if node_event_path(node_id, event_id):
                event_ids.append(event_id)
        current = base.to_dict()
        old_score = float(base.get("Risk_Score", 0) or 0)
        old_event_exposure = float(base.get("Event_Exposure", 0) or 0)
        old_business_risk = float(base.get("Business_Impact_Risk", 0) or 0)
        old_severity = {"Critical": 1.0, "High": 0.8, "Medium": 0.5, "Low": 0.25}.get(
            str(base.get("Max_Event_Severity", "")), 0.0
        )
        old_confidence = float(base.get("Average_Event_Confidence", 0) or 0)
        structural_score = (
            float(base.get("Structural_Baseline", 0) or 0)
            if str(base.get("Structural_Baseline", "")).strip()
            else max(
                0.0,
                old_score
                - (0.30 * old_severity)
                - (0.20 * old_confidence)
                - (0.20 * old_business_risk)
                - (0.15 * old_event_exposure),
            )
        )
        event_records = [graph.nodes[event_id] for event_id in event_ids]
        event_scores = [float(record.get("risk_score", 0) or 0) for record in event_records]
        weather_score = max(
            (
                score
                for score, record in zip(event_scores, event_records)
                if record.get("event_domain") == "WEATHER"
            ),
            default=0.0,
        )
        news_score = max(
            (
                score
                for score, record in zip(event_scores, event_records)
                if record.get("event_domain") != "WEATHER"
            ),
            default=0.0,
        )
        impact_values = {"High": 1.0, "Medium": 0.7, "Low": 0.4, "Unknown": 0.0}
        business_impact_score = max(
            (impact_values.get(str(record.get("impact_level", "Unknown")), 0.0) for record in event_records),
            default=0.0,
        )
        event_exposure_component = min(0.15, 0.05 * len(event_ids))
        weather_component = 0.30 * weather_score
        news_component = 0.20 * news_score
        business_impact_component = 0.15 * business_impact_score
        current_score = min(
            1.0,
            structural_score
            + weather_component
            + news_component
            + business_impact_component
            + event_exposure_component,
        )
        current["Risk_Score"] = str(round(current_score, 3))
        current["Risk_Level"] = risk_level(current_score)
        current["Connected_Events"] = str(len(event_ids))
        current["Event_Exposure"] = str(round(event_exposure_component, 3))
        current["Structural_Baseline"] = str(round(structural_score, 3))
        current["Weather_Event_Component"] = str(round(weather_component, 3))
        current["News_Event_Component"] = str(round(news_component, 3))
        current["Business_Impact_Component"] = str(round(business_impact_component, 3))
        current["Event_Exposure_Component"] = str(round(event_exposure_component, 3))
        current["Final_Scenario_Risk_Score"] = str(round(current_score, 3))
        current["Risk_Provenance"] = "DERIVED / RULE_BASED"
        current["Current"] = "true"
        current["Risk_Reason"] = (
            f"Current rule-based exposure to {len(event_ids)} event(s)."
            if event_ids
            else "No current valid event exposure detected."
        )
        rows.append(current)
    return pd.DataFrame(rows)


def refresh_current_state() -> dict[str, Any]:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_run_timestamp = datetime.now(timezone.utc).isoformat()
    weather, news, source_status = _collect_external()
    accepted_records = sum(
        int(details.get("accepted_count", 0))
        for sources in source_status.values()
        for details in sources.values()
    )
    if accepted_records == 0:
        failures = [
            f"{category}.{source}: {details.get('error')}"
            for category, sources in source_status.items()
            for source, details in sources.items()
            if not details.get("success")
        ]
        raise RuntimeError(
            "Current-data refresh returned no usable provider records; "
            "the previous snapshot was preserved. " + "; ".join(failures)
        )
    _write_json(CURRENT_DIR / "current_weather.json", weather)
    _write_json(CURRENT_DIR / "current_news.json", news)
    events = _extract_events(weather, news)
    for event in events:
        event["event_domain"], event["source_channel"] = _event_semantics(event)
        event.update(apply_runtime_temporal_status(event))
    impacts = _apply_business_rules(events)
    if not impacts.empty:
        impact_by_id = impacts.set_index("Event_ID").to_dict(orient="index")
        for event in events:
            event.update(impact_by_id.get(event.get("Event_ID"), {}))
    if not impacts.empty:
        impacts.to_csv(CURRENT_DIR / "current_business_impacts.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(CURRENT_DIR / "current_business_impacts.csv", index=False)
    _write_json(CURRENT_DIR / "current_events.json", events)
    graph = _build_current_graph(events)
    graph_path = CURRENT_DIR / "current_supply_chain_graph.graphml"
    nx.write_graphml(graph, graph_path)
    risk = _current_risk(graph, impacts)
    risk.to_csv(CURRENT_DIR / "current_risk_scores.csv", index=False, encoding="utf-8-sig")
    operational_events = [event for event in events if is_currently_eligible(event)]
    excluded_events = len(events) - len(operational_events)
    location_resolved = sum(
        1 for event in operational_events if event.get("Location") not in ("", "Unknown")
    )
    graph_entity_resolved = 0
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Event" or attrs.get("operationally_relevant") != "true":
            continue
        if any(
            graph.nodes[neighbor].get("node_type") in {"Warehouse", "Location"}
            for neighbor in graph.successors(node_id)
        ):
            graph_entity_resolved += 1
    graph_entity_unresolved = len(operational_events) - graph_entity_resolved
    metadata = {
        "pipeline_run_timestamp": pipeline_run_timestamp,
        "run_timestamp": pipeline_run_timestamp,
        "weather_fetch_timestamp": weather.get("collected_at"),
        "news_fetch_timestamp": news.get("collected_at"),
        "event_count": len(events),
        "raw_event_count": len(events),
        "operational_event_count": len(operational_events),
        "excluded_event_count": excluded_events,
        "resolved_event_count": graph_entity_resolved,
        "unresolved_event_count": graph_entity_unresolved,
        "location_resolved_count": location_resolved,
        "graph_entity_resolved_count": graph_entity_resolved,
        "graph_entity_unresolved_count": graph_entity_unresolved,
        "collector_sources": source_status,
        "collector_failures": [
            f"{category}.{source}: {details.get('error')}"
            for category, sources in source_status.items()
            for source, details in sources.items()
            if not details.get("success")
        ],
        "current_only": True,
        "historical_files_used_as_current": False,
    }
    _write_json(CURRENT_DIR / "current_run_metadata.json", metadata)
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    metadata = refresh_current_state()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
