"""Deterministic news-to-warehouse exposure boundaries."""
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.dynamic_risk.location_resolution import recognition_aliases, resolve_event_location
from src.dynamic_risk.refresh_current_state import _apply_business_rules, _current_risk, _event_locations
from src.event_extraction.event_extractor import article_to_events, detect_locations
from src.external_data.unified_sources import deduplicate_news_articles
from src.graphrag.retriever import EvidenceRetriever


NOW = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)


@pytest.fixture
def graph(monkeypatch):
    from src.event_extraction import temporal_validity
    monkeypatch.setattr(temporal_validity, "utc_now", lambda: NOW)
    graph = nx.DiGraph()
    for warehouse, city, state in (
        ("WH_REAL_066", "Thiruvananthapuram", "Kerala"),
        ("WH_COIMBATORE", "Coimbatore", "Tamil Nadu"),
        ("WH_BENGALURU", "Bengaluru", "Karnataka"),
        ("WH_CHENNAI", "Chennai", "Tamil Nadu"),
        ("WH_NOIDA", "Noida", "Uttar Pradesh"),
        ("WH_GREATER_NOIDA", "Greater Noida", "Uttar Pradesh"),
    ):
        location = f"LOC:{city}"
        graph.add_node(warehouse, node_type="Warehouse", name=f"{city} Warehouse", city=city, state=state)
        graph.add_node(location, node_type="Location", name=city, city=city, state=state, country="India")
        graph.add_edge(warehouse, location, relationship="LOCATED_IN")
    graph.add_node("PRODUCT", node_type="Product")
    graph.add_node("MODEL", node_type="ProductModel")
    graph.add_edge("WH_REAL_066", "MODEL", relationship="CATALOG_ASSOCIATED_WITH")
    return graph


def event(location, *, start="2026-10-10T10:00:00+00:00", end="2026-10-10T14:00:00+00:00", operational=True):
    return {
        "Location": location, "Operationally_Relevant": operational,
        "event_start": start, "event_end": end, "Event_Type": "Transport Disruption",
    }


def affected(graph, location, **kwargs):
    graph.add_node("EVENT", node_type="Event")
    _event_locations(graph, "EVENT", event(location, **kwargs))
    return {
        target for _, target, attrs in graph.out_edges("EVENT", data=True)
        if attrs.get("relationship") == "AFFECTS" and graph.nodes[target].get("node_type") == "Warehouse"
    }


@pytest.mark.parametrize("location, expected", [
    ("Thiruvananthapuram", {"WH_REAL_066"}),
    ("Coimbatore", {"WH_COIMBATORE"}),
    ("Karnataka", set()),
    ("Bengaluru, Karnataka", {"WH_BENGALURU"}),
    ("Unknown", set()),
    ("India", set()),
    ("London", set()),
])
def test_only_geographically_resolved_warehouses(graph, location, expected):
    assert affected(graph, location) == expected
    assert "WH_CHENNAI" not in affected(graph, "Karnataka")


@pytest.mark.parametrize("kwargs", [
    {"start": "2026-10-09T10:00:00+00:00", "end": "2026-10-09T14:00:00+00:00"},
    {"start": "2026-10-11T10:00:00+00:00", "end": "2026-10-11T14:00:00+00:00"},
    {"start": None, "end": None},
    {"operational": False},
])
def test_non_current_or_informational_events_have_no_exposure(graph, kwargs):
    assert affected(graph, "Thiruvananthapuram", **kwargs) == set()


def test_catalog_only_warehouse_has_no_product_or_supplier_exposure(graph):
    assert affected(graph, "Thiruvananthapuram") == {"WH_REAL_066"}
    graph.nodes["EVENT"].update(event_status="ACTIVE", operational_current="true", risk_score="0.8", impact_level="High", event_domain="NEWS")
    baseline = pd.DataFrame([{
        "Node_ID": "PRODUCT", "Node_Type": "Product", "Risk_Score": "0.1",
        "Risk_Level": "Low", "Connected_Events": "0", "Structural_Baseline": "0.1",
    }])
    row = _current_risk(graph, pd.DataFrame(), baseline_override=baseline).iloc[0]
    assert row["Connected_Events"] == "0"
    assert row["Risk_Score"] == ""
    assert row["Risk_Level"] == "Unavailable"
    assert not graph.has_edge("PRODUCT", "WH_REAL_066")
    retriever = EvidenceRetriever.__new__(EvidenceRetriever)
    retriever.graph = graph
    assert retriever._event_path("PRODUCT", "EVENT") is None
    assert retriever._event_path("MODEL", "EVENT") is None
    assert retriever._event_path("WH_REAL_066", "EVENT") == ["WH_REAL_066", "EVENT"]


def test_graph_vocabulary_preserves_distinct_cities(graph):
    aliases = recognition_aliases(graph)
    assert detect_locations("Greater Noida depot closed", aliases) == ["Greater Noida"]
    assert resolve_event_location(graph, "Greater Noida")["warehouse_ids"] == ["WH_GREATER_NOIDA"]
    assert resolve_event_location(graph, "Noida")["warehouse_ids"] == ["WH_NOIDA"]
    assert detect_locations("Bangalore transport halted", aliases) == ["Bengaluru"]


def test_news_keyword_without_operational_effect_remains_informational(graph):
    aliases = recognition_aliases(graph)
    items = article_to_events({
        "title": "Heavy rain warning for Coimbatore today",
        "description": "Supply chain observers monitor conditions.",
        "published_at": NOW.isoformat(),
    }, Path("current_news.json"), aliases)
    assert items and items[0]["Operationally_Relevant"] is False
    assert affected(graph, items[0]["Location"], operational=False) == set()
    vague = article_to_events({
        "title": "Supply chain disruption in Karnataka",
        "description": "Industry discussion covers potential disruption.",
        "published_at": NOW.isoformat(),
    }, Path("current_news.json"), aliases)
    assert vague and all(item["Operationally_Relevant"] is False for item in vague)


def test_business_impact_warehouse_and_location_use_graph_resolution(graph):
    rows = _apply_business_rules([{
        "Event_ID": 1, "Event_Type": "Transport Disruption", "Severity": "High",
        "Confidence": 0.9, "Location": "Karnataka", "Affected_Entity": "Unknown",
        "Operationally_Relevant": True,
        "event_start": "2026-10-10T10:00:00+00:00", "event_end": "2026-10-10T14:00:00+00:00",
    }], graph=graph)
    row = rows.iloc[0]
    assert row["Affected_Warehouse"] == "Unknown"
    assert row["Affected_Location"] == "Unknown"
    assert row["matched_warehouse_ids"] == ""
    assert row["location_resolution_basis"] == "UNRESOLVED"


def test_statewide_business_impact_requires_explicit_scope(graph):
    rows = _apply_business_rules([{
        "Event_ID": 3, "Event_Type": "Transport Disruption", "Severity": "High",
        "Confidence": 0.9, "Location": "Karnataka", "Location_Scope": "STATEWIDE",
        "Affected_Entity": "Unknown", "Operationally_Relevant": True,
        "event_start": "2026-10-10T10:00:00+00:00", "event_end": "2026-10-10T14:00:00+00:00",
    }], graph=graph)
    row = rows.iloc[0]
    assert row["Affected_Warehouse"] == "Bengaluru Warehouse"
    assert row["matched_warehouse_ids"] == "WH_BENGALURU"
    assert row["location_resolution_basis"] == "STATE_LOCATION"


def test_business_impact_unknown_does_not_invent_warehouse(graph):
    row = _apply_business_rules([{
        "Event_ID": 2, "Event_Type": "Transport Disruption", "Severity": "High",
        "Confidence": 0.8, "Location": "India", "Affected_Entity": "Unknown",
    }], graph=graph).iloc[0]
    assert row["Affected_Warehouse"] == "Unknown"
    assert row["Affected_Location"] == "Unknown"
    assert row["matched_warehouse_ids"] == ""


def test_cross_provider_dedup_retains_all_references_with_distinct_urls():
    articles = deduplicate_news_articles([
        {"title": "Coimbatore warehouse closed", "url": "https://a.test/1", "published_at": "2026-10-10T10:00:00Z", "source_channel": "NEWS_API"},
        {"title": "Coimbatore warehouse closed", "url": "https://b.test/2", "published_at": "2026-10-10T11:00:00Z", "source_channel": "CURRENTS_API"},
    ])
    assert len(articles) == 1
    assert {reference["source_channel"] for reference in articles[0]["source_references"]} == {"NEWS_API", "CURRENTS_API"}
