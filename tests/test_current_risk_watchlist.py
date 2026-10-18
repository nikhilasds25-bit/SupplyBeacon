"""Fixed-fixture tests for current rule-based risk availability and ranking."""
from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from src.api.app import _risk_payload
from src.dynamic_risk.refresh_current_state import _current_risk, _extract_events
from src.dynamic_risk.risk_views import network_summary, operational_event_ids, watchlist
from src.event_extraction.event_extractor import weather_to_events
from pathlib import Path


@pytest.fixture
def fixture_state():
    graph = nx.DiGraph()
    for warehouse_id, city in (("W_ACTIVE", "Coimbatore"), ("WH_REAL_066", "Thiruvananthapuram"), ("W_OTHER", "Chennai")):
        graph.add_node(warehouse_id, node_type="Warehouse", name=f"{city} Warehouse", city=city, state="Tamil Nadu" if city != "Thiruvananthapuram" else "Kerala", catalog_only=warehouse_id == "WH_REAL_066")
        location_id = f"LOC:{city}"
        graph.add_node(location_id, node_type="Location", city=city, state=graph.nodes[warehouse_id]["state"])
        graph.add_edge(warehouse_id, location_id, relationship="LOCATED_IN")
    graph.add_node("P", node_type="Product", name="Inventory product")
    graph.add_node("MODEL", node_type="ProductModel", name="Catalog model")
    graph.add_node("S", node_type="Supplier", name="Supplier")
    graph.add_node("B", node_type="Brand", name="Brand")
    graph.add_edge("P", "W_ACTIVE", relationship="STORED_AT")
    graph.add_edge("S", "B", relationship="SUPPLIES")
    graph.add_edge("B", "P", relationship="PRODUCES")
    graph.add_edge("WH_REAL_066", "MODEL", relationship="CATALOG_ASSOCIATED_WITH")
    for event_id, status, operational, target in (
        ("E_ACTIVE", "ACTIVE", "true", "W_ACTIVE"),
        ("E_INFO", "ACTIVE", "false", "W_ACTIVE"),
        ("E_EXPIRED", "EXPIRED", "false", "W_ACTIVE"),
        ("E_UPCOMING", "UPCOMING", "false", "W_ACTIVE"),
        ("E_UNKNOWN", "UNKNOWN", "false", "W_ACTIVE"),
    ):
        graph.add_node(event_id, node_type="Event", name=event_id, event_status=status,
                       operational_current=operational, operationally_relevant="true" if event_id != "E_INFO" else "false",
                       risk_score="0.8", impact_level="High", event_domain="WEATHER", severity="High")
        graph.add_edge(event_id, target, relationship="AFFECTS")
    baseline = pd.DataFrame([
        {"Node_ID": node_id, "Node_Type": kind, "Node_Name": node_id,
         "Risk_Score": "0.1", "Risk_Level": "Low", "Structural_Baseline": "0.1",
         "Connected_Events": "0", "Event_Exposure": "0"}
        for node_id, kind in (("W_ACTIVE", "Warehouse"), ("P", "Product"), ("S", "Supplier"))
    ])
    risk = _current_risk(graph, pd.DataFrame([{"Affected_Warehouse": "Chennai Warehouse", "Risk_Score": 1}]), baseline_override=baseline)
    events = [
        {"Event_ID": 1, "Title": "Coimbatore operational disruption", "Location": "Coimbatore",
         "Operationally_Relevant": True, "event_start": "2026-10-10T10:00:00+00:00", "event_end": "2026-10-10T14:00:00+00:00"},
        {"Event_ID": 2, "Title": "Chennai commentary", "Location": "Chennai",
         "Operationally_Relevant": False, "event_start": "2026-10-10T10:00:00+00:00", "event_end": "2026-10-10T14:00:00+00:00"},
    ]
    return {"graph": graph, "risk": risk, "events": events, "metadata": {"pipeline_run_timestamp": "2026-10-10T12:00:00+00:00"}}


def test_only_active_operational_graph_event_contributes(fixture_state):
    graph, risk = fixture_state["graph"], fixture_state["risk"].set_index("Node_ID")
    assert operational_event_ids(graph, "W_ACTIVE") == ["E_ACTIVE"]
    assert operational_event_ids(graph, "W_OTHER") == []
    assert operational_event_ids(graph, "P") == ["E_ACTIVE"]
    assert operational_event_ids(graph, "S") == ["E_ACTIVE"]
    assert operational_event_ids(graph, "MODEL") == []
    assert risk.loc["W_ACTIVE", "Connected_Events"] == "1"
    assert risk.loc["W_OTHER", "Risk_Level"] == "Unavailable"
    assert risk.loc["WH_REAL_066", "Risk_Level"] == "Unavailable"
    assert risk.loc["P", "Connected_Events"] == "1"
    assert risk.loc["S", "Connected_Events"] == "1"


def test_component_breakdown_equals_final_score(fixture_state):
    row = fixture_state["risk"].set_index("Node_ID").loc["W_ACTIVE"]
    components = [float(row[field]) for field in (
        "Structural_Baseline", "Weather_Event_Component", "News_Event_Component",
        "Business_Impact_Component", "Event_Exposure_Component",
    )]
    assert float(row["Final_Scenario_Risk_Score"]) == pytest.approx(min(1.0, sum(components)), abs=0.001)
    assert float(row["Risk_Score"]) == float(row["Final_Scenario_Risk_Score"])
    assert float(row["Weather_Event_Component"]) == pytest.approx(0.24)
    assert float(row["News_Event_Component"]) == 0.0


def test_structural_baseline_uses_source_features_not_stale_event_score(fixture_state):
    graph = fixture_state["graph"].copy()
    graph.remove_edges_from([(u, v) for u, v, edge in graph.edges(data=True) if edge.get("relationship") == "AFFECTS"])
    baseline = pd.DataFrame([{
        "Node_ID": "P", "Node_Type": "Product", "Node_Name": "Inventory product",
        "Risk_Score": "0.9", "Graph_Degree": "10", "Connected_Products": "4",
        "Max_Event_Severity": "Critical", "Average_Event_Confidence": "1",
        "Business_Impact_Risk": "1", "Event_Exposure": "1",
    }])
    row = _current_risk(graph, pd.DataFrame(), baseline_override=baseline).iloc[0]
    assert float(row["Structural_Baseline"]) == pytest.approx(0.06)
    assert float(row["Risk_Score"]) == pytest.approx(0.06)
    assert row["Connected_Events"] == "0"


def test_catalog_warehouse_can_have_event_only_risk_without_product_risk(fixture_state):
    graph = fixture_state["graph"].copy()
    graph.add_node("E_CURATED", node_type="Event", name="Kerala disruption",
                   event_status="ACTIVE", operational_current="true", operationally_relevant="true",
                   risk_score="0.8", impact_level="High", event_domain="NEWS", severity="High")
    graph.add_edge("E_CURATED", "WH_REAL_066", relationship="AFFECTS")
    risk = _current_risk(graph, pd.DataFrame(), baseline_override=fixture_state["risk"]).set_index("Node_ID")
    assert risk.loc["WH_REAL_066", "Risk_Availability"] == "EVENT_ONLY"
    assert float(risk.loc["WH_REAL_066", "Risk_Score"]) > 0
    assert operational_event_ids(graph, "MODEL") == []
    assert not any(node_id == "MODEL" for node_id in risk.index)


def test_network_excludes_unavailable_and_watchlist_prioritizes_exposure(monkeypatch, fixture_state):
    from src.event_extraction import temporal_validity
    from datetime import datetime, timezone
    monkeypatch.setattr(temporal_validity, "utc_now", lambda: datetime(2026, 10, 10, 12, tzinfo=timezone.utc))
    summary = network_summary(fixture_state)
    assert summary["monitored_entity_count"] == 5
    assert summary["available_risk_entity_count"] == 3
    assert summary["unavailable_risk_entity_count"] == 2
    assert summary["operationally_exposed_entity_count"] == 3
    assert summary["active_signal_count"] == 2
    ranked = watchlist(fixture_state)
    assert ranked[0]["entity_id"] == "W_ACTIVE"
    assert ranked[0]["operational_event_count"] == 1
    assert ranked[0]["risk_components"]["Weather_Event_Component"] == pytest.approx(0.24)
    assert all(item["risk_score"] is None for item in ranked[1:])
    assert next(item for item in ranked if item["entity_id"] == "WH_REAL_066")["catalog_only"] is True


def test_api_does_not_coerce_null_to_zero():
    payload = _risk_payload({"Risk_Score": "", "Risk_Level": "Low", "Connected_Events": "0"})
    assert payload["risk_score"] is None
    assert payload["risk_level"] == "Unavailable"
    assert payload["risk_availability"] == "UNAVAILABLE"
    assert payload["risk_components"]["Final_Scenario_Risk_Score"] is None


def test_api_watchlist_and_network_counts(monkeypatch, fixture_state):
    from src.api import app as api
    from fastapi.testclient import TestClient
    from src.event_extraction import temporal_validity
    from datetime import datetime, timezone
    monkeypatch.setattr(temporal_validity, "utc_now", lambda: datetime(2026, 10, 10, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(api, "load_current_state", lambda: fixture_state)
    summary = api.current_summary()
    assert summary["available_risk_entity_count"] == 3
    assert summary["unavailable_risk_entity_count"] == 2
    assert api.current_risk_watchlist("Warehouse")[0]["entity_id"] == "W_ACTIVE"
    assert next(row for row in api.current_risk_watchlist("Warehouse") if row["entity_id"] == "WH_REAL_066")["risk_score"] is None
    response = TestClient(api.app).get("/current/risk-watchlist")
    assert response.status_code == 200
    assert next(row for row in response.json() if row["entity_id"] == "WH_REAL_066")["risk_score"] is None


def test_rejected_weather_has_no_event_to_contribute():
    assert weather_to_events("Coimbatore", {
        "source_channel": "WEATHERAPI", "location_validation_status": "REJECTED",
        "weather": [{"main": "Cyclone", "description": "cyclone"}],
        "alerts": [{"headline": "Cyclone alert"}],
    }, Path("current_weather.json")) == []
    assert _extract_events({"cities": {"Coimbatore:OPENWEATHER": {
        "source_channel": "OPENWEATHER", "location_validation_status": "REJECTED",
        "weather": [{"main": "Cyclone", "description": "cyclone"}],
        "dt": 1790236800,
    }}}, {"articles": []}) == []
