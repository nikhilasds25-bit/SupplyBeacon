from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.event_extraction import temporal_validity


def test_health_is_available_without_artifact_access() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_refresh_requires_admin_key(monkeypatch) -> None:
    monkeypatch.delenv("SUPPLYBEACON_ADMIN_API_KEY", raising=False)
    client = TestClient(api_module.app)
    response = client.post("/current/refresh")
    assert response.status_code == 401


def test_refresh_rejects_wrong_admin_key(monkeypatch) -> None:
    monkeypatch.setenv("SUPPLYBEACON_ADMIN_API_KEY", "expected")
    client = TestClient(api_module.app)
    response = client.post("/current/refresh", headers={"X-SupplyBeacon-Admin-Key": "wrong"})
    assert response.status_code == 401


def test_refresh_accepts_correct_admin_key_without_external_calls(monkeypatch) -> None:
    monkeypatch.setenv("SUPPLYBEACON_ADMIN_API_KEY", "expected")
    monkeypatch.setattr(api_module, "refresh_current_state", lambda: None)
    monkeypatch.setattr(api_module, "invalidate_retriever_cache", lambda: None)
    monkeypatch.setattr(api_module, "current_metadata", lambda: {"status": "refreshed"})
    client = TestClient(api_module.app)
    response = client.post("/current/refresh", headers={"X-SupplyBeacon-Admin-Key": "expected"})
    assert response.status_code == 200
    assert response.json() == {"status": "refreshed"}


def test_agentic_query_does_not_require_admin_key(monkeypatch) -> None:
    monkeypatch.delenv("SUPPLYBEACON_ADMIN_API_KEY", raising=False)
    monkeypatch.setattr(api_module, "run_agentic_query", lambda query: {"query": query, "fallback_used": False})
    client = TestClient(api_module.app)
    response = client.post("/query/agentic", json={"query": "What is current network risk?"})
    assert response.status_code == 200
    assert response.json()["query"] == "What is current network risk?"


def test_development_cors_origin_is_allowed() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_current_read_only_endpoints_return_current_artifacts() -> None:
    client = TestClient(api_module.app)
    assert client.get("/current/weather").status_code == 200
    assert client.get("/current/impact").status_code == 200
    assert client.get("/current/events").status_code == 200
    assert client.get("/current/products").status_code == 200
    assert client.get("/current/metadata").status_code == 200


def test_current_impact_resolves_event_ids_and_uses_valid_graph_paths(monkeypatch) -> None:
    events = json.loads(Path("data/current/current_events.json").read_text(encoding="utf-8"))
    active = [event for event in events if event.get("event_status") == "ACTIVE" and event.get("Operationally_Relevant") is True]
    # The persisted snapshot is a fixed fixture, not a live clock. Keep this
    # structural-path assertion inside its recorded event-validity window.
    if active:
        fixed_now = datetime.fromisoformat(active[0]["event_start"]) + timedelta(minutes=1)
        monkeypatch.setattr(temporal_validity, "utc_now", lambda: fixed_now)
    client = TestClient(api_module.app)
    traces = client.get("/current/impact").json()
    graph = nx.read_graphml(Path("data/current/current_supply_chain_graph.graphml"))
    assert len(traces) == len(active)
    for trace in traces:
        graph_event_id = trace["graph_event_node_id"]
        assert graph_event_id in graph
        assert graph.nodes[graph_event_id]["node_type"] == "Event"
        assert str(graph.nodes[graph_event_id].get("event_id")) == str(trace["event"]["event_id"])
        assert set(trace["network_risk_levels"]).issuperset({"Medium", "High", "Critical"})
        assert trace["network_risk_levels"]["Medium"] >= trace["product_risk_levels"]["Medium"]
        for path in trace["product_paths"]:
            product_id = path["product_id"]
            warehouse_id = path["warehouse_id"]
            assert graph.nodes[product_id]["node_type"] == "Product"
            assert graph.nodes[warehouse_id]["node_type"] == "Warehouse"
            assert (
                graph.get_edge_data(product_id, warehouse_id, {}).get("relationship") == "STORED_AT"
                or graph.get_edge_data(warehouse_id, product_id, {}).get("relationship") == "STORED_AT"
            )
            direct_event_exposure = (
                graph.get_edge_data(warehouse_id, graph_event_id, {}).get("relationship") == "AFFECTS"
                or graph.get_edge_data(graph_event_id, warehouse_id, {}).get("relationship") == "AFFECTS"
            )
            location_event_exposure = any(
                (
                    graph.get_edge_data(warehouse_id, location_id, {}).get("relationship") == "LOCATED_IN"
                    or graph.get_edge_data(location_id, warehouse_id, {}).get("relationship") == "LOCATED_IN"
                )
                and (
                    graph.get_edge_data(location_id, graph_event_id, {}).get("relationship") == "AFFECTS"
                    or graph.get_edge_data(graph_event_id, location_id, {}).get("relationship") == "AFFECTS"
                )
                for location_id in path["location_ids"]
            )
            assert direct_event_exposure or location_event_exposure
        for path in trace["supplier_paths"]:
            assert graph.nodes[path["supplier_id"]]["node_type"] == "Supplier"
            assert graph.nodes[path["brand_id"]]["node_type"] == "Brand"
            assert graph.nodes[path["product_id"]]["node_type"] == "Product"
            assert (
                graph.get_edge_data(path["supplier_id"], path["brand_id"], {}).get("relationship") == "SUPPLIES"
                or graph.get_edge_data(path["brand_id"], path["supplier_id"], {}).get("relationship") == "SUPPLIES"
            )
            assert (
                graph.get_edge_data(path["brand_id"], path["product_id"], {}).get("relationship") == "PRODUCES"
                or graph.get_edge_data(path["product_id"], path["brand_id"], {}).get("relationship") == "PRODUCES"
            )
            assert path["product_id"] in next(
                warehouse["product_ids"] for warehouse in trace["warehouses"]
                if warehouse["node_id"] == path["warehouse_id"]
            )
            assert (
                graph.get_edge_data(path["product_id"], path["warehouse_id"], {}).get("relationship") == "STORED_AT"
                or graph.get_edge_data(path["warehouse_id"], path["product_id"], {}).get("relationship") == "STORED_AT"
            )
            assert any(
                (
                    graph.get_edge_data(path["warehouse_id"], location_id, {}).get("relationship") == "LOCATED_IN"
                    or graph.get_edge_data(location_id, path["warehouse_id"], {}).get("relationship") == "LOCATED_IN"
                )
                and (
                    graph.get_edge_data(location_id, path["event_id"], {}).get("relationship") == "AFFECTS"
                    or graph.get_edge_data(path["event_id"], location_id, {}).get("relationship") == "AFFECTS"
                )
                for location_id in path["location_ids"]
            ) or (
                graph.get_edge_data(path["warehouse_id"], path["event_id"], {}).get("relationship") == "AFFECTS"
                or graph.get_edge_data(path["event_id"], path["warehouse_id"], {}).get("relationship") == "AFFECTS"
            )


def test_current_impact_severity_matches_independent_risk_artifact_counts(monkeypatch) -> None:
    # Isolated fixture: this test must not depend on yesterday's live alert #6.
    now = datetime.now(timezone.utc)
    event = {"Event_ID": 1, "Title": "Test flood", "Operationally_Relevant": True,
             "event_start": (now - timedelta(hours=1)).isoformat(),
             "event_end": (now + timedelta(hours=1)).isoformat()}
    graph = nx.DiGraph()
    for node, kind in (("event", "Event"), ("warehouse", "Warehouse"), ("location", "Location"),
                       ("product", "Product"), ("supplier", "Supplier"), ("brand", "Brand")):
        graph.add_node(node, node_type=kind, name="Test flood" if kind == "Event" else node)
    graph.nodes["event"]["event_id"] = "1"
    for source, target, relationship in (("event", "warehouse", "AFFECTS"), ("event", "location", "AFFECTS"),
                                         ("warehouse", "location", "LOCATED_IN"), ("product", "warehouse", "STORED_AT"),
                                         ("supplier", "brand", "SUPPLIES"), ("brand", "product", "PRODUCES")):
        graph.add_edge(source, target, relationship=relationship)
    state = {"events": [event], "graph": graph, "risk": pd.DataFrame([
        {"Node_ID": "product", "Risk_Level": "Medium"},
        {"Node_ID": "warehouse", "Risk_Level": "High"},
        {"Node_ID": "location", "Risk_Level": "High"},
        {"Node_ID": "supplier", "Risk_Level": "Critical"},
    ])}
    monkeypatch.setattr(api_module, "load_current_state", lambda: state)
    client = TestClient(api_module.app)
    trace = next(
        item for item in client.get("/current/impact").json()
        if item["graph_event_node_id"] == "event"
    )
    state = api_module.load_current_state()
    risk_map = api_module._risk_map(state)
    node_ids_by_type = {
        "Product": set(trace["product_ids"]),
        "Warehouse": {warehouse["node_id"] for warehouse in trace["warehouses"]},
        "Supplier": set(trace["supplier_ids"]),
        "Location": {
            location_id
            for warehouse in trace["warehouses"]
            for location_id in warehouse.get("location_ids", [])
        },
    }
    expected_by_type = {
        node_type: {
            level: sum(
                1
                for node_id in node_ids
                if risk_map.get(node_id, {}).get("Risk_Level") == level
            )
            for level in ("Medium", "High", "Critical")
        }
        for node_type, node_ids in node_ids_by_type.items()
    }
    expected_network = {
        level: sum(counts[level] for counts in expected_by_type.values())
        for level in ("Medium", "High", "Critical")
    }
    assert trace["network_risk_levels"] == expected_network
    assert trace["product_risk_levels"] == expected_by_type["Product"]
    assert trace["network_risk_levels"] == {"Medium": 1, "High": 2, "Critical": 1}


def test_event_payload_preserves_provider_channel():
    for filename, channel in (("current_weather.json", "WEATHERAPI"), ("current_news.json", "NEWSDATA_IO"),
                              ("current_news.json", "CURRENTS_API")):
        assert api_module._event_payload({"Source": filename, "source_channel": channel})["source_channel"] == channel


def test_runtime_metadata_does_not_report_expired_events_as_operational():
    state = {"metadata": {"operational_event_count": 1, "excluded_event_count": 0,
                           "graph_entity_resolved_count": 1},
             "graph": nx.DiGraph(), "events": [{"Event_ID": 1, "Operationally_Relevant": True,
                                                "event_start": "2000-01-01T00:00:00Z",
                                                "event_end": "2000-01-02T00:00:00Z"}]}
    payload = api_module._metadata_payload(state)
    assert payload["operational_event_count"] == 0
    assert payload["excluded_event_count"] == 1
    assert payload["graph_entity_resolved_count"] == 0
    assert payload["graph_entity_unresolved_count"] == 0
