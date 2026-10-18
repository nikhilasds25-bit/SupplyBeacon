"""Focused, read-only tests for Scenario Lab graph simulation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.agentic import grounded_domains
from src.dynamic_risk.current_state import BASE_DIR, load_current_state


CLIENT = TestClient(api_module.app)
CURRENT_FILES = [
    "current_supply_chain_graph.graphml",
    "current_risk_scores.csv",
    "current_events.json",
    "current_weather.json",
    "current_news.json",
    "current_business_impacts.csv",
    "current_run_metadata.json",
]


def _warehouse() -> str:
    state = load_current_state()
    return next(
        node_id
        for node_id, attrs in state["graph"].nodes(data=True)
        if attrs.get("node_type") == "Warehouse"
    )


def _simulate() -> dict:
    response = CLIENT.post(
        "/scenario/simulate",
        json={
            "scenario_type": "WAREHOUSE_UNAVAILABLE",
            "target_node_id": _warehouse(),
            "duration_hours": 48,
            "scenario_severity": "Medium",
        },
    )
    assert response.status_code == 200
    return response.json()


def _neighbors(graph: nx.Graph, node_id: str, relationship: str, node_type: str) -> set[str]:
    values = set()
    for source, target, edge in graph.edges(node_id, data=True):
        other = target if source == node_id else source
        if edge.get("relationship") == relationship and graph.nodes[other].get("node_type") == node_type:
            values.add(str(other))
    return values


def _hashes() -> dict[str, str]:
    return {
        filename: hashlib.sha256((BASE_DIR / "data" / "current" / filename).read_bytes()).hexdigest()
        for filename in CURRENT_FILES
    }


def test_valid_warehouse_simulation_returns_200() -> None:
    result = _simulate()
    assert result["scenario"]["scenario_type"] == "WAREHOUSE_UNAVAILABLE"
    assert result["provenance"].startswith("DERIVED")


def test_invalid_warehouse_returns_400() -> None:
    response = CLIENT.post(
        "/scenario/simulate",
        json={
            "scenario_type": "WAREHOUSE_UNAVAILABLE",
            "target_node_id": "WAREHOUSE:does-not-exist",
            "duration_hours": 48,
            "scenario_severity": "Medium",
        },
    )
    assert response.status_code == 400


def test_scenario_products_use_explicit_stored_at_and_no_category_leakage() -> None:
    result = _simulate()
    graph = load_current_state()["graph"]
    disrupted = result["scenario"]["target_node_id"]
    assert all(disrupted in _neighbors(graph, product_id, "STORED_AT", "Warehouse") for product_id in result["product_ids"])


def test_scenario_suppliers_use_supplies_and_produces() -> None:
    result = _simulate()
    graph = load_current_state()["graph"]
    products = set(result["product_ids"])
    for supplier_id in result["supplier_ids"]:
        valid = False
        for brand_id in _neighbors(graph, supplier_id, "SUPPLIES", "Brand"):
            produced = _neighbors(graph, brand_id, "PRODUCES", "Product")
            valid = valid or bool(produced.intersection(products))
        assert valid


def test_delta_sets_are_exact() -> None:
    result = _simulate()
    current_products = {
        product_id
        for trace in api_module.current_impact()
        for product_id in trace["product_ids"]
    }
    scenario_products = set(result["product_ids"])
    assert set(result["newly_exposed_product_ids"]) == scenario_products - current_products
    assert set(result["overlap_product_ids"]) == scenario_products & current_products


def test_alternative_warehouses_have_direct_matching_product_relationships() -> None:
    result = _simulate()
    graph = load_current_state()["graph"]
    disrupted = result["scenario"]["target_node_id"]
    products = set(result["product_ids"])
    for candidate in result["alternative_warehouse_candidates"]:
        warehouse_id = candidate["warehouse_id"]
        assert warehouse_id != disrupted
        matching = {
            product_id
            for product_id in products
            if warehouse_id in _neighbors(graph, product_id, "STORED_AT", "Warehouse")
        }
        assert candidate["matching_product_count"] == len(matching)
        assert set(candidate["matching_product_ids"]) == matching


def test_simulation_does_not_modify_current_artifacts() -> None:
    before = _hashes()
    _simulate()
    assert _hashes() == before


def test_simulation_does_not_call_refresh_or_external_collectors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("Scenario simulation must not refresh or collect external data")

    monkeypatch.setattr(api_module, "refresh_current_state", fail)
    result = _simulate()
    assert result["scenario"]["scenario_type"] == "WAREHOUSE_UNAVAILABLE"


def test_scenario_explanation_uses_deterministic_fallback_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SUPPLYBEACON_LLM_ENABLED", raising=False)
    response = CLIENT.post(
        "/scenario/explain",
        json={
            "candidate_type": "warehouse",
            "evidence": {
                "candidate_id": "WAREHOUSE:example",
                "candidate_name": "Example Warehouse",
                "matching_product_count": 3,
                "matching_product_ids": ["PRODUCT:1", "PRODUCT:2", "PRODUCT:3"],
                "explicit_relationship": "STORED_AT",
                "current_risk": "Medium",
                "current_event_exposure": 0,
                "unsupported_claim": "must not be forwarded",
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_mode"] == "DETERMINISTIC_FALLBACK"
    assert "POTENTIAL MODELED ALTERNATIVE" in body["answer"]
    assert all(limit in body["answer"] for limit in body["limitations"])


def _synthetic_scenario(monkeypatch: pytest.MonkeyPatch) -> nx.DiGraph:
    graph = nx.DiGraph()
    for warehouse in ("W_TARGET", "W_ALTERNATIVE", "W_CATALOG", "W_SAME_BRAND", "W_INACTIVE"):
        graph.add_node(warehouse, node_type="Warehouse", name=warehouse, catalog_only=warehouse == "W_CATALOG")
    graph.nodes["W_INACTIVE"]["status"] = "INACTIVE"
    for product in ("P_AFFECTED", "P_UNRELATED"):
        graph.add_node(product, node_type="Product", name=product)
    graph.add_node("MODEL", node_type="ProductModel")
    graph.add_node("BRAND", node_type="Brand")
    graph.add_node("S_LINKED", node_type="Supplier")
    graph.add_node("S_UNRELATED", node_type="Supplier")
    graph.add_edge("P_AFFECTED", "W_TARGET", relationship="STORED_AT")
    graph.add_edge("P_AFFECTED", "W_ALTERNATIVE", relationship="STORED_AT")
    graph.add_edge("P_UNRELATED", "W_SAME_BRAND", relationship="STORED_AT")
    graph.add_edge("BRAND", "P_AFFECTED", relationship="PRODUCES")
    graph.add_edge("BRAND", "P_UNRELATED", relationship="PRODUCES")
    graph.add_edge("S_LINKED", "BRAND", relationship="SUPPLIES")
    graph.add_edge("S_UNRELATED", "W_TARGET", relationship="ASSOCIATED_WITH")
    graph.add_edge("W_CATALOG", "MODEL", relationship="CATALOG_ASSOCIATED_WITH")
    graph.add_edge("W_SAME_BRAND", "MODEL", relationship="CATALOG_ASSOCIATED_WITH")
    state_loader = lambda: {"graph": graph, "risk": pd.DataFrame([
        {"Node_ID": "W_TARGET", "Risk_Score": 42, "Risk_Level": "Medium"}
    ]), "events": []}
    monkeypatch.setattr(api_module, "load_current_state", state_loader)
    monkeypatch.setattr(grounded_domains, "load_current_state", state_loader)
    monkeypatch.setattr(api_module, "current_impact", lambda: [])
    return graph


def _post_target(target: str, duration: int = 48, severity: str = "High"):
    return CLIENT.post("/scenario/simulate", json={"scenario_type": "WAREHOUSE_UNAVAILABLE", "target_node_id": target, "duration_hours": duration, "scenario_severity": severity})


def test_synthetic_paths_catalog_and_current_risk_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_scenario(monkeypatch)
    body = _post_target("W_TARGET").json()
    assert body["affected_product_ids"] == ["P_AFFECTED"]
    assert body["affected_supplier_ids"] == ["S_LINKED"]
    assert [item["warehouse_id"] for item in body["alternative_warehouse_candidates"]] == ["W_ALTERNATIVE"]
    assert body["current_risk_score"] == 42
    assert body["scenario_state"]["exposed_products"] == 1
    assert "DataCo prediction is not used" in body["explanation"]
    assert "Transport routes" in body["explanation"]
    assert body["evidence_basis"] == ["Product → STORED_AT → target Warehouse", "Supplier → SUPPLIES → Brand → PRODUCES → affected Product", "Affected Product → STORED_AT → alternative Warehouse", "CATALOG_ASSOCIATED_WITH is contextual only"]


def test_catalog_only_warehouse_does_not_fabricate_impact(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_scenario(monkeypatch)
    body = _post_target("W_CATALOG").json()
    assert body["catalog_only"] is True
    assert body["catalog_association_count"] == 1
    assert body["inventory_relationship_available"] is False
    assert body["impact_status"] == "UNAVAILABLE_VERIFIED_INVENTORY"
    assert body["affected_product_ids"] == body["affected_supplier_ids"] == []
    assert body["alternative_warehouse_candidates"] == []
    assert "cannot claim specific affected products" in body["explanation"]
    assert body["current_risk_score"] is None


def test_inactive_and_nonwarehouse_targets_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_scenario(monkeypatch)
    assert "inactive" in _post_target("W_INACTIVE").json()["detail"]
    assert "Warehouse" in _post_target("P_AFFECTED").json()["detail"]


def test_duration_and_severity_are_descriptive_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_scenario(monkeypatch)
    first = _post_target("W_TARGET", 12, "Medium").json()
    second = _post_target("W_TARGET", 72, "Critical").json()
    for field in ("affected_product_ids", "affected_supplier_ids", "alternative_warehouse_candidates", "current_risk_score"):
        assert first[field] == second[field]
    assert first["scenario"]["duration_hours"] == 12
    assert second["scenario"]["severity_assumption"] == "Critical"


def test_candidate_explanation_revalidates_graph_and_llm_cannot_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_scenario(monkeypatch)
    monkeypatch.setenv("SUPPLYBEACON_LLM_ENABLED", "true")
    monkeypatch.setattr(api_module.OpenAICompatibleProvider, "generate", lambda *_args: '{"answer":"POTENTIAL MODELED ALTERNATIVE: guaranteed rerouting capacity"}')
    valid = CLIENT.post("/scenario/explain", json={"candidate_type": "warehouse", "evidence": {"target_node_id": "W_TARGET", "candidate_id": "W_ALTERNATIVE", "matching_product_count": 999}})
    assert valid.status_code == 200
    assert valid.json()["explanation_mode"] == "DETERMINISTIC_FALLBACK"
    assert "1 scenario product(s)" in valid.json()["answer"]
    assert "999" not in valid.json()["answer"]
    invalid = CLIENT.post("/scenario/explain", json={"candidate_type": "warehouse", "evidence": {"target_node_id": "W_TARGET", "candidate_id": "W_CATALOG"}})
    assert invalid.status_code == 400


def test_real_wh_real_066_is_catalog_context_not_stock() -> None:
    body = _post_target("WH_REAL_066").json()
    assert body["scenario"]["target_name"] == "Allied Electronics Corporation Warehouse / Depot - Trivandrum"
    assert body["catalog_only"] is True
    assert body["inventory_relationship_available"] is False
    assert body["affected_product_ids"] == body["affected_supplier_ids"] == []
    assert "no verified operational STORED_AT" in body["explanation"]


def test_real_delhi_warehouse_uses_verified_stored_at_products() -> None:
    body = _post_target("WAREHOUSE:Delhi_Warehouse").json()
    graph = load_current_state()["graph"]
    assert body["inventory_relationship_available"] is True
    assert body["affected_product_count"] > 0
    assert all(
        "WAREHOUSE:Delhi_Warehouse" in _neighbors(graph, product_id, "STORED_AT", "Warehouse")
        for product_id in body["affected_product_ids"]
    )
    assert all(
        candidate["warehouse_id"] in _neighbors(graph, product_id, "STORED_AT", "Warehouse")
        for candidate in body["alternative_warehouse_candidates"]
        for product_id in candidate["matching_product_ids"]
    )


@pytest.mark.parametrize("route", ["/query", "/query/agentic"])
@pytest.mark.parametrize("wording", ["What if W_CATALOG warehouse is unavailable?", "What happens to W_CATALOG warehouse?", "Are there alternatives for W_CATALOG warehouse?"])
def test_what_if_routes_use_authoritative_scenario(monkeypatch: pytest.MonkeyPatch, route: str, wording: str) -> None:
    _synthetic_scenario(monkeypatch)
    response = CLIENT.post(route, json={"query": wording})
    assert response.status_code == 200
    body = response.json()
    assert body["scenario_result"]["impact_status"] == "UNAVAILABLE_VERIFIED_INVENTORY"
    assert body["scenario_result"]["affected_product_ids"] == []
    assert "cannot claim specific affected products" in body["answer"]
