"""Deterministic grounding checks for safety-critical Ask domains."""
from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agentic import grounded_domains, service
from src.agentic import tools as agentic_tools
from src.api import app as api_module


CLIENT = TestClient(api_module.app)


@pytest.fixture
def controlled_graph(monkeypatch: pytest.MonkeyPatch) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("WAREHOUSE:Delhi_Warehouse", node_type="Warehouse", name="Delhi Warehouse")
    graph.add_node("WH_REAL_066", node_type="Warehouse", name="Allied Depot - Trivandrum",
                   warehouse_id="WH_REAL_066", city="Thiruvananthapuram", state="Kerala", catalog_only=True)
    graph.add_node("W_TWIN_A", node_type="Warehouse", name="Twin Warehouse")
    graph.add_node("W_TWIN_B", node_type="Warehouse", name="Twin Warehouse")
    graph.add_node("MODEL_A", node_type="ProductModel", name="Catalog Model A")
    graph.add_node("P_DELHI", node_type="Product", name="Operational Product")
    graph.add_node("B_DELHI", node_type="Brand", name="Operational Brand")
    graph.add_node("S_DELHI", node_type="Supplier", name="Operational Supplier")
    graph.add_edge("WH_REAL_066", "MODEL_A", relationship="CATALOG_ASSOCIATED_WITH")
    graph.add_edge("P_DELHI", "WAREHOUSE:Delhi_Warehouse", relationship="STORED_AT")
    graph.add_edge("B_DELHI", "P_DELHI", relationship="PRODUCES")
    graph.add_edge("S_DELHI", "B_DELHI", relationship="SUPPLIES")
    state = {"graph": graph, "risk": pd.DataFrame([
        {"Node_ID": "WAREHOUSE:Delhi_Warehouse", "Risk_Score": 52, "Risk_Level": "High", "Risk_Reason": "Controlled current risk"},
        {"Node_ID": "WH_REAL_066", "Risk_Score": None, "Risk_Level": "Unavailable"},
    ]), "events": [], "metadata": {}}
    monkeypatch.setattr(grounded_domains, "load_current_state", lambda: state)
    monkeypatch.setattr(api_module, "load_current_state", lambda: state)
    monkeypatch.setattr(api_module, "current_impact", lambda: [])
    return graph


@pytest.mark.parametrize("route", ["/query", "/query/agentic"])
def test_catalog_context_never_claims_stock(controlled_graph: nx.DiGraph, route: str) -> None:
    body = CLIENT.post(route, json={"query": "What products are associated with WH_REAL_066?"}).json()
    assert body["query_intent"] == "CATALOG_LOOKUP"
    assert body["catalog_context"]["product_model_ids"] == ["MODEL_A"]
    assert body["catalog_context"]["verified_stored_product_ids"] == []
    assert "physical stock" in body["answer"].lower()
    assert all(item["source_type"] != "SCENARIO_ENGINE" for item in body["evidence"])


def test_catalog_followup_without_context_clarifies(controlled_graph: nx.DiGraph) -> None:
    body = service.run_agentic_query("Are those products actually stored there?")
    assert body["scenario_result"] is None
    assert "Specify one exact warehouse" in body["answer"]


def test_catalog_store_question_does_not_confirm_physical_stock(controlled_graph: nx.DiGraph) -> None:
    body = service.run_agentic_query("Does WH_REAL_066 store product Catalog Model A?")
    assert body["catalog_context"]["product_model_ids"] == ["MODEL_A"]
    assert body["catalog_context"]["inventory_relationship_available"] is False
    assert "do not verify physical stock" in body["answer"]


def test_curated_risk_null_is_unavailable(controlled_graph: nx.DiGraph) -> None:
    body = service.run_agentic_query("What is the current risk of WH_REAL_066?")
    assert body["query_intent"] == "CURRENT_RISK"
    assert body["risk_evidence"][1]["value"] is None
    assert "Unavailable" in body["answer"]
    assert "probability" not in body["answer"]


def test_multi_intent_keeps_current_risk_and_scenario_separate(controlled_graph: nx.DiGraph) -> None:
    body = service.run_agentic_query("Why is Delhi Warehouse risky and what happens if it closes?")
    assert body["query_intent"] == "MULTI_INTENT"
    assert body["risk_evidence"][1]["value"] == 52
    assert body["scenario_result"]["affected_product_ids"] == ["P_DELHI"]
    assert body["scenario_result"]["affected_supplier_ids"] == ["S_DELHI"]
    assert {item["source_type"] for item in body["evidence"]} == {"CURRENT_GRAPH", "CURRENT_RISK_ENGINE", "SCENARIO_ENGINE"}
    assert body["llm_used"] is False


def test_curated_warehouse_scenario_has_no_operational_product(controlled_graph: nx.DiGraph) -> None:
    body = service.run_agentic_query("What happens to WH_REAL_066 if it closes?")
    assert body["scenario_result"]["inventory_relationship_available"] is False
    assert body["scenario_result"]["affected_product_ids"] == []
    assert body["scenario_result"]["affected_supplier_ids"] == []
    assert "cannot claim specific affected products" in body["answer"]


def test_exact_city_lookup_explains_missing_product_impact(controlled_graph: nx.DiGraph) -> None:
    body = service.run_agentic_query("Why is Thiruvananthapuram warehouse not showing product impact?")
    assert body["query_plan"]["resolved_entity_ids"] == ["WH_REAL_066"]
    assert body["catalog_context"]["inventory_relationship_available"] is False
    assert "No verified STORED_AT" in body["answer"]


def test_exact_id_unknown_and_ambiguous_warehouse(controlled_graph: nx.DiGraph) -> None:
    exact = service.run_agentic_query("What if WAREHOUSE:Delhi_Warehouse is unavailable?")
    assert exact["query_plan"]["resolved_entity_ids"] == ["WAREHOUSE:Delhi_Warehouse"]
    unknown = service.run_agentic_query("What if W_UNKNOWN warehouse is unavailable?")
    assert unknown["scenario_result"] is None
    assert "Specify one exact warehouse" in unknown["answer"]
    ambiguous = service.run_agentic_query("What if Twin Warehouse is unavailable?")
    assert ambiguous["scenario_result"] is None
    assert len(ambiguous["entities"]) == 2
    assert "More than one" in ambiguous["answer"]


def test_authoritative_evidence_excludes_warnings(controlled_graph: nx.DiGraph) -> None:
    body = service.run_agentic_query("What products are associated with WH_REAL_066?")
    assert body["evidence_count"] == len(body["evidence"])
    assert all(item["source_type"] != "WARNING" for item in body["evidence"])
    assert all({"source_type", "entity_id", "relation", "value", "timestamp", "temporal_status", "provenance"} <= item.keys() for item in body["evidence"])


def test_dataco_prediction_without_order_features_is_unavailable() -> None:
    body = service.run_agentic_query("What does the DataCo model predict for shipping?")
    assert body["query_plan"]["primary_intent"] == "HISTORICAL_SHIPPING"
    assert "No inference was run" in body["answer"]
    assert "current disruption" in body["answer"]
    assert body["historical_evidence"]["required_features"]


def test_dataco_feature_question_uses_audited_feature_list() -> None:
    body = service.run_agentic_query("What features does the historical DataCo model use?")
    assert "Days for shipment (scheduled)" in body["answer"]
    assert "Order City" in body["answer"]
    assert "Days for shipping (real)" in body["answer"]


def test_highest_warehouse_risk_uses_current_ranked_rows() -> None:
    body = service.run_agentic_query("What warehouses are currently at highest risk?")
    assert body["query_plan"]["primary_intent"] == "WAREHOUSE_RISK"
    assert body["tools_used"] == ["current_risk_ranking_tool"]
    assert all(entity.get("risk_score") is not None for entity in body["entities"])
    assert "probability" not in body["answer"].lower()


def test_cache_signature_changes_with_runtime_temporal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    current = {"event_status": "ACTIVE"}
    monkeypatch.setattr(agentic_tools, "load_current_events", lambda: {"events": [{"Event_ID": "E1", **current}]})
    active_signature = agentic_tools._retriever_signature()
    current["event_status"] = "EXPIRED"
    assert agentic_tools._retriever_signature() != active_signature


def test_llm_cannot_replace_authoritative_risk_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    class InventingProvider:
        def generate(self, *_args):
            return '{"answer":"Delhi Warehouse has spare capacity and 70% disruption probability."}'

    monkeypatch.setenv("SUPPLYBEACON_LLM_ENABLED", "true")
    monkeypatch.setattr(service, "OpenAICompatibleProvider", InventingProvider)
    body = service.run_agentic_query("Why is Delhi Warehouse risky today?")
    assert body["llm_used"] is False
    assert "spare capacity" not in body["answer"]
    assert "70%" not in body["answer"]


def test_current_warehouse_events_require_active_operational_affects_path(controlled_graph: nx.DiGraph) -> None:
    controlled_graph.add_node("E_ACTIVE", node_type="Event", name="Active flood", event_status="ACTIVE",
                              operational_current="true", operationally_relevant="true")
    controlled_graph.add_node("E_EXPIRED", node_type="Event", name="Expired flood", event_status="EXPIRED",
                              operational_current="false", operationally_relevant="true")
    controlled_graph.add_node("E_SIGNAL", node_type="Event", name="Light rain", event_status="ACTIVE",
                              operational_current="false", operationally_relevant="false")
    for event_id in ("E_ACTIVE", "E_EXPIRED", "E_SIGNAL"):
        controlled_graph.add_edge(event_id, "WAREHOUSE:Delhi_Warehouse", relationship="AFFECTS")
    body = service.run_agentic_query("Which events are currently affecting warehouses?")
    assert body["query_intent"] == "CURRENT_EVENTS"
    assert [event["node_id"] for event in body["events"]] == ["E_ACTIVE"]
    assert body["evidence"][0]["relation"] == "AFFECTS"
    assert body["evidence"][0]["temporal_status"] == "ACTIVE"


def test_system_provenance_and_stateless_followup() -> None:
    policy = service.run_agentic_query("Is this current risk a probability?")
    assert policy["query_intent"] == "SYSTEM_PROVENANCE"
    assert "not a calibrated disruption probability" in policy["answer"]
    followup = service.run_agentic_query("Show me the evidence behind this answer")
    assert "no verified prior-answer context" in followup["answer"]


def test_lookup_and_graph_structure_do_not_claim_operational_exposure(controlled_graph: nx.DiGraph) -> None:
    lookup = service.run_agentic_query("Tell me about WH_REAL_066")
    assert lookup["query_intent"] == "WAREHOUSE_LOOKUP"
    assert "Thiruvananthapuram, Kerala" in lookup["answer"]
    structure = service.run_agentic_query("Show graph structure for WH_REAL_066")
    assert structure["query_intent"] == "GRAPH_STRUCTURE"
    assert any(item["relation"] == "CATALOG_ASSOCIATED_WITH" for item in structure["evidence"])
    assert "do not establish operational exposure or inventory" in structure["answer"]


@pytest.mark.parametrize("query", [
    "Which warehouses are currently at highest risk?",
    "Which events are currently affecting the supply network?",
    "What does the DataCo model represent?",
])
def test_legacy_query_routes_phase6_questions_to_grounded_workflow(
    monkeypatch: pytest.MonkeyPatch, query: str,
) -> None:
    calls: list[str] = []

    def run_grounded(question: str) -> dict:
        calls.append(question)
        return {"query": question, "answer": "Grounded answer"}

    monkeypatch.setattr(api_module, "run_agentic_query", run_grounded)
    response = CLIENT.post("/query", json={"query": query})
    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer"
    assert calls == [query]
