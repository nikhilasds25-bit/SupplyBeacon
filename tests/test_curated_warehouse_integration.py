from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.api import app as api_module
from src.data_integration.curated_warehouse_loader import load_curated_warehouse_data
from src.dynamic_risk.refresh_current_state import _dynamic_location_aliases, _event_locations
from src.event_extraction.event_extractor import article_to_events
from src.external_data.weather_collector import warehouse_weather_locations
from src.graph.graph_builder import build_graph
from src.graphrag.current_retriever import CurrentEvidenceRetriever


@pytest.fixture(scope="module")
def curated():
    return load_curated_warehouse_data()


@pytest.fixture(scope="module")
def graph():
    return build_graph()


def relationship_count(graph: nx.Graph, relationship: str, curated_only: bool = False) -> int:
    return sum(
        1 for source, target, attrs in graph.edges(data=True)
        if attrs.get("relationship") == relationship
        and (not curated_only or str(source).startswith("WH_REAL_") or str(target).startswith("WH_REAL_"))
    )


def test_loader_reads_expected_rows_and_warehouses(curated) -> None:
    assert len(curated.associations) == 1298
    assert len(curated.warehouses) == 66
    assert curated.conflicts == ()
    assert set(curated.associations["Warehouse_Stock_Verified"]) == {"NO"}


def test_curated_nodes_locations_and_safety(graph) -> None:
    warehouses = {node_id: attrs for node_id, attrs in graph.nodes(data=True) if attrs.get("node_type") == "Warehouse"}
    curated = {node_id: attrs for node_id, attrs in warehouses.items() if str(node_id).startswith("WH_REAL_")}
    assert len(warehouses) == 74
    assert len(curated) == 66
    assert curated["WH_REAL_066"]["city"] == "Thiruvananthapuram"
    assert curated["WH_REAL_066"]["synthetic"] is False
    assert all(attrs["catalog_only"] is True for attrs in curated.values())
    assert all(attrs["inventory_relationship_available"] is False for attrs in curated.values())
    assert relationship_count(graph, "LOCATED_IN", curated_only=True) == 66
    assert relationship_count(graph, "ASSOCIATED_WITH_BRAND", curated_only=True) == 66
    assert relationship_count(graph, "CATALOG_ASSOCIATED_WITH", curated_only=True) == 1298
    assert relationship_count(graph, "STORED_AT", curated_only=True) == 0
    cities = {attrs.get("city") for _, attrs in graph.nodes(data=True) if attrs.get("node_type") == "Location" and attrs.get("city")}
    assert len(cities.intersection(set(load_curated_warehouse_data().warehouses["City"]))) == 51


def test_product_models_are_catalog_only(graph) -> None:
    models = [(node_id, attrs) for node_id, attrs in graph.nodes(data=True) if attrs.get("node_type") == "ProductModel"]
    assert len(models) == 482
    assert all(attrs["warehouse_stock_verified"] is False for _, attrs in models)
    assert relationship_count(graph, "OFFERS") == 482


def test_weather_locations_are_dynamic_and_deduplicated(graph) -> None:
    cities = warehouse_weather_locations(nx.DiGraph(graph))
    assert len(cities) == 55
    assert len(cities) == len({city.casefold() for city in cities})
    assert "Bhiwandi" in cities
    assert "Thiruvananthapuram" in cities


def test_dynamic_news_locations_and_located_in_resolution(graph) -> None:
    directed = nx.DiGraph(graph)
    aliases = _dynamic_location_aliases(directed)
    bhiwandi = article_to_events({
        "title": "Flooding in Bhiwandi disrupts warehouse logistics",
        "description": "Electronics supply chain disruption reported in Bhiwandi.",
        "publishedAt": "2026-10-09T10:00:00Z",
    }, api_module.BASE_DIR / "data" / "current" / "current_news.json", aliases)
    assert bhiwandi and bhiwandi[0]["Location"] == "Bhiwandi"
    trivandrum = article_to_events({
        "title": "Flooding in Thiruvananthapuram disrupts logistics",
        "description": "Warehouse transport disruption reported.",
        "publishedAt": "2026-10-09T10:00:00Z",
    }, api_module.BASE_DIR / "data" / "current" / "current_news.json", aliases)
    assert trivandrum and trivandrum[0]["Location"] == "Thiruvananthapuram"
    directed.add_node("TEST_EVENT", node_type="Event", name="Flooding in Bhiwandi")
    now = datetime.now(timezone.utc)
    _event_locations(directed, "TEST_EVENT", {
        **bhiwandi[0], "event_status": "ACTIVE", "Operationally_Relevant": True,
        "event_start": (now - timedelta(hours=1)).isoformat(),
        "event_end": (now + timedelta(hours=1)).isoformat(),
    })
    affected = [target for _, target, attrs in directed.out_edges("TEST_EVENT", data=True) if attrs.get("relationship") == "AFFECTS"]
    assert "WH_REAL_004" in affected
    assert directed["TEST_EVENT"]["WH_REAL_004"]["resolution_provenance"] == "DERIVED_FROM_LOCATED_IN"

    directed.add_node("STATE_EVENT", node_type="Event", name="Flooding in Kerala")
    _event_locations(directed, "STATE_EVENT", {
        **bhiwandi[0], "Location": "Kerala", "event_status": "ACTIVE",
        "Location_Scope": "STATEWIDE",
        "Operationally_Relevant": True,
        "event_start": (now - timedelta(hours=1)).isoformat(),
        "event_end": (now + timedelta(hours=1)).isoformat(),
    })
    state_warehouses = {
        target for _, target, attrs in directed.out_edges("STATE_EVENT", data=True)
        if attrs.get("relationship") == "AFFECTS" and str(target).startswith("WH_REAL_")
    }
    assert state_warehouses == {"WH_REAL_017", "WH_REAL_036", "WH_REAL_060", "WH_REAL_066"}
    assert all(directed["STATE_EVENT"][node_id]["exposure_scope"] == "STATE_LOCATION" for node_id in state_warehouses)


def _state(graph: nx.Graph) -> dict:
    columns = ["Node_ID", "Node_Type", "Node_Name", "Risk_Score", "Risk_Level", "Connected_Events", "Risk_Reason"]
    return {"graph": nx.DiGraph(graph), "risk": pd.DataFrame(columns=columns), "events": [], "metadata": {}, "weather": {}, "news": {}, "business_impacts": pd.DataFrame(), "available": True}


def test_api_and_scenario_expose_catalog_warehouse_without_inventory(graph, monkeypatch) -> None:
    state = _state(graph)
    monkeypatch.setattr(api_module, "load_current_state", lambda: state)
    warehouses = api_module.current_warehouses()
    assert len(warehouses) == 74
    real = next(item for item in warehouses if item["node_id"] == "WH_REAL_001")
    assert real["risk_score"] is None
    assert real["risk_level"] == "Unavailable"
    assert real["catalog_only"] is True
    assert real["inventory_relationship_available"] is False
    catalog = api_module.current_warehouse_catalog("WH_REAL_001")
    assert catalog and all(item["inventory_verified"] is False for item in catalog)
    scenario = api_module.simulate_scenario(api_module.ScenarioRequest(
        scenario_type="WAREHOUSE_UNAVAILABLE", target_node_id="WH_REAL_001",
        duration_hours=48, scenario_severity="Medium",
    ))
    assert scenario.product_ids == []
    assert scenario.scenario_state["exposed_products"] == 0
    assert "No verified operational product inventory relationships are available for this warehouse." in scenario.assumptions


def test_graphrag_never_claims_catalog_stock(graph) -> None:
    retriever = CurrentEvidenceRetriever.__new__(CurrentEvidenceRetriever)
    retriever.graph = nx.DiGraph(graph)
    retriever.current_metadata = {}
    result = retriever._catalog_query("Is Galaxy S26 physically stocked in WH_REAL_001?")
    assert result is not None
    assert result["answer"] == "The dataset provides a catalog association only. Physical warehouse stock is not verified."


def test_graphrag_resolves_curated_location_brand_and_model_queries(graph) -> None:
    retriever = CurrentEvidenceRetriever.__new__(CurrentEvidenceRetriever)
    retriever.graph = nx.DiGraph(graph)
    retriever.current_metadata = {}

    kerala = retriever._catalog_query("What monitored warehouses are in Kerala?")
    assert kerala is not None
    assert len(kerala["entities"]) == 4

    brand = retriever._catalog_query("Which brands are associated with the Noida Samsung warehouse?")
    assert brand is not None
    assert brand["entities"][0]["node_id"] == "WH_REAL_001"
    assert "Samsung" in brand["entities"][0]["associated_brands"]

    models = retriever._catalog_query("Which product models are catalog-associated with WH_REAL_001?")
    assert models is not None
    assert any(entity["name"] == "Galaxy S26" for entity in models["entities"])
