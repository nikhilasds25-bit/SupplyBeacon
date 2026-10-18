from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.graphrag.current_retriever import CurrentEvidenceRetriever
from src.master_data import graph_overlay


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    return TestClient(app)


def _publish_fixture(client: TestClient):
    location = client.post("/master/locations", json={"name": "Kochi", "city": "Kochi", "region": "Kerala", "country": "India"}).json()
    warehouse = client.post("/master/warehouses", json={"name": "Kochi Company Warehouse", "location_id": location["id"]}).json()
    product = client.post("/master/products", json={"name": "SupplyBeacon Demo Laptop", "brand": "HP"}).json()
    supplier = client.post("/master/suppliers", json={"name": "Demo Electronics Supplier"}).json()
    assert client.post("/master/relationships", json={"source_id": product["id"], "source_type": "Product", "relationship_type": "STORED_AT", "target_id": warehouse["id"], "target_type": "Warehouse"}).status_code == 200
    assert client.post("/master/relationships", json={"source_id": supplier["id"], "source_type": "Supplier", "relationship_type": "SUPPLIES", "target_id": "BRAND:HP", "target_type": "Brand"}).status_code == 200
    assert client.post("/master/publish").status_code == 200
    return location, warehouse, product, supplier


def test_failed_publish_preserves_overlay_manifest_and_metadata(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _publish_fixture(client)
    generated = tmp_path / "generated"
    overlay = generated / "company_overlay.graphml"
    manifest = generated / "publish_manifest.json"
    before_overlay = hashlib.sha256(overlay.read_bytes()).hexdigest()
    before_manifest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    before_meta = client.get("/master/status").json()
    client.post("/master/products", json={"name": "Second Product", "brand": "HP"})

    def fail_replace(*_args, **_kwargs):
        raise OSError("forced promotion failure")

    monkeypatch.setattr(graph_overlay.os, "replace", fail_replace)
    response = TestClient(app, raise_server_exceptions=False).post("/master/publish")
    assert response.status_code == 500
    assert hashlib.sha256(overlay.read_bytes()).hexdigest() == before_overlay
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before_manifest
    after_meta = client.get("/master/status").json()
    assert after_meta["published_version"] == before_meta["published_version"]
    assert after_meta["published_at"] == before_meta["published_at"]


def test_warehouse_location_selector_semantics_are_single_and_replaced(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    first = client.post("/master/locations", json={"name": "Kochi", "city": "Kochi"}).json()
    second = client.post("/master/locations", json={"name": "Chennai", "city": "Chennai"}).json()
    warehouse = client.post("/master/warehouses", json={"name": "Central", "location_id": first["id"]}).json()
    rows = client.get("/master/relationships").json()
    assert len([row for row in rows if row["source_id"] == warehouse["id"] and row["relationship_type"] == "LOCATED_IN" and row["status"] == "ACTIVE"]) == 1
    updated = client.put(f"/master/warehouses/{warehouse['id']}", json={"location_id": second["id"]})
    assert updated.status_code == 200
    rows = client.get("/master/relationships").json()
    active = [row for row in rows if row["source_id"] == warehouse["id"] and row["relationship_type"] == "LOCATED_IN" and row["status"] == "ACTIVE"]
    assert len(active) == 1
    assert active[0]["target_id"] == second["id"]
    cleared = client.put(f"/master/warehouses/{warehouse['id']}", json={"location_id": None})
    assert cleared.status_code == 200
    assert not [row for row in client.get("/master/relationships").json() if row["source_id"] == warehouse["id"] and row["relationship_type"] == "LOCATED_IN" and row["status"] == "ACTIVE"]
    assert any(row["action"] == "RELATIONSHIP_DEACTIVATE" for row in client.get("/master/change-log").json())


def test_public_runtime_scenario_and_graphrag_see_published_company_graph(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    location, warehouse, product, supplier = _publish_fixture(client)
    products = client.get("/current/products").json()
    warehouses = client.get("/current/warehouses").json()
    suppliers = client.get("/current/suppliers").json()
    assert any(row["node_id"] == product["id"] for row in products)
    assert any(row["node_id"] == warehouse["id"] for row in warehouses)
    assert any(row["node_id"] == supplier["id"] for row in suppliers)
    company_product = next(row for row in products if row["node_id"] == product["id"])
    assert company_product["risk_level"] == "Unavailable"
    assert company_product["risk_score"] is None
    scenario = client.post("/scenario/simulate", json={"scenario_type": "WAREHOUSE_UNAVAILABLE", "target_node_id": warehouse["id"], "duration_hours": 24, "scenario_severity": "High"})
    assert scenario.status_code == 200
    assert product["id"] in scenario.json()["product_ids"]
    assert location["id"] in scenario.json()["location_ids"]
    retriever = CurrentEvidenceRetriever()
    assert product["id"] in retriever.match_entities("SupplyBeacon Demo Laptop", limit=10)
    assert client.get("/master/brands").json()[0]["node_type"] == "Brand"
