from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from src.api.app import app
from src.dynamic_risk.current_state import load_current_state
from src.dynamic_risk.current_state import BASE_DIR


def test_publish_creates_manifest_and_runtime_overlay_without_current_graph_rewrite(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    generated = tmp_path / "generated"
    monkeypatch.setenv("SUPPLYBEACON_MASTER_GENERATED_DIR", str(generated))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    current_graph = BASE_DIR / "data" / "current" / "current_supply_chain_graph.graphml"
    current_hash = hashlib.sha256(current_graph.read_bytes()).hexdigest()
    location = client.post("/master/locations", json={"name": "Company City", "city": "Company City", "region": "State", "country": "India"}).json()
    warehouse = client.post("/master/warehouses", json={"name": "Company Warehouse", "warehouse_code": "CW-1", "location_id": location["id"]}).json()
    product = client.post("/master/products", json={"name": "Company Product", "sku": "CP-1", "brand": "Company Brand"}).json()
    relation = client.post("/master/relationships", json={"source_id": product["id"], "source_type": "Product", "relationship_type": "STORED_AT", "target_id": warehouse["id"], "target_type": "Warehouse"})
    assert relation.status_code == 200
    manifest = client.post("/master/publish")
    assert manifest.status_code == 200
    payload = manifest.json()
    overlay = generated / "company_overlay.graphml"
    assert overlay.exists()
    assert (generated / "publish_manifest.json").exists()
    assert payload["sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()
    assert json.loads((generated / "publish_manifest.json").read_text())["version"] == 1
    state = load_current_state()
    assert product["id"] in state["graph"]
    assert warehouse["id"] in state["graph"]
    assert state["graph"].has_edge(product["id"], warehouse["id"])
    assert hashlib.sha256(current_graph.read_bytes()).hexdigest() == current_hash
