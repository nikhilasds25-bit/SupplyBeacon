from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app


def test_master_reads_and_write_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    client = TestClient(app)
    assert client.get("/master/status").status_code == 200
    assert client.get("/master/products").status_code == 200
    response = client.post("/master/products", json={"name": "Blocked Product"})
    assert response.status_code == 403


def test_master_crud_duplicate_deactivate_and_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    payload = {"name": "Company Product", "sku": "SKU-1", "brand": "HP", "category": "Electronics"}
    created = client.post("/master/products", json=payload)
    assert created.status_code == 200
    duplicate = client.post("/master/products", json=payload)
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "POSSIBLE DUPLICATE"
    product_id = created.json()["id"]
    updated = client.put(f"/master/products/{product_id}", json={**payload, "name": "Updated Product"})
    assert updated.status_code == 200
    deactivated = client.post(f"/master/products/{product_id}/deactivate")
    assert deactivated.status_code == 200
    assert deactivated.json()["status"] == "INACTIVE"
    log = client.get("/master/change-log").json()
    assert [entry["action"] for entry in log[:3]] == ["DEACTIVATE", "UPDATE", "CREATE"]


def test_relationship_type_and_endpoint_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    response = client.post("/master/relationships", json={
        "source_id": "PRODUCT:missing", "source_type": "Product",
        "relationship_type": "SUPPLIES", "target_id": "SUPPLIER:missing", "target_type": "Supplier",
    })
    assert response.status_code == 422


def test_update_preserves_omitted_fields_and_relationship_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    location = client.post("/master/locations", json={"name": "Kochi", "city": "Kochi", "region": "Kerala", "country": "India"}).json()
    warehouse = client.post("/master/warehouses", json={"name": "Kochi Warehouse", "warehouse_code": "KW01", "location_id": location["id"], "capacity": 25000, "capacity_unit": "sqft"}).json()
    product = client.post("/master/products", json={"name": "Laptop", "sku": "L1", "brand": "HP"}).json()
    updated = client.put(f"/master/warehouses/{warehouse['id']}", json={"name": "Kochi Central Warehouse"})
    assert updated.status_code == 200
    assert updated.json()["warehouse_code"] == "KW01"
    assert updated.json()["location_id"] == location["id"]
    assert updated.json()["capacity"] == 25000
    relationship = client.post("/master/relationships", json={"source_id": product["id"], "source_type": "Product", "relationship_type": "STORED_AT", "target_id": warehouse["id"], "target_type": "Warehouse"})
    assert relationship.status_code == 200
    duplicate = client.post("/master/relationships", json=relationship.json())
    assert duplicate.status_code == 409
    deactivated = client.post(f"/master/relationships/{relationship.json()['id']}/deactivate")
    assert deactivated.status_code == 200
    assert deactivated.json()["status"] == "INACTIVE"


def test_duplicate_override_is_explicit_and_audited(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    payload = {"name": "Duplicate Candidate", "sku": "DUP-1", "brand": "HP"}
    assert client.post("/master/products", json=payload).status_code == 200
    duplicate = client.post("/master/products", json=payload)
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate_reasons"]
    overridden = client.post("/master/products", json={**payload, "continue_on_duplicate": True})
    assert overridden.status_code == 200
    assert any(row["action"] == "DUPLICATE_OVERRIDE" for row in client.get("/master/change-log").json())


def test_brand_registry_contract_and_base_brand_relationship(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    brands = client.get("/master/brands")
    assert brands.status_code == 200
    hp = next(item for item in brands.json() if item["id"] == "BRAND:HP")
    assert hp == {"id": "BRAND:HP", "name": "HP", "node_type": "Brand", "source": "BASE"}
    supplier = client.post("/master/suppliers", json={"name": "HP Supplier"}).json()
    relationship = client.post("/master/relationships", json={"source_id": supplier["id"], "source_type": "Supplier", "relationship_type": "SUPPLIES", "target_id": "BRAND:HP", "target_type": "Brand"})
    assert relationship.status_code == 200
