from __future__ import annotations

import io
import sqlite3

from fastapi.testclient import TestClient

from src.api.app import app


def test_csv_preview_does_not_mutate_and_confirm_creates_draft(monkeypatch, tmp_path):
    database = tmp_path / "master.db"
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(database))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    csv_data = b"name,sku,brand,category\nImported Product,IMP-1,HP,Electronics\n,BAD,,\n"
    preview = client.post("/master/import/preview?import_type=Products", files={"file": ("products.csv", io.BytesIO(csv_data), "text/csv")})
    assert preview.status_code == 200
    body = preview.json()
    assert body["total_rows"] == 2
    assert body["valid_rows"] == 1
    assert client.get("/master/products").json() == []
    confirmed = client.post(f"/master/import/{body['batch_id']}/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["draft"] is True
    assert len(client.get("/master/products").json()) == 1
    assert client.get("/master/status").json()["published_version"] == 0


def test_invalid_csv_confirmation_rolls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    preview = client.post("/master/import/preview?import_type=Products", files={"file": ("products.csv", io.BytesIO(b"name,sku\nFirst,1\nSecond,2\n"), "text/csv")})
    assert preview.status_code == 200
    import sqlite3
    connection = sqlite3.connect(tmp_path / "master.db")
    connection.execute("UPDATE import_batches SET payload=?", ('[{"name":"First","sku":"1"},{"bad":"row"}]',))
    connection.commit()
    connection.close()
    response = client.post(f"/master/import/{preview.json()['batch_id']}/confirm")
    assert response.status_code == 422
    assert client.get("/master/products").json() == []


def test_import_counts_are_disjoint_and_duplicate_warning_is_visible(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(tmp_path / "master.db"))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "true")
    client = TestClient(app)
    assert client.post("/master/products", json={"name": "Existing", "sku": "EX-1"}).status_code == 200
    csv_data = b"name,sku\nExisting,EX-1\nNew,NEW-1\n,BAD\n"
    response = client.post("/master/import/preview?import_type=Products", files={"file": ("products.csv", io.BytesIO(csv_data), "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == body["valid_rows"] + body["warning_rows"] + body["invalid_rows"]
    assert body["warning_rows"] == 1
    assert any(row["status"] == "WARNING" for row in body["rows"])
