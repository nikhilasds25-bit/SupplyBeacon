"""Exercise only Docker COPY inputs in an isolated process, without providers."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def test_docker_copy_runtime_and_persistent_brand_registry(tmp_path):
    # Derive the fixture from Dockerfile so new missing-file assumptions fail here.
    with tempfile.TemporaryDirectory(prefix="image-runtime-", dir=tmp_path) as directory:
        image = Path(directory)
        for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
            if not line.startswith("COPY "):
                continue
            _, source, destination = shlex.split(line)
            source_path = ROOT / source
            target = image / destination
            if source_path.is_dir():
                shutil.copytree(source_path, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                if destination == ".":
                    target = image / source_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
        assert not (image / ".env").exists()
        assert not (image / "data" / "current").exists()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "SUPPLYBEACON_CURRENT_DATA_DIR": str(image / "persistent" / "current"),
            "SUPPLYBEACON_MASTER_DB_PATH": str(image / "persistent" / "master.db"),
            "SUPPLYBEACON_MASTER_GENERATED_DIR": str(image / "persistent" / "generated"),
            "SUPPLYBEACON_MASTER_WRITE_ENABLED": "false",
            "SUPPLYBEACON_LLM_ENABLED": "false",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "SUPPLYBEACON_ALLOWED_ORIGINS": "https://supply-beacon.vercel.app",
        })
        for name in ("SUPPLYBEACON_ADMIN_API_KEY", "NEWS_API_KEY", "NEWSDATA_API_KEY", "CURRENTS_API_KEY",
                     "OPENWEATHER_API_KEY", "WEATHERAPI_KEY", "ELECTROGRAPH_LLM_API_KEY", "LANGSMITH_API_KEY"):
            env.pop(name, None)
        audit = subprocess.run([sys.executable, "-B", "deploy/audit_seed.py"], cwd=image, env=env,
                               capture_output=True, text=True, timeout=30)
        assert audit.returncode == 0, audit.stdout + audit.stderr
        result = subprocess.run(
            [sys.executable, "-B", "-c", RUNTIME_CHECK], cwd=image, env=env,
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "CLEAN_RUNTIME_PASS" in result.stdout
    assert not image.exists()


RUNTIME_CHECK = r'''
import socket
def blocked(*args, **kwargs):
    raise AssertionError("External network access forbidden in clean-runtime test")
original_connect = socket.socket.connect
def local_connect(self, address):
    # Windows asyncio uses a loopback socket pair internally.
    if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
        return original_connect(self, address)
    return blocked()
socket.socket.connect = local_connect
import requests
requests.sessions.Session.request = blocked
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import app
from src.dynamic_risk import current_state
from src.graphrag.current_retriever import CurrentEvidenceRetriever

assert current_state.BASE_DIR == Path.cwd()
assert current_state.CURRENT_DIR == Path.cwd() / "persistent" / "current"
with TestClient(app) as client:
    for path in ["/health", "/current/summary", "/current/warehouses/WH_REAL_066",
                 "/current/graph/entity/WH_REAL_066", "/historical/model-metrics",
                 "/current/events", "/current/weather", "/current/metadata",
                 "/current/products", "/current/suppliers", "/current/warehouses",
                 "/current/risk-watchlist", "/current/live-situation", "/master/status"]:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
    snapshot = client.get("/current/snapshot").json()
    assert snapshot["summary"]["graph_nodes"] > 0
    assert "summary" in snapshot["liveSituation"]
    assert isinstance(snapshot["events"], list)
    assert isinstance(snapshot["products"], list)
    assert isinstance(snapshot["warehouses"], list)
    assert isinstance(snapshot["suppliers"], list)
    warehouse_066 = next(row for row in snapshot["warehouses"] if row["node_id"] == "WH_REAL_066")
    assert str(warehouse_066["catalog_only"]).lower() == "true"
    assert warehouse_066["inventory_relationship_available"] is False
    assert client.get("/master/status").json()["write_enabled"] is False
    assert client.post("/master/products", json={"name": "Blocked"}).status_code == 403
    assert client.post("/master/publish").status_code == 403
    assert client.post("/master/import/preview?import_type=Products",
                       files={"file": ("products.csv", b"name\nBlocked\n", "text/csv")}).status_code == 403
    assert client.post("/current/refresh").status_code == 401
    allowed = client.options("/query", headers={"Origin": "https://supply-beacon.vercel.app",
                                              "Access-Control-Request-Method": "POST"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://supply-beacon.vercel.app"
    denied = client.options("/query", headers={"Origin": "https://untrusted.example",
                                             "Access-Control-Request-Method": "POST"})
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers
    metrics = client.get("/historical/model-metrics").json()
    assert metrics["model"] == "hist_gradient_boosting"
    # A fresh process gets CURRENT_DIR from the environment; no default graph exists.
    brands = client.get("/master/brands").json()
    assert any(row["id"] == "BRAND:HP" and row["source"] == "BASE" for row in brands)
    assert not (Path.cwd() / "data" / "current").exists()
    retriever = CurrentEvidenceRetriever()
    assert not retriever.nodes.empty and not retriever.edges.empty
    assert not retriever.embeddings.empty
    assert "WH_REAL_066" in retriever.graph
    evidence = retriever.retrieve("WH_REAL_066 associated catalog product models")
    assert evidence["entities"], evidence
    # Use a Product query to exercise ordinary retrieval rather than only domain shortcuts.
    product_id = next(node for node, attrs in retriever.graph.nodes(data=True)
                      if attrs.get("node_type") == "Product")
    for path in ["/query", "/query/agentic"]:
        response = client.post(path, json={"query": "What is the risk for " + product_id + "?"})
        assert response.status_code == 200, (path, response.text)
        payload = response.json()
        assert payload["answer"], payload
        assert payload.get("entities"), payload
    scenario = client.post("/scenario/simulate", json={
        "scenario_type": "WAREHOUSE_UNAVAILABLE", "target_node_id": "WH_REAL_066",
        "duration_hours": 48, "scenario_severity": "High",
    })
    assert scenario.status_code == 200, scenario.text
    assert scenario.json()["affected_product_count"] == 0
    assert scenario.json()["catalog_association_count"] == 10
print("CLEAN_RUNTIME_PASS: imports health summary entity query agentic GraphRAG scenario metrics persistent-brands")
'''
