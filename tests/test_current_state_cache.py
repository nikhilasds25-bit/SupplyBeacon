"""Long-lived query processes must see time rollover and published artifacts."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.agentic import tools
from src.dynamic_risk import current_state
from src.event_extraction import temporal_validity


@pytest.fixture
def cached_state(monkeypatch, tmp_path):
    current_dir = tmp_path / "data" / "current"
    current_dir.mkdir(parents=True)
    events_path = current_dir / "current_events.json"
    events_path.write_text(json.dumps([{
        "Event_ID": 1, "Operationally_Relevant": True,
        "event_start": "2026-09-22T03:00:00+00:00",
        "event_end": "2026-09-22T04:00:00+00:00",
    }]), encoding="utf-8")
    monkeypatch.setattr(tools, "BASE_DIR", tmp_path)
    monkeypatch.setattr(current_state, "CURRENT_DIR", current_dir)
    monkeypatch.setenv("SUPPLYBEACON_MASTER_GENERATED_DIR", str(tmp_path / "overlay"))
    monkeypatch.setattr(tools, "CurrentEvidenceRetriever", lambda: SimpleNamespace(
        events=current_state.load_current_events()["events"]))
    tools.invalidate_retriever_cache()
    yield tmp_path
    tools.invalidate_retriever_cache()


def test_cached_queries_roll_forward_without_file_changes(cached_state, monkeypatch):
    clock = [datetime(2026, 9, 22, 2, tzinfo=timezone.utc)]
    monkeypatch.setattr(temporal_validity, "utc_now", lambda: clock[0])
    upcoming = tools.current_retriever()
    assert upcoming.events[0]["event_status"] == "UPCOMING"
    assert tools.current_retriever() is upcoming
    clock[0] = datetime(2026, 9, 22, 3, 30, tzinfo=timezone.utc)
    active = tools.current_retriever()
    assert active is not upcoming
    assert active.events[0]["event_status"] == "ACTIVE"
    clock[0] = datetime(2026, 9, 22, 4, 0, 1, tzinfo=timezone.utc)
    expired = tools.current_retriever()
    assert expired is not active
    assert expired.events[0]["event_status"] == "EXPIRED"


def test_published_overlay_invalidates_cached_queries(cached_state):
    first = tools.current_retriever()
    overlay = cached_state / "overlay" / "company_overlay.graphml"
    overlay.parent.mkdir()
    overlay.write_text("published version", encoding="utf-8")
    assert tools.current_retriever() is not first


def test_refreshed_artifacts_invalidate_cached_queries(cached_state):
    first = tools.current_retriever()
    risk_path = cached_state / "data" / "current" / "current_risk_scores.csv"
    risk_path.write_text("new risk snapshot", encoding="utf-8")
    assert tools.current_retriever() is not first


def test_legacy_query_resolves_retriever_on_each_request(monkeypatch):
    from src.api import app as api
    version = ["before refresh"]
    def retrieve(_query):
        return {"entities": [], "graph_paths": [], "events": [], "risk_evidence": [],
                "limitations": [], "current_metadata": {"version": version[0]}}
    calls = []
    def resolve():
        calls.append(version[0])
        return SimpleNamespace(retrieve=retrieve)
    monkeypatch.setattr(api, "current_retriever", resolve)
    monkeypatch.setattr(api, "build_explanation", lambda _: {"answer": "grounded", "provenance_summary": []})
    client = TestClient(api.app)
    first = client.post("/query", json={"query": "current risks"})
    version[0] = "after refresh"
    second = client.post("/query", json={"query": "current risks"})
    assert first.json()["current_metadata"]["version"] == "before refresh"
    assert second.json()["current_metadata"]["version"] == "after refresh"
    assert calls == ["before refresh", "after refresh"]
