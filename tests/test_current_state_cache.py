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



def test_current_state_cache_reuses_state_and_invalidates_on_file_change(cached_state, monkeypatch):
    current_dir = cached_state / "data" / "current"
    calls = []

    def build_state():
        calls.append(len(calls) + 1)
        return {"version": len(calls)}

    monkeypatch.setattr(current_state, "_load_current_state_uncached", build_state)

    first = current_state.load_current_state()
    assert current_state.load_current_state() is first
    assert calls == [1]

    # A newly published risk artifact changes the cache signature.
    (current_dir / "current_risk_scores.csv").write_text(
        "Node_ID,Risk_Score\nWH_TEST,0.25\n", encoding="utf-8"
    )
    second = current_state.load_current_state()
    assert second is not first
    assert second == {"version": 2}
    assert current_state.load_current_state() is second
    assert calls == [1, 2]


def test_current_state_cache_invalidates_when_event_temporal_status_changes(cached_state, monkeypatch):
    statuses = ["UPCOMING"]
    calls = []

    monkeypatch.setattr(
        current_state,
        "apply_runtime_temporal_status",
        lambda event: {**event, "event_status": statuses[0]},
    )
    monkeypatch.setattr(
        current_state,
        "_load_current_state_uncached",
        lambda: {"version": len(calls) + 1} if not calls.append("load") else {"version": len(calls)},
    )

    upcoming = current_state.load_current_state()
    assert current_state.load_current_state() is upcoming
    assert len(calls) == 1

    statuses[0] = "ACTIVE"
    active = current_state.load_current_state()
    assert active is not upcoming
    assert len(calls) == 2
