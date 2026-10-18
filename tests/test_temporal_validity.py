import json
from datetime import datetime, timezone

import networkx as nx
import pandas as pd

from src.dynamic_risk import current_state
from src.event_extraction.temporal_validity import (
    extract_news_validity,
    temporal_status,
)


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_explicit_past_date_is_expired() -> None:
    fields = extract_news_validity("Forecast for September 19, 2026", "", "2026-09-18T10:00:00+00:00")
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-20T12:00:00")) == "EXPIRED"


def test_same_day_event_is_active() -> None:
    fields = extract_news_validity("Forecast for September 19, 2026", "", "2026-09-18T10:00:00+00:00")
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-19T12:00:00")) == "ACTIVE"


def test_future_date_is_upcoming() -> None:
    fields = extract_news_validity("Forecast for September 21, 2026", "", "2026-09-20T10:00:00+00:00")
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-20T12:00:00")) == "UPCOMING"


def test_range_is_active_then_expired() -> None:
    fields = extract_news_validity("Rain expected September 20-22, 2026", "", "2026-09-19T10:00:00+00:00")
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-20T12:00:00")) == "ACTIVE"
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-23T12:00:00")) == "EXPIRED"


def test_missing_validity_date_is_unknown() -> None:
    fields = extract_news_validity("Heavy rain warning", "", "2026-09-20T10:00:00+00:00")
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-20T12:00:00")) == "UNKNOWN"


def test_tomorrow_uses_publication_date() -> None:
    fields = extract_news_validity(
        "Weather tomorrow, September 19",
        "",
        "2026-09-18T10:00:00+00:00",
    )
    assert fields["event_start"].startswith("2026-09-19")
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-20T12:00:00")) == "EXPIRED"


def test_source_freshness_does_not_change_expired_status() -> None:
    fields = extract_news_validity("Forecast for September 19, 2026", "", "2026-09-18T10:00:00+00:00")
    assert temporal_status(fields["event_start"], fields["event_end"], instant("2026-09-20T12:00:00")) == "EXPIRED"


def test_rollover_without_refresh_removes_current_risk_and_graph_exposure(monkeypatch, tmp_path) -> None:
    current_dir = tmp_path / "current"
    baseline_dir = tmp_path / "data" / "risk_prediction"
    current_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    event = {
        "Event_ID": 1,
        "Date": "2026-09-19T10:00:00+00:00",
        "Event_Type": "Heavy Rain",
        "Title": "Delhi rain forecast",
        "Description": "Heavy rain in Delhi",
        "Location": "Delhi",
        "Category": "Weather",
        "Severity": "High",
        "Source": "current_news.json",
        "Confidence": 0.9,
        "Operationally_Relevant": True,
        "event_start": "2026-09-19T00:00:00+00:00",
        "event_end": "2026-09-19T23:59:59+00:00",
        "Risk_Score": 0.8,
        "Impact_Level": "High",
    }
    (current_dir / "current_events.json").write_text(json.dumps([event]), encoding="utf-8")
    (current_dir / "current_run_metadata.json").write_text(json.dumps({}), encoding="utf-8")
    pd.DataFrame([{
        "Node_ID": "PRODUCT:1", "Node_Type": "Product", "Node_Name": "Test Product",
        "Risk_Score": "0.9", "Risk_Level": "High", "Connected_Events": "1",
        "Event_Exposure": "0.05", "Max_Event_Severity": "High",
        "Average_Event_Confidence": "0.9", "Business_Impact_Risk": "0.15",
    }]).to_csv(current_dir / "current_risk_scores.csv", index=False)
    pd.DataFrame(columns=["Event_ID"]).to_csv(current_dir / "current_business_impacts.csv", index=False)
    pd.DataFrame([{
        "Node_ID": "PRODUCT:1", "Node_Type": "Product", "Node_Name": "Test Product",
        "Risk_Score": "0.9", "Risk_Level": "High", "Connected_Events": "1",
        "Event_Exposure": "0.05", "Max_Event_Severity": "High",
        "Average_Event_Confidence": "0.9", "Business_Impact_Risk": "0.15",
    }]).to_csv(baseline_dir / "risk_scores.csv", index=False)

    base_graph = nx.DiGraph()
    base_graph.add_node("PRODUCT:1", node_type="Product", name="Test Product")
    base_graph.add_node("WAREHOUSE:Delhi", node_type="Warehouse", name="Delhi Warehouse")
    base_graph.add_node("LOCATION:Delhi", node_type="Location", name="Delhi")
    base_graph.add_edge("PRODUCT:1", "WAREHOUSE:Delhi", relationship="STORED_AT")
    base_graph.add_edge("WAREHOUSE:Delhi", "LOCATION:Delhi", relationship="LOCATED_IN")
    base_graph_path = tmp_path / "base.graphml"
    nx.write_graphml(base_graph, base_graph_path)
    graph_dir = tmp_path / "data" / "graph"
    graph_dir.mkdir(parents=True)
    pd.DataFrame([
        {"Node_ID": "PRODUCT:1", "Node_Type": "Product", "Node_Name": "Test Product"},
        {"Node_ID": "WAREHOUSE:Delhi", "Node_Type": "Warehouse", "Node_Name": "Delhi Warehouse"},
        {"Node_ID": "LOCATION:Delhi", "Node_Type": "Location", "Node_Name": "Delhi"},
    ]).to_csv(graph_dir / "nodes.csv", index=False)
    pd.DataFrame([
        {"Source": "PRODUCT:1", "Target": "WAREHOUSE:Delhi", "Relationship": "STORED_AT"},
        {"Source": "WAREHOUSE:Delhi", "Target": "LOCATION:Delhi", "Relationship": "LOCATED_IN"},
    ]).to_csv(graph_dir / "edges.csv", index=False)

    persisted_graph = base_graph.copy()
    persisted_graph.add_node(
        "CURRENT_EVENT:1", node_type="Event", name="Delhi rain forecast",
        event_status="ACTIVE", operational_current="true", event_id="1",
        event_type="Heavy Rain", event_domain="WEATHER", risk_score="0.8",
        impact_level="High",
    )
    persisted_graph.add_edge("CURRENT_EVENT:1", "WAREHOUSE:Delhi", relationship="AFFECTS")
    nx.write_graphml(persisted_graph, current_dir / "current_supply_chain_graph.graphml")

    monkeypatch.setattr(current_state, "CURRENT_DIR", current_dir)
    monkeypatch.setattr(current_state, "BASE_DIR", tmp_path)
    from src.dynamic_risk import refresh_current_state
    monkeypatch.setattr(refresh_current_state, "CURRENT_DIR", current_dir)
    monkeypatch.setattr(refresh_current_state, "BASE_DIR", tmp_path)
    monkeypatch.setattr(refresh_current_state, "BASE_GRAPH", base_graph_path)

    from src.event_extraction import temporal_validity
    monkeypatch.setattr(temporal_validity, "utc_now", lambda: instant("2026-09-19T18:00:00"))
    at_t1 = current_state.load_current_state()
    assert at_t1["events"][0]["event_status"] == "ACTIVE"
    assert at_t1["risk"].iloc[0]["Connected_Events"] == "1"

    monkeypatch.setattr(temporal_validity, "utc_now", lambda: instant("2026-09-20T10:00:00"))
    at_t2 = current_state.load_current_state()
    assert at_t2["events"][0]["event_status"] == "EXPIRED"
    assert not [event for event in at_t2["events"] if event["Operationally_Relevant"] and event["event_status"] == "ACTIVE"]
    assert not any(attrs.get("relationship") == "AFFECTS" for _, _, attrs in at_t2["graph"].edges(data=True))
    assert at_t2["risk"].iloc[0]["Connected_Events"] == "0"
    assert "No current valid event exposure" in at_t2["risk"].iloc[0]["Risk_Reason"]

    from src.api import app
    monkeypatch.setattr(app, "load_current_state", lambda: at_t2)
    assert app.current_events(operational_only=True) == []

    from src.graphrag import current_retriever
    monkeypatch.setattr(current_retriever, "CURRENT_DIR", current_dir)
    monkeypatch.setattr(current_retriever, "BASE_DIR", tmp_path)
    retrieved = current_retriever.CurrentEvidenceRetriever().retrieve(
        "Why is Delhi Warehouse risky today?"
    )
    assert retrieved["events"] == []
