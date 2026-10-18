from __future__ import annotations

from datetime import datetime, timedelta, timezone

import networkx as nx
import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import app
from src.dynamic_risk.live_situation import build_live_situation


def _state() -> dict:
    now = datetime.now(timezone.utc)
    graph = nx.DiGraph()
    graph.add_node("WAREHOUSE:Chennai", node_type="Warehouse", name="Chennai Warehouse")
    graph.add_node("LOCATION:Chennai", node_type="Location", name="Chennai")
    graph.add_edge("WAREHOUSE:Chennai", "LOCATION:Chennai", relationship="LOCATED_IN")
    event = {
        "Event_ID": 1,
        "Title": "Light Rain in Chennai",
        "Event_Type": "Light Rain",
        "Category": "Weather",
        "Location": "Chennai",
        "Date": now.isoformat(),
        "Severity": "Low",
        "Source": "current_weather.json",
        "source_channel": "OPENWEATHER",
        "Event_Class": "RISK_WARNING",
        "Operationally_Relevant": False,
        "Rule_Triggered": "Light rain is monitored but does not by itself establish disruption.",
        "event_start": (now - timedelta(minutes=5)).isoformat(),
        "event_end": (now + timedelta(hours=2)).isoformat(),
        "temporal_basis": "PROJECT OPERATIONAL VALIDITY WINDOW",
    }
    return {
        "graph": graph,
        "events": [event],
        "weather": {"cities": {
            "Chennai": {
                "weather": [{"main": "Rain", "description": "light rain"}],
                "main": {"temp": 29.0},
                "rain": {"1h": 0.3},
                "wind": {"speed": 3.2},
                "dt": int(now.timestamp()),
            },
            "Chennai:WEATHERAPI": {
                "weather": [{"main": "Rain", "description": "patchy rain nearby"}],
                "record_kind": "DAILY_FORECAST",
                "source_channel": "WEATHERAPI",
                "location_validation_status": "VALIDATED",
                "dt": int(now.timestamp()),
            },
        }},
        "metadata": {"pipeline_run_timestamp": now.isoformat()},
        "risk": pd.DataFrame([{"Risk_Level": "Low"}]),
    }


def test_active_non_operational_signal_is_visible_without_exposure() -> None:
    live = build_live_situation(_state(), {"overall_stale": False, "weather_fresh": True, "news_fresh": True})
    assert live["summary"]["active_signals"] == 1
    assert live["summary"]["operational_events"] == 0
    assert live["summary"]["graph_resolved_events"] == 0
    assert live["summary"]["affected_warehouses"] == 0
    signal = live["active_signals"][0]
    assert signal["operational_eligible"] is False
    assert signal["graph_resolution"]["resolved"] is False
    assert signal["risk_contribution"] == "None"


def test_weather_observation_and_forecast_remain_separate() -> None:
    live = build_live_situation(_state(), {"overall_stale": False, "weather_fresh": True, "news_fresh": True})
    city = live["weather"][0]
    assert city["observations"][0]["record_kind"] == "OBSERVATION"
    assert city["observations"][0]["temperature_c"] == 29.0
    assert city["forecasts"][0]["record_kind"] == "DAILY_FORECAST"
    assert city["forecasts"][0]["temperature_c"] is None


def test_freshness_and_dynamic_coverage_propagate() -> None:
    freshness = {"overall_stale": False, "weather_fresh": True, "news_fresh": True}
    live = build_live_situation(_state(), freshness)
    assert live["freshness"] == freshness
    assert live["coverage"] == {
        "required": 1,
        "represented": 1,
        "missing": [],
        "locations": ["Chennai"],
    }


def test_live_situation_endpoint_uses_current_artifacts() -> None:
    response = TestClient(app).get("/current/live-situation")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["active_signals"] >= payload["summary"]["operational_events"]
    assert payload["coverage"]["required"] == len(payload["coverage"]["locations"])
