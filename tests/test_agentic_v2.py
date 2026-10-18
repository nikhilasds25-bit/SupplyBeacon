"""Behavioral regression tests for the grounded LangGraph V2 workflow."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.agentic import service
from src.agentic import tools as agentic_tools
from src.agentic import weather as agentic_weather
from src.agentic.service import EvidenceBundle, EvidenceItem, run_agentic_query
from src.event_extraction import temporal_validity


@pytest.fixture
def fixed_utc_now(monkeypatch) -> datetime:
    fixed_now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(temporal_validity, "utc_now", lambda: fixed_now)
    return fixed_now


@pytest.fixture
def controlled_events(fixed_utc_now: datetime) -> dict[str, object]:
    active_non_operational = [
        {
            "Event_ID": 900 + index,
            "Title": f"Active light rain signal in {city}",
            "Event_Type": "Light Rain",
            "Location": city,
            "Source": "Weather API",
            "source_channel": "OPENWEATHER",
            "Operationally_Relevant": False,
            "event_start": (fixed_utc_now - timedelta(hours=1)).isoformat(),
            "event_end": (fixed_utc_now + timedelta(hours=1)).isoformat(),
            "temporal_confidence": "HIGH",
            "temporal_basis": "Controlled test validity window",
        }
        for index, city in enumerate(("Delhi", "Mumbai", "Chennai"), start=1)
    ]
    return {
        "active_non_operational": active_non_operational,
        "active_operational": {
            "Event_ID": 910,
            "Title": "Active operational flood in Chennai",
            "Event_Type": "Flood",
            "Location": "Chennai",
            "Source": "Weather API",
            "source_channel": "WEATHERAPI",
            "Operationally_Relevant": True,
            "event_start": (fixed_utc_now - timedelta(hours=2)).isoformat(),
            "event_end": (fixed_utc_now + timedelta(hours=2)).isoformat(),
            "temporal_confidence": "HIGH",
            "temporal_basis": "Controlled test validity window",
        },
        "upcoming": {
            "Event_ID": 911,
            "Title": "Upcoming storm",
            "Event_Type": "Storm",
            "Location": "Mumbai",
            "Operationally_Relevant": True,
            "event_start": (fixed_utc_now + timedelta(hours=1)).isoformat(),
            "event_end": (fixed_utc_now + timedelta(hours=4)).isoformat(),
        },
        "expired": {
            "Event_ID": 912,
            "Title": "Expired flood warning",
            "Event_Type": "Flood",
            "Location": "Kolkata",
            "Operationally_Relevant": True,
            "event_start": (fixed_utc_now - timedelta(days=1)).isoformat(),
            "event_end": (fixed_utc_now - timedelta(hours=2)).isoformat(),
        },
    }


def _install_event_loader(monkeypatch, events: list[dict[str, object]]) -> None:
    loader = lambda: {"events": events, "available": True}
    monkeypatch.setattr(agentic_tools, "load_current_events", loader)
    monkeypatch.setattr(service, "load_current_events", loader)


@pytest.fixture
def active_non_operational_event_loader(monkeypatch, controlled_events) -> list[dict[str, object]]:
    events = [
        *controlled_events["active_non_operational"],
        controlled_events["upcoming"],
        controlled_events["expired"],
    ]
    _install_event_loader(monkeypatch, events)
    return events


@pytest.fixture
def active_operational_event_loader(monkeypatch, controlled_events) -> list[dict[str, object]]:
    events = [
        controlled_events["active_operational"],
        controlled_events["upcoming"],
        controlled_events["expired"],
    ]
    _install_event_loader(monkeypatch, events)
    return events


@pytest.fixture
def controlled_low_risk_snapshot(monkeypatch, active_non_operational_event_loader) -> None:
    monkeypatch.setattr(
        service,
        "load_current_state",
        lambda: {"risk": pd.DataFrame([{"Node_ID": "TEST:1", "Risk_Level": "Low"}])},
    )


@pytest.fixture
def controlled_weather_snapshot(monkeypatch, tmp_path, fixed_utc_now: datetime) -> dict[str, object]:
    current_dir = tmp_path / "data" / "current"
    current_dir.mkdir(parents=True)
    observed_at = int(fixed_utc_now.timestamp())

    def observation(city: str, description: str, temperature: float, rain: float = 0.0) -> dict[str, object]:
        record: dict[str, object] = {
            "city": city,
            "source_channel": "OPENWEATHER",
            "weather": [{"main": "Rain" if rain else "Clear", "description": description}],
            "main": {"temp": temperature},
            "wind": {"speed": 2.5},
            "dt": observed_at,
        }
        if rain:
            record["rain"] = {"1h": rain}
        return record

    cities: dict[str, dict[str, object]] = {
        "Chennai": observation("Chennai", "light rain", 28.0, 0.4),
        "Kolkata": observation("Kolkata", "moderate rain", 27.0, 1.2),
        "Mumbai": observation("Mumbai", "drizzle", 26.0, 0.2),
        "Delhi": observation("Delhi", "clear sky", 31.0),
        "Bengaluru": observation("Bengaluru", "overcast clouds", 24.0),
    }
    for city in ("Chennai", "Kolkata", "Mumbai", "Delhi", "Bengaluru"):
        cities[f"{city}:WEATHERAPI"] = {
            "city": city,
            "source_channel": "WEATHERAPI",
            "record_kind": "DAILY_FORECAST",
            "location_validation_status": "VALIDATED",
            "weather": [{"main": "Forecast", "description": "controlled daily forecast"}],
        }
    payload: dict[str, object] = {
        "collected_at": fixed_utc_now.isoformat(),
        "cities": cities,
    }
    (current_dir / "current_weather.json").write_text(json.dumps(payload), encoding="utf-8")
    (current_dir / "current_run_metadata.json").write_text(json.dumps({
        "weather_fetch_timestamp": fixed_utc_now.isoformat(),
        "news_fetch_timestamp": fixed_utc_now.isoformat(),
    }), encoding="utf-8")
    monkeypatch.setattr(agentic_weather.current_state, "CURRENT_DIR", current_dir)
    return payload


def test_warehouse_risk_routes_graph_risk_and_events() -> None:
    result = run_agentic_query("Why is Delhi Warehouse risky today?")
    assert result["query_plan"]["primary_intent"] == "WAREHOUSE_RISK"
    assert "graph_context_tool" in result["tools_used"]
    assert "current_risk_tool" in result["tools_used"]
    assert result["fallback_used"] is False


def test_product_ranking_uses_risk_artifact_only() -> None:
    result = run_agentic_query("Which products currently have Medium or higher risk?")
    assert result["query_plan"]["primary_intent"] == "PRODUCT_RISK"
    assert result["tools_used"] == ["current_risk_ranking_tool"]
    assert result["answer"]
    assert "unavailable" not in result["answer"].lower()


def test_product_delivery_question_is_multi_intent() -> None:
    result = run_agentic_query("Why is Samsung Galaxy F05 risky and could its delivery be delayed?")
    assert result["query_plan"]["primary_intent"] == "PRODUCT_RISK"
    assert result["query_plan"]["secondary_intents"] == ["DELIVERY"]
    assert "delivery_scenario_tool" in result["tools_used"]
    assert result["fallback_used"] is False


def test_supplier_ranking_uses_risk_artifact() -> None:
    result = run_agentic_query("What suppliers are currently exposed?")
    assert result["query_plan"]["primary_intent"] == "SUPPLIER_RISK"
    assert result["tools_used"] == ["current_risk_ranking_tool"]


def test_events_bypass_entity_resolution(active_operational_event_loader) -> None:
    result = run_agentic_query("What current operational events are affecting the supply network?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_EVENTS"
    assert result["timings"]["entity_resolution_ms"] == 0.0
    assert "current_events_tool" in result["tools_used"]
    assert result["evidence_count"] > 0
    if result["events"]:
        assert any(item.startswith("events-") for item in result["evidence_used"])


def test_active_operational_events_produce_event_evidence(monkeypatch) -> None:
    class ActiveEventsTool:
        def invoke(self, _arguments):
            return {
                "events": [{
                    "Event_ID": 99,
                    "Title": "Active flood",
                    "Operationally_Relevant": True,
                    "event_status": "ACTIVE",
                }],
                "retrieval_ok": True,
            }

    monkeypatch.setattr(service, "current_events", ActiveEventsTool())
    result = run_agentic_query("What current operational events are affecting the supply network?")
    assert result["evidence_count"] > 0
    assert "events-0" in result["evidence_used"]


def test_zero_operational_events_is_valid(monkeypatch) -> None:
    class EmptyEventsTool:
        def invoke(self, _arguments):
            return {"events": [], "retrieval_ok": True}

    monkeypatch.setattr(service, "current_events", EmptyEventsTool())
    result = run_agentic_query("What current operational events are affecting the supply network?")
    assert result["fallback_used"] is False
    assert result["answer"] == "No current operational events were found."
    assert "CURRENT DATA UNAVAILABLE" not in result["answer"]


def test_historical_bypasses_current_retrieval() -> None:
    result = run_agentic_query("What was the historical DataCo model performance?")
    assert result["query_plan"]["primary_intent"] == "HISTORICAL_SHIPPING"
    assert result["tools_used"] == ["historical_shipping_tool"]
    assert result["historical_evidence"]["test_metrics"]


def test_current_and_historical_domains_remain_separate() -> None:
    result = run_agentic_query("Compare current network risk with historical DataCo model performance.")
    assert result["query_plan"]["primary_intent"] == "HISTORICAL_SHIPPING"
    assert "NETWORK_RISK" in result["query_plan"]["secondary_intents"]
    assert result["historical_evidence"]["test_metrics"]
    assert result["risk_evidence"]
    assert "r2" not in result["risk_evidence"][0]


def test_unknown_entity_fails_closed() -> None:
    result = run_agentic_query("Why is a nonexistent warehouse called XYZ Warehouse risky?")
    assert result["fallback_used"] is True
    assert result["query_plan"]["resolved_entity_ids"] == []
    assert "not found" in " ".join(result["limitations"]).lower()


def test_evidence_bundle_excludes_warnings_from_iteration() -> None:
    bundle = EvidenceBundle(
        graph=[EvidenceItem(evidence_id="g", evidence_type="graph", fact_name="path", value={}, provenance="DERIVED")],
        warnings=["Current external-signal artifacts are stale; results are not live."],
    )
    assert [item.evidence_id for item in bundle.all_evidence_items()] == ["g"]


def test_stale_warning_does_not_crash(monkeypatch) -> None:
    class FakeTool:
        def invoke(self, _arguments):
            return {"overall_stale": True, "weather_fresh": False, "news_fresh": False}

    monkeypatch.setattr(service, "freshness", FakeTool())
    result = run_agentic_query("Why is Delhi Warehouse risky today?")
    assert result["warnings"]
    assert "stale" in result["warnings"][0].lower()
    assert isinstance(result["evidence_used"], list)


def test_llm_disabled_is_deterministic(monkeypatch) -> None:
    monkeypatch.delenv("SUPPLYBEACON_LLM_ENABLED", raising=False)
    result = run_agentic_query("Why is Delhi Warehouse risky today?")
    assert result["llm_used"] is False
    assert result["answer"]


def test_llm_failure_preserves_answer(monkeypatch) -> None:
    class FailingProvider:
        def __init__(self):
            pass

        def generate(self, *_args, **_kwargs):
            raise service.ProviderError("offline")

    monkeypatch.setenv("SUPPLYBEACON_LLM_ENABLED", "true")
    monkeypatch.setattr(service, "OpenAICompatibleProvider", FailingProvider)
    result = run_agentic_query("Why is Delhi Warehouse risky today?")
    assert result["llm_used"] is False
    assert result["answer"]


def test_malformed_llm_output_preserves_answer(monkeypatch) -> None:
    class BadProvider:
        def __init__(self):
            pass

        def generate(self, *_args, **_kwargs):
            return json.dumps({"answer": 42})

    monkeypatch.setenv("SUPPLYBEACON_LLM_ENABLED", "true")
    monkeypatch.setattr(service, "OpenAICompatibleProvider", BadProvider)
    result = run_agentic_query("Why is Delhi Warehouse risky today?")
    assert result["llm_used"] is False
    assert result["answer"]


def test_non_json_llm_output_preserves_answer(monkeypatch) -> None:
    class BadProvider:
        def __init__(self):
            pass

        def generate(self, *_args, **_kwargs):
            return "not JSON"

    monkeypatch.setenv("SUPPLYBEACON_LLM_ENABLED", "true")
    monkeypatch.setattr(service, "OpenAICompatibleProvider", BadProvider)
    result = run_agentic_query("Why is Delhi Warehouse risky today?")
    assert result["llm_used"] is False
    assert result["answer"]


def test_empty_or_null_llm_answer_preserves_answer(monkeypatch) -> None:
    for value in ({}, {"answer": None}):
        class BadProvider:
            def __init__(self):
                pass

            def generate(self, *_args, **_kwargs):
                return json.dumps(value)

        monkeypatch.setenv("SUPPLYBEACON_LLM_ENABLED", "true")
        monkeypatch.setattr(service, "OpenAICompatibleProvider", BadProvider)
        result = run_agentic_query("Why is Delhi Warehouse risky today?")
        assert result["llm_used"] is False
        assert result["answer"]


def test_failed_risk_artifact_is_unavailable_not_zero_or_low(monkeypatch) -> None:
    class MissingRiskTool:
        def invoke(self, _arguments):
            return {"entities": [], "retrieval_ok": False, "limitations": ["Current risk artifact is unavailable."]}

    monkeypatch.setattr(service, "current_risk_ranking", MissingRiskTool())
    result = run_agentic_query("Which products currently have Medium or higher risk?")
    assert result["fallback_used"] is True
    assert result["entities"] == []
    assert "unavailable" in " ".join(result["limitations"]).lower()
    assert "Low" not in result["answer"]


def test_structured_answer_keeps_python_authoritative_fields() -> None:
    result = run_agentic_query("Why is Delhi Warehouse risky today?")
    structured = result["structured_answer"]
    assert structured["answer"] == result["answer"]
    for field in (
        "primary_intent", "secondary_intents", "entities", "risk_level",
        "evidence_used", "warnings", "current_data_fresh", "llm_used", "fallback_used",
    ):
        assert field in structured
    assert structured["primary_intent"] == result["query_plan"]["primary_intent"]


def test_product_graph_paths_are_entity_scoped() -> None:
    result = run_agentic_query("Why is Samsung Galaxy F05 risky?")
    product_ids = set(result["query_plan"]["resolved_entity_ids"])
    paths = result["graph_paths"]
    assert product_ids
    assert all(not path.get("nodes") or product_ids.intersection(path["nodes"]) for path in paths)


def test_refresh_cache_can_be_invalidated() -> None:
    from src.agentic.tools import current_retriever, invalidate_retriever_cache

    first = current_retriever()
    invalidate_retriever_cache()
    second = current_retriever()
    assert first is not second


def test_active_events_query_returns_active_non_operational_events(active_non_operational_event_loader) -> None:
    result = run_agentic_query("What active events are there?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_EVENTS"
    assert result["fallback_used"] is False
    assert len(result["events"]) == 3
    assert "active signal" in result["answer"].lower()
    for event in result["events"]:
        assert event["event_status"] == "ACTIVE"
        assert event["active_flag"] == "YES"
        assert event["operational_flag"] == "NO"
        assert event["operational_eligible"] is False
        assert "light rain" in event["event_type"].lower()
        assert event["operational_reason"]


def test_current_operational_events_query_returns_none_when_zero(active_non_operational_event_loader) -> None:
    result = run_agentic_query("What current operational events are there?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_EVENTS"
    assert result["fallback_used"] is False
    assert len(result["events"]) == 0
    assert result["answer"] == "No current operational events were found."


def test_happening_now_distinguishes_active_signals_from_operational_events(active_non_operational_event_loader) -> None:
    result = run_agentic_query("What is happening right now?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_EVENTS"
    assert "active signal" in result["answer"].lower()
    assert "operational" in result["answer"].lower()


def test_why_current_risk_low_is_grounded_network_answer(controlled_low_risk_snapshot) -> None:
    result = run_agentic_query("Why is current risk low?")
    assert result["query_plan"]["primary_intent"] == "NETWORK_RISK"
    assert "operational disruption" in result["answer"].lower()


def test_active_light_rain_remains_operationally_ineligible(active_non_operational_event_loader) -> None:
    from src.agentic.tools import current_events_tool

    res = current_events_tool(operational_only=False)
    assert res["active_event_count"] == 3
    assert res["operational_event_count"] == 0
    for event in res["events"]:
        assert event["operational_eligible"] is False
        assert event["active"] == "YES"
        assert event["operational"] == "NO"


def test_current_weather_returns_grounded_data(controlled_weather_snapshot) -> None:
    result = run_agentic_query("What is the current weather?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_WEATHER"
    assert result["fallback_used"] is False
    assert "current_weather_tool" in result["tools_used"]
    assert "OpenWeather observations" in result["answer"]
    assert len(result["weather"]) >= 7


def test_weather_in_delhi_filters_by_location(controlled_weather_snapshot) -> None:
    result = run_agentic_query("What is the weather in Delhi?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_WEATHER"
    assert result["fallback_used"] is False
    assert "Delhi" in result["answer"]
    assert "Mumbai" not in result["answer"]
    for record in result["weather"]:
        assert record["location"] == "Delhi"


def test_forecast_data_not_mislabeled_as_current_observation(controlled_weather_snapshot) -> None:
    result = run_agentic_query("What is the weather in Delhi?")
    weather_records = result["weather"]
    openweather = next(r for r in weather_records if r["source_channel"] == "OPENWEATHER")
    weatherapi = next(r for r in weather_records if r["source_channel"] == "WEATHERAPI")
    assert openweather["record_kind"] == "OBSERVATION"
    assert openweather["temperature_c"] is not None
    assert openweather["observed_at"] is not None
    assert weatherapi["record_kind"] == "DAILY_FORECAST_OR_ALERT"
    assert weatherapi["temperature_c"] is None
    assert weatherapi["observed_at"] is None


def test_locations_currently_have_rain(controlled_weather_snapshot) -> None:
    result = run_agentic_query("Which monitored locations currently have rain?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_WEATHER"
    assert result["fallback_used"] is False
    locations = {r["location"] for r in result["weather"]}
    assert {"Chennai", "Kolkata", "Mumbai"}.issubset(locations)
    assert "Delhi" not in locations
    assert all(r["rain_reported"] for r in result["weather"])


def test_warehouse_locations_bad_weather(controlled_weather_snapshot) -> None:
    result = run_agentic_query("Which warehouse locations have bad weather?")
    assert result["query_plan"]["primary_intent"] == "CURRENT_WEATHER"
    assert result["fallback_used"] is False
    assert "Chennai" in result["answer"]
    assert "Kolkata" in result["answer"]
    assert "Mumbai" in result["answer"]


def test_why_is_network_currently_low_risk(controlled_low_risk_snapshot) -> None:
    result = run_agentic_query("Why is the network currently Low risk?")
    assert result["query_plan"]["primary_intent"] == "NETWORK_RISK"
    assert result["fallback_used"] is False
    assert "Low" in result["answer"]
    assert "0 operational disruption events" in result["answer"]
    assert "3 active weather event(s) are present" in result["answer"]
