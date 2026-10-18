from unittest.mock import patch

from src.external_data.unified_sources import (
    deduplicate_news_articles,
    normalize_news_articles,
)
from src.event_extraction.event_extractor import weather_to_events
from src.event_extraction.event_extractor import article_to_events
from pathlib import Path
import networkx as nx
import pytest

from src.external_data.weather_collector import collect_weather, warehouse_weather_locations
from src.external_data.utils import sanitize_external_error


def test_news_sources_normalize_common_schema() -> None:
    articles = normalize_news_articles({
        "results": [{
            "title": "Rain warning",
            "description": "Heavy rain expected",
            "link": "https://example.test/rain",
            "pubDate": "2026-09-20T10:00:00Z",
            "source_id": "example",
        }],
    }, "NEWSDATA_IO")
    assert articles[0]["source_channel"] == "NEWSDATA_IO"
    assert articles[0]["url"] == "https://example.test/rain"
    assert articles[0]["published_at"] == "2026-09-20T10:00:00Z"


def test_cross_source_news_dedup_retains_provenance() -> None:
    rows = deduplicate_news_articles([
        {"title": "Same story", "url": "https://example.test/story", "published_at": "2026-09-20", "source_channel": "NEWS_API"},
        {"title": "Same story", "url": "https://example.test/story/", "published_at": "2026-09-20", "source_channel": "CURRENTS_API"},
    ])
    assert len(rows) == 1
    assert set(rows[0]["source_channels"]) == {"NEWS_API", "CURRENTS_API"}
    assert len(rows[0]["source_references"]) == 2


def test_news_source_failure_isolated() -> None:
    from src.external_data import unified_sources

    with patch.object(unified_sources, "collect_news", side_effect=RuntimeError("news down")), \
         patch.object(unified_sources, "_request_json", side_effect=RuntimeError("secondary down")):
        articles, status = unified_sources.collect_unified_news("configured")
    assert articles == []
    assert status["NEWS_API"]["success"] is False
    assert status["NEWSDATA_IO"]["success"] is False
    assert status["CURRENTS_API"]["success"] is False


def test_duplicate_articles_produce_one_extracted_event_input() -> None:
    rows = deduplicate_news_articles([
        {"title": "Heavy rain", "description": "Delhi", "url": "", "published_at": "2026-09-20", "source_channel": "NEWS_API"},
        {"title": "Heavy rain", "description": "Delhi", "url": "", "published_at": "2026-09-20", "source_channel": "NEWSDATA_IO"},
    ])
    assert len(rows) == 1


def test_weatherapi_alert_dates_feed_temporal_validity() -> None:
    events = weather_to_events(
        "Delhi",
        {
            "weather": [{"main": "Heavy Rain", "description": "heavy rain"}],
            "dt": 1790236800,
            "source_channel": "WEATHERAPI",
            "location_validation_status": "VALIDATED",
            "alerts": [{
                "event": "Heavy rain warning",
                "effective": "2026-09-20T00:00:00+00:00",
                "expires": "2026-09-21T00:00:00+00:00",
            }],
        },
        Path("current_weather.json"),
    )
    assert events[0]["source_channel"] == "WEATHERAPI"
    assert events[0]["event_start"].startswith("2026-09-20")
    assert events[0]["event_end"].startswith("2026-09-21")


def test_negated_weather_statement_does_not_create_event() -> None:
    events = article_to_events(
        {
            "title": "Delhi expected to remain largely dry",
            "description": "No rain expected through the week",
            "published_at": "2026-09-20T10:00:00Z",
            "source_channel": "NEWS_API",
        },
        Path("current_news.json"),
    )
    assert events == []


def test_unrelated_supply_chain_story_is_not_operational() -> None:
    events = article_to_events(
        {
            "title": "Karnataka fake-drug investigation and license suspensions",
            "description": "The case triggered supply chain scrutiny in the pharmaceutical sector.",
            "published_at": "2026-09-20T10:00:00Z",
            "source_channel": "CURRENTS_API",
        },
        Path("current_news.json"),
    )
    assert len(events) == 1
    assert events[0]["Event_Type"] == "Irrelevant"
    assert events[0]["Operationally_Relevant"] is False


def test_currents_publication_timestamp_survives_normalization():
    articles = normalize_news_articles({"news": [{
        "title": "Factory shutdown today in Delhi",
        "description": "Electronics manufacturing halted today.",
        "published": "2026-09-20 04:18:54 +0000",
    }]}, "CURRENTS_API")
    assert articles[0]["published_at"] == "2026-09-20 04:18:54 +0000"
    event = article_to_events(articles[0], Path("current_news.json"))[0]
    assert event["Date"].startswith("2026-09-20")
    assert event["event_start"].startswith("2026-09-20")
    from src.event_extraction.temporal_validity import temporal_status
    from datetime import datetime, timezone
    assert temporal_status(event["event_start"], event["event_end"],
                           datetime(2026, 9, 22, tzinfo=timezone.utc)) == "EXPIRED"


def test_missing_news_publication_does_not_use_collection_clock():
    event = article_to_events({
        "title": "Factory shutdown today in Delhi",
        "description": "Electronics manufacturing halted today.",
    }, Path("current_news.json"))[0]
    assert event["Date"] is None
    assert event["event_start"] is None
    assert event["event_end"] is None


def test_explicit_component_shortage_is_operational_but_negation_is_not():
    event = article_to_events({
        "title": "Component shortage hits smartphone production today",
        "description": "Electronics suppliers cannot meet manufacturing demand.",
        "published_at": "2026-09-20T10:00:00Z",
    }, Path("current_news.json"))[0]
    assert event["Event_Type"] == "Component Shortage"
    assert event["Operationally_Relevant"] is True
    assert article_to_events({
        "title": "No component shortage in electronics manufacturing today",
        "description": "Suppliers meet demand.",
        "published_at": "2026-09-20T10:00:00Z",
    }, Path("current_news.json")) == []


def test_runtime_warehouse_cities_drive_weather_monitoring() -> None:
    from src.dynamic_risk.current_state import load_current_state

    cities = warehouse_weather_locations(load_current_state()["graph"])
    assert len(cities) == 55
    assert len(cities) == len({city.casefold() for city in cities})
    assert "Bhiwandi" in cities
    assert "Thiruvananthapuram" in cities
    assert "Hyderabad" in cities
    assert "Pune" not in cities
    assert "South India" not in cities


def test_company_overlay_warehouse_location_is_included() -> None:
    graph = nx.DiGraph()
    graph.add_node("warehouse", node_type="Warehouse", name="Company Warehouse", provenance="USER_ENTERED")
    graph.add_node("location", node_type="Location", name="Company City", city="Company City", provenance="USER_ENTERED")
    graph.add_edge("warehouse", "location", relationship="LOCATED_IN", provenance="USER_ENTERED")
    assert warehouse_weather_locations(graph) == ["Company City"]


def test_light_rain_remains_non_operational() -> None:
    event = weather_to_events("Mumbai", {
        "weather": [{"id": 500, "main": "Rain", "description": "light rain"}],
        "rain": {"1h": 0.4},
        "dt": 1790236800,
    }, Path("current_weather.json"))[0]
    assert event["Event_Type"] == "Light Rain"
    assert event["Operationally_Relevant"] is False


def test_equivalent_alert_terms_map_without_broadening() -> None:
    cases = {
        "Very heavy rainfall warning": "Heavy Rain",
        "Flood alert": "Flood",
        "Severe thunderstorm warning": "Thunderstorm",
        "Strong wind alert": "Storm",
    }
    for label, expected in cases.items():
        event = weather_to_events("Delhi", {
            "weather": [{"main": label, "description": label}],
            "alerts": [{"event": label}],
            "dt": 1790236800,
            "source_channel": "WEATHERAPI",
            "location_validation_status": "VALIDATED",
        }, Path("current_weather.json"))[0]
        assert event["Event_Type"] == expected
        assert event["Operationally_Relevant"] is False
        assert event["Evidence_Level"] == "POTENTIAL_DISRUPTION"


def test_moderate_rain_code_is_not_downgraded_by_low_hourly_amount() -> None:
    event = weather_to_events("Pune", {
        "weather": [{"id": 501, "main": "Rain", "description": "moderate rain"}],
        "rain": {"1h": 1.1},
        "dt": 1790236800,
    }, Path("current_weather.json"))[0]
    assert event["Event_Type"] == "Moderate Rain"
    assert event["Operationally_Relevant"] is False


def test_provider_failure_does_not_create_weather_evidence() -> None:
    with patch("src.external_data.weather_collector.fetch_weather_for_city", side_effect=RuntimeError("provider unavailable")):
        payload = collect_weather("configured", ["Delhi"])
    assert payload["cities"] == {}
    assert payload["successful_count"] == 0
    assert payload["failed_cities"] == ["Delhi"]


def test_provider_errors_redact_credentials() -> None:
    error = "failed https://example.test/data?apiKey=secret-value&q=Delhi&appid=other-secret"
    safe = sanitize_external_error(error)
    assert "secret-value" not in safe
    assert "other-secret" not in safe
    assert safe.count("<redacted>") == 2


def test_zero_record_refresh_preserves_previous_snapshot(monkeypatch, tmp_path: Path) -> None:
    from src.dynamic_risk import refresh_current_state as refresh_module

    current_dir = tmp_path / "current"
    current_dir.mkdir()
    existing = current_dir / "current_weather.json"
    existing.write_text('{"snapshot": "last-good"}', encoding="utf-8")
    monkeypatch.setattr(refresh_module, "CURRENT_DIR", current_dir)
    monkeypatch.setattr(refresh_module, "_collect_external", lambda: (
        {"cities": {}},
        {"articles": []},
        {
            "news": {"NEWS_API": {"success": False, "accepted_count": 0, "error": "offline"}},
            "weather": {"OPENWEATHER": {"success": False, "accepted_count": 0, "error": "offline"}},
        },
    ))

    with pytest.raises(RuntimeError, match="previous snapshot was preserved"):
        refresh_module.refresh_current_state()

    assert existing.read_text(encoding="utf-8") == '{"snapshot": "last-good"}'
