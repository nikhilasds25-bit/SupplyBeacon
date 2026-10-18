from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import networkx as nx
import pandas as pd
import pytest

from src.dynamic_risk.refresh_current_state import _current_risk, _event_locations
from src.event_extraction.event_extractor import weather_to_events
from src.external_data.unified_sources import collect_unified_weather
from src.external_data.weather_collector import (
    MAX_CROSS_PROVIDER_DISTANCE_KM,
    _cached_coordinates,
    cache_validated_resolution,
    collect_weather,
    fetch_weather_for_city,
    validate_provider_location,
    warehouse_weather_location_specs,
    weather_location_query,
)


FIXED_LOCATION = {"city": "Maheshtala", "state": "West Bengal", "country": "India"}


def weatherapi_payload(
    *,
    city: str,
    region: str,
    country: str = "India",
    alerts: list[dict] | None = None,
    latitude: float = 22.51,
    longitude: float = 88.25,
) -> dict:
    return {
        "location": {
            "name": city,
            "region": region,
            "country": country,
            "lat": latitude,
            "lon": longitude,
            "tz_id": "Asia/Kolkata" if country == "India" else "America/Mexico_City",
        },
        "forecast": {"forecastday": [{
            "date": "2026-10-10",
            "day": {
                "condition": {"text": "Light rain"},
                "totalprecip_mm": 2.0,
                "maxwind_kph": 20.0,
            },
        }]},
        "alerts": {"alert": alerts or []},
    }


def foreign_cyclone_payload(city: str) -> dict:
    return weatherapi_payload(
        city=city,
        region="Guerrero",
        country="Mexico",
        alerts=[{
            "event": "Aviso de lluvias",
            "headline": "Ciclón tropical Simon",
            "areas": "COL, GRO",
            "effective": "2026-10-10T01:00:00-06:00",
            "expires": "2026-10-10T18:00:00-06:00",
        }],
    )


def test_location_query_includes_city_state_and_india() -> None:
    assert weather_location_query(FIXED_LOCATION) == "Maheshtala, West Bengal, India"


def test_openweather_prefers_qualified_query_over_bare_city() -> None:
    response = Mock(status_code=200)
    response.json.return_value = {
        "name": "Maheshtala",
        "sys": {"country": "IN"},
        "coord": {"lat": 22.51, "lon": 88.25},
        "timezone": 19800,
        "weather": [{"main": "Clouds", "description": "overcast clouds"}],
    }
    with patch("src.external_data.weather_collector.requests_with_retry", return_value=response) as request:
        fetch_weather_for_city(Mock(), "configured", FIXED_LOCATION, {})
    params = request.call_args.kwargs["params"]
    assert params["q"] == "Maheshtala, West Bengal, India"
    assert params["q"] != "Maheshtala"


def test_non_india_weatherapi_response_is_rejected() -> None:
    valid, metadata, reason = validate_provider_location(
        FIXED_LOCATION, foreign_cyclone_payload("Maheshtala"), "WEATHERAPI",
        weather_location_query(FIXED_LOCATION),
    )
    assert valid is False
    assert metadata["location_validation_status"] == "REJECTED_COUNTRY"
    assert "non-India" in str(reason)


def test_wrong_weatherapi_region_is_rejected() -> None:
    payload = weatherapi_payload(city="Maheshtala", region="Karnataka")
    valid, metadata, _ = validate_provider_location(
        FIXED_LOCATION, payload, "WEATHERAPI", weather_location_query(FIXED_LOCATION)
    )
    assert valid is False
    assert metadata["location_validation_status"] == "REJECTED_REGION"


def test_valid_india_weatherapi_response_is_accepted_with_audit_metadata() -> None:
    payload = weatherapi_payload(city="Maheshtala", region="West Bengal")
    valid, metadata, reason = validate_provider_location(
        FIXED_LOCATION, payload, "WEATHERAPI", weather_location_query(FIXED_LOCATION)
    )
    assert valid is True
    assert reason is None
    assert metadata == {
        "requested_city": "Maheshtala",
        "requested_state": "West Bengal",
        "requested_country": "India",
        "resolved_city": "Maheshtala",
        "resolved_region": "West Bengal",
        "resolved_country": "India",
        "latitude": 22.51,
        "longitude": 88.25,
        "timezone": "Asia/Kolkata",
        "geocode_source": "WEATHERAPI",
        "geocode_query": "Maheshtala, West Bengal, India",
        "geocode_country": "India",
        "geocode_region": "West Bengal",
        "geocode_confidence": "HIGH",
        "location_validation_status": "VALIDATED",
        "city_validation_basis": "EXACT_CITY",
        "reference_source": None,
        "reference_distance_km": None,
        "distance_threshold_km": MAX_CROSS_PROVIDER_DISTANCE_KM,
    }


def test_ramanagara_exact_city_is_accepted() -> None:
    location = {"city": "Ramanagara", "state": "Karnataka", "country": "India"}
    payload = weatherapi_payload(
        city="Ramanagara", region="Karnataka", latitude=12.7253, longitude=77.2805
    )
    valid, metadata, reason = validate_provider_location(
        location, payload, "WEATHERAPI", weather_location_query(location)
    )
    assert valid is True
    assert reason is None
    assert metadata["city_validation_basis"] == "EXACT_CITY"


def test_ramanagara_far_abbigeri_resolution_is_rejected_against_openweather_reference() -> None:
    location = {"city": "Ramanagara", "state": "Karnataka", "country": "India"}
    openweather_reference = {
        "requested_city": "Ramanagara",
        "requested_state": "Karnataka",
        "requested_country": "India",
        "resolved_city": "Ramanagara",
        "resolved_region": "",
        "resolved_country": "IN",
        "latitude": 12.7253,
        "longitude": 77.2805,
        "geocode_source": "OPENWEATHER",
        "location_validation_status": "VALIDATED",
    }
    payload = weatherapi_payload(
        city="Abbigeri", region="Karnataka", latitude=15.5833, longitude=75.7667
    )
    valid, metadata, reason = validate_provider_location(
        location,
        payload,
        "WEATHERAPI",
        weather_location_query(location),
        reference_resolution=openweather_reference,
    )
    assert valid is False
    assert metadata["location_validation_status"] == "REJECTED_CITY_DISTANCE"
    assert metadata["reference_distance_km"] == pytest.approx(357.249, abs=0.001)
    assert "maximum allowed is 25.0 km" in str(reason)


def test_unified_collection_rejects_ramanagara_abbigeri_using_openweather_reference() -> None:
    location = {"city": "Ramanagara", "state": "Karnataka", "country": "India"}
    openweather_record = {
        "weather": [{"main": "Clouds", "description": "overcast clouds"}],
        "requested_city": "Ramanagara",
        "requested_state": "Karnataka",
        "requested_country": "India",
        "resolved_city": "Ramanagara",
        "resolved_region": "",
        "resolved_country": "IN",
        "latitude": 12.7253,
        "longitude": 77.2805,
        "geocode_source": "OPENWEATHER",
        "location_validation_status": "VALIDATED",
        "city": "Ramanagara",
        "state": "Karnataka",
        "country": "India",
    }
    openweather_result = {
        "cities": {"Ramanagara": openweather_record},
        "rejected_locations": [],
        "provider_request_failures": [],
    }
    abbigeri = weatherapi_payload(
        city="Abbigeri", region="Karnataka", latitude=15.5833, longitude=75.7667
    )
    with patch("src.external_data.unified_sources.collect_weather", return_value=openweather_result), \
         patch("src.external_data.unified_sources._configured_key", return_value="configured"), \
         patch("src.external_data.unified_sources._request_json", return_value=abbigeri):
        weather, status = collect_unified_weather(
            "configured", [location], cache_path=None, resolution_cache={}
        )

    assert list(weather["cities"]) == ["Ramanagara"]
    assert status["OPENWEATHER"]["accepted_count"] == 1
    assert status["WEATHERAPI"]["accepted_count"] == 0
    assert status["WEATHERAPI"]["rejected_count"] == 1
    failure = weather["location_resolution_failures"][0]
    assert failure["provider"] == "WEATHERAPI"
    assert failure["location_validation_status"] == "REJECTED_CITY_DISTANCE"
    assert failure["reference_distance_km"] == pytest.approx(357.249, abs=0.001)


def test_jammu_and_kashmir_ampersand_is_equivalent_to_and() -> None:
    location = {"city": "Jammu", "state": "Jammu & Kashmir", "country": "India"}
    payload = weatherapi_payload(city="Jammu", region="Jammu and Kashmir")
    valid, metadata, reason = validate_provider_location(
        location, payload, "WEATHERAPI", weather_location_query(location)
    )
    assert valid is True
    assert reason is None
    assert metadata["location_validation_status"] == "VALIDATED"


def test_controlled_bengaluru_bangalore_alias_is_accepted() -> None:
    location = {"city": "Bengaluru", "state": "Karnataka", "country": "India"}
    payload = weatherapi_payload(city="Bangalore", region="Karnataka")
    valid, metadata, reason = validate_provider_location(
        location, payload, "WEATHERAPI", weather_location_query(location)
    )
    assert valid is True
    assert reason is None
    assert metadata["city_validation_basis"] == "CONTROLLED_CITY_ALIAS"


def test_unapproved_different_city_is_not_accepted_without_independent_reference() -> None:
    location = {"city": "Ramanagara", "state": "Karnataka", "country": "India"}
    payload = weatherapi_payload(
        city="Nearby Locality", region="Karnataka", latitude=12.73, longitude=77.29
    )
    valid, metadata, reason = validate_provider_location(
        location, payload, "WEATHERAPI", weather_location_query(location)
    )
    assert valid is False
    assert metadata["location_validation_status"] == "REJECTED_CITY"
    assert "no qualifying independent distance validation" in str(reason)


def test_nearby_different_city_requires_and_uses_independent_distance_validation() -> None:
    location = {"city": "Ramanagara", "state": "Karnataka", "country": "India"}
    reference = {
        "requested_city": "Ramanagara",
        "requested_state": "Karnataka",
        "requested_country": "India",
        "resolved_city": "Ramanagara",
        "resolved_region": "Karnataka",
        "resolved_country": "India",
        "latitude": 12.7253,
        "longitude": 77.2805,
        "geocode_source": "OPENWEATHER",
        "location_validation_status": "VALIDATED",
    }
    payload = weatherapi_payload(
        city="Nearby Locality", region="Karnataka", latitude=12.73, longitude=77.29
    )
    valid, metadata, reason = validate_provider_location(
        location,
        payload,
        "WEATHERAPI",
        weather_location_query(location),
        reference_resolution=reference,
    )
    assert valid is True
    assert reason is None
    assert metadata["city_validation_basis"] == "DISTANCE_CORROBORATED"
    assert metadata["reference_distance_km"] < MAX_CROSS_PROVIDER_DISTANCE_KM


def test_stale_mismatched_city_cache_coordinates_are_not_reused() -> None:
    location = {"city": "Ramanagara", "state": "Karnataka", "country": "India"}
    cache = {
        "ramanagara|karnataka|india": {
            "resolved_city": "Abbigeri",
            "latitude": 15.5833,
            "longitude": 75.7667,
            "location_validation_status": "VALIDATED",
        }
    }
    assert _cached_coordinates(cache, location) is None


@pytest.mark.parametrize(
    ("city", "state"),
    (("Maheshtala", "West Bengal"), ("Mamring", "Sikkim"), ("Ramanagara", "Karnataka")),
)
def test_foreign_cyclone_alert_is_rejected_for_indian_warehouse_locations(city: str, state: str) -> None:
    location = {"city": city, "state": state, "country": "India"}
    with patch("src.external_data.unified_sources._configured_key", return_value="configured"), \
         patch("src.external_data.unified_sources._request_json", return_value=foreign_cyclone_payload(city)):
        weather, status = collect_unified_weather(
            None, [location], cache_path=None, resolution_cache={}
        )
    assert weather["cities"] == {}
    assert weather["location_resolution_failures"][0]["location_validation_status"] == "REJECTED_COUNTRY"
    assert status["WEATHERAPI"]["accepted_count"] == 0
    assert status["WEATHERAPI"]["rejected_count"] == 1


def test_thiruvananthapuram_india_resolution_is_accepted() -> None:
    location = {"city": "Thiruvananthapuram", "state": "Kerala", "country": "India"}
    payload = weatherapi_payload(city="Thiruvananthapuram", region="Kerala")
    with patch("src.external_data.unified_sources._configured_key", return_value="configured"), \
         patch("src.external_data.unified_sources._request_json", return_value=payload) as request:
        weather, status = collect_unified_weather(
            None, [location], cache_path=None, resolution_cache={}
        )
    assert request.call_args.args[1]["q"] == "Thiruvananthapuram, Kerala, India"
    record = weather["cities"]["Thiruvananthapuram:WEATHERAPI"]
    assert record["location_validation_status"] == "VALIDATED"
    assert record["resolved_country"] == "India"
    assert status["WEATHERAPI"]["accepted_count"] == 1


def test_duplicate_city_state_warehouses_share_one_provider_query() -> None:
    graph = nx.DiGraph()
    graph.add_node("WH:1", node_type="Warehouse", name="First", state="West Bengal")
    graph.add_node("WH:2", node_type="Warehouse", name="Second", state="West Bengal")
    graph.add_node(
        "LOCATION:Maheshtala", node_type="Location", name="Maheshtala",
        city="Maheshtala", state="West Bengal", country="India",
    )
    graph.add_edge("WH:1", "LOCATION:Maheshtala", relationship="LOCATED_IN")
    graph.add_edge("WH:2", "LOCATION:Maheshtala", relationship="LOCATED_IN")
    locations = warehouse_weather_location_specs(graph)
    assert locations == [FIXED_LOCATION]

    payload = weatherapi_payload(city="Maheshtala", region="West Bengal")
    with patch("src.external_data.unified_sources._configured_key", return_value="configured"), \
         patch("src.external_data.unified_sources._request_json", return_value=payload) as request:
        weather, _ = collect_unified_weather(None, locations, cache_path=None, resolution_cache={})
    assert request.call_count == 1
    assert len(weather["cities"]) == 1


def test_weatherapi_request_failure_is_persisted_with_safe_audit_fields() -> None:
    location = {"city": "Ramanagara", "state": "Karnataka", "country": "India"}
    error = RuntimeError(
        "failed https://api.weatherapi.com/v1/forecast.json?key=secret-value&q=Ramanagara"
    )
    with patch("src.external_data.unified_sources._configured_key", return_value="configured"), \
         patch("src.external_data.unified_sources._request_json", side_effect=error):
        weather, status = collect_unified_weather(
            None, [location], cache_path=None, resolution_cache={}
        )

    assert weather["cities"] == {}
    assert len(weather["provider_request_failures"]) == 1
    failure = weather["provider_request_failures"][0]
    assert failure["provider"] == "WEATHERAPI"
    assert failure["requested_city"] == "Ramanagara"
    assert failure["requested_state"] == "Karnataka"
    assert failure["requested_country"] == "India"
    assert failure["query"] == "Ramanagara, Karnataka, India"
    assert failure["failure_type"] == "REQUEST_ERROR"
    assert failure["exception_type"] == "RuntimeError"
    assert "secret-value" not in failure["reason"]
    assert "<redacted>" in failure["reason"]
    assert failure["timestamp"].endswith("+00:00")
    assert status["WEATHERAPI"]["request_failure_count"] == 1


def test_openweather_request_failure_is_exposed_for_unified_persistence() -> None:
    with patch(
        "src.external_data.weather_collector.fetch_weather_for_city",
        side_effect=RuntimeError("provider unavailable"),
    ):
        payload = collect_weather("configured", [FIXED_LOCATION], {})

    assert len(payload["provider_request_failures"]) == 1
    failure = payload["provider_request_failures"][0]
    assert failure["provider"] == "OPENWEATHER"
    assert failure["query"] == "Maheshtala, West Bengal, India"
    assert failure["failure_type"] == "REQUEST_ERROR"
    assert failure["reason"] == "provider unavailable"


def test_validated_coordinates_are_cached_and_preferred_on_later_requests() -> None:
    response = Mock(status_code=200)
    response.json.return_value = {
        "name": "Maheshtala",
        "sys": {"country": "IN"},
        "coord": {"lat": 22.51, "lon": 88.25},
        "timezone": 19800,
        "weather": [{"main": "Clouds", "description": "overcast clouds"}],
    }
    cache: dict[str, dict] = {}
    with patch("src.external_data.weather_collector.requests_with_retry", return_value=response) as request:
        fetch_weather_for_city(Mock(), "configured", FIXED_LOCATION, cache)
        valid, metadata, _ = validate_provider_location(
            FIXED_LOCATION,
            weatherapi_payload(city="Maheshtala", region="West Bengal"),
            "WEATHERAPI",
            weather_location_query(FIXED_LOCATION),
        )
        assert valid is True
        cache_validated_resolution(cache, FIXED_LOCATION, metadata)
        fetch_weather_for_city(Mock(), "configured", FIXED_LOCATION, cache)
    first_params = request.call_args_list[0].kwargs["params"]
    second_params = request.call_args_list[1].kwargs["params"]
    assert first_params["q"] == "Maheshtala, West Bengal, India"
    assert "q" not in second_params
    assert (second_params["lat"], second_params["lon"]) == (22.51, 88.25)
    assert all("key" not in entry for entry in cache.values())


def rejected_alert_record() -> dict:
    return {
        "weather": [{"main": "Cyclone", "description": "Ciclón tropical Simon"}],
        "alerts": [{"headline": "Ciclón tropical Simon"}],
        "source_channel": "WEATHERAPI",
        "location_validation_status": "REJECTED_COUNTRY",
        "resolved_country": "Mexico",
        "dt": 1791590400,
    }


def test_rejected_weather_data_creates_no_event() -> None:
    assert weather_to_events(
        "Maheshtala", rejected_alert_record(), Path("current_weather.json")
    ) == []


def test_rejected_alert_creates_no_affects_edge() -> None:
    graph = nx.DiGraph()
    graph.add_node("WH_REAL_TEST", node_type="Warehouse", name="Test Warehouse")
    graph.add_node("LOCATION:Maheshtala", node_type="Location", name="Maheshtala", city="Maheshtala")
    graph.add_edge("WH_REAL_TEST", "LOCATION:Maheshtala", relationship="LOCATED_IN")
    events = weather_to_events("Maheshtala", rejected_alert_record(), Path("current_weather.json"))
    for index, event in enumerate(events):
        event_id = f"EVENT:{index}"
        graph.add_node(event_id, node_type="Event", name=event["Title"])
        _event_locations(graph, event_id, event)
    assert not any(attrs.get("relationship") == "AFFECTS" for _, _, attrs in graph.edges(data=True))


def test_rejected_alert_creates_no_risk_contribution() -> None:
    graph = nx.DiGraph()
    graph.add_node("PRODUCT:TEST", node_type="Product", name="Test Product")
    graph.add_node("WH_REAL_TEST", node_type="Warehouse", name="Test Warehouse")
    graph.add_edge("PRODUCT:TEST", "WH_REAL_TEST", relationship="STORED_AT")
    baseline = pd.DataFrame([{
        "Node_ID": "PRODUCT:TEST", "Node_Type": "Product", "Node_Name": "Test Product",
        "Risk_Score": "0.1", "Risk_Level": "Low", "Connected_Events": "0",
        "Event_Exposure": "0", "Business_Impact_Risk": "0", "Max_Event_Severity": "Low",
        "Average_Event_Confidence": "0", "Structural_Baseline": "0.1",
    }])
    risk = _current_risk(graph, pd.DataFrame(), baseline_override=baseline).iloc[0]
    assert risk["Connected_Events"] == "0"
    assert risk["Event_Exposure_Component"] == "0.0"
    assert risk["Final_Scenario_Risk_Score"] == "0.1"
