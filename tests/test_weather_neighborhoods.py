from unittest.mock import Mock, patch
from pathlib import Path

import pytest
import networkx as nx
import pandas as pd
from src.dynamic_risk.refresh_current_state import _current_risk, _event_locations
from src.event_extraction.event_extractor import weather_to_events

from src.external_data.weather_collector import (
    _cached_coordinates, cache_validated_resolution, fetch_weather_for_city,
    validate_provider_location, weather_location_cache_key,
)
from src.external_data.unified_sources import collect_unified_weather


def anchor(city="Chennai", state="Tamil Nadu", lat=13.083, lon=80.283, provider="WEATHERAPI"):
    return dict(requested_city=city, requested_state=state, requested_country="India",
                resolved_city=city, resolved_region=state, resolved_country="India",
                latitude=lat, longitude=lon, geocode_source=provider,
                geocode_query=f"{city}, {state}, India", location_validation_status="VALIDATED",
                city_validation_basis="EXACT_CITY")


def spec(a):
    return dict(city=a["requested_city"], state=a["requested_state"], country="India")


def response(city, state, lat, lon, country="India"):
    return {"location": dict(name=city, region=state, lat=lat, lon=lon, country=country)}


@pytest.mark.parametrize("a,city,lat,lon", [
    (anchor(), "Sowcarpet", 13.09, 80.28),
    (anchor("Mumbai", "Maharashtra", 18.975, 72.8258), "Bombay Wadala", 19.017, 72.85),
    (anchor("Vikhroli", "Maharashtra", 19.1, 72.9167), "Ghatkopar", 19.083, 72.9),
])
def test_trusted_coordinate_locality_keeps_anchor(a, city, lat, lon):
    location = spec(a)
    query = f'{a["latitude"]},{a["longitude"]}'
    ok, metadata, _ = validate_provider_location(
        location, response(city, a["resolved_region"], lat, lon), "WEATHERAPI", query,
        reference_resolution=a,
    )
    assert ok
    assert metadata["city_validation_basis"] == "DISTANCE_CORROBORATED"
    cache = {}
    cache_validated_resolution(cache, location, metadata)
    assert _cached_coordinates(cache, location) == (a["latitude"], a["longitude"])
    assert cache[weather_location_cache_key(location)]["resolved_city"] == a["resolved_city"]
    # A second observation must still be measured from the original anchor.
    distant = response(city, a["resolved_region"], a["latitude"] + 0.3, a["longitude"])
    assert not validate_provider_location(location, distant, "WEATHERAPI", query,
                                         reference_resolution=cache[weather_location_cache_key(location)])[0]


@pytest.mark.parametrize("city,state,lat,lon", [
    ("Sowcarpet", "Tamil Nadu", 13.09, 80.28),
    ("Chennai", "Tamil Nadu", 15.0, 80.28),
    ("Sowcarpet", "Karnataka", 13.09, 80.28),
])
def test_coordinate_query_rejects_far_coordinates_even_exact_label_or_wrong_state(city,state,lat,lon):
    if lat == 13.09 and state == "Tamil Nadu":
        country = "Mexico"
    else:
        country = "India"
    assert not validate_provider_location(spec(anchor()), response(city,state,lat,lon,country),
        "WEATHERAPI", "13.083,80.283", reference_resolution=anchor())[0]


def test_string_neighborhood_cannot_self_corroborate():
    a = anchor()
    assert not validate_provider_location(spec(a), response("Sowcarpet","Tamil Nadu",13.09,80.28),
        "WEATHERAPI", "Chennai, Tamil Nadu, India", reference_resolution=a)[0]


@pytest.mark.parametrize("change", [
    {"resolved_city":"Sowcarpet"}, {"requested_city":"Other"},
    {"requested_state":"Karnataka"}, {"resolved_country":"Mexico"},
    {"resolved_region":"Karnataka"}, {"latitude":float("nan")},
    {"latitude":100}, {"longitude":float("inf")},
])
def test_bad_cache_never_establishes_coordinate_identity(change):
    a = anchor()
    a.update(change)
    location = spec(anchor())
    assert _cached_coordinates({weather_location_cache_key(location):a},location) is None


@pytest.mark.parametrize("city,resolved,state", [
    ("Gurugram","Gurgaon","Haryana"), ("Ernakulam","Ernacolum","Kerala"),
    ("Ernakulam","Ernākulam","Kerala"), ("Karnal","Karnāl","Haryana"),
])
def test_explicit_aliases_and_diacritics(city,resolved,state):
    assert validate_provider_location(dict(city=city,state=state),
        response(resolved,state,13,77),"WEATHERAPI",f"{city}, {state}, India")[0]


@pytest.mark.parametrize("city,resolved,state", [
    ("Secunderabad","Hyderabad","Telangana"), ("Greater Noida","Noida","Uttar Pradesh"),
    ("Ernakulam","Kochi","Kerala"), ("Vasai East","Vasai","Maharashtra"),
])
def test_nearby_names_are_not_identity_aliases(city,resolved,state):
    assert not validate_provider_location(dict(city=city,state=state),
        response(resolved,state,13,77),"WEATHERAPI",f"{city}, {state}, India")[0]


def test_openweather_uses_cache_anchor_for_neighborhood():
    a = anchor()
    location = spec(a)
    cache = {weather_location_cache_key(location):a}
    raw = dict(name="Park Town",sys={"country":"IN"},coord={"lat":13.0833,"lon":80.2833},
               weather=[{"main":"Clouds"}])
    r = Mock(status_code=200)
    r.json.return_value = raw
    with patch("src.external_data.weather_collector.requests_with_retry",return_value=r):
        record = fetch_weather_for_city(Mock(),"configured",location,cache)
    assert record["city_validation_basis"] == "DISTANCE_CORROBORATED"
    assert _cached_coordinates(cache, location) == (13.083,80.283)


def test_string_rejection_is_reconsidered_after_independent_provider_exact_match():
    a = anchor()
    location = spec(a)
    r = Mock(status_code=200)
    r.json.return_value = dict(name="Park Town",sys={"country":"IN"},
        coord={"lat":13.0833,"lon":80.2833},weather=[{"main":"Clouds"}])
    with patch("src.external_data.weather_collector.requests_with_retry",return_value=r) as ow, \
         patch("src.external_data.unified_sources._configured_key",return_value="configured"), \
         patch("src.external_data.unified_sources._request_json",
               return_value=response("Chennai","Tamil Nadu",13.083,80.283)) as wa:
        weather, status = collect_unified_weather("configured",[location],cache_path=None,resolution_cache={})
    assert len(weather["cities"]) == 2
    assert weather["cities"]["Chennai"]["city_validation_basis"] == "DISTANCE_CORROBORATED"
    assert weather["location_resolution_failures"] == []
    assert status["OPENWEATHER"]["accepted_count"] == 1
    assert status["OPENWEATHER"]["rejected_count"] == 0
    assert ow.call_count == wa.call_count == 1


@pytest.mark.parametrize("a,locality,lat,lon,expected_distance", [
    (anchor("Ramanagara","Karnataka",12.7253,77.2805,"OPENWEATHER"),"Abbigeri",15.5833,75.7667,357.249),
    (anchor("Luhari","Haryana",23.6025,75.0994,"OPENWEATHER"),"Haryana",29.6161,76.9806,694.304),
    (anchor("South Kalamassery","Kerala",10.0448,76.3151,"OPENWEATHER"),"Kadakal",8.8333,76.9167,150.007),
])
def test_far_snapshot_response_creates_no_event_exposure_or_risk(a,locality,lat,lon,expected_distance):
    location = spec(a)
    ok, metadata, _ = validate_provider_location(location,
        response(locality,location["state"],lat,lon),"WEATHERAPI",
        f'{location["city"]}, {location["state"]}, India',reference_resolution=a)
    assert not ok
    assert metadata["reference_distance_km"] == pytest.approx(expected_distance,abs=0.001)
    record = {**metadata,"source_channel":"WEATHERAPI","weather":[{"main":"Cyclone"}],
              "alerts":[{"headline":"Cyclone warning"}]}
    events = weather_to_events(location["city"],record,Path("current_weather.json"))
    assert events == []
    graph = nx.DiGraph()
    graph.add_node("P",node_type="Product")
    graph.add_node("W",node_type="Warehouse")
    graph.add_node("L",node_type="Location",city=location["city"],state=location["state"])
    graph.add_edge("P","W",relationship="STORED_AT")
    graph.add_edge("W","L",relationship="LOCATED_IN")
    for event in events:
        graph.add_node("E",node_type="Event")
        _event_locations(graph,"E",event)
    assert not any(e.get("relationship")=="AFFECTS" for _,_,e in graph.edges(data=True))
    baseline = pd.DataFrame([dict(Node_ID="P",Node_Type="Product",Node_Name="Test",
        Risk_Score="0.1",Risk_Level="Low",Connected_Events="0",Structural_Baseline="0.1")])
    risk = _current_risk(graph,pd.DataFrame(),baseline_override=baseline).iloc[0]
    assert risk["Connected_Events"] == "0"
    assert risk["Weather_Event_Component"] == "0.0"
    assert risk["Final_Scenario_Risk_Score"] == "0.1"


def test_legacy_distance_verdict_without_anchor_cannot_establish_identity():
    a = anchor()
    a.update(resolved_city="Neighborhood",city_validation_basis="CROSS_PROVIDER_DISTANCE",
             reference_distance_km=1.0,reference_source="OPENWEATHER")
    assert _cached_coordinates({weather_location_cache_key(spec(a)):a},spec(a)) is None
