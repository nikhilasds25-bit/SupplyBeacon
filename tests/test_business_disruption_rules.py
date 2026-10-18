from pathlib import Path
from datetime import datetime, timezone

import networkx as nx
import pandas as pd
import pytest

from src.dynamic_risk.refresh_current_state import _current_risk, _event_locations
from src.dynamic_risk.location_resolution import recognition_aliases
from src.event_extraction import temporal_validity
from src.event_extraction.business_disruption_rules import (
    INFORMATIONAL_SIGNAL,
    OPERATIONAL_DISRUPTION,
    POTENTIAL_DISRUPTION,
    classify_news_evidence,
)
from src.event_extraction.event_extractor import article_to_events, weather_to_events
from src.event_extraction.temporal_validity import is_currently_eligible
from src.api.app import _event_payload


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(
        temporal_validity,
        "utc_now",
        lambda: datetime(2026, 10, 10, 12, tzinfo=timezone.utc),
    )
    graph = nx.DiGraph()
    for warehouse, city, state in (
        ("WH_CHENNAI", "Chennai", "Tamil Nadu"),
        ("WH_MUMBAI", "Mumbai", "Maharashtra"),
        ("WH_AHMEDABAD", "Ahmedabad", "Gujarat"),
        ("WH_SURAT", "Surat", "Gujarat"),
    ):
        location = f"LOC:{city}"
        graph.add_node(warehouse, node_type="Warehouse", name=f"{city} Warehouse", city=city, state=state)
        graph.add_node(location, node_type="Location", name=city, city=city, state=state, country="India")
        graph.add_edge(warehouse, location, relationship="LOCATED_IN")
    return graph


OPERATIONAL_CASES = [
    ("WAREHOUSE_FIRE", "A fire damaged the Chennai warehouse and forced operations to stop."),
    ("WAREHOUSE_CLOSURE", "The Chennai warehouse was closed after a safety inspection."),
    ("WAREHOUSE_FLOODING", "Floodwater flooded the Chennai warehouse and damaged its stock."),
    ("WAREHOUSE_EVACUATION", "Workers were evacuated from the Chennai warehouse."),
    ("WAREHOUSE_EQUIPMENT_FAILURE", "A conveyor failure stopped sorting at the Chennai warehouse."),
    ("ROAD_CLOSURE", "A road closure blocked freight movement near Chennai."),
    ("ROAD_CLOSURE", "Highway closure halted truck movement in Chennai."),
    ("BRIDGE_CLOSURE", "Freight traffic was suspended after the bridge closed near Chennai."),
    ("BRIDGE_CLOSURE", "A damaged bridge was closed to truck traffic near Chennai."),
    ("RAIL_DISRUPTION", "Rail freight services were suspended in Chennai."),
    ("PORT_DISRUPTION", "The Chennai port shut down and cargo handling was suspended."),
    ("AIR_CARGO_DISRUPTION", "Cargo operations at Chennai airport were suspended."),
    ("TRANSPORT_STRIKE", "A logistics workers strike halted freight transport in Chennai."),
    ("TRANSPORT_STRIKE", "A transport strike stopped goods movement in Chennai."),
    ("TRUCKER_STRIKE", "Truckers went on strike and deliveries were halted in Chennai."),
    ("WAREHOUSE_STRIKE", "A warehouse strike stopped dispatch operations in Chennai."),
    ("POWER_OUTAGE", "A power outage stopped warehouse operations in Chennai."),
    ("FUEL_SHORTAGE", "A diesel shortage halted freight transport in Chennai."),
    ("CYCLONE", "The cyclone damaged roads and suspended cargo movement in Chennai."),
    ("FLOOD", "Flooding closed roads and suspended cargo movement in Chennai."),
    ("SEVERE_STORM", "A severe storm disrupted freight movement in Chennai."),
    ("EXTREME_RAINFALL", "Extreme rainfall flooded roads and suspended cargo movement in Chennai."),
    ("LANDSLIDE", "A landslide blocked the highway and halted freight near Chennai."),
    ("LANDSLIDE", "A landslide blocked the cargo corridor near Chennai."),
    ("FACTORY_SHUTDOWN", "The Chennai factory was shut down and production stopped."),
    ("PRODUCTION_HALT", "Production halted at the Chennai plant and shipments were suspended."),
    ("FACTORY_FIRE", "A fire damaged the Chennai factory and production stopped."),
    ("CYBER_LOGISTICS_DISRUPTION", "A cyberattack disrupted warehouse routing and halted freight in Chennai."),
    ("CURFEW", "A curfew suspended freight transport in Chennai."),
    ("BORDER_RESTRICTION", "The border was closed and cross-border freight suspended."),
    ("CUSTOMS_DISRUPTION", "Customs processing was suspended and freight clearance halted."),
    ("FREIGHT_RESTRICTION", "Freight movement was restricted and cargo transport suspended."),
    ("PRODUCT_RECALL", "A product recall halted distribution of the affected device in Chennai."),
    ("PRODUCTION_SUSPENSION", "Production was suspended at the Chennai plant."),
    ("COMPONENT_SHORTAGE", "A component shortage halted electronics production in Chennai."),
    ("SEMICONDUCTOR_SHORTAGE", "A semiconductor shortage stopped chip production in Chennai."),
    ("SUPPLY_SHORTAGE", "A raw material shortage halted manufacturing in Chennai."),
    ("SHIPPING_DELAY", "Shipping delays disrupted cargo delivery in Chennai."),
]


@pytest.mark.parametrize(("family", "text"), OPERATIONAL_CASES)
def test_disruption_families_require_and_recognize_direct_evidence(family, text):
    result = classify_news_evidence(text, "", family)
    assert result["event_family"] == family
    assert result["evidence_level"] == OPERATIONAL_DISRUPTION
    assert result["operationally_relevant"] is True
    assert result["directness"] == "DIRECT_OPERATIONAL"
    assert result["impact_type"]
    assert "explicitly reports" in result["classification_reason"]


def test_registry_detection_creates_structured_operational_event(graph):
    event = article_to_events(
        {
            "title": "Chennai warehouse closed today",
            "description": "The Chennai warehouse was closed and dispatch operations stopped.",
            "published_at": "2026-10-10T11:00:00+00:00",
        },
        Path("current_news.json"),
        recognition_aliases(graph),
    )[0]
    assert event["Event_Family"] == "WAREHOUSE_CLOSURE"
    assert event["Evidence_Level"] == OPERATIONAL_DISRUPTION
    assert event["Impact_Type"] == "WAREHOUSE_OPERATION"
    assert event["Evidence_Directness"] == "DIRECT_OPERATIONAL"
    assert event["Operationally_Relevant"] is True
    assert event["Classification_Reason"]
    event.update({
        "event_start": "2026-10-10T10:00:00+00:00",
        "event_end": "2026-10-10T14:00:00+00:00",
    })
    graph.add_node("TEST_EVENT", node_type="Event")
    _event_locations(graph, "TEST_EVENT", event)
    assert graph.has_edge("TEST_EVENT", "WH_CHENNAI")
    assert graph["TEST_EVENT"]["WH_CHENNAI"]["relationship"] == "AFFECTS"


@pytest.mark.parametrize(("text", "family"), [
    ("Highway closure halted truck movement in Chennai.", "ROAD_CLOSURE"),
    ("Freight traffic was suspended after the bridge closed near Chennai.", "BRIDGE_CLOSURE"),
    ("A landslide blocked the cargo corridor near Chennai.", "LANDSLIDE"),
    ("A transport strike stopped goods movement in Chennai.", "TRANSPORT_STRIKE"),
    ("Highway restriction banned commercial vehicles in Chennai.", "ROAD_CLOSURE"),
    ("Port access roads were closed to cargo vehicles in Chennai.", "ROAD_CLOSURE"),
])
def test_direct_freight_transport_disruptions_are_operational_and_city_resolved(graph, text, family):
    event = article_to_events({
        "title": text,
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    assert event["Event_Family"] == family
    assert event["Evidence_Level"] == OPERATIONAL_DISRUPTION
    assert event["Operationally_Relevant"] is True
    assert event["Location"] == "Chennai"
    assert event["Location_Scope"] == "LOCALIZED"
    event.update({
        "event_start": "2026-10-10T10:00:00+00:00",
        "event_end": "2026-10-10T14:00:00+00:00",
    })
    graph.add_node(f"POSITIVE_{family}", node_type="Event")
    _event_locations(graph, f"POSITIVE_{family}", event)
    assert graph.has_edge(f"POSITIVE_{family}", "WH_CHENNAI")
    assert not graph.has_edge(f"POSITIVE_{family}", "WH_MUMBAI")


@pytest.mark.parametrize("text", [
    "A road closed and commuters faced delays in Chennai.",
    "A flyover closure caused traffic snarls in Chennai.",
    "Motorists faced congestion near Chennai.",
    "Passenger bus delays were reported in Chennai.",
    "Tourist traffic disruption was reported in Chennai.",
    "A road accident affected commuters in Chennai.",
    "Road closure delayed commuters heading to Gujarat.",
])
def test_commuter_only_transport_events_are_never_operational(graph, text):
    events = article_to_events({
        "title": text,
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))
    for index, event in enumerate(events):
        assert event["Evidence_Level"] != OPERATIONAL_DISRUPTION
        assert event["Operationally_Relevant"] is False
        if event["Event_Family"] in {"ROAD_CLOSURE", "BRIDGE_CLOSURE", "TRAFFIC_DISRUPTION"}:
            assert event["Evidence_Level"] == POTENTIAL_DISRUPTION
        event.update({
            "event_start": "2026-10-10T10:00:00+00:00",
            "event_end": "2026-10-10T14:00:00+00:00",
        })
        graph.add_node(f"COMMUTER_ONLY_{index}", node_type="Event")
        _event_locations(graph, f"COMMUTER_ONLY_{index}", event)
    assert not any(
        attrs.get("relationship") == "AFFECTS"
        for _, _, attrs in graph.edges(data=True)
    )


def test_saved_october_13_flyover_article_is_non_operational_without_waiting_for_unknown_time(graph):
    events = article_to_events({
        "title": "Structural defects shut new flyover, commuters heading to Nashik, Gujarat face snarls",
        "description": (
            "Delays are plaguing commuters on the Ghodbunder highway as the Kasarwadavli flyover "
            "remains closed due to structural flaws uncovered last month. Inspections prompted the "
            "shutdown, leading to heavy Gaimukh-bound traffic being redirected to adjacent slip roads,\u2026"
        ),
        "published_at": "2026-10-12T05:51:55Z",
        "source_channel": "NEWS_API",
    }, Path("current_news.json"), recognition_aliases(graph))
    event = next(item for item in events if item["Event_Family"] == "ROAD_CLOSURE")
    assert event["Event_Family"] == "ROAD_CLOSURE"
    assert event["Evidence_Level"] == POTENTIAL_DISRUPTION
    assert event["Operationally_Relevant"] is False
    assert "no direct freight, cargo, delivery" in event["Classification_Reason"]
    assert event["Location"] == "Unknown"
    assert {mention["role"] for mention in event["Location_Mentions"]} == {"DESTINATION"}

    event.update({
        "event_start": "2026-10-10T10:00:00+00:00",
        "event_end": "2026-10-10T14:00:00+00:00",
    })
    assert not is_currently_eligible(event)
    graph.add_node("OCT13_FLYOVER_EVENT", node_type="Event")
    _event_locations(graph, "OCT13_FLYOVER_EVENT", event)
    for index, signal in enumerate(events):
        graph.add_node(f"OCT13_FLYOVER_SIGNAL_{index}", node_type="Event")
        _event_locations(graph, f"OCT13_FLYOVER_SIGNAL_{index}", signal)
    assert not any(
        attrs.get("relationship") == "AFFECTS"
        for _, _, attrs in graph.edges(data=True)
    )


def test_commuter_article_does_not_propagate_product_or_supplier_risk(graph):
    event = article_to_events({
        "title": "Structural defects shut new flyover, commuters heading to Nashik, Gujarat face snarls",
        "description": (
            "Delays are plaguing commuters on the Ghodbunder highway as the Kasarwadavli flyover "
            "remains closed due to structural flaws uncovered last month."
        ),
        "published_at": "2026-10-12T05:51:55Z",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    graph.add_node("PRODUCT:CHENNAI", node_type="Product", name="Stored Product")
    graph.add_node("SUPPLIER:CHENNAI", node_type="Supplier", name="Supplier")
    graph.add_node("BRAND:CHENNAI", node_type="Brand", name="Brand")
    graph.add_edge("PRODUCT:CHENNAI", "WH_CHENNAI", relationship="STORED_AT")
    graph.add_edge("SUPPLIER:CHENNAI", "BRAND:CHENNAI", relationship="SUPPLIES")
    graph.add_edge("BRAND:CHENNAI", "PRODUCT:CHENNAI", relationship="PRODUCES")
    event.update({
        "event_start": "2026-10-10T10:00:00+00:00",
        "event_end": "2026-10-10T14:00:00+00:00",
    })
    graph.add_node("OCT13_RISK_EVENT", node_type="Event")
    _event_locations(graph, "OCT13_RISK_EVENT", event)
    graph.nodes["OCT13_RISK_EVENT"].update({
        "event_status": "ACTIVE",
        "operational_current": "true",
        "operationally_relevant": "false",
    })
    assert not graph.has_edge("OCT13_RISK_EVENT", "WH_CHENNAI")
    assert not graph.has_edge("PRODUCT:CHENNAI", "OCT13_RISK_EVENT")
    assert not graph.has_edge("SUPPLIER:CHENNAI", "OCT13_RISK_EVENT")
    baseline = pd.DataFrame([
        {"Node_ID": node_id, "Node_Type": node_type, "Node_Name": node_id,
         "Risk_Score": "0.1", "Risk_Level": "Low", "Connected_Events": "0",
         "Structural_Baseline": "0.1", "Graph_Degree": "1", "Connected_Products": "1"}
        for node_id, node_type in (
            ("PRODUCT:CHENNAI", "Product"),
            ("SUPPLIER:CHENNAI", "Supplier"),
            ("WH_CHENNAI", "Warehouse"),
        )
    ])
    risk = _current_risk(graph, pd.DataFrame(), baseline_override=baseline)
    assert set(risk["Connected_Events"].astype(str)) == {"0"}
    assert set(risk["Event_Exposure_Component"].astype(str)) == {"0.0"}


def test_destination_state_is_not_used_as_transport_disruption_location(graph):
    event = article_to_events({
        "title": "Road closure stopped truck movement for commuters heading to Nashik, Gujarat.",
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    assert event["Evidence_Level"] == OPERATIONAL_DISRUPTION
    assert event["Location"] == "Unknown"
    assert event["Location_Scope"] == "UNRESOLVED"
    assert any(
        mention == {"location": "Gujarat", "role": "DESTINATION"}
        for mention in event["Location_Mentions"]
    )
    event.update({
        "event_start": "2026-10-10T10:00:00+00:00",
        "event_end": "2026-10-10T14:00:00+00:00",
    })
    graph.add_node("DESTINATION_ONLY_EVENT", node_type="Event")
    _event_locations(graph, "DESTINATION_ONLY_EVENT", event)
    assert not any(
        attrs.get("relationship") == "AFFECTS"
        for _, _, attrs in graph.edges(data=True)
    )


def test_explicit_statewide_transport_event_may_resolve_state_locations(graph):
    event = article_to_events({
        "title": "All major highways across Gujarat closed and truck movement suspended.",
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    assert event["Location"] == "Gujarat"
    payload = _event_payload(event)
    assert payload["location_scope"] == "STATEWIDE"
    assert payload["location_mentions"] == [{"location": "Gujarat", "role": "DISRUPTION_LOCATION"}]


def test_statewide_scope_must_be_about_the_disruption_not_an_unrelated_mention(graph):
    event = article_to_events({
        "title": "Gujarat held statewide elections. A road closure halted truck movement near Ahmedabad.",
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    assert event["Evidence_Level"] == OPERATIONAL_DISRUPTION
    assert event["Location"] == "Ahmedabad"
    assert event["Location_Scope"] == "LOCALIZED"


def test_non_transport_statewide_facility_event_is_explicitly_scoped(graph):
    event = article_to_events({
        "title": "Statewide warehouse shutdown across Gujarat halted distribution operations.",
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    assert event["Event_Family"] == "WAREHOUSE_CLOSURE"
    assert event["Location"] == "Gujarat"
    assert event["Location_Scope"] == "STATEWIDE"


def test_state_location_does_not_fan_out_without_explicit_statewide_evidence(graph):
    local = article_to_events({
        "title": "Gujarat highway closed and truck movement stopped.",
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    assert local["Evidence_Level"] == OPERATIONAL_DISRUPTION
    assert local["Location"] == "Unknown"
    assert local["Location_Scope"] == "UNRESOLVED"

    statewide = article_to_events({
        "title": "All major highways across Gujarat closed and truck movement suspended.",
        "description": "",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"), recognition_aliases(graph))[0]
    assert statewide["Evidence_Level"] == OPERATIONAL_DISRUPTION
    assert statewide["Location"] == "Gujarat"
    assert statewide["Location_Scope"] == "STATEWIDE"
    statewide.update({
        "event_start": "2026-10-10T10:00:00+00:00",
        "event_end": "2026-10-10T14:00:00+00:00",
    })
    graph.add_node("STATEWIDE_EVENT", node_type="Event")
    _event_locations(graph, "STATEWIDE_EVENT", statewide)
    targets = {
        target for _, target, attrs in graph.out_edges("STATEWIDE_EVENT", data=True)
        if attrs.get("relationship") == "AFFECTS"
        and graph.nodes[target].get("node_type") == "Warehouse"
    }
    assert targets == {"WH_AHMEDABAD", "WH_SURAT"}


def test_event_payload_exposes_classification_reason_to_consumers():
    payload = _event_payload({
        "Event_ID": 42,
        "Event_Type": "Warehouse Closure",
        "Event_Class": "ACTUAL_DISRUPTION",
        "Evidence_Level": OPERATIONAL_DISRUPTION,
        "Event_Family": "WAREHOUSE_CLOSURE",
        "Evidence_Directness": "DIRECT_OPERATIONAL",
        "Classification_Reason": "Source explicitly reports warehouse closure.",
        "matched_operational_evidence": "warehouse was closed",
        "Operationally_Relevant": True,
    })
    assert payload["evidence_level"] == OPERATIONAL_DISRUPTION
    assert payload["event_family"] == "WAREHOUSE_CLOSURE"
    assert payload["evidence_directness"] == "DIRECT_OPERATIONAL"
    assert payload["classification_reason"] == "Source explicitly reports warehouse closure."
    assert payload["matched_operational_evidence"] == "warehouse was closed"


@pytest.mark.parametrize("text", [
    "Light rain in Chennai.",
    "Ordinary rain forecast for Chennai tomorrow.",
    "Company earnings rose as the share price increased.",
    "A company launched a new product in Chennai.",
    "A generic protest took place in Chennai.",
    "Stock-market shares fell during trading.",
    "Political leaders held a debate in Delhi.",
    "A cyberattack disrupted hospital systems in Chennai.",
    "A fire damaged a private residence in Chennai.",
    "Traffic was slow on city streets, with no freight disruption.",
])
def test_unrelated_or_informational_news_never_qualifies_as_operational(graph, text):
    events = article_to_events(
        {
            "title": text,
            "description": text,
            "published_at": "2026-10-10T11:00:00+00:00",
        },
        Path("current_news.json"),
        recognition_aliases(graph),
    )
    assert all(event["Operationally_Relevant"] is False for event in events)
    for index, event in enumerate(events):
        graph.add_node(f"NEGATIVE_EVENT_{index}", node_type="Event")
        event.update({
            "event_start": "2026-10-10T10:00:00+00:00",
            "event_end": "2026-10-10T14:00:00+00:00",
        })
        _event_locations(graph, f"NEGATIVE_EVENT_{index}", event)
    assert not any(data.get("relationship") == "AFFECTS" for _, _, data in graph.edges(data=True))


def test_road_closure_with_india_only_scope_does_not_resolve_warehouse(graph):
    event = article_to_events(
        {
            "title": "Road closure across India suspended freight movement.",
            "description": "Road closure across India suspended freight movement.",
            "published_at": "2026-10-10T11:00:00+00:00",
        },
        Path("current_news.json"),
        recognition_aliases(graph),
    )[0]
    assert event["Operationally_Relevant"] is True
    event.update({
        "event_start": "2026-10-10T10:00:00+00:00",
        "event_end": "2026-10-10T14:00:00+00:00",
    })
    graph.add_node("INDIA_ONLY_EVENT", node_type="Event")
    _event_locations(graph, "INDIA_ONLY_EVENT", event)
    assert not [
        target for _, target, data in graph.out_edges("INDIA_ONLY_EVENT", data=True)
        if data.get("relationship") == "AFFECTS"
    ]


def test_potential_weather_language_does_not_become_operational():
    result = classify_news_evidence(
        "Red alert for severe rainfall in Chennai.",
        "Transport disruption is possible, but freight is still moving.",
        "EXTREME_RAINFALL",
    )
    assert result["evidence_level"] == POTENTIAL_DISRUPTION
    assert result["operationally_relevant"] is False
    assert result["directness"] == "EXPECTED_OPERATIONAL_IMPACT"


@pytest.mark.parametrize("text", [
    "A fire was reported at the Chennai warehouse.",
    "A Chennai warehouse closure could disrupt deliveries.",
    "The Chennai warehouse is not closed.",
    "A Chennai warehouse fire occurred. Separately, a road was closed elsewhere.",
])
def test_topic_alone_or_unrelated_effect_does_not_qualify_as_operational(text):
    result = classify_news_evidence(text, "")
    assert result["evidence_level"] != OPERATIONAL_DISRUPTION
    assert result["operationally_relevant"] is False


def test_port_congestion_without_explicit_cargo_effect_remains_potential():
    result = classify_news_evidence("Port congestion reported in Chennai.", "")
    assert result["evidence_level"] == POTENTIAL_DISRUPTION
    assert result["operationally_relevant"] is False


def test_potential_event_cannot_become_currently_eligible_even_if_flag_is_stale():
    from datetime import timedelta

    start = datetime.now(timezone.utc) - timedelta(minutes=1)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    active_event = {
        "Operationally_Relevant": True,
        "event_start": start.isoformat(),
        "event_end": end.isoformat(),
    }
    assert is_currently_eligible({
        **active_event,
        "Evidence_Level": OPERATIONAL_DISRUPTION,
    })
    assert not is_currently_eligible({
        **active_event,
        "Evidence_Level": POTENTIAL_DISRUPTION,
    })


def test_weather_observation_and_ordinary_forecast_are_signals_only():
    observation = weather_to_events("Chennai", {
        "weather": [{"id": 500, "main": "Rain", "description": "light rain"}],
        "dt": 1791626400,
        "location_validation_status": "VALIDATED",
    }, Path("current_weather.json"))[0]
    forecast = weather_to_events("Chennai", {
        "weather": [{"id": 501, "main": "Rain", "description": "moderate rain"}],
        "dt": 1791626400,
        "location_validation_status": "VALIDATED",
        "alerts": [],
    }, Path("current_weather.json"))[0]
    assert observation["Evidence_Level"] == INFORMATIONAL_SIGNAL
    assert forecast["Evidence_Level"] == INFORMATIONAL_SIGNAL
    assert observation["Operationally_Relevant"] is False
    assert forecast["Operationally_Relevant"] is False


def test_news_light_rain_and_ordinary_rain_keep_non_severe_event_types():
    light_rain = article_to_events({
        "title": "Light rain in Chennai",
        "description": "Light rain expected today.",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"))[0]
    ordinary_rain = article_to_events({
        "title": "Rain forecast in Chennai",
        "description": "Ordinary rain is expected today.",
        "published_at": "2026-10-10T11:00:00+00:00",
    }, Path("current_news.json"))[0]
    assert light_rain["Event_Type"] == "Light Rain"
    assert ordinary_rain["Event_Type"] == "Rain"
    assert light_rain["Evidence_Level"] == INFORMATIONAL_SIGNAL
    assert ordinary_rain["Evidence_Level"] == INFORMATIONAL_SIGNAL
    assert not light_rain["Operationally_Relevant"]
    assert not ordinary_rain["Operationally_Relevant"]


def test_validated_severe_weather_alert_is_potential_without_direct_effect():
    event = weather_to_events("Chennai", {
        "weather": [{"id": 502, "main": "Rain", "description": "heavy rain"}],
        "dt": 1791626400,
        "source_channel": "WEATHERAPI",
        "location_validation_status": "VALIDATED",
        "alerts": [{"event": "Red alert for very heavy rainfall"}],
    }, Path("current_weather.json"))[0]
    assert event["Evidence_Level"] == POTENTIAL_DISRUPTION
    assert event["Operationally_Relevant"] is False


def test_validated_weather_alert_with_explicit_logistics_impact_is_operational():
    event = weather_to_events("Chennai", {
        "weather": [{"id": 502, "main": "Rain", "description": "heavy rain"}],
        "dt": 1791626400,
        "source_channel": "WEATHERAPI",
        "location_validation_status": "VALIDATED",
        "alerts": [{
            "event": "Flooding closed roads and suspended cargo transport",
            "effective": "2026-10-10T10:00:00+00:00",
            "expires": "2026-10-10T14:00:00+00:00",
        }],
    }, Path("current_weather.json"))[0]
    assert event["Evidence_Level"] == OPERATIONAL_DISRUPTION
    assert event["Operationally_Relevant"] is True
    assert "validated weather alert explicitly reports" in event["Classification_Reason"]
