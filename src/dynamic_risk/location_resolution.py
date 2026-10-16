"""Resolve event geography only through warehouse LOCATED_IN graph evidence."""
from __future__ import annotations

import re
from typing import Any

import networkx as nx


CITY_ALIASES = {
    "bangalore": "bengaluru",
    "gurgaon": "gurugram",
    "calcutta": "kolkata",
    "vizag": "visakhapatnam",
    "trivandrum": "thiruvananthapuram",
}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value or "").casefold())).strip()


def city_key(value: Any) -> str:
    key = normalize(value)
    return CITY_ALIASES.get(key, key)


def location_index(graph: nx.Graph) -> list[dict[str, str]]:
    """Include only locations actually connected to a warehouse by LOCATED_IN."""
    entries = []
    for warehouse_id, warehouse in graph.nodes(data=True):
        if warehouse.get("node_type") != "Warehouse":
            continue
        for location_id in graph.neighbors(warehouse_id):
            edge = graph.get_edge_data(warehouse_id, location_id) or graph.get_edge_data(location_id, warehouse_id) or {}
            if edge.get("relationship") != "LOCATED_IN" or graph.nodes[location_id].get("node_type") != "Location":
                continue
            location = graph.nodes[location_id]
            entries.append({
                "warehouse_id": str(warehouse_id),
                "location_id": str(location_id),
                "city": str(location.get("city") or warehouse.get("city") or location.get("name") or "").strip(),
                "state": str(location.get("state") or warehouse.get("state") or "").strip(),
                "country": str(location.get("country") or warehouse.get("country") or "India").strip(),
            })
    return entries


def recognition_aliases(graph: nx.Graph) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for entry in location_index(graph):
        for field in ("city", "state", "country"):
            value = entry[field]
            if value:
                aliases.setdefault(value, [value])
    for alias, canonical in CITY_ALIASES.items():
        matching = next((city for city in aliases if normalize(city) == canonical), None)
        if matching:
            aliases[matching].append(alias)
    return aliases


def resolve_event_location(
    graph: nx.Graph,
    location_text: Any,
    *,
    allow_state: bool = False,
) -> dict[str, Any]:
    """Prefer an explicit city; only expand a state when evidence permits it."""
    tokens = [token.strip() for token in str(location_text or "").split(",") if token.strip()]
    entries = location_index(graph)
    cities = [entry for entry in entries if any(city_key(entry["city"]) == city_key(token) for token in tokens)]
    if cities:
        matched = [token for token in tokens if any(city_key(entry["city"]) == city_key(token) for entry in cities)]
        return _result(cities, "EXACT_LOCATION", ", ".join(matched))
    if not allow_state:
        return _result([], "UNRESOLVED", "")
    states = [entry for entry in entries if any(normalize(entry["state"]) == normalize(token) for token in tokens)]
    if states:
        matched = [token for token in tokens if any(normalize(entry["state"]) == normalize(token) for entry in states)]
        return _result(states, "STATE_LOCATION", ", ".join(matched))
    return _result([], "UNRESOLVED", "")


def _result(entries: list[dict[str, str]], basis: str, matched_text: str) -> dict[str, Any]:
    return {
        "basis": basis,
        "confidence": "HIGH" if basis == "EXACT_LOCATION" else "MEDIUM" if basis == "STATE_LOCATION" else "NONE",
        "matched_text": matched_text,
        "location_ids": sorted({entry["location_id"] for entry in entries}),
        "warehouse_ids": sorted({entry["warehouse_id"] for entry in entries}),
        "entries": entries,
    }
