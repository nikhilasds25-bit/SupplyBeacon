"""Feature extraction for graph-based supply chain risk prediction."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx

from risk_prediction.risk_models import severity_value
from risk_prediction.utils import GRAPHML_PATH, normalize_text, safe_float

MAX_EVENT_DISTANCE = 4

NODE_TYPES_OF_INTEREST = {
    "Supplier",
    "Brand",
    "Product",
    "Warehouse",
    "Location",
    "Category",
}

EVENT_EDGE_RELATIONSHIP = "AFFECTS"


def load_graph() -> nx.Graph:
    """Load the existing supply chain graph from GraphML."""
    return nx.read_graphml(str(GRAPHML_PATH))


def get_event_nodes(graph: nx.Graph) -> List[str]:
    """Return the list of event node identifiers."""
    return [node for node, attrs in graph.nodes(data=True) if attrs.get("node_type") == "Event"]


def event_distance_weight(distance: int) -> float:
    """Assign a score weight to event proximity based on graph distance."""
    if distance <= 0:
        return 0.0
    return max(0.0, 1.0 - 0.2 * (distance - 1))


def compute_event_distances(graph: nx.Graph, event_nodes: List[str]) -> Dict[str, Dict[str, int]]:
    """Compute shortest path distances from each event node to other nodes."""
    undirected = graph.to_undirected()
    event_distances: Dict[str, Dict[str, int]] = {}
    for event in event_nodes:
        if event in undirected:
            event_distances[event] = nx.single_source_shortest_path_length(undirected, event, cutoff=MAX_EVENT_DISTANCE)
        else:
            event_distances[event] = {}
    return event_distances


def get_two_hop_neighbors(undirected: nx.Graph, node_id: str) -> tuple[set[str], set[str]]:
    """Return the first- and second-hop neighbor sets for a node."""
    first_hop = set(undirected[node_id]) if node_id in undirected else set()
    second_hop: set[str] = set()
    for neighbor in first_hop:
        second_hop.update(undirected[neighbor])
    second_hop.discard(node_id)
    second_hop.difference_update(first_hop)
    return first_hop, second_hop


def get_connected_nodes_by_type(
    graph: nx.Graph, undirected: nx.Graph, node_id: str, node_type: str
) -> int:
    """Count connected nodes of a given type within one or two hops."""
    if node_type not in NODE_TYPES_OF_INTEREST or node_id not in graph:
        return 0
    first_hop, second_hop = get_two_hop_neighbors(undirected, node_id)
    connected = first_hop | second_hop
    return sum(1 for target in connected if graph.nodes[target].get("node_type") == node_type)


def get_direct_event_ids(graph: nx.Graph, node_id: str) -> List[str]:
    """Return event IDs that are directly connected to a node via AFFECTS."""
    direct_events = set()
    if node_id not in graph:
        return []
    for source, target, attrs in graph.in_edges(node_id, data=True):
        if attrs.get("relationship") == EVENT_EDGE_RELATIONSHIP and graph.nodes[source].get("node_type") == "Event":
            direct_events.add(source)
    for source, target, attrs in graph.out_edges(node_id, data=True):
        if attrs.get("relationship") == EVENT_EDGE_RELATIONSHIP and graph.nodes[target].get("node_type") == "Event":
            direct_events.add(target)
    return sorted(direct_events)


def compute_exposure_metrics(
    graph: nx.Graph,
    node_id: str,
    event_nodes: List[str],
    event_distances: Dict[str, Dict[str, int]],
) -> Dict[str, Optional[float]]:
    """Compute event exposure metrics for a node from nearby event nodes."""
    if node_id not in graph:
        return {
            "event_exposure": 0.0,
            "connected_events": 0,
            "direct_event_count": 0,
            "min_event_distance": None,
            "event_ids": [],
        }

    connected_events = []
    for event in event_nodes:
        distance = event_distances.get(event, {}).get(node_id)
        if distance is not None and distance <= MAX_EVENT_DISTANCE:
            connected_events.append((event, distance))

    if not connected_events:
        return {
            "event_exposure": 0.0,
            "connected_events": 0,
            "direct_event_count": 0,
            "min_event_distance": None,
            "event_ids": [],
        }

    weights = [event_distance_weight(distance) for _, distance in connected_events]
    event_exposure = min(1.0, sum(weights) / 2.0)
    direct_event_count = sum(1 for _, distance in connected_events if distance == 1)
    min_distance = min(distance for _, distance in connected_events)
    return {
        "event_exposure": round(event_exposure, 3),
        "connected_events": len(connected_events),
        "direct_event_count": direct_event_count,
        "min_event_distance": min_distance,
        "event_ids": [event for event, _ in connected_events],
    }


def get_event_values(graph: nx.Graph, event_ids: List[str]) -> Dict[str, object]:
    """Extract aggregate values from the connected event nodes."""
    if not event_ids:
        return {
            "max_severity": 0.0,
            "max_severity_label": "Unknown",
            "average_confidence": 0.0,
            "business_risk": 0.0,
        }

    severities = []
    confidences: List[float] = []
    business_risks: List[float] = []
    severity_labels = []
    for event_id in event_ids:
        attrs = graph.nodes.get(event_id, {})
        severity_label = normalize_text(attrs.get("severity", "Unknown")) or "Unknown"
        severity_labels.append(severity_label.title())
        severities.append(severity_value(severity_label))
        confidences.append(safe_float(attrs.get("confidence", 0.0)))
        business_risks.append(safe_float(attrs.get("risk_score", 0.0)))

    return {
        "max_severity": max(severities) if severities else 0.0,
        "max_severity_label": max(severity_labels, key=lambda label: severity_value(label)),
        "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "business_risk": round(max(business_risks) if business_risks else 0.0, 3),
    }


def build_node_features(
    graph: nx.Graph,
    undirected: nx.Graph,
    node_id: str,
    event_nodes: List[str],
    event_distances: Dict[str, Dict[str, int]],
) -> Dict[str, object]:
    """Build the risk feature dictionary for a single graph node."""
    attrs = graph.nodes.get(node_id, {})
    node_type = attrs.get("node_type", "Unknown")
    node_name = normalize_text(attrs.get("name", ""))

    exposure_metrics = compute_exposure_metrics(graph, node_id, event_nodes, event_distances)
    event_data = get_event_values(graph, exposure_metrics["event_ids"])

    graph_degree = graph.degree(node_id)
    product_count = get_connected_nodes_by_type(graph, undirected, node_id, "Product")
    supplier_count = get_connected_nodes_by_type(graph, undirected, node_id, "Supplier")
    warehouse_count = get_connected_nodes_by_type(graph, undirected, node_id, "Warehouse")
    location_count = get_connected_nodes_by_type(graph, undirected, node_id, "Location")

    return {
        "Node_ID": node_id,
        "Node_Type": node_type,
        "Node_Name": node_name,
        "Graph_Degree": int(graph_degree),
        "Connected_Products": int(product_count),
        "Connected_Suppliers": int(supplier_count),
        "Connected_Warehouses": int(warehouse_count),
        "Connected_Locations": int(location_count),
        "Connected_Events": int(exposure_metrics["connected_events"]),
        "Event_Exposure": float(exposure_metrics["event_exposure"]),
        "Direct_Event_Count": int(exposure_metrics["direct_event_count"]),
        "Min_Event_Distance": exposure_metrics["min_event_distance"],
        "Max_Event_Severity": event_data["max_severity_label"],
        "Average_Event_Confidence": float(event_data["average_confidence"]),
        "Business_Impact_Risk": float(event_data["business_risk"]),
        "max_severity": float(event_data["max_severity"]),
        "average_confidence": float(event_data["average_confidence"]),
        "business_risk": float(event_data["business_risk"]),
        "product_connectivity": round(min(1.0, product_count / 20.0), 3),
        "degree_connectivity": round(min(1.0, graph_degree / 20.0), 3),
    }


def build_feature_rows(graph: nx.Graph) -> List[Dict[str, object]]:
    """Build risk features for all relevant nodes in the graph."""
    event_nodes = get_event_nodes(graph)
    event_distances = compute_event_distances(graph, event_nodes)
    undirected = graph.to_undirected()
    rows: List[Dict[str, object]] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "Event":
            continue
        if attrs.get("node_type") not in NODE_TYPES_OF_INTEREST:
            continue
        rows.append(build_node_features(graph, undirected, node_id, event_nodes, event_distances))
    return rows
