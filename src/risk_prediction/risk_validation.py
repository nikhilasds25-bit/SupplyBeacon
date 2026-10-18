"""Validate and explain event propagation used by the current risk baseline.

Produces reports showing direct, 1-hop, 2-hop, and MAX_DISTANCE-based exposure for each event,
plus human-readable explanations for top risky nodes.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import pandas as pd

from graph.graph_schema import (
    EDGE_TYPE_AFFECTS,
    EDGE_TYPE_SUPPLIES,
    EDGE_TYPE_PRODUCES,
    EDGE_TYPE_BELONGS_TO,
    EDGE_TYPE_STORED_AT,
    EDGE_TYPE_LOCATED_IN,
    NODE_TYPE_EVENT,
)
from risk_prediction.risk_features import MAX_EVENT_DISTANCE
from risk_prediction.risk_features import get_event_nodes
from risk_prediction.risk_features import compute_event_distances
from risk_prediction.risk_features import event_distance_weight
from risk_prediction.risk_features import get_direct_event_ids
from risk_prediction.risk_features import get_event_values
from risk_prediction.risk_features import get_connected_nodes_by_type
from risk_prediction.risk_features import build_node_features
from risk_prediction.utils import GRAPHML_PATH, RISK_DIR, create_directory, normalize_text, save_json


LOG = logging.getLogger(__name__)

VALIDATION_DIR = RISK_DIR / "validation"
REPORT_JSON = VALIDATION_DIR / "risk_validation_report.json"
EVENT_REPORT_CSV = VALIDATION_DIR / "event_propagation_report.csv"
TOP_EXPLAIN_CSV = VALIDATION_DIR / "top_risk_explanations.csv"


PROPAGATION_EDGE_SET = {EDGE_TYPE_AFFECTS, EDGE_TYPE_SUPPLIES, EDGE_TYPE_PRODUCES, EDGE_TYPE_BELONGS_TO, EDGE_TYPE_STORED_AT, EDGE_TYPE_LOCATED_IN}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_graph() -> nx.Graph:
    if not GRAPHML_PATH.exists():
        raise FileNotFoundError(f"GraphML not found: {GRAPHML_PATH}")
    return nx.read_graphml(str(GRAPHML_PATH))


def neighbors_by_distance(graph: nx.Graph, source: str, max_distance: int) -> Dict[int, Set[str]]:
    undirected = graph.to_undirected()
    lengths = nx.single_source_shortest_path_length(undirected, source, cutoff=max_distance)
    buckets: Dict[int, Set[str]] = defaultdict(set)
    for node, dist in lengths.items():
        if node == source:
            continue
        buckets[dist].add(node)
    return buckets


def edge_relationships_along_path(graph: nx.Graph, path: List[str]) -> List[str]:
    rels: List[str] = []
    # path is list of node ids [n0,event,...,nk]
    for u, v in zip(path, path[1:]):
        data = graph.get_edge_data(u, v)
        # get_edge_data may be dict of keys -> attrs for MultiGraph/DiGraph
        if data is None:
            data = graph.get_edge_data(v, u)
            if data is None:
                rels.append("UNKNOWN")
                continue
        if isinstance(data, dict) and any(isinstance(vv, dict) for vv in data.values()):
            # Multi-edge
            attrs = list(data.values())[0]
        else:
            attrs = data
        rels.append(attrs.get("relationship", "UNKNOWN"))
    return rels


def propagation_edge_types_for_event(graph: nx.Graph, event: str, exposed_nodes: Set[str]) -> Set[str]:
    # collect relationship types used in shortest paths from event to exposed nodes (limit path length 4)
    rel_types: Set[str] = set()
    undirected = graph.to_undirected()
    for node in exposed_nodes:
        try:
            path = nx.shortest_path(undirected, source=event, target=node)
        except nx.NetworkXNoPath:
            continue
        # only consider paths up to MAX_EVENT_DISTANCE
        if len(path) - 1 > MAX_EVENT_DISTANCE:
            continue
        rels = edge_relationships_along_path(graph, path)
        rel_types.update(rels)
    return rel_types


def nodes_reachable_by_supply_chain(graph: nx.Graph, start_nodes: Set[str], max_distance: int = 4) -> Set[str]:
    """Reach nodes by walking only supply-chain relationship types.
    This takes the undirected view but only traverses edges whose relationship is in PROPAGATION_EDGE_SET.
    """
    undirected = graph.to_undirected()
    visited: Set[str] = set()
    frontier: Set[str] = set(start_nodes)
    dist = 0
    while frontier and dist < max_distance:
        next_frontier: Set[str] = set()
        for node in frontier:
            visited.add(node)
            for nbr in undirected[node]:
                # inspect original directed edges between node and nbr to find relationship types
                data = graph.get_edge_data(node, nbr) or graph.get_edge_data(nbr, node)
                if data is None:
                    continue
                # pick first relationship present
                found = False
                if isinstance(data, dict) and any(isinstance(vv, dict) for vv in data.values()):
                    for key_attrs in data.values():
                        if key_attrs.get("relationship") in PROPAGATION_EDGE_SET:
                            found = True
                            break
                else:
                    if data.get("relationship") in PROPAGATION_EDGE_SET:
                        found = True
                if found and nbr not in visited:
                    next_frontier.add(nbr)
        frontier = next_frontier
        dist += 1
    return visited - start_nodes


def analyze_events(graph: nx.Graph) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    event_nodes = get_event_nodes(graph)
    event_distances = compute_event_distances(graph, event_nodes)

    rows: List[Dict[str, Any]] = []
    overall_direct: Set[str] = set()
    overall_supply_exposed: Set[str] = set()
    overall_broad_reachable: Set[str] = set()

    for event in event_nodes:
        attrs = graph.nodes.get(event, {})
        event_id = event
        event_type = attrs.get("event_type", "Unknown")
        severity = attrs.get("severity", "Unknown")
        location = attrs.get("location", attrs.get("Affected_Location", attrs.get("Location", "Unknown")))
        affected_entity = attrs.get("affected_entity", attrs.get("Affected_Entity", "Unknown"))

        # directly connected nodes via AFFECTS (outgoing edges)
        direct_targets = set()
        for _, target, ed in graph.out_edges(event, data=True):
            if ed.get("relationship") == EDGE_TYPE_AFFECTS:
                direct_targets.add(target)
        # also include incoming AFFECTS? typically events have outgoing AFFECTS
        for source, _, ed in graph.in_edges(event, data=True):
            if ed.get("relationship") == EDGE_TYPE_AFFECTS:
                direct_targets.add(source)

        overall_direct.update(direct_targets)

        # 1-hop and 2-hop neighbors (undirected shortest path lengths)
        buckets = neighbors_by_distance(graph, event, max_distance=2)
        one_hop = buckets.get(1, set())
        two_hop = buckets.get(2, set())

        # Broad reachability (all nodes reachable via undirected traversal)
        undirected = graph.to_undirected()
        try:
            reachable = set(nx.node_connected_component(undirected, event))
        except Exception:
            reachable = {event}

        overall_broad_reachable.update(reachable - {event})

        # Current model exposure set: nodes within MAX_EVENT_DISTANCE (1..MAX)
        reach_within_max: Set[str] = set()
        distances = event_distances.get(event, {})
        for node, d in distances.items():
            if node == event:
                continue
            if d >= 1 and d <= MAX_EVENT_DISTANCE:
                reach_within_max.add(node)
        # supply-chain exposure: from direct_targets expand along supply-chain relationships
        supply_exposed = nodes_reachable_by_supply_chain(graph, direct_targets, max_distance=MAX_EVENT_DISTANCE)
        overall_supply_exposed.update(supply_exposed)
        overall_direct.update(direct_targets)
        # collect relationship types used for propagation
        prop_rels = propagation_edge_types_for_event(graph, event, reach_within_max)

        rows.append(
            {
                "Event_ID": event_id,
                "Event_Type": event_type,
                "Severity": severity,
                "Location": location,
                "Affected_Entity": affected_entity,
                "Directly_Connected_Nodes": len(direct_targets),
                "One_Hop_Nodes": len(one_hop),
                "Two_Hop_Nodes": len(two_hop),
                "Total_Reachable_Nodes": len(reachable) - 1,
                "Total_Exposed_Nodes": len(reach_within_max),
                "Supply_Exposed_Nodes": len(supply_exposed),
                "Propagation_Path_Types": ",".join(sorted(prop_rels)),
                "Propagation_Status": "computed",
            }
        )

    report = {
        "total_nodes": graph.number_of_nodes(),
        "total_events": len(event_nodes),
        "overall_directly_exposed_nodes": len(overall_direct),
        "overall_supply_exposed_nodes": len(overall_supply_exposed),
        "overall_broad_reachable_nodes": len(overall_broad_reachable),
        "model_exposure_count": sum(1 for n, a in graph.nodes(data=True) if n not in event_nodes and any(event_distances.get(ev, {}).get(n) for ev in event_nodes)),
        "model_exposure_percentage": round(100.0 * sum(1 for n, a in graph.nodes(data=True) if n not in event_nodes and any(event_distances.get(ev, {}).get(n) for ev in event_nodes)) / max(1, graph.number_of_nodes()), 2),
        "max_event_distance": MAX_EVENT_DISTANCE,
    }

    return rows, report


def explain_top_nodes(graph: nx.Graph, top_n: int = 20) -> List[Dict[str, Any]]:
    # read risk scores CSV to find top nodes
    risk_csv = RISK_DIR / "risk_scores.csv"
    if not risk_csv.exists():
        raise FileNotFoundError("risk_scores.csv not found in data/risk_prediction")
    df = pd.read_csv(risk_csv, dtype=str, keep_default_na=False)
    df["Risk_Score"] = df["Risk_Score"].astype(float)
    top = df.sort_values(by="Risk_Score", ascending=False).head(top_n)

    explanations: List[Dict[str, Any]] = []
    undirected = graph.to_undirected()
    event_nodes = get_event_nodes(graph)
    event_distances = compute_event_distances(graph, event_nodes)

    for _, row in top.iterrows():
        node_id = row["Node_ID"]
        node_type = row["Node_Type"]
        name = row["Node_Name"]
        score = row["Risk_Score"]
        level = row["Risk_Level"]
        connected_events = []
        paths_info: List[str] = []
        # find events within MAX_EVENT_DISTANCE
        for ev in event_nodes:
            dist = event_distances.get(ev, {}).get(node_id)
            if dist is None or dist > MAX_EVENT_DISTANCE:
                continue
            connected_events.append(ev)
            try:
                path = nx.shortest_path(undirected, source=ev, target=node_id)
            except nx.NetworkXNoPath:
                path = []
            if path:
                rels = edge_relationships_along_path(graph, path)
                path_str = " -> ".join([f"{p}({graph.nodes[p].get('node_type')})" for p in path])
                paths_info.append(f"Path from {ev}: dist={len(path)-1}, rels={rels}, path={path_str}")
        # gather event severities and business risk
        ev_details: List[Dict[str, Any]] = []
        for ev in connected_events:
            attrs = graph.nodes.get(ev, {})
            ev_details.append({
                "event_id": ev,
                "event_type": attrs.get("event_type"),
                "severity": attrs.get("severity"),
                "confidence": safe_float(attrs.get("confidence", 0.0)),
                "business_risk": safe_float(attrs.get("risk_score", 0.0)),
            })
        # neighbors and relationships for evidence
        neighs = []
        for u, v, ed in graph.edges(node_id, data=True):
            neighs.append({"to": v, "relationship": ed.get("relationship")})
        for u, v, ed in graph.in_edges(node_id, data=True):
            neighs.append({"from": u, "relationship": ed.get("relationship")})

        explanations.append(
            {
                "Node_ID": node_id,
                "Node_Type": node_type,
                "Node_Name": name,
                "Risk_Score": score,
                "Risk_Level": level,
                "Connected_Events": connected_events,
                "Event_Details": ev_details,
                "Paths": paths_info,
                "Neighbor_Evidence_Sample": neighs[:10],
            }
        )
    return explanations


# safe float helper duplicated to avoid import cycles

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(str(value).strip())
        except Exception:
            return default


def analyze_unresolved_events(graph: nx.Graph) -> List[Dict[str, Any]]:
    # Look into business impact rows and event nodes to see why unresolved
    rows = []
    business_csv = Path(__file__).resolve().parents[2] / "data" / "processed" / "business_impacts" / "business_impact_events.csv"
    if not business_csv.exists():
        return rows
    df = pd.read_csv(business_csv, dtype=str, keep_default_na=False)
    for _, r in df.iterrows():
        location = str(r.get("Location", "")).strip()
        affected_entity = str(r.get("Affected_Entity", "")).strip()
        if (not location or location.lower() == "unknown") or (not affected_entity or affected_entity.lower() == "unknown"):
            # check whether a node exists with exact name match
            possible_match = None
            if location and location.lower() != "unknown":
                # check location node
                for n, a in graph.nodes(data=True):
                    if a.get("node_type") == "Location" and normalize_text(a.get("name", "")).lower() == normalize_text(location).lower():
                        possible_match = {"type": "Location", "node": n}
                        break
            if affected_entity and affected_entity.lower() != "unknown":
                for n, a in graph.nodes(data=True):
                    if a.get("node_type") in ("Brand", "Supplier") and normalize_text(a.get("name", "")).lower() == normalize_text(affected_entity).lower():
                        possible_match = {"type": a.get("node_type"), "node": n}
                        break
            rows.append(
                {
                    "Event_ID": r.get("Event_ID"),
                    "Location": location or "",
                    "Affected_Entity": affected_entity or "",
                    "Resolved_in_Graph": bool(possible_match),
                    "Resolved_Node": possible_match["node"] if possible_match else "",
                    "Resolved_Node_Type": possible_match["type"] if possible_match else "",
                }
            )
    return rows


def main() -> None:
    configure_logging()
    LOG.info("Starting risk propagation validation")
    try:
        graph = load_graph()
    except Exception as exc:
        LOG.exception("Failed to load graph: %s", exc)
        raise

    event_rows, report = analyze_events(graph)
    explanations = explain_top_nodes(graph, top_n=20)
    unresolved = analyze_unresolved_events(graph)

    create_directory(VALIDATION_DIR)
    # save event propagation csv
    pd.DataFrame(event_rows).to_csv(EVENT_REPORT_CSV, index=False, encoding="utf-8-sig")
    # save top node explanations
    pd.DataFrame(explanations).to_csv(TOP_EXPLAIN_CSV, index=False, encoding="utf-8-sig")
    # save overall json
    save_json(REPORT_JSON, report)
    # save unresolved events
    save_json(VALIDATION_DIR / "unresolved_events.json", unresolved)

    # Print concise console report
    LOG.info("Risk Validation Report")
    LOG.info("======================")
    LOG.info("Total nodes: %s", report["total_nodes"])
    LOG.info("Total events: %s", report["total_events"])
    LOG.info("Directly exposed nodes (union): %s", report["overall_directly_exposed_nodes"])
    LOG.info("Supply-chain exposed nodes (union): %s", report["overall_supply_exposed_nodes"])
    LOG.info("Broadly reachable nodes (union): %s", report["overall_broad_reachable_nodes"])
    LOG.info("Current model exposure count: %s", report["model_exposure_count"])
    LOG.info("Exposure percentage: %s%%", report["model_exposure_percentage"])
    # assessment: too broad if model_exposure_percentage > 50% and much larger than direct
    perc = report["model_exposure_percentage"]
    assessment = "Reasonable"
    if perc > 75:
        assessment = "Too Broad"
    elif perc < 5:
        assessment = "Too Narrow"
    LOG.info("Current assessment: %s", assessment)

    LOG.info("Unresolved events count: %s", len(unresolved))
    LOG.info("Validation artifacts saved to %s", str(VALIDATION_DIR))


if __name__ == "__main__":
    main()
