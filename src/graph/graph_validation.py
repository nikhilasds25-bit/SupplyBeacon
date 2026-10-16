"""Validate and explore the generated supply chain graph."""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from graph_schema import (
    EDGE_TYPE_ASSOCIATED_WITH_BRAND,
    EDGE_TYPE_AFFECTS,
    EDGE_TYPE_BELONGS_TO,
    EDGE_TYPE_CATALOG_ASSOCIATED_WITH,
    EDGE_TYPE_LOCATED_IN,
    EDGE_TYPE_OFFERS,
    EDGE_TYPE_PRODUCES,
    EDGE_TYPE_STORED_AT,
    EDGE_TYPE_SUPPLIES,
    NODE_TYPE_BRAND,
    NODE_TYPE_CATEGORY,
    NODE_TYPE_EVENT,
    NODE_TYPE_LOCATION,
    NODE_TYPE_PRODUCT,
    NODE_TYPE_PRODUCT_MODEL,
    NODE_TYPE_SUPPLIER,
    NODE_TYPE_WAREHOUSE,
)
from graph_utils import GRAPHML_PATH, GRAPH_DIR, create_directory, load_dataframe, normalize_text

VALIDATION_JSON = GRAPH_DIR / "validation_report.json"
EVENT_REPORT_CSV = GRAPH_DIR / "event_graph_report.csv"
EVENT_SUBGRAPH_PNG = GRAPH_DIR / "event_subgraph.png"

ALLOWED_NODE_TYPES = {
    NODE_TYPE_BRAND,
    NODE_TYPE_CATEGORY,
    NODE_TYPE_EVENT,
    NODE_TYPE_LOCATION,
    NODE_TYPE_PRODUCT,
    NODE_TYPE_PRODUCT_MODEL,
    NODE_TYPE_SUPPLIER,
    NODE_TYPE_WAREHOUSE,
}

ALLOWED_EDGE_TYPES = {
    EDGE_TYPE_SUPPLIES,
    EDGE_TYPE_PRODUCES,
    EDGE_TYPE_BELONGS_TO,
    EDGE_TYPE_STORED_AT,
    EDGE_TYPE_LOCATED_IN,
    EDGE_TYPE_AFFECTS,
    EDGE_TYPE_ASSOCIATED_WITH_BRAND,
    EDGE_TYPE_CATALOG_ASSOCIATED_WITH,
    EDGE_TYPE_OFFERS,
}

REQUIRED_NODE_ATTRS = {"node_type", "name"}
EVENT_REQUIRED_ATTRS = {"event_type", "date", "severity", "risk_score"}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_graph() -> nx.Graph:
    if not GRAPHML_PATH.exists():
        raise FileNotFoundError(f"GraphML file not found: {GRAPHML_PATH}")
    graph = nx.read_graphml(str(GRAPHML_PATH))
    return graph


def validate_node_types(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    unexpected = []
    node_type_counts: Counter[str] = Counter()
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        node_type_counts[str(node_type)] += 1
        if node_type not in ALLOWED_NODE_TYPES:
            unexpected.append({"node_id": node_id, "node_type": node_type})
    return {"counts": dict(node_type_counts), "unexpected": unexpected}


def validate_edge_types(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    unexpected = []
    edge_type_counts: Counter[str] = Counter()
    for u, v, attrs in graph.edges(data=True):
        relationship = attrs.get("relationship")
        edge_type_counts[str(relationship)] += 1
        if relationship not in ALLOWED_EDGE_TYPES:
            unexpected.append({"source": u, "target": v, "relationship": relationship})
    return {"counts": dict(edge_type_counts), "unexpected": unexpected}


def validate_graph(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    node_ids = list(graph.nodes())
    unique_ids = set(node_ids)
    edges = list(graph.edges(data=True))
    missing_endpoints = []
    for u, v, _ in edges:
        if u not in graph:
            missing_endpoints.append({"missing": u, "edge": [u, v]})
        if v not in graph:
            missing_endpoints.append({"missing": v, "edge": [u, v]})
    self_loops = []
    for u, v, attrs in graph.edges(data=True):
        if u == v:
            self_loops.append({"node": u, "relationship": attrs})

    node_attr_issues = []
    event_attr_issues = []
    for node_id, attrs in graph.nodes(data=True):
        if not REQUIRED_NODE_ATTRS.issubset(attrs.keys()):
            node_attr_issues.append({"node_id": node_id, "missing_attrs": list(REQUIRED_NODE_ATTRS - set(attrs.keys()))})
        if attrs.get("node_type") == NODE_TYPE_EVENT:
            missing = [attr for attr in EVENT_REQUIRED_ATTRS if not attrs.get(attr)]
            if missing:
                event_attr_issues.append({"node_id": node_id, "missing_attrs": missing})

    node_validation = validate_node_types(graph)
    edge_validation = validate_edge_types(graph)
    curated_inventory_violations = [
        {"source": source, "target": target}
        for source, target, attrs in edges
        if attrs.get("relationship") == EDGE_TYPE_STORED_AT
        and (str(source).startswith("WH_REAL_") or str(target).startswith("WH_REAL_"))
    ]

    return {
        "unique_node_ids": len(node_ids) == len(unique_ids),
        "node_count": len(node_ids),
        "edge_count": len(edges),
        "missing_endpoints": missing_endpoints,
        "unexpected_node_types": node_validation["unexpected"],
        "unexpected_edge_types": edge_validation["unexpected"],
        "self_loops": self_loops,
        "node_attr_issues": node_attr_issues,
        "event_attr_issues": event_attr_issues,
        "node_type_counts": node_validation["counts"],
        "edge_type_counts": edge_validation["counts"],
        "curated_inventory_violations": curated_inventory_violations,
    }


def node_degree_stats(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, float]] = {}
    type_groups: Dict[str, List[int]] = defaultdict(list)
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "Unknown")
        type_groups[node_type].append(graph.degree(node_id))
    for node_type, degrees in type_groups.items():
        stats[node_type] = {
            "min": min(degrees),
            "max": max(degrees),
            "avg": round(sum(degrees) / len(degrees), 2),
            "count": len(degrees),
        }
    return stats


def compute_edge_stats(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    edge_counts = Counter(attrs.get("relationship", "Unknown") for _, _, attrs in graph.edges(data=True))
    event_edges = 0
    affects_edges = 0
    temporal_edges = 0
    for u, v, attrs in graph.edges(data=True):
        if graph.nodes[u].get("node_type") == NODE_TYPE_EVENT or graph.nodes[v].get("node_type") == NODE_TYPE_EVENT:
            event_edges += 1
        if attrs.get("relationship") == EDGE_TYPE_AFFECTS:
            affects_edges += 1
        if attrs.get("date"):
            temporal_edges += 1
    return {
        "edges_by_relationship": dict(edge_counts),
        "event_edges": event_edges,
        "affects_edges": affects_edges,
        "temporal_edges": temporal_edges,
    }


def event_analysis(graph: nx.MultiDiGraph) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    connected_events = 0
    unconnected_events = 0
    affects_count = 0
    suspicious = []

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != NODE_TYPE_EVENT:
            continue
        event_type = attrs.get("event_type", "")
        severity = attrs.get("severity", "")
        risk_score = attrs.get("risk_score", "")
        affected_entity_names: List[str] = []
        location_names: List[str] = []
        connected_nodes: List[str] = []
        connected_relationships: List[str] = []

        neighbors = set(graph.predecessors(node_id)) | set(graph.successors(node_id))
        for neighbor in neighbors:
            neighbor_type = graph.nodes[neighbor].get("node_type", "")
            neighbor_name = graph.nodes[neighbor].get("name", "")
            connected_nodes.append(f"{neighbor_type}:{neighbor_name}")
        for _, target, attrs_edge in graph.out_edges(node_id, data=True):
            connected_relationships.append(f"{attrs_edge.get('relationship')}->{graph.nodes[target].get('node_type')}:{graph.nodes[target].get('name')}")
            if attrs_edge.get("relationship") == EDGE_TYPE_AFFECTS:
                affects_count += 1
        for source, _, attrs_edge in graph.in_edges(node_id, data=True):
            connected_relationships.append(f"{attrs_edge.get('relationship')}<-{graph.nodes[source].get('node_type')}:{graph.nodes[source].get('name')}")
            if attrs_edge.get("relationship") == EDGE_TYPE_AFFECTS:
                affects_count += 1

        if connected_nodes:
            connected_events += 1
        else:
            unconnected_events += 1
            suspicious.append({"event_id": node_id, "reason": "Event has no connected nodes"})

        event_location = None
        event_affected_entity = None
        for neighbor in neighbors:
            t = graph.nodes[neighbor].get("node_type")
            if t == NODE_TYPE_LOCATION:
                event_location = graph.nodes[neighbor].get("name")
            if t in {NODE_TYPE_BRAND, NODE_TYPE_SUPPLIER}:
                affected_entity_names.append(graph.nodes[neighbor].get("name"))
        if not event_location:
            suspicious.append({"event_id": node_id, "reason": "Event with Unknown location"})
        if not affected_entity_names:
            suspicious.append({"event_id": node_id, "reason": "Event with Unknown affected entity"})

        rows.append(
            {
                "Event_ID": node_id.split(":", 1)[-1],
                "Event_Type": event_type,
                "Location": event_location or "Unknown",
                "Severity": severity,
                "Risk_Score": risk_score,
                "Affected_Entity": ", ".join(sorted(set(affected_entity_names))) or "Unknown",
                "Connected_Nodes": "; ".join(sorted(set(connected_nodes))) or "None",
                "Connected_Relationships": "; ".join(sorted(set(connected_relationships))) or "None",
            }
        )

    return rows, {
        "total_events": connected_events + unconnected_events,
        "connected_events": connected_events,
        "unconnected_events": unconnected_events,
        "affects_relationships": affects_count,
        "suspicious_event_issues": suspicious,
    }


def detect_suspicious_relationships(graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type == NODE_TYPE_PRODUCT:
            neighbors = set(graph.successors(node_id)) | set(graph.predecessors(node_id))
            has_category = False
            has_warehouse = False
            for nbr in neighbors:
                edge_data = graph.get_edge_data(node_id, nbr)
                if edge_data is None:
                    continue
                if isinstance(edge_data, dict) and any(isinstance(v, dict) for v in edge_data.values()):
                    for key_attrs in edge_data.values():
                        relationship = key_attrs.get("relationship")
                        if relationship == EDGE_TYPE_BELONGS_TO:
                            has_category = True
                        if relationship == EDGE_TYPE_STORED_AT:
                            has_warehouse = True
                else:
                    relationship = edge_data.get("relationship")
                    if relationship == EDGE_TYPE_BELONGS_TO:
                        has_category = True
                    if relationship == EDGE_TYPE_STORED_AT:
                        has_warehouse = True
            if not has_category:
                issues.append({"node_id": node_id, "issue": "Product with no category"})
            if not has_warehouse:
                issues.append({"node_id": node_id, "issue": "Product with no warehouse"})
        if node_type == NODE_TYPE_SUPPLIER:
            if graph.degree(node_id) == 0:
                issues.append({"node_id": node_id, "issue": "Supplier with no relationships"})
    return issues


def generate_subgraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    event_nodes = [n for n, attrs in graph.nodes(data=True) if attrs.get("node_type") == NODE_TYPE_EVENT]
    sub_nodes = set(event_nodes)
    for event in event_nodes:
        for _, neighbor, attrs in graph.edges(event, data=True):
            if graph.nodes[neighbor].get("node_type") in {NODE_TYPE_WAREHOUSE, NODE_TYPE_LOCATION, NODE_TYPE_SUPPLIER, NODE_TYPE_BRAND}:
                sub_nodes.add(neighbor)
                for _, second_neighbor, second_attrs in graph.edges(neighbor, data=True):
                    if graph.nodes[second_neighbor].get("node_type") in {NODE_TYPE_PRODUCT, NODE_TYPE_SUPPLIER, NODE_TYPE_LOCATION}:
                        sub_nodes.add(second_neighbor)
    return graph.subgraph(sub_nodes).copy()


def draw_subgraph(subgraph: nx.Graph, path: Path) -> None:
    if not subgraph.nodes:
        return
    create_directory(path.parent)
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)
    node_colors = []
    color_map = {
        NODE_TYPE_EVENT: "red",
        NODE_TYPE_LOCATION: "orange",
        NODE_TYPE_WAREHOUSE: "blue",
        NODE_TYPE_SUPPLIER: "green",
        NODE_TYPE_BRAND: "purple",
        NODE_TYPE_PRODUCT: "grey",
        NODE_TYPE_CATEGORY: "cyan",
    }
    labels = {}
    for node_id, attrs in subgraph.nodes(data=True):
        node_colors.append(color_map.get(attrs.get("node_type"), "lightgrey"))
        labels[node_id] = f"{attrs.get('node_type')}\n{attrs.get('name')}"
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_edges(subgraph, pos, arrows=True, arrowstyle="-|>", arrowsize=12)
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
    edge_labels = {(u, v): data.get("relationship", "") for u, v, data in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Event Subgraph: Event → Location/Warehouse → Product/Supplier")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_report(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    validation = validate_graph(graph)
    degree_stats = node_degree_stats(graph)
    edge_stats = compute_edge_stats(graph)
    event_rows, event_summary = event_analysis(graph)
    suspicious_relationships = detect_suspicious_relationships(graph)

    report = {
        "graph_validation": validation,
        "degree_stats": degree_stats,
        "edge_stats": edge_stats,
        "event_summary": event_summary,
        "suspicious_relationships": suspicious_relationships,
    }
    return report, event_rows


def save_reports(report: Dict[str, Any], event_rows: List[Dict[str, Any]]) -> None:
    create_directory(GRAPH_DIR)
    with VALIDATION_JSON.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    import pandas as pd

    pd.DataFrame(event_rows).to_csv(EVENT_REPORT_CSV, index=False, encoding="utf-8-sig")


def print_console_report(graph: nx.MultiDiGraph, report: Dict[str, Any]) -> None:
    validation = report["graph_validation"]
    edge_stats = report["edge_stats"]
    event_summary = report["event_summary"]

    print("GRAPH VALIDATION REPORT")
    print("=======================")
    print(f"Total nodes: {validation['node_count']}")
    print(f"Total edges: {validation['edge_count']}")
    print(f"Connected components: {nx.number_connected_components(graph.to_undirected())}")
    isolated_nodes = list(nx.isolates(graph))
    print(f"Isolated nodes: {len(isolated_nodes)}")
    print()
    print("Node types:")
    for node_type, count in validation["node_type_counts"].items():
        print(f"  {node_type}: {count}")
    print()
    print("Relationship types:")
    for rel_type, count in edge_stats["edges_by_relationship"].items():
        print(f"  {rel_type}: {count}")
    print()
    print("Event analysis:")
    print(f"  Total events: {event_summary['total_events']}")
    print(f"  Connected events: {event_summary['connected_events']}")
    print(f"  Unconnected events: {event_summary['unconnected_events']}")
    print(f"  AFFECTS relationships: {edge_stats['affects_edges']}")
    print()
    issues = []
    issues.extend(validation["unexpected_node_types"])
    issues.extend(validation["unexpected_edge_types"])
    issues.extend(validation["self_loops"])
    issues.extend(validation["node_attr_issues"])
    issues.extend(validation["event_attr_issues"])
    issues.extend(report["suspicious_relationships"])
    issues.extend(event_summary["suspicious_event_issues"])

    print("Potential data-quality issues:")
    if not issues:
        print("  None detected.")
    else:
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ...and {len(issues) - 20} more issues")


def main() -> None:
    configure_logging()
    try:
        graph = load_graph()
        report, event_rows = build_report(graph)
        save_reports(report, event_rows)
        subgraph = generate_subgraph(graph)
        draw_subgraph(subgraph, EVENT_SUBGRAPH_PNG)
        print_console_report(graph, report)
    except Exception as exc:
        logging.exception("Graph validation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
