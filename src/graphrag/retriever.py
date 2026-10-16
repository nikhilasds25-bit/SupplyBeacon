"""Evidence-first GraphRAG retrieval without an LLM or graph mutation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
GRAPH_PATH = BASE_DIR / "data" / "graph" / "supply_chain_graph.graphml"
NODES_PATH = BASE_DIR / "data" / "graph" / "nodes.csv"
EDGES_PATH = BASE_DIR / "data" / "graph" / "edges.csv"
RISK_PATH = BASE_DIR / "data" / "risk_prediction" / "risk_scores.csv"
BUSINESS_EVENTS_PATH = BASE_DIR / "data" / "processed" / "business_impacts" / "business_impact_events.csv"
EMBEDDINGS_PATH = BASE_DIR / "data" / "gnn_prediction" / "node_embeddings.csv"
OUTPUT_DIR = BASE_DIR / "data" / "analysis" / "graphrag"

OBSERVED = "OBSERVED"
SIMULATED = "SIMULATED"
DERIVED = "DERIVED"
MODEL_OUTPUT = "MODEL_OUTPUT"
EXTERNAL_API = "EXTERNAL_API"

SIMULATED_RELATIONSHIPS = {"SUPPLIES", "STORED_AT"}
EXTERNAL_EVENT_SOURCES = {"News", "Weather"}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize(value)) if len(token) > 1}


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


class EvidenceRetriever:
    """Retrieve bounded graph evidence and attach source classifications."""

    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        nodes_path: Path = NODES_PATH,
        edges_path: Path = EDGES_PATH,
        risk_path: Path = RISK_PATH,
        business_events_path: Path = BUSINESS_EVENTS_PATH,
        graph_source_label: str = "data/graph/supply_chain_graph.graphml",
    ) -> None:
        self.graph_source_label = graph_source_label
        self.graph = nx.read_graphml(str(graph_path))
        self.nodes = load_csv(nodes_path)
        self.edges = load_csv(edges_path)
        self.risk = load_csv(risk_path)
        self.business_events = load_csv(business_events_path)
        self.embeddings = load_csv(EMBEDDINGS_PATH)
        self.risk_by_node = {
            str(row["Node_ID"]): row.to_dict()
            for _, row in self.risk.iterrows()
            if row.get("Node_ID")
        }
        self.event_by_id = {
            str(row["Event_ID"]): row.to_dict()
            for _, row in self.business_events.iterrows()
            if row.get("Event_ID")
        }
        self.embedding_ids = set(self.embeddings["Node_ID"].astype(str)) if "Node_ID" in self.embeddings else set()

    def _node_provenance(self, attrs: dict[str, Any], field: str) -> str:
        node_type = attrs.get("node_type", "")
        if node_type in {"Supplier", "Warehouse"}:
            return SIMULATED
        if node_type == "Event":
            return EXTERNAL_API
        if field in {"risk_score", "Risk_Score"}:
            return MODEL_OUTPUT
        if node_type in {"Product", "Brand", "Category"}:
            return OBSERVED
        return DERIVED

    def _fact(self, subject: str, predicate: str, value: Any, provenance: str, source: str) -> dict[str, Any]:
        return {
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "provenance": provenance,
            "source": source,
        }

    def _node_facts(self, node_id: str) -> list[dict[str, Any]]:
        attrs = dict(self.graph.nodes[node_id])
        facts = [
            self._fact(node_id, "node_type", attrs.get("node_type", ""), self._node_provenance(attrs, "node_type"), "graphml"),
            self._fact(node_id, "name", attrs.get("name", ""), self._node_provenance(attrs, "name"), "graphml"),
        ]
        for field, value in attrs.items():
            if field in {"node_type", "name"} or value in ("", None):
                continue
            provenance = self._node_provenance(attrs, field)
            facts.append(self._fact(node_id, field, value, provenance, "graphml"))
        risk = self.risk_by_node.get(node_id)
        if risk:
            for field in ("Risk_Score", "Risk_Level", "Event_Exposure", "Connected_Events", "Risk_Reason"):
                if risk.get(field, "") != "":
                    facts.append(self._fact(node_id, field, risk[field], MODEL_OUTPUT, "data/risk_prediction/risk_scores.csv"))
        if node_id in self.embedding_ids:
            facts.append(self._fact(node_id, "embedding_available", True, MODEL_OUTPUT, "data/gnn_prediction/node_embeddings.csv"))
        return facts

    def _edge_fact(self, source: str, target: str, attrs: dict[str, Any]) -> dict[str, Any]:
        relationship = attrs.get("relationship", "")
        provenance = SIMULATED if relationship in SIMULATED_RELATIONSHIPS else DERIVED
        return {
            "source": source,
            "target": target,
            "relationship": relationship,
            "provenance": provenance,
            "source_artifact": self.graph_source_label,
            **{
                key: attrs.get(key)
                for key in ("exposure_scope", "resolution_provenance", "exposure_location")
                if attrs.get(key) not in ("", None)
            },
        }

    def _edge_attributes(self, source: str, target: str) -> list[tuple[str, str, dict[str, Any]]]:
        """Return attributes for a directed edge in either observed direction."""
        data = self.graph.get_edge_data(source, target)
        actual_source, actual_target = source, target
        if data is None:
            data = self.graph.get_edge_data(target, source)
            actual_source, actual_target = target, source
        if data is None:
            return []
        if self.graph.is_multigraph():
            return [
                (actual_source, actual_target, value)
                for value in data.values()
                if isinstance(value, dict)
            ]
        return [(actual_source, actual_target, data)]

    def _matching_entities(self, query: str, limit: int = 10) -> list[str]:
        query_tokens = tokens(query)
        scored: list[tuple[int, str]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            searchable = tokens(f"{node_id} {attrs.get('name', '')} {attrs.get('node_type', '')}")
            score = len(query_tokens & searchable)
            if score:
                scored.append((score, node_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [node_id for _, node_id in scored[:limit]]

    def _risk_entities(self, limit: int = 10) -> list[str]:
        rows = sorted(
            self.risk_by_node.values(),
            key=lambda row: float(row.get("Risk_Score", 0) or 0),
            reverse=True,
        )
        return [str(row["Node_ID"]) for row in rows[:limit] if str(row.get("Node_ID", "")) in self.graph]

    def _event_entities(self) -> list[str]:
        return [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") == "Event"
        ]

    def _valid_event_path(
        self,
        entity_id: str,
        path: list[str],
        path_edges: list[dict[str, Any]],
    ) -> bool:
        """Accept only defensible supply-chain exposure routes to an event."""
        if not path or path[0] != entity_id or not path_edges:
            return False
        entity_type = self.graph.nodes[entity_id].get("node_type")
        relationships = [edge.get("relationship") for edge in path_edges]
        node_types = [self.graph.nodes[node_id].get("node_type") for node_id in path]
        if node_types[-1] != "Event" or "AFFECTS" not in relationships:
            return False

        affects_index = relationships.index("AFFECTS")
        affected_node = path[affects_index]
        affected_type = self.graph.nodes[affected_node].get("node_type")
        if affected_type not in {"Warehouse", "Location"}:
            return False

        if entity_type == "Product":
            # A product may expose only through its own warehouse and that
            # warehouse's directly related location; peer warehouses/products
            # cannot be used as a bridge.
            return (
                len(path) in {3, 4}
                and node_types.count("Product") == 1
                and node_types.count("Warehouse") == 1
                and node_types[1] == "Warehouse"
                and relationships[0] == "STORED_AT"
                and all(node_type in {"Product", "Warehouse", "Location", "Event"} for node_type in node_types)
                and all(relationship in {"STORED_AT", "LOCATED_IN", "AFFECTS"} for relationship in relationships)
            )

        if entity_type == "Warehouse":
            return (
                all(node_type in {"Warehouse", "Location", "Event"} for node_type in node_types)
                and node_types.count("Warehouse") == 1
                and all(relationship in {"LOCATED_IN", "AFFECTS"} for relationship in relationships)
            )

        if entity_type == "Supplier":
            # Supplier exposure must flow through a supplied brand/product and its
            # warehouse or location, rather than through an unrelated peer product.
            return (
                "SUPPLIES" in relationships
                and "PRODUCES" in relationships
                and "STORED_AT" in relationships
                and all(
                    relationship in {"SUPPLIES", "PRODUCES", "STORED_AT", "LOCATED_IN", "AFFECTS"}
                    for relationship in relationships
                )
                and node_types.count("Product") >= 1
                and node_types.count("Warehouse") >= 1
            )

        return False

    def _related_typed(self, node_id: str, relationship: str, node_type: str) -> list[str]:
        neighbors = set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id))
        return sorted(
            neighbor for neighbor in neighbors
            if self.graph.nodes[neighbor].get("node_type") == node_type
            and any(edge.get("relationship") == relationship for _, _, edge in self._edge_attributes(node_id, neighbor))
        )

    def _event_path(self, entity_id: str, event_id: str) -> list[str] | None:
        """Traverse only approved inventory-backed exposure relationships."""
        node_type = self.graph.nodes[entity_id].get("node_type")
        if node_type == "Location":
            return [entity_id, event_id] if event_id in self._related_typed(entity_id, "AFFECTS", "Event") else None
        if node_type == "Warehouse":
            if event_id in self._related_typed(entity_id, "AFFECTS", "Event"):
                return [entity_id, event_id]
            for location_id in self._related_typed(entity_id, "LOCATED_IN", "Location"):
                if event_id in self._related_typed(location_id, "AFFECTS", "Event"):
                    return [entity_id, location_id, event_id]
        if node_type == "Product":
            for warehouse_id in self._related_typed(entity_id, "STORED_AT", "Warehouse"):
                suffix = self._event_path(warehouse_id, event_id)
                if suffix:
                    return [entity_id, *suffix]
        if node_type == "Supplier":
            for brand_id in self._related_typed(entity_id, "SUPPLIES", "Brand"):
                for product_id in self._related_typed(brand_id, "PRODUCES", "Product"):
                    suffix = self._event_path(product_id, event_id)
                    if suffix:
                        return [entity_id, brand_id, *suffix]
        return None

    def _selected_entities(self, query: str) -> list[str]:
        query_lower = normalize(query)
        if "which products" in query_lower and ("event" in query_lower or "exposed" in query_lower):
            return [
                node_id
                for node_id, row in sorted(
                    self.risk_by_node.items(),
                    key=lambda item: float(item[1].get("Risk_Score", 0) or 0),
                    reverse=True,
                )
                if node_id in self.graph
                and self.graph.nodes[node_id].get("node_type") == "Product"
                and int(float(row.get("Connected_Events", 0) or 0)) > 0
            ][:10]
        if "which warehouses" in query_lower:
            undirected = self.graph.to_undirected()
            return [
                node_id for node_id, attrs in self.graph.nodes(data=True)
                if attrs.get("node_type") == "Warehouse"
                and any(
                    self.graph.nodes[neighbor].get("node_type") == "Event"
                    for neighbor in undirected.neighbors(node_id)
                )
            ]
        if "which suppliers" in query_lower and "risk" in query_lower:
            return [
                node_id
                for node_id in sorted(
                    (
                        node_id
                        for node_id, attrs in self.graph.nodes(data=True)
                        if attrs.get("node_type") == "Supplier" and node_id in self.risk_by_node
                    ),
                    key=lambda node_id: float(self.risk_by_node[node_id].get("Risk_Score", 0) or 0),
                    reverse=True,
                )
            ][:10]
        if "risk" in query_lower and not self._matching_entities(query):
            return self._risk_entities()
        matches = self._matching_entities(query)
        if query_lower.startswith(("why is ", "what evidence supports ")):
            return matches[:1]
        return matches

    def retrieve(
        self,
        query: str,
        max_entities: int = 10,
        max_paths_per_entity: int = 5,
        max_path_length: int = 4,
    ) -> dict[str, Any]:
        selected = self._selected_entities(query)[:max_entities]
        graph_paths: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        risk_evidence: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        entities: list[dict[str, Any]] = []
        path_keys: set[tuple[tuple[str, str], ...]] = set()
        event_ids: set[str] = set()

        for node_id in selected:
            attrs = dict(self.graph.nodes[node_id])
            entities.append({"node_id": node_id, "node_type": attrs.get("node_type"), "name": attrs.get("name", "")})
            facts = self._node_facts(node_id)
            provenance.extend(facts)
            risk_evidence.extend([fact for fact in facts if fact["provenance"] == MODEL_OUTPUT])
            neighbors = list(self.graph.neighbors(node_id))
            for neighbor in neighbors:
                for edge_source, edge_target, edge_attrs in self._edge_attributes(node_id, neighbor):
                    edge = self._edge_fact(edge_source, edge_target, edge_attrs)
                    path_key = ((edge["source"], edge["target"]),)
                    if path_key in path_keys:
                        continue
                    path_keys.add(path_key)
                    graph_paths.append(
                        {
                            "nodes": [node_id, neighbor],
                            "edges": [edge],
                        }
                    )
            for event_id in self._event_entities():
                path = self._event_path(node_id, event_id) if event_id != node_id else None
                if path is None:
                    continue
                if len(path) - 1 > max_path_length:
                    continue
                path_edges = []
                for source, target in zip(path, path[1:]):
                    edge_attrs = self._edge_attributes(source, target)
                    if edge_attrs:
                        edge_source, edge_target, attrs = edge_attrs[0]
                        path_edges.append(self._edge_fact(edge_source, edge_target, attrs))
                if not self._valid_event_path(node_id, path, path_edges):
                    continue
                path_key = tuple((edge["source"], edge["target"]) for edge in path_edges)
                if path_key in path_keys:
                    continue
                path_keys.add(path_key)
                graph_paths.append({"nodes": path, "edges": path_edges})
                if len([item for item in graph_paths if item["nodes"][0] == node_id]) >= max_paths_per_entity:
                    break

            for event_id in self._event_entities():
                event_attrs = self.graph.nodes[event_id]
                event_path = self._event_path(node_id, event_id) if event_id not in event_ids and event_id != node_id else None
                if event_path is None:
                    continue
                if len(event_path) - 1 > max_path_length:
                    continue
                event_path_edges = []
                for source, target in zip(event_path, event_path[1:]):
                    edge_attrs = self._edge_attributes(source, target)
                    if edge_attrs:
                        edge_source, edge_target, attrs = edge_attrs[0]
                        event_path_edges.append(self._edge_fact(edge_source, edge_target, attrs))
                if not self._valid_event_path(node_id, event_path, event_path_edges):
                    continue
                event_ids.add(event_id)
                event_row = next(
                    (
                        row
                        for row in self.event_by_id.values()
                        if normalize(row.get("Title")) == normalize(event_attrs.get("name"))
                    ),
                    None,
                )
                item = {
                    "node_id": event_id,
                    "name": event_attrs.get("name", ""),
                    "event_type": event_attrs.get("event_type", ""),
                    "provenance": EXTERNAL_API,
                    "source": event_attrs.get("source", "graphml"),
                    "unresolved": str(event_attrs.get("affected_entity", "")).lower() in {"", "unknown"},
                    "event_domain": event_attrs.get("event_domain", ""),
                    "source_channel": event_attrs.get("source_channel", ""),
                }
                if event_row:
                    item["business_impact"] = {
                        key: event_row.get(key, "")
                        for key in ("Impact_Type", "Impact_Level", "Risk_Score", "Possible_Supply_Chain_Effect")
                    }
                    item["business_impact_provenance"] = DERIVED
                events.append(item)
        return {
            "query": query,
            "entities": entities,
            "graph_paths": graph_paths,
            "events": events,
            "risk_evidence": risk_evidence,
            "provenance": provenance,
            "limitations": [
                "Retrieval uses existing graph connectivity and does not infer missing relationships.",
                "Unknown or unresolved event entities remain unresolved.",
                "GraphSAGE embeddings are reported only as availability metadata; no vector similarity is performed.",
                f"Event evidence is limited to actual graph paths of at most {max_path_length} edges.",
            ],
        }


EXAMPLE_QUERIES = [
    "Which products are exposed to current supply-chain events?",
    "Why is Samsung Galaxy F05 considered risky?",
    "Which warehouses are affected by external events?",
    "Which suppliers have the highest current graph risk?",
    "What evidence supports the Samsung Galaxy F05 risk assessment?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=EXAMPLE_QUERIES[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = EvidenceRetriever().retrieve(args.query)
    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
