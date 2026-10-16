"""Current-state GraphRAG retrieval over data/current artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from src.dynamic_risk.risk_views import numeric as risk_numeric

from src.dynamic_risk.current_state import CURRENT_DIR, load_current_state
from src.event_extraction.temporal_validity import is_currently_eligible
from src.dynamic_risk.delivery_risk import estimate_delivery_risk
from .retriever import (
    BASE_DIR,
    DERIVED,
    EXTERNAL_API,
    MODEL_OUTPUT,
    OBSERVED,
    SIMULATED,
    EvidenceRetriever,
    normalize,
)


class CurrentEvidenceRetriever(EvidenceRetriever):
    """Use only current artifacts while retaining baseline path semantics."""

    def __init__(self) -> None:
        super().__init__(
            graph_path=CURRENT_DIR / "current_supply_chain_graph.graphml",
            nodes_path=BASE_DIR / "data" / "graph" / "nodes.csv",
            edges_path=BASE_DIR / "data" / "graph" / "edges.csv",
            risk_path=CURRENT_DIR / "current_risk_scores.csv",
            business_events_path=CURRENT_DIR / "current_business_impacts.csv",
            graph_source_label="data/current/current_supply_chain_graph.graphml",
        )
        self.current_state = load_current_state()
        # Use the runtime-temporally evaluated local state rather than stale
        # persisted current graph/risk artifacts.
        self.graph = self.current_state["graph"] or self.graph
        self.risk = self.current_state["risk"]
        self.risk_by_node = {
            str(row["Node_ID"]): row.to_dict()
            for _, row in self.risk.iterrows()
            if str(row.get("Node_ID", ""))
        }
        self.current_events = self.current_state["events"]
        self.event_by_id = {
            str(event.get("Event_ID")): event
            for event in self.current_events
            if event.get("Event_ID") is not None
        }
        self.current_metadata = self.current_state["metadata"]

    def _event_record(self, event_id: str, event_attrs: dict[str, Any]) -> dict[str, Any] | None:
        event_name = normalize(event_attrs.get("name"))
        for event in self.current_events:
            if normalize(event.get("Title")) == event_name:
                return event
        return self.event_by_id.get(event_id)

    def _event_entities(self) -> list[str]:
        """Expose only active operational events to current-risk traversal."""
        active_ids = {
            str(event.get("Event_ID"))
            for event in self.current_events
            if is_currently_eligible(event)
        }
        return [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") == "Event"
            and (
                attrs.get("event_status") == "ACTIVE"
                and attrs.get("operational_current") == "true"
            )
            and (
                not attrs.get("event_id")
                or str(attrs.get("event_id")) in active_ids
            )
        ]

    def match_entities(self, query: str, limit: int = 10) -> list[str]:
        """Stable adapter for entity resolution without exposing private internals."""
        return self._matching_entities(query, limit=limit)

    def node_facts(self, node_id: str) -> list[dict[str, Any]]:
        """Stable adapter for validated node evidence."""
        return self._node_facts(node_id)

    def _catalog_query(self, query: str) -> dict[str, Any] | None:
        query_lower = normalize(query)
        warehouses = [
            (str(node_id), attrs) for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") == "Warehouse" and attrs.get("catalog_only") in {True, "true", "True", 1, "1"}
        ]
        mentioned = [
            (node_id, attrs) for node_id, attrs in warehouses
            if normalize(node_id) in query_lower or normalize(attrs.get("name")) in query_lower
        ]
        if not mentioned:
            location_matches = [
                (node_id, attrs) for node_id, attrs in warehouses
                if (
                    normalize(attrs.get("city")) in query_lower
                    or normalize(attrs.get("state")) in query_lower
                )
            ]
            brand_matches = []
            for node_id, attrs in location_matches:
                brand_names = [
                    normalize(self.graph.nodes[neighbor].get("name"))
                    for neighbor in self.graph.successors(node_id)
                    if (self.graph.get_edge_data(node_id, neighbor) or {}).get("relationship")
                    == "ASSOCIATED_WITH_BRAND"
                ]
                if any(brand_name and brand_name in query_lower for brand_name in brand_names):
                    brand_matches.append((node_id, attrs))
            mentioned = brand_matches or (location_matches if len(location_matches) == 1 else [])
        if ("warehouse" in query_lower or "warehouses" in query_lower) and " in " in query_lower:
            matches = [
                (node_id, attrs) for node_id, attrs in warehouses
                if normalize(attrs.get("state")) in query_lower or normalize(attrs.get("city")) in query_lower
            ]
            if matches:
                names = ", ".join(str(attrs.get("name", node_id)) for node_id, attrs in matches)
                return {
                    "query": query, "query_type": "curated_warehouse_location", "answer": names,
                    "entities": [{"node_id": node_id, "node_type": "Warehouse", "name": attrs.get("name"), "city": attrs.get("city"), "state": attrs.get("state"), "provenance": OBSERVED} for node_id, attrs in matches],
                    "graph_paths": [], "events": [], "risk_evidence": [], "provenance": [],
                    "limitations": ["Catalog/reference locations do not establish operational inventory."],
                    "current_metadata": self.current_metadata,
                }
        if mentioned and ("brand" in query_lower or "associated" in query_lower):
            node_id, attrs = mentioned[0]
            brands = []
            models = []
            for neighbor in self.graph.successors(node_id):
                edge = self.graph.get_edge_data(node_id, neighbor) or {}
                relationship = edge.get("relationship")
                if relationship == "ASSOCIATED_WITH_BRAND":
                    brands.append(self.graph.nodes[neighbor].get("name", neighbor))
                elif relationship == "CATALOG_ASSOCIATED_WITH":
                    model = self.graph.nodes[neighbor]
                    models.append({"node_id": neighbor, "name": model.get("name"), "brand": model.get("brand"), "provenance": OBSERVED})
            values = models if "product model" in query_lower or "catalog" in query_lower else brands
            return {
                "query": query, "query_type": "catalog_association",
                "answer": ", ".join(str(item.get("name")) for item in values) if models and isinstance(values[0], dict) else ", ".join(str(item) for item in values),
                "entities": models if isinstance(values, list) and values and isinstance(values[0], dict) else [{"node_id": node_id, "node_type": "Warehouse", "name": attrs.get("name"), "associated_brands": brands, "provenance": OBSERVED}],
                "graph_paths": [], "events": [], "risk_evidence": [], "provenance": [],
                "limitations": ["These are catalog associations, not verified physical stock."],
                "current_metadata": self.current_metadata,
            }
        stock_terms = {"stocked", "physically stocked", "physical stock", "inventory"}
        if any(term in query_lower for term in stock_terms):
            models = [
                (str(node_id), attrs) for node_id, attrs in self.graph.nodes(data=True)
                if attrs.get("node_type") == "ProductModel" and normalize(attrs.get("name")) in query_lower
            ]
            if models:
                return {
                    "query": query, "query_type": "catalog_stock_verification",
                    "answer": "The dataset provides a catalog association only. Physical warehouse stock is not verified.",
                    "entities": [{"node_id": node_id, "node_type": "ProductModel", "name": attrs.get("name"), "warehouse_stock_verified": False, "provenance": OBSERVED} for node_id, attrs in models],
                    "graph_paths": [], "events": [], "risk_evidence": [], "provenance": [],
                    "limitations": ["CATALOG_ASSOCIATED_WITH is not an operational STORED_AT relationship."],
                    "current_metadata": self.current_metadata,
                }
        return None

    def _node_facts(self, node_id: str) -> list[dict[str, Any]]:
        facts = super()._node_facts(node_id)
        current = self.risk_by_node.get(node_id, {})
        component_fields = (
            "Structural_Baseline",
            "Weather_Event_Component",
            "News_Event_Component",
            "Business_Impact_Component",
            "Event_Exposure_Component",
            "Final_Scenario_Risk_Score",
        )
        for field in component_fields:
            if current.get(field, "") != "":
                facts.append(
                    self._fact(
                        node_id,
                        field,
                        current[field],
                        DERIVED,
                        "data/current/current_risk_scores.csv",
                    )
                )
        for fact in facts:
            if fact["predicate"] in {
                "Risk_Score",
                "Risk_Level",
                "Event_Exposure",
                "Connected_Events",
                "Risk_Reason",
            }:
                fact["source"] = "data/current/current_risk_scores.csv"
        return facts

    def _ranking(
        self,
        node_type: str,
        query: str,
        limit: int = 10,
        *,
        affected_only: bool = False,
        risky_only: bool = False,
    ) -> dict[str, Any]:
        rows = self.risk[self.risk["Node_Type"].astype(str) == node_type].copy()
        rows["score_numeric"] = pd.to_numeric(rows["Risk_Score"], errors="coerce")
        if affected_only:
            rows = rows[pd.to_numeric(rows["Connected_Events"], errors="coerce").fillna(0) > 0]
        if risky_only:
            rows = rows[rows["Risk_Level"].isin(["Medium", "High", "Critical"])]
        ranked = rows.sort_values(["score_numeric", "Node_Name"], ascending=[False, True], na_position="last")
        selected = ranked.head(limit)
        informational = ranked if not risky_only else self.risk[
            self.risk["Node_Type"].astype(str) == node_type
        ].assign(score_numeric=lambda frame: pd.to_numeric(frame["Risk_Score"], errors="coerce")).sort_values(
            ["score_numeric", "Node_Name"], ascending=[False, True], na_position="last"
        ).head(limit)
        entities = []
        provenance = []
        graph_paths = []
        events = []
        for _, row in selected.iterrows():
            node_id = str(row["Node_ID"])
            attrs = dict(self.graph.nodes[node_id]) if node_id in self.graph else {}
            entity = {
                "node_id": node_id,
                "node_type": node_type,
                "name": row.get("Node_Name", attrs.get("name", node_id)),
                "risk_score": risk_numeric(row.get("Risk_Score")),
                "risk_level": row.get("Risk_Level", "Unavailable") if risk_numeric(row.get("Risk_Score")) is not None else "Unavailable",
                "risk_availability": row.get("Risk_Availability", "AVAILABLE") if risk_numeric(row.get("Risk_Score")) is not None else "UNAVAILABLE",
                "connected_events": row.get("Connected_Events", "0"),
                "reason": row.get("Risk_Reason", ""),
                "provenance": MODEL_OUTPUT,
            }
            for component in (
                "Structural_Baseline",
                "Weather_Event_Component",
                "News_Event_Component",
                "Business_Impact_Component",
                "Event_Exposure_Component",
                "Final_Scenario_Risk_Score",
            ):
                entity[component] = row.get(component, "")
            if node_type == "Product" and node_id in self.graph:
                entity["brand"] = attrs.get("brand", "")
                entity["warehouse"] = next(
                    (
                        self.graph.nodes[neighbor].get("name", neighbor)
                        for neighbor in self.graph.successors(node_id)
                        if self.graph.nodes[neighbor].get("node_type") == "Warehouse"
                    ),
                    "",
                )
            if node_type in {"Supplier", "Warehouse"}:
                entity["provenance"] = SIMULATED
            entities.append(entity)
            evidence = super().retrieve(
                f"Why is {entity['name']} considered risky?",
                max_entities=1,
            )
            entity["current_events"] = evidence.get("events", [])
            events.extend(evidence.get("events", []))
            graph_paths.extend(evidence.get("graph_paths", []))
            if entity["risk_score"] is not None:
                provenance.append(self._fact(node_id, "risk", entity["risk_score"], MODEL_OUTPUT, "data/current/current_risk_scores.csv"))
            if node_type == "Supplier":
                provenance.append(self._fact(node_id, "supplier_mapping", attrs.get("name", ""), SIMULATED, "current graph"))
        return {
            "query": query,
            "query_type": f"{node_type.lower()}_ranking",
            "entities": entities,
            "informational_entities": informational.drop(columns=["score_numeric"]).astype(object).where(
                pd.notna(informational.drop(columns=["score_numeric"])), None
            ).to_dict(orient="records"),
            "graph_paths": graph_paths,
            "events": list({event.get("node_id"): event for event in events}.values()),
            "risk_evidence": provenance,
            "provenance": provenance,
            "limitations": [
                "Ranking uses current risk scores and exact node-type filtering.",
                "Current event evidence is limited to valid relationship-aware paths.",
            ],
            "risk_threshold": "Medium or above" if risky_only else "At least one connected current event",
            "current_metadata": self.current_metadata,
        }

    def retrieve(self, query: str, max_entities: int = 10, **kwargs: Any) -> dict[str, Any]:
        query_lower = normalize(query)
        catalog_result = self._catalog_query(query)
        if catalog_result is not None:
            return catalog_result
        if "weather risks" in query_lower or "weather risk" in query_lower:
            events = []
            for event in self.current_events:
                if (
                    event.get("event_status") != "ACTIVE"
                    or (
                        event.get("Category") != "Weather"
                        and event.get("Impact_Type") != "Weather Disruption"
                    )
                ):
                    continue
                item = {**event, "provenance": EXTERNAL_API, "business_impact_provenance": DERIVED}
                item["business_impact"] = {
                    key: event.get(key, "")
                    for key in ("Impact_Type", "Impact_Level", "Risk_Score", "Possible_Supply_Chain_Effect")
                }
                events.append(item)
            return {
                "query": query,
                "query_type": "weather_ranking",
                "entities": [],
                "graph_paths": [],
                "events": events,
                "risk_evidence": [],
                "provenance": events,
                "limitations": ["Weather results are read only from current_events.json."],
                "current_metadata": self.current_metadata,
            }
        if "which products" in query_lower and ("risky" in query_lower or "risk" in query_lower):
            return self._ranking("Product", query, risky_only=True)
        if "which warehouses" in query_lower and ("affected" in query_lower or "risky" in query_lower):
            return self._ranking(
                "Warehouse",
                query,
                affected_only="affected" in query_lower,
                risky_only="risky" in query_lower,
            )
        if "which suppliers" in query_lower and ("risky" in query_lower or "risk" in query_lower):
            return self._ranking("Supplier", query, risky_only=True)
        if "delivery delay" in query_lower or "face delivery" in query_lower:
            matches = self._matching_entities(query, limit=1)
            if matches:
                result = estimate_delivery_risk(matches[0])
                return {
                    "query": query,
                    "query_type": "delivery_risk",
                    "entities": [{"node_id": matches[0], "node_type": self.graph.nodes[matches[0]].get("node_type"), "name": self.graph.nodes[matches[0]].get("name", "")}],
                    "graph_paths": [],
                    "events": [],
                    "risk_evidence": result.get("evidence", []),
                    "provenance": result.get("evidence", []),
                    "delivery_risk": result,
                    "limitations": [result.get("message", "")],
                    "current_metadata": self.current_metadata,
                }
        single_entity = next(
            (
                (node_id, attrs.get("name", ""))
                for node_id, attrs in sorted(
                    self.graph.nodes(data=True),
                    key=lambda item: len(str(item[1].get("name", ""))),
                    reverse=True,
                )
                if attrs.get("name") and normalize(attrs.get("name")) in query_lower
            ),
            None,
        )
        if single_entity:
            result = super().retrieve(
                f"Why is {single_entity[1]} considered risky?",
                max_entities=1,
                **kwargs,
            )
            result["query"] = query
        else:
            result = super().retrieve(query, max_entities=max_entities, **kwargs)
        result["risk_evidence"].extend(
            fact
            for fact in result.get("provenance", [])
            if fact.get("predicate") in {
                "Structural_Baseline",
                "Weather_Event_Component",
                "News_Event_Component",
                "Business_Impact_Component",
                "Event_Exposure_Component",
                "Final_Scenario_Risk_Score",
            }
        )
        for event in result.get("events", []):
            event_id = event.get("node_id")
            event_row = self._event_record(
                str(event_id).replace("CURRENT_EVENT:", ""),
                self.graph.nodes[event_id] if event_id in self.graph else {},
            )
            if event_row:
                for field in (
                    "event_start",
                    "event_end",
                    "event_status",
                    "temporal_confidence",
                    "temporal_basis",
                ):
                    event[field] = event_row.get(field, event.get(field, ""))
            if event_id in self.graph:
                event["unresolved"] = not any(
                    self.graph.nodes[neighbor].get("node_type") in {"Warehouse", "Location"}
                    for neighbor in self.graph.successors(event_id)
                )
        result["current_metadata"] = self.current_metadata
        return result
