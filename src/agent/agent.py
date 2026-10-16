"""Deterministic single-agent router over existing ElectroGraph tools."""
from __future__ import annotations

import argparse
import json
import re
from typing import Any

from .tools import (
    get_entity_risk,
    get_external_events,
    get_shipping_model_info,
    retrieve_graph_evidence,
    search_entities,
)

RISK_QUERY = "RISK_QUERY"
GRAPH_QUERY = "GRAPH_QUERY"
EVENT_QUERY = "EVENT_QUERY"
SHIPPING_QUERY = "SHIPPING_QUERY"
GENERAL_SUPPLY_CHAIN_QUERY = "GENERAL_SUPPLY_CHAIN_QUERY"


def classify_intent(query: str) -> str:
    text = query.lower()
    if any(term in text for term in ("shipping model", "shipping performance", "dataco", "days for shipping")):
        return SHIPPING_QUERY
    if any(term in text for term in ("external event", "events affecting", "affect", "happening", "weather", "news")):
        return EVENT_QUERY
    if any(term in text for term in ("warehouse", "supplier", "stored", "where", "connected to", "graph path", "relationship")):
        return GRAPH_QUERY
    if any(term in text for term in ("risk", "risky", "risk score", "why is")):
        return RISK_QUERY
    return GENERAL_SUPPLY_CHAIN_QUERY


def _entity_from_search(query: str) -> dict[str, Any] | None:
    result = search_entities(query)
    entities = result.get("entities", [])
    return entities[0] if entities else None


def _graph_answer(evidence: dict[str, Any]) -> str:
    paths = evidence.get("graph_paths", [])
    if not paths:
        return "No graph relationship evidence was found for this query."
    descriptions = [" -> ".join(path.get("nodes", [])) for path in paths[:5]]
    return "Existing graph paths: " + "; ".join(descriptions) + "."


def _event_answer(events: list[dict[str, Any]]) -> str:
    if not events:
        return "No relevant external event was connected through a valid supply-chain exposure path."
    names = [event.get("name", event.get("node_id", "unresolved event")) for event in events]
    return (
        "Relevant external events include "
        + "; ".join(names)
        + ". They may contribute to risk or be associated with the current scenario; causation is not established."
    )


class SupplyChainAgent:
    """Route questions to existing deterministic, evidence-producing tools."""

    def run(self, query: str) -> dict[str, Any]:
        intent = classify_intent(query)
        tools_used: list[str] = []
        evidence: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        limitations: list[str] = []
        answer = ""

        if intent == SHIPPING_QUERY:
            shipping = get_shipping_model_info()
            tools_used.append("get_shipping_model_info")
            evidence.append(shipping)
            provenance.extend(shipping["provenance"])
            limitations.extend(shipping["limitations"])
            best = next(
                (row for row in shipping.get("test_metrics", []) if row.get("model") == shipping.get("best_model")),
                None,
            )
            answer = (
                f"The frozen DataCo model predicts {shipping.get('target', 'historical shipping duration')}. "
                f"The best test baseline is {shipping.get('best_model', 'unavailable')}."
            )
            if best:
                answer += f" Test MAE is {best['mae']:.4f} days, RMSE is {best['rmse']:.4f}, and R² is {best['r2']:.4f}."
        else:
            search = search_entities(query)
            tools_used.append("search_entities")
            entity = search["entities"][0] if search["entities"] else None
            if entity:
                entity_id = entity["node_id"]
                if intent == EVENT_QUERY:
                    event_result = get_external_events(entity_id)
                    tools_used.append("get_external_events")
                    evidence.append(event_result)
                    provenance.extend(event_result["provenance"])
                    limitations.extend(event_result["limitations"])
                    answer = _event_answer(event_result["events"])
                elif intent == GRAPH_QUERY:
                    graph_result = retrieve_graph_evidence(
                        f"What evidence supports {entity.get('name', entity_id)}?"
                    )
                    tools_used.append("retrieve_graph_evidence")
                    evidence.append(graph_result)
                    provenance.extend(graph_result["provenance"])
                    limitations.extend(graph_result["limitations"])
                    answer = _graph_answer(graph_result)
                else:
                    graph_result = retrieve_graph_evidence(query)
                    risk_result = get_entity_risk(entity_id)
                    tools_used.extend(["retrieve_graph_evidence", "get_entity_risk"])
                    evidence.extend([graph_result, risk_result])
                    provenance.extend(graph_result["provenance"])
                    provenance.extend(risk_result["provenance"])
                    limitations.extend(graph_result["limitations"] + risk_result["limitations"])
                    answer = graph_result["answer"]
            else:
                graph_result = retrieve_graph_evidence(query)
                tools_used.append("retrieve_graph_evidence")
                evidence.append(graph_result)
                provenance.extend(graph_result["provenance"])
                limitations.extend(graph_result["limitations"])
                answer = graph_result["answer"] if graph_result["entities"] else "No evidence was found for this query."

        return {
            "query": query,
            "intent": intent,
            "tools_used": tools_used,
            "answer": answer,
            "evidence": evidence,
            "provenance": provenance,
            "limitations": list(dict.fromkeys(limitations)),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    print(json.dumps(SupplyChainAgent().run(args.query), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
