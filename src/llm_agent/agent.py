"""LLM orchestration over deterministic, read-only ElectroGraph tools."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from src.agent.agent import (
    GENERAL_SUPPLY_CHAIN_QUERY,
    SupplyChainAgent,
    classify_intent,
)
from src.agent.tools import (
    get_entity_risk,
    get_external_events,
    get_shipping_model_info,
    retrieve_graph_evidence,
    search_entities,
)

from .prompts import SYSTEM_PROMPT, TOOL_SELECTION_PROMPT
from .provider import LLMProvider, OpenAICompatibleProvider, ProviderError

TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "search_entities": search_entities,
    "get_entity_risk": get_entity_risk,
    "retrieve_graph_evidence": retrieve_graph_evidence,
    "get_external_events": get_external_events,
    "get_shipping_model_info": get_shipping_model_info,
}


class LLMToolPlanError(ValueError):
    """Raised when the model requests an unsafe or malformed tool plan."""


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _flatten_provenance(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            provenance = item.get("provenance")
            if isinstance(provenance, str):
                records.append(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique[json.dumps(record, sort_keys=True, default=str)] = record
    return list(unique.values())


class LLMToolAgent:
    """Execute only validated tool calls requested by an LLM provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def select_intent(self, query: str) -> str:
        raw = self.provider.generate(TOOL_SELECTION_PROMPT, query, {})
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMToolPlanError("Provider returned a non-JSON intent.") from exc
        valid_intents = {
            "RISK_QUERY",
            "GRAPH_QUERY",
            "EVENT_QUERY",
            "SHIPPING_QUERY",
            "GENERAL_SUPPLY_CHAIN_QUERY",
        }
        if not isinstance(plan, dict) or plan.get("intent") not in valid_intents:
            raise LLMToolPlanError("Provider returned an unsupported intent.")
        if "tool_calls" in plan or "entity_id" in plan or "node_id" in plan:
            raise LLMToolPlanError("Provider returned tool calls or an entity identifier during intent selection.")
        return str(plan["intent"])

    def execute_for_intent(self, intent: str, query: str) -> tuple[list[str], list[dict[str, Any]]]:
        tools_used: list[str] = []
        evidence: list[dict[str, Any]] = []
        if intent == "SHIPPING_QUERY":
            result = get_shipping_model_info()
            return ["get_shipping_model_info"], [{"tool": "get_shipping_model_info", "result": result}]

        search_result = search_entities(query)
        tools_used.append("search_entities")
        evidence.append({"tool": "search_entities", "result": search_result})
        entity = next(iter(search_result.get("entities", [])), None)
        entity_id = entity.get("node_id") if isinstance(entity, dict) else None
        if not isinstance(entity_id, str):
            if intent in {"RISK_QUERY", "EVENT_QUERY", "GRAPH_QUERY"}:
                raise LLMToolPlanError("Deterministic entity search returned no entity.")
            graph_result = retrieve_graph_evidence(query)
            return tools_used + ["retrieve_graph_evidence"], evidence + [
                {"tool": "retrieve_graph_evidence", "result": graph_result}
            ]

        if intent == "RISK_QUERY":
            graph_result = retrieve_graph_evidence(query)
            risk_result = get_entity_risk(entity_id)
            return (
                tools_used + ["retrieve_graph_evidence", "get_entity_risk"],
                evidence
                + [
                    {"tool": "retrieve_graph_evidence", "result": graph_result},
                    {"tool": "get_entity_risk", "result": risk_result},
                ],
            )
        if intent == "EVENT_QUERY":
            event_result = get_external_events(entity_id)
            return tools_used + ["get_external_events"], evidence + [
                {"tool": "get_external_events", "result": event_result}
            ]
        graph_result = retrieve_graph_evidence(query)
        return tools_used + ["retrieve_graph_evidence"], evidence + [
            {"tool": "retrieve_graph_evidence", "result": graph_result}
        ]


class LLMSupplyChainAgent:
    """Use an LLM for routing and prose, with deterministic fallback."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        fallback: SupplyChainAgent | None = None,
    ) -> None:
        self.provider = provider or OpenAICompatibleProvider()
        self.fallback = fallback or SupplyChainAgent()

    def _fallback_result(self, query: str, reason: str) -> dict[str, Any]:
        result = self.fallback.run(query)
        result["limitations"] = _unique(result.get("limitations", []) + [f"LLM layer unavailable; deterministic fallback used: {reason}"])
        return result

    def run(self, query: str) -> dict[str, Any]:
        try:
            orchestrator = LLMToolAgent(self.provider)
            intent = orchestrator.select_intent(query)
            tools_used, evidence = orchestrator.execute_for_intent(intent, query)
            grounded_evidence = {"query": query, "intent": intent, "tool_results": evidence}
            answer = self.provider.generate(SYSTEM_PROMPT, query, grounded_evidence)
            return {
                "query": query,
                "intent": intent,
                "tools_used": tools_used,
                "answer": answer,
                "evidence": evidence,
                "provenance": _flatten_provenance(evidence),
                "limitations": [
                    "The LLM generates wording only; factual claims must be supported by the listed tool evidence.",
                    "Event exposure is limited to the validated GraphRAG paths returned by the existing retriever.",
                ],
            }
        except (ProviderError, LLMToolPlanError, OSError, ValueError) as exc:
            return self._fallback_result(query, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    print(json.dumps(LLMSupplyChainAgent().run(args.query), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
