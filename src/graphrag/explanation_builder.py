"""Deterministic human-readable explanations over structured GraphRAG results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .retriever import EvidenceRetriever
except ImportError:
    from retriever import EvidenceRetriever

BASE_DIR = Path(__file__).resolve().parents[2]
ANSWER_DIR = BASE_DIR / "data" / "analysis" / "graphrag" / "answers"


def _facts(result: dict[str, Any], predicate: str) -> list[dict[str, Any]]:
    return [fact for fact in result.get("risk_evidence", []) if fact.get("predicate") == predicate]


def _path_text(path: dict[str, Any]) -> str:
    return " -> ".join(path.get("nodes", []))


def build_explanation(result: dict[str, Any]) -> dict[str, Any]:
    """Create an answer without adding claims beyond retrieved evidence."""
    entities = result.get("entities", [])
    entity = entities[0] if entities else {"name": "No matching entity", "node_id": ""}
    name = entity.get("name") or entity.get("node_id") or "the requested entity"
    score = _facts(result, "Risk_Score")
    level = _facts(result, "Risk_Level")
    score_text = (
        f"{level[0]['value']} scenario risk score of {score[0]['value']}"
        if score and level
        else "no current graph risk score available"
    )
    relevant_paths = [
        path for path in result.get("graph_paths", [])
        if entity.get("node_id") in path.get("nodes", [])
    ]
    path_text = "; ".join(_path_text(path) for path in relevant_paths[:3])
    path_text = path_text or "no bounded graph path was retrieved"

    events = result.get("events", [])
    regional_paths = [
        path for path in relevant_paths
        if any(edge.get("exposure_scope") == "REGIONAL_RULE" for edge in path.get("edges", []))
    ]
    regional_text = ""
    if regional_paths:
        regional_edges = [
            edge
            for path in regional_paths
            for edge in path.get("edges", [])
            if edge.get("exposure_scope") == "REGIONAL_RULE"
        ]
        locations = ", ".join(dict.fromkeys(
            str(edge.get("exposure_location"))
            for edge in regional_edges
            if edge.get("exposure_location")
        ))
        regional_text = (
            f" {name} has regional scenario exposure because {locations} was included "
            "in the current warning; this does not prove the event occurred at this warehouse."
        )
    if events:
        event_text = (
            " Relevant external events include "
            + "; ".join(event.get("name", event.get("node_id", "unresolved event")) for event in events[:3])
            + ". These events may contribute to risk or be associated with the current scenario; causation is not established."
            + regional_text
        )
        channels = [
            f"{event.get('event_domain')} warning sourced from {event.get('source_channel')}"
            for event in events[:3]
            if event.get("event_domain") and event.get("source_channel")
        ]
        if channels:
            event_text += " Evidence channels: " + "; ".join(channels) + "."
    else:
        event_text = " No relevant external event was connected through a valid supply-chain exposure path."

    impacts = []
    for event in events:
        impact = event.get("business_impact", {})
        if impact:
            impacts.append(
                f"{impact.get('Impact_Type', 'unspecified impact')} ({impact.get('Possible_Supply_Chain_Effect', 'unspecified effect')})"
            )
    impact_text = (
        " Derived business-impact evidence includes " + "; ".join(dict.fromkeys(impacts)) + "."
        if impacts
        else " No derived business-impact record was retrieved."
    )
    simulated = [
        str(fact.get("value"))
        for fact in result.get("provenance", [])
        if fact.get("provenance") == "SIMULATED"
    ]
    simulated_text = (
        " Simulated evidence includes " + ", ".join(dict.fromkeys(simulated[:5])) + "."
        if simulated
        else ""
    )
    unresolved = [event for event in events if event.get("unresolved")]
    unresolved_text = (
        " Unresolved evidence remains for: "
        + "; ".join(event.get("name", event.get("node_id", "unknown event")) for event in unresolved[:3])
        + "."
        if unresolved
        else ""
    )
    answer = (
        f"{name} has {score_text}. Relevant graph evidence includes {path_text}."
        f"{event_text}{impact_text}{simulated_text} "
        "Provenance warning: OBSERVED facts come from existing graph artifacts; "
        "SIMULATED facts include supplier/warehouse mappings; DERIVED facts include graph paths and business impacts; "
        "MODEL_OUTPUT facts include the existing risk result; EXTERNAL_API facts come from retained event records. "
        "This retrieval does not establish that an event caused a delay."
        f"{unresolved_text}"
    )
    return {
        "query": result.get("query", ""),
        "answer": answer,
        "provenance_summary": {
            "OBSERVED": "Existing product/entity facts from graph artifacts.",
            "SIMULATED": "Existing supplier/warehouse mapping evidence.",
            "DERIVED": "Graph paths and business-impact interpretations.",
            "MODEL_OUTPUT": "Existing risk score, level, and risk explanation.",
            "EXTERNAL_API": "Existing NewsAPI/OpenWeather event records.",
        },
        "structured_evidence": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Why is Samsung Galaxy F05 considered risky?")
    parser.add_argument("--input", type=Path, help="Existing structured retriever JSON.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (
        json.loads(args.input.read_text(encoding="utf-8"))
        if args.input
        else EvidenceRetriever().retrieve(args.query)
    )
    answer = build_explanation(result)
    output = args.output or ANSWER_DIR / "samsung_galaxy_f05_risk_answer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(answer, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(answer["answer"])
    print(f"Saved answer: {output}")


if __name__ == "__main__":
    main()
