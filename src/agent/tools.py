"""Read-only tool wrappers around existing ElectroGraph capabilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.graphrag.explanation_builder import build_explanation
from src.graphrag.retriever import EvidenceRetriever

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_METADATA_DIR = BASE_DIR / "deploy" / "seed" / "analysis" / "dataco" / "modeling"
METRICS_PATH = MODEL_METADATA_DIR / "baseline_metrics.csv"
RUN_SUMMARY_PATH = MODEL_METADATA_DIR / "run_summary.json"
FEATURE_AUDIT_PATH = MODEL_METADATA_DIR / "feature_audit.json"

PROVENANCE_VALUES = {"OBSERVED", "SIMULATED", "DERIVED", "MODEL_OUTPUT", "EXTERNAL_API"}


def _retriever() -> EvidenceRetriever:
    return EvidenceRetriever()


def _provenance(evidence: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("provenance"), str) and value.get("provenance") in PROVENANCE_VALUES:
                records.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = json.dumps(record, sort_keys=True, default=str)
        unique[key] = record
    return list(unique.values())


def search_entities(query: str) -> dict[str, Any]:
    """Find graph entities using the retriever's existing matching logic."""
    retriever = _retriever()
    entity_ids = retriever._matching_entities(query, limit=10)
    entities = [
        {
            "node_id": node_id,
            "node_type": retriever.graph.nodes[node_id].get("node_type", ""),
            "name": retriever.graph.nodes[node_id].get("name", ""),
            "provenance": "OBSERVED",
        }
        for node_id in entity_ids
    ]
    return {
        "query": query,
        "entities": entities,
        "provenance": _provenance(entities),
        "limitations": ["Entity search uses token matching over existing graph node IDs, names, and types."],
    }


def get_entity_risk(entity_id: str) -> dict[str, Any]:
    """Return existing risk-model evidence for one graph entity."""
    retriever = _retriever()
    if entity_id not in retriever.graph:
        return {
            "entity_id": entity_id,
            "risk_evidence": [],
            "provenance": [],
            "limitations": ["The requested entity does not exist in the current graph."],
        }
    facts = retriever._node_facts(entity_id)
    risk_evidence = [fact for fact in facts if fact.get("provenance") == "MODEL_OUTPUT"]
    return {
        "entity_id": entity_id,
        "risk_evidence": risk_evidence,
        "provenance": _provenance(risk_evidence),
        "limitations": ["Risk values are existing model outputs; this tool does not recompute them."],
    }


def retrieve_graph_evidence(query: str) -> dict[str, Any]:
    """Use the validated GraphRAG retriever and deterministic explanation."""
    result = _retriever().retrieve(query)
    explanation = build_explanation(result)
    return {
        "query": query,
        "answer": explanation["answer"],
        "entities": result["entities"],
        "graph_paths": result["graph_paths"],
        "events": result["events"],
        "risk_evidence": result["risk_evidence"],
        "provenance": _provenance(result),
        "provenance_summary": explanation["provenance_summary"],
        "limitations": result["limitations"],
    }


def get_external_events(entity_id: str) -> dict[str, Any]:
    """Retrieve only events connected through the retriever's valid path rules."""
    retriever = _retriever()
    if entity_id not in retriever.graph:
        return {
            "entity_id": entity_id,
            "events": [],
            "provenance": [],
            "limitations": ["The requested entity does not exist in the current graph."],
        }
    entity_name = retriever.graph.nodes[entity_id].get("name", entity_id)
    result = retriever.retrieve(f"Are there external events affecting {entity_name}?", max_entities=1)
    if not result["entities"] or result["entities"][0]["node_id"] != entity_id:
        return {
            "entity_id": entity_id,
            "events": [],
            "provenance": [],
            "limitations": ["No matching entity-specific retrieval result was available."],
        }
    return {
        "entity_id": entity_id,
        "events": result["events"],
        "provenance": _provenance(result["events"]),
        "limitations": result["limitations"],
    }


def get_shipping_model_info() -> dict[str, Any]:
    """Return frozen DataCo shipping-model analysis artifacts."""
    if not METRICS_PATH.exists() or not RUN_SUMMARY_PATH.exists():
        return {
            "target": "Days for shipping (real)",
            "records": 180519,
            "metrics": [],
            "provenance": [{"provenance": "MODEL_OUTPUT", "source": "DataCo analysis artifacts"}],
            "limitations": ["Frozen DataCo modeling artifacts are unavailable."],
        }
    metrics = pd.read_csv(METRICS_PATH)
    summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    feature_audit = json.loads(FEATURE_AUDIT_PATH.read_text(encoding="utf-8-sig")) if FEATURE_AUDIT_PATH.exists() else {}
    required_features = [name for name, details in feature_audit.get("features", {}).items() if details.get("included") is True]
    test_metrics = metrics[metrics["split"] == "test"].to_dict(orient="records")
    return {
        "target": summary.get("target", "Days for shipping (real)"),
        "records": 180519,
        "best_model": summary.get("best_model_by_test_mae"),
        "train_rows": summary.get("train_rows"),
        "validation_rows": summary.get("validation_rows"),
        "test_rows": summary.get("test_rows"),
        "test_metrics": test_metrics,
        "required_features": required_features,
        "prediction_point": feature_audit.get("prediction_point", "Unavailable"),
        "provenance": [{"provenance": "MODEL_OUTPUT", "source": "deploy/seed/analysis/dataco/modeling"}],
        "limitations": [
            "This predicts historical DataCo shipping duration and is not equivalent to ElectroGraph Lead_Time_Days.",
            "No inference is run by this tool.",
        ],
    }
