"""Entry point for graph-based supply chain risk prediction."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import networkx as nx

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from risk_features import build_feature_rows, load_graph
from risk_models import build_risk_reason, compute_risk_score, risk_level
from utils import BUSINESS_IMPACT_EVENTS_CSV, RISK_DIR, create_directory, save_json


OUTPUT_CSV = RISK_DIR / "risk_scores.csv"
OUTPUT_JSON = RISK_DIR / "risk_scores.json"
SUMMARY_JSON = RISK_DIR / "risk_summary.json"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_unresolved_event_count() -> int:
    mapping = pd.read_csv(BUSINESS_IMPACT_EVENTS_CSV, dtype=str, keep_default_na=False)
    unresolved = 0
    for _, row in mapping.iterrows():
        location = str(row.get("Location", "")).strip().lower()
        affected_entity = str(row.get("Affected_Entity", "")).strip().lower()
        if location == "unknown" or affected_entity == "unknown" or not location or not affected_entity:
            unresolved += 1
    return unresolved


def assemble_risk_rows(feature_rows: List[Dict[str, Any]], unresolved_event_count: int) -> List[Dict[str, Any]]:
    risk_rows = []
    for feature in feature_rows:
        score = compute_risk_score(
            {
                "max_severity": feature["max_severity"],
                "average_confidence": feature["average_confidence"],
                "business_risk": feature["business_risk"],
                "event_exposure": feature["Event_Exposure"],
                "degree": feature["degree_connectivity"],
                "product_connectivity": feature["product_connectivity"],
            }
        )
        level = risk_level(score)
        reason = build_risk_reason(
            {
                **feature,
                "min_event_distance": feature.get("Min_Event_Distance"),
                "connected_events": feature.get("Connected_Events", 0),
                "direct_event_count": feature.get("Direct_Event_Count", 0),
                "average_confidence": feature.get("average_confidence", 0.0),
                "max_severity_label": feature.get("Max_Event_Severity", "Unknown"),
                "business_risk": feature.get("business_risk", 0.0),
                "connected_warehouses": feature.get("Connected_Warehouses", 0),
                "connected_locations": feature.get("Connected_Locations", 0),
                "connected_products": feature.get("Connected_Products", 0),
                "connected_suppliers": feature.get("Connected_Suppliers", 0),
            }
        )
        risk_rows.append(
            {
                "Node_ID": feature["Node_ID"],
                "Node_Type": feature["Node_Type"],
                "Node_Name": feature["Node_Name"],
                "Risk_Score": round(score, 3),
                "Risk_Level": level,
                "Event_Exposure": round(feature["Event_Exposure"], 3),
                "Connected_Events": feature["Connected_Events"],
                "Max_Event_Severity": feature["Max_Event_Severity"],
                "Average_Event_Confidence": round(feature["Average_Event_Confidence"], 3),
                "Business_Impact_Risk": round(feature["Business_Impact_Risk"], 3),
                "Graph_Degree": feature["Graph_Degree"],
                "Connected_Products": feature["Connected_Products"],
                "Connected_Suppliers": feature["Connected_Suppliers"],
                "Connected_Warehouses": feature["Connected_Warehouses"],
                "Connected_Locations": feature["Connected_Locations"],
                "Risk_Reason": reason,
            }
        )
    if unresolved_event_count > 0:
        for row in risk_rows:
            if "unresolved_event_note" not in row:
                row["Risk_Reason"] += f" There are {unresolved_event_count} unresolved external event(s) with Unknown location or affected entity."
    return risk_rows


def build_summary(risk_rows: List[Dict[str, Any]], unresolved_event_count: int) -> Dict[str, Any]:
    total_nodes = len(risk_rows)
    levels = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    node_types = {}
    event_exposure_nodes = 0
    scores = []
    for row in risk_rows:
        levels[row["Risk_Level"]] = levels.get(row["Risk_Level"], 0) + 1
        node_types[row["Node_Type"]] = node_types.get(row["Node_Type"], 0) + 1
        if row["Connected_Events"] > 0:
            event_exposure_nodes += 1
        scores.append(row["Risk_Score"])

    sorted_top = sorted(risk_rows, key=lambda item: item["Risk_Score"], reverse=True)[:20]
    return {
        "total_nodes_scored": total_nodes,
        "risk_level_counts": levels,
        "node_type_counts": node_types,
        "nodes_with_event_exposure": event_exposure_nodes,
        "unresolved_event_count": unresolved_event_count,
        "min_risk_score": min(scores) if scores else 0.0,
        "max_risk_score": max(scores) if scores else 0.0,
        "average_risk_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "top_20_highest_risk_nodes": [
            {
                "Node_ID": row["Node_ID"],
                "Node_Type": row["Node_Type"],
                "Node_Name": row["Node_Name"],
                "Risk_Score": row["Risk_Score"],
                "Risk_Level": row["Risk_Level"],
                "Risk_Reason": row["Risk_Reason"],
            }
            for row in sorted_top
        ],
    }


def print_report(summary: Dict[str, Any]) -> None:
    logging.info("Graph-Based Supply Chain Risk Scoring Baseline")
    logging.info("========================================")
    logging.info("Total nodes scored: %s", summary["total_nodes_scored"])
    logging.info("Risk levels: %s", summary["risk_level_counts"])
    logging.info("Nodes with event exposure: %s", summary["nodes_with_event_exposure"])
    logging.info("Unresolved events: %s", summary["unresolved_event_count"])
    logging.info("Risk score range: %.3f to %.3f", summary["min_risk_score"], summary["max_risk_score"])
    logging.info("Average risk score: %.3f", summary["average_risk_score"])
    logging.info("Top 5 highest-risk nodes:")
    for row in summary["top_20_highest_risk_nodes"][:5]:
        logging.info(
            "  %s (%s) - %.3f [%s]",
            row["Node_Name"],
            row["Node_Type"],
            row["Risk_Score"],
            row["Risk_Level"],
        )


def main() -> None:
    configure_logging()
    logging.info("Starting graph-based risk prediction")
    try:
        graph = load_graph()
    except Exception as exc:
        logging.exception("Failed to load graph: %s", exc)
        sys.exit(1)

    try:
        feature_rows = build_feature_rows(graph)
        unresolved_event_count = load_unresolved_event_count()
        risk_rows = assemble_risk_rows(feature_rows, unresolved_event_count)
    except Exception as exc:
        logging.exception("Failed to compute risk rows: %s", exc)
        sys.exit(1)

    if not risk_rows:
        logging.error("No risk rows were generated.")
        sys.exit(1)

    create_directory(RISK_DIR)
    pd.DataFrame(risk_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    save_json(OUTPUT_JSON, risk_rows)
    summary = build_summary(risk_rows, unresolved_event_count)
    save_json(SUMMARY_JSON, summary)
    print_report(summary)


if __name__ == "__main__":
    main()
