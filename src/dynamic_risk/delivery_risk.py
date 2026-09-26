"""Transparent rule-based current delivery-risk scenarios."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .current_state import BASE_DIR, load_current_state


def estimate_delivery_risk(product_id: str) -> dict[str, Any]:
    state = load_current_state()
    dataset_path = BASE_DIR / "data" / "processed" / "electronics_supplychain_dataset.csv"
    if not dataset_path.exists():
        return {
            "product_id": product_id,
            "status": "insufficient_evidence",
            "message": "Baseline delivery information is unavailable.",
            "evidence": [],
            "provenance": ["DERIVED", "SIMULATED"],
        }
    dataset = pd.read_csv(dataset_path, dtype=str).fillna("")
    numeric_id = str(product_id).split(":", 1)[-1]
    rows = dataset[dataset["Product_ID"].astype(str) == numeric_id]
    if rows.empty or not rows.iloc[0].get("Lead_Time_Days"):
        return {
            "product_id": product_id,
            "status": "insufficient_evidence",
            "message": "Baseline Lead_Time_Days is unavailable for this product.",
            "evidence": [],
            "provenance": ["DERIVED", "SIMULATED"],
        }
    row = rows.iloc[0]
    baseline = float(row["Lead_Time_Days"])
    risk_rows = state["risk"]
    risk_row = risk_rows[risk_rows["Node_ID"].astype(str) == product_id] if not risk_rows.empty else pd.DataFrame()
    risk = risk_row.iloc[0].to_dict() if not risk_row.empty else {}
    level = risk.get("Risk_Level", "Unknown")
    event_count = int(float(risk.get("Connected_Events", 0) or 0))
    delay_max = {"Critical": 3, "High": 2, "Medium": 1}.get(level, 0) if event_count else 0
    return {
        "product_id": product_id,
        "status": "scenario_estimate",
        "baseline_delivery_days": baseline,
        "current_risk_level": level,
        "scenario_delay_days_min": 0 if delay_max == 0 else 1,
        "scenario_delay_days_max": delay_max,
        "estimated_delivery_days_min": baseline if delay_max == 0 else baseline + 1,
        "estimated_delivery_days_max": baseline + delay_max,
        "primary_risk_driver": risk.get("Risk_Reason", "No current event exposure detected."),
        "evidence": [
            {"field": "Lead_Time_Days", "value": baseline, "provenance": "SIMULATED"},
            {"field": "current_risk_level", "value": level, "provenance": "MODEL_OUTPUT"},
            {"field": "current_event_exposure", "value": event_count, "provenance": "DERIVED"},
        ],
        "message": "This is a rule-based scenario estimate, not a learned causal prediction.",
        "provenance": ["SIMULATED", "DERIVED", "MODEL_OUTPUT"],
    }
