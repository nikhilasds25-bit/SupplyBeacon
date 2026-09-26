"""Current-state ranking queries."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .current_state import load_current_state


def _rank(state: dict[str, Any], node_type: str, limit: int = 10) -> list[dict[str, Any]]:
    risk = state["risk"]
    if risk.empty:
        return []
    rows = risk[risk.get("Node_Type", pd.Series(dtype=str)) == node_type].copy()
    if rows.empty:
        return []
    rows["score_numeric"] = pd.to_numeric(rows["Risk_Score"], errors="coerce").fillna(0.0)
    if node_type == "Warehouse":
        rows = rows[pd.to_numeric(rows["Connected_Events"], errors="coerce").fillna(0) > 0]
    else:
        rows = rows[rows["Risk_Level"].isin(["Medium", "High", "Critical"])]
    rows = rows.sort_values(["score_numeric", "Node_Name"], ascending=[False, True]).head(limit)
    return rows.drop(columns=["score_numeric"]).to_dict(orient="records")


def rank_products(limit: int = 10) -> list[dict[str, Any]]:
    return _rank(load_current_state(), "Product", limit)


def rank_warehouses(limit: int = 10) -> list[dict[str, Any]]:
    return _rank(load_current_state(), "Warehouse", limit)


def rank_suppliers(limit: int = 10) -> list[dict[str, Any]]:
    return _rank(load_current_state(), "Supplier", limit)


def current_summary() -> dict[str, Any]:
    state = load_current_state()
    risk = state["risk"]
    if risk.empty:
        return {
            "active_weather_events": 0,
            "relevant_news_alerts": 0,
            "affected_warehouses": 0,
            "affected_suppliers": 0,
            "high_risk_products": 0,
            "critical_nodes": 0,
        }
    current_events = state["events"]
    weather = sum(1 for event in current_events if event.get("Category") == "Weather")
    news = sum(1 for event in current_events if event.get("Category") == "News")
    exposed = risk[pd.to_numeric(risk.get("Connected_Events", 0), errors="coerce").fillna(0) > 0]
    return {
        "active_weather_events": weather,
        "relevant_news_alerts": news,
        "affected_warehouses": int((exposed.get("Node_Type", "") == "Warehouse").sum()),
        "affected_suppliers": int((exposed.get("Node_Type", "") == "Supplier").sum()),
        "high_risk_products": int(
            ((risk.get("Node_Type", "") == "Product") & risk.get("Risk_Level", "").isin(["High", "Critical"])).sum()
        ),
        "critical_nodes": int((risk.get("Risk_Level", "") == "Critical").sum()),
    }
