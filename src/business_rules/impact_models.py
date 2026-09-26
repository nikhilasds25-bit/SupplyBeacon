"""Impact model constants for the business rules engine."""
from __future__ import annotations

from typing import List

OUTPUT_FIELDS: List[str] = [
    "Event_ID",
    "Date",
    "Event_Type",
    "Title",
    "Description",
    "Location",
    "Affected_Entity",
    "Category",
    "Severity",
    "Source",
    "Confidence",
    "Impact_Type",
    "Impact_Level",
    "Risk_Score",
    "Possible_Supply_Chain_Effect",
    "Affected_Supplier",
    "Affected_Warehouse",
    "Affected_Location",
    "Affected_Category",
    "Rule_Triggered",
]

RISK_BUCKETS = ["High", "Medium", "Low"]
