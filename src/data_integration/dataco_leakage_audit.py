"""Classify DataCo fields for prediction-time availability and privacy."""
from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from .dataco_loader import load_main_dataset
    from .dataco_profile import ANALYSIS_DIR, write_json
except ImportError:
    from dataco_loader import load_main_dataset
    from dataco_profile import ANALYSIS_DIR, write_json

POST_OUTCOME = {
    "Days for shipping (real)", "Delivery Status", "Late_delivery_risk",
    "Days for shipment (real)", "Actual Shipping Days", "shipping date (DateOrders)",
    "Order Status", "Order Profit Per Order",
}
IDENTIFIERS = {
    "Order Id", "Order Item Id", "Order Customer Id", "Customer Id",
    "Product Card Id", "Product Image", "Category Id", "Department Id",
    "Product Category Id",
}
PII_TERMS = ("customer", "email", "password", "street", "address", "latitude", "longitude", "zip", "postal")
SHIPMENT_TERMS = ("days for shipment (scheduled)",)


def classify_column(column: str) -> dict[str, Any]:
    label = str(column).strip()
    lower = label.lower()
    if any(term in lower for term in PII_TERMS):
        category = "E. possible PII"
    elif label in IDENTIFIERS or lower.endswith(" id") or lower == "id":
        category = "D. identifiers"
    elif label in POST_OUTCOME:
        category = "F. target/outcome variables"
    elif any(term in lower for term in SHIPMENT_TERMS):
        category = "B. available during shipment"
    else:
        category = "A. available before prediction"
    if label in {"Late_delivery_risk"}:
        category = "F. target/outcome variables"
    return {
        "classification": category,
        "recommended_feature_use": "exclude",
        "reason": "Contains an identifier, possible PII, or outcome/post-outcome information.",
    } if category in {"D. identifiers", "E. possible PII", "F. target/outcome variables"} else {
        "classification": category,
        "recommended_feature_use": "candidate subject to prediction-time validation",
        "reason": "Potentially usable only if it is known at the declared prediction timestamp.",
    }


def audit_leakage(df: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in df.columns]
    classifications = {column: classify_column(column) for column in columns}
    exclusions = [
        column for column, item in classifications.items()
        if item["recommended_feature_use"] == "exclude"
    ]
    explicit_fields = [
        "Days for shipping (real)", "Days for shipment (scheduled)", "Delivery Status",
        "Late_delivery_risk", "shipping date (DateOrders)", "Order Status",
        "Shipping Mode", "Product Name", "Category Name", "Market", "Order Region",
        "Order Country", "Order City", "Order Item Quantity", "Sales",
        "Order Item Total", "Order Profit Per Order",
    ]
    return {
        "status": "available" if not df.empty else "unavailable",
        "prediction_time": "Must be declared before feature selection; this phase assumes historical pre-shipment prediction.",
        "classifications": classifications,
        "explicit_field_classifications": {
            field: classifications.get(field, classify_column(field)) for field in explicit_fields
        },
        "recommended_exclusion_list": exclusions,
        "privacy_findings": [
            "Do not place customer names, email, address/street, customer IDs, passwords, coordinates, or other direct identifiers in graph nodes, embeddings, RAG documents, or model features.",
            "Retain order and product identifiers only in a controlled audit mapping; use irreversible or scoped pseudonyms for later experiments.",
        ],
    }


def main() -> None:
    df, _ = load_main_dataset()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS_DIR / "dataco_leakage_report.json", audit_leakage(df))


if __name__ == "__main__":
    main()
