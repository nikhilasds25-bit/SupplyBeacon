"""Business rule engine for supply chain event impact assessment."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from impact_models import OUTPUT_FIELDS
from rules import (
    compute_risk_score,
    derive_impact_type,
    derive_possible_effect,
    derive_rule_triggered,
    derive_affected_category as derive_default_category,
    get_rule_config,
    impact_level_from_score,
)
from utils import (
    BRAND_SUPPLIER_CSV,
    DEFAULT_CATEGORY,
    DEFAULT_LOCATION,
    DEFAULT_SUPPLIER,
    DEFAULT_WAREHOUSE,
    EVENTS_CSV,
    OUTPUT_CSV,
    OUTPUT_DIR,
    OUTPUT_JSON,
    SUPPLYCHAIN_CSV,
    create_directory,
    derive_affected_category as derive_category_from_brand,
    find_matching_mapping_by_brand,
    find_matching_mapping_by_location,
    load_dataframe,
    normalize_text,
    save_dataframe_outputs,
    split_location_tokens,
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events_df = load_dataframe(EVENTS_CSV)
    mapping_df = load_dataframe(BRAND_SUPPLIER_CSV)
    supply_df = load_dataframe(SUPPLYCHAIN_CSV)
    return events_df, mapping_df, supply_df


def choose_affected_location(location: str, location_match: Dict[str, str]) -> str:
    if location_match and location_match.get("Warehouse") not in (None, "", DEFAULT_WAREHOUSE):
        warehouse_name = str(location_match["Warehouse"]).strip()
        if warehouse_name.endswith("Warehouse"):
            return warehouse_name.replace("Warehouse", "").strip()
        return warehouse_name

    tokens = split_location_tokens(location)
    for token in tokens:
        normalized = normalize_text(token)
        if normalized and normalized != "india":
            return token
    if tokens:
        return tokens[0]
    return DEFAULT_LOCATION


def choose_affected_supplier(entity_match: Dict[str, str], _) -> str:
    supplier = entity_match.get("Supplier") if entity_match else ""
    return supplier or DEFAULT_SUPPLIER


def choose_affected_warehouse(
    entity_match: Dict[str, str], location_match: Dict[str, str]
) -> str:
    warehouse = entity_match.get("Warehouse") if entity_match else ""
    if warehouse and warehouse != DEFAULT_WAREHOUSE:
        return warehouse
    warehouse = location_match.get("Warehouse") if location_match else ""
    return warehouse or DEFAULT_WAREHOUSE


def choose_affected_region(
    entity_match: Dict[str, str], location_match: Dict[str, str]
) -> str:
    region = entity_match.get("Region") if entity_match else ""
    if region and region != DEFAULT_CATEGORY:
        return region
    region = location_match.get("Region") if location_match else ""
    return region or DEFAULT_CATEGORY


def extract_event_row(event: pd.Series, mapping_df: pd.DataFrame, supply_df: pd.DataFrame) -> Dict[str, object]:
    event_type = str(event.get("Event_Type", "")).strip()
    severity = str(event.get("Severity", "")).strip()
    confidence = float(event.get("Confidence", 0.0) or 0.0)
    location_value = str(event.get("Location", "")).strip()
    affected_entity = str(event.get("Affected_Entity", "")).strip()

    entity_match = find_matching_mapping_by_brand(affected_entity, mapping_df)
    location_match = find_matching_mapping_by_location(location_value, mapping_df)

    impacted_location = choose_affected_location(location_value, location_match)
    impacted_supplier = choose_affected_supplier(entity_match, location_match)
    impacted_warehouse = choose_affected_warehouse(entity_match, location_match)

    rule_config = get_rule_config(event_type)
    default_category = derive_default_category(event_type, DEFAULT_CATEGORY)
    affected_category = derive_category_from_brand(affected_entity, supply_df, default_category)

    risk_score = compute_risk_score(
        severity=severity,
        confidence=confidence,
        base_risk=float(rule_config.get("Base_Risk", 0.6)),
    )
    impact_level = impact_level_from_score(risk_score)

    business_event = {
        "Impact_Type": derive_impact_type(event_type),
        "Impact_Level": impact_level,
        "Risk_Score": risk_score,
        "Possible_Supply_Chain_Effect": derive_possible_effect(event_type),
        "Affected_Supplier": impacted_supplier,
        "Affected_Warehouse": impacted_warehouse,
        "Affected_Location": impacted_location,
        "Affected_Category": affected_category,
        "Rule_Triggered": derive_rule_triggered(event_type),
    }

    output_event = {**event.to_dict(), **business_event}
    return {key: output_event.get(key, DEFAULT_CATEGORY) for key in OUTPUT_FIELDS}


def generate_report(rows: List[Dict[str, object]]) -> str:
    total_events = len(rows)
    impact_levels = [row.get("Impact_Level", "Unknown") for row in rows]

    high_risk = sum(1 for level in impact_levels if level == "High")
    medium_risk = sum(1 for level in impact_levels if level == "Medium")
    low_risk = sum(1 for level in impact_levels if level == "Low")
    matched_suppliers = sum(1 for row in rows if row.get("Affected_Supplier") not in (DEFAULT_SUPPLIER, ""))
    matched_warehouses = sum(1 for row in rows if row.get("Affected_Warehouse") not in (DEFAULT_WAREHOUSE, ""))
    no_reliable_match = sum(
        1
        for row in rows
        if row.get("Affected_Supplier") in (DEFAULT_SUPPLIER, "")
        and row.get("Affected_Warehouse") in (DEFAULT_WAREHOUSE, "")
        and row.get("Affected_Location") in (DEFAULT_LOCATION, "")
    )

    report_lines = [
        "Business Rule Engine Report",
        "---------------------------",
        f"Total events processed: {total_events}",
        f"Total impacted events: {total_events}",
        f"High-risk events: {high_risk}",
        f"Medium-risk events: {medium_risk}",
        f"Low-risk events: {low_risk}",
        f"Events matched to suppliers: {matched_suppliers}",
        f"Events matched to warehouses: {matched_warehouses}",
        f"Events with no reliable match: {no_reliable_match}",
    ]
    return "\n".join(report_lines)


def main() -> None:
    configure_logging()
    logging.info("Starting business rule engine")

    try:
        events_df, mapping_df, supply_df = load_inputs()
    except Exception as exc:
        logging.exception("Failed to load input data: %s", exc)
        sys.exit(1)

    business_rows = []
    for _, event in events_df.iterrows():
        row = extract_event_row(event, mapping_df, supply_df)
        business_rows.append(row)

    if not business_rows:
        logging.warning("No events found to process.")
        sys.exit(0)

    output_df = pd.DataFrame(business_rows, columns=OUTPUT_FIELDS)
    create_directory(OUTPUT_DIR)
    save_dataframe_outputs(output_df, OUTPUT_CSV, OUTPUT_JSON)

    report = generate_report(business_rows)
    logging.info("\n%s", report)


if __name__ == "__main__":
    main()
