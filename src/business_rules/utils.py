"""Utility helpers for the business rules engine."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EVENTS_CSV = BASE_DIR / "data" / "external" / "events" / "events.csv"
EVENTS_JSON = BASE_DIR / "data" / "external" / "events" / "events.json"
SUPPLYCHAIN_CSV = BASE_DIR / "data" / "processed" / "electronics_supplychain_dataset.csv"
BRAND_SUPPLIER_CSV = BASE_DIR / "data" / "processed" / "brand_supplier_mapping.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "business_impacts"
OUTPUT_CSV = OUTPUT_DIR / "business_impact_events.csv"
OUTPUT_JSON = OUTPUT_DIR / "business_impact_events.json"

# Common Indian supply chain cities and states for location inference
WAREHOUSE_CITY_KEYWORDS = [
    "Chennai",
    "Bengaluru",
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Kochi",
    "Hyderabad",
    "Pune",
    "Noida",
    "Gurugram",
    "Gurgaon",
    "Ahmedabad",
    "Surat",
    "Kolkata",
    "Visakhapatnam",
    "Tamil Nadu",
    "Karnataka",
    "Maharashtra",
    "Kerala",
    "Telangana",
    "Gujarat",
    "Uttar Pradesh",
    "West Bengal",
    "Andhra Pradesh",
]

STATE_TO_REGION = {
    "Tamil Nadu": "South India",
    "Karnataka": "South India",
    "Kerala": "South India",
    "Telangana": "South India",
    "Andhra Pradesh": "South India",
    "Maharashtra": "West India",
    "Gujarat": "West India",
    "West Bengal": "East India",
    "Uttar Pradesh": "North India",
    "Delhi": "North India",
}

DEFAULT_LOCATION = "Unknown"
DEFAULT_SUPPLIER = "Unknown"
DEFAULT_WAREHOUSE = "Unknown"
DEFAULT_CATEGORY = "Unknown"


def create_directory(path: Path) -> None:
    """Create a directory and its parents if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def save_dataframe_outputs(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    """Save a DataFrame to CSV and JSON output paths."""
    create_directory(csv_path.parent)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    logging.info("Saved business impact outputs to %s and %s", csv_path, json_path)


def normalize_text(value: Any) -> str:
    """Normalize text for deterministic matching."""
    if value is None:
        return ""
    return str(value).strip().lower()


def split_location_tokens(location_text: Any) -> List[str]:
    """Split a comma-separated location string into normalized tokens."""
    if not location_text:
        return []
    raw_tokens = str(location_text).split(",")
    tokens = []
    for token in raw_tokens:
        normalized = token.strip()
        if normalized:
            tokens.append(normalized)
    return tokens


def find_first_match(text: str, candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate that appears in text using fuzzy case-insensitive matching."""
    lower_text = normalize_text(text)
    for candidate in candidates:
        if normalize_text(candidate) and normalize_text(candidate) in lower_text:
            return candidate
    return None


def infer_region_from_state(token: str) -> Optional[str]:
    """Infer a logistics region from a state or city token."""
    if not token:
        return None
    canonical = token.strip()
    return STATE_TO_REGION.get(canonical)


def find_matching_mapping_by_brand(affected_entity: str, mapping_df: pd.DataFrame) -> Dict[str, str]:
    """Find supplier mapping row by brand name."""
    if not affected_entity:
        return {}
    normalized = normalize_text(affected_entity)
    for _, row in mapping_df.iterrows():
        if normalize_text(row.get("Brand")) == normalized:
            return {
                "Brand": row.get("Brand", DEFAULT_CATEGORY),
                "Supplier": row.get("Supplier", DEFAULT_SUPPLIER),
                "Warehouse": row.get("Warehouse", DEFAULT_WAREHOUSE),
                "Region": row.get("Region", DEFAULT_CATEGORY),
            }
    return {}


def find_matching_mapping_by_location(location: str, mapping_df: pd.DataFrame) -> Dict[str, str]:
    """Find a supplier mapping row that best matches an event location."""
    tokens = split_location_tokens(location)
    # First prefer exact warehouse city matches
    for token in tokens:
        for _, row in mapping_df.iterrows():
            warehouse = str(row.get("Warehouse", ""))
            if normalize_text(token) in normalize_text(warehouse):
                return {
                    "Brand": row.get("Brand", DEFAULT_CATEGORY),
                    "Supplier": row.get("Supplier", DEFAULT_SUPPLIER),
                    "Warehouse": row.get("Warehouse", DEFAULT_WAREHOUSE),
                    "Region": row.get("Region", DEFAULT_CATEGORY),
                }
    # Next prefer region matches based on state keywords
    for token in tokens:
        region = infer_region_from_state(token)
        if region:
            matches = mapping_df[mapping_df["Region"].str.lower() == region.lower()]
            if not matches.empty:
                row = matches.iloc[0]
                return {
                    "Brand": row.get("Brand", DEFAULT_CATEGORY),
                    "Supplier": row.get("Supplier", DEFAULT_SUPPLIER),
                    "Warehouse": row.get("Warehouse", DEFAULT_WAREHOUSE),
                    "Region": row.get("Region", DEFAULT_CATEGORY),
                }
    # Finally match on explicit region names
    for token in tokens:
        for _, row in mapping_df.iterrows():
            if normalize_text(token) == normalize_text(row.get("Region")):
                return {
                    "Brand": row.get("Brand", DEFAULT_CATEGORY),
                    "Supplier": row.get("Supplier", DEFAULT_SUPPLIER),
                    "Warehouse": row.get("Warehouse", DEFAULT_WAREHOUSE),
                    "Region": row.get("Region", DEFAULT_CATEGORY),
                }
    return {}


def derive_affected_category(affected_entity: str, supply_df: pd.DataFrame, default_category: str) -> str:
    """Derive a likely affected category from the brand present in the supply dataset."""
    if not affected_entity or affected_entity == "Unknown":
        return default_category
    normalized = normalize_text(affected_entity)
    matches = supply_df[supply_df["Brand"].fillna("").apply(normalize_text) == normalized]
    categories = matches["Category"].dropna().unique().tolist()
    if categories:
        unique_categories = sorted(set(categories))
        return ", ".join(unique_categories)
    return default_category
