"""Utility helpers for dynamic supply chain graph construction."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EVENTS_CSV = BASE_DIR / "data" / "external" / "events" / "events.csv"
BUSINESS_EVENTS_CSV = BASE_DIR / "data" / "processed" / "business_impacts" / "business_impact_events.csv"
SUPPLYCHAIN_CSV = BASE_DIR / "data" / "processed" / "electronics_supplychain_dataset.csv"
BRAND_SUPPLIER_CSV = BASE_DIR / "data" / "processed" / "brand_supplier_mapping.csv"
GRAPH_DIR = BASE_DIR / "data" / "graph"
GRAPHML_PATH = GRAPH_DIR / "supply_chain_graph.graphml"
NODES_CSV_PATH = GRAPH_DIR / "nodes.csv"
EDGES_CSV_PATH = GRAPH_DIR / "edges.csv"
SUMMARY_JSON_PATH = GRAPH_DIR / "graph_summary.json"


def create_directory(path: Path) -> None:
    """Create a directory and its parents if needed."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with safe string handling."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: Any) -> str:
    """Normalize text values for deterministic matching."""
    if value is None:
        return ""
    normalized = str(value).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_id(value: Any) -> str:
    """Normalize a value for use in stable node IDs."""
    value_text = normalize_text(value)
    return re.sub(r"[^A-Za-z0-9_-]", "_", value_text)


def build_node_id(prefix: str, name: Any) -> str:
    """Create a stable node ID from a prefix and a name."""
    normalized = normalize_id(name)
    return f"{prefix}:{normalized}"


def build_event_node_id(event_id: Any) -> str:
    return build_node_id("EVENT", event_id)


def build_product_node_id(product_id: Any) -> str:
    return build_node_id("PRODUCT", product_id)


def build_product_model_node_id(brand: Any, product_model: Any) -> str:
    return build_node_id("PRODUCT_MODEL", f"{normalize_text(brand)}__{normalize_text(product_model)}")


def build_brand_node_id(brand: Any) -> str:
    return build_node_id("BRAND", brand)


def build_supplier_node_id(supplier: Any) -> str:
    return build_node_id("SUPPLIER", supplier)


def build_warehouse_node_id(warehouse: Any) -> str:
    return build_node_id("WAREHOUSE", warehouse)


def build_category_node_id(category: Any) -> str:
    return build_node_id("CATEGORY", category)


def build_location_node_id(location: Any) -> str:
    return build_node_id("LOCATION", location)


def safe_node_attr(value: Any) -> str:
    """Serialize node attributes to a string-safe representation."""
    if value is None:
        return ""
    return normalize_text(value)


def get_first_location_token(location: str) -> str:
    """Extract a meaningful location token from a comma-separated location string."""
    if not location:
        return ""
    tokens = [normalize_text(token) for token in str(location).split(",") if normalize_text(token)]
    return tokens[0] if tokens else ""


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required input CSV files for graph construction."""
    events_df = load_dataframe(EVENTS_CSV)
    business_df = load_dataframe(BUSINESS_EVENTS_CSV)
    supply_df = load_dataframe(SUPPLYCHAIN_CSV)
    mapping_df = load_dataframe(BRAND_SUPPLIER_CSV)
    return events_df, business_df, supply_df, mapping_df


def to_json_string(attributes: Dict[str, Any]) -> str:
    """Serialize a dictionary to JSON with stable key ordering."""
    return json.dumps(attributes, ensure_ascii=False, sort_keys=True)
