from __future__ import annotations

import re
from typing import Any
import networkx as nx

from src.dynamic_risk import current_state

ALLOWED = {
    ("Supplier", "SUPPLIES", "Brand"),
    ("Product", "STORED_AT", "Warehouse"),
    ("Warehouse", "LOCATED_IN", "Location"),
}
ENTITY_TABLES = {"Product": "products", "Supplier": "suppliers", "Warehouse": "warehouses", "Location": "locations"}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def brand_id(value: Any) -> str:
    return f"BRAND:{re.sub(r'[^A-Z0-9]+', '_', str(value or '').strip().upper()).strip('_')}"


def brand_registry(connection: Any) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    base_path = current_state.CURRENT_DIR / "current_supply_chain_graph.graphml"
    if base_path.exists():
        base_graph = nx.read_graphml(str(base_path))
        for node_id, attrs in base_graph.nodes(data=True):
            if attrs.get("node_type") == "Brand":
                registry[brand_id(node_id.removeprefix("BRAND:"))] = {"id": str(node_id), "name": attrs.get("name") or node_id.removeprefix("BRAND:"), "node_type": "Brand", "source": "BASE"}
            elif str(node_id).upper().startswith("BRAND:"):
                registry[brand_id(node_id.removeprefix("BRAND:"))] = {"id": str(node_id), "name": attrs.get("name") or node_id.removeprefix("BRAND:"), "node_type": attrs.get("node_type"), "source": "BASE"}
    for row in connection.execute("SELECT brand FROM products WHERE status='ACTIVE' AND brand IS NOT NULL"):
        key = brand_id(row["brand"])
        registry.setdefault(key, {"id": key, "name": str(row["brand"]).strip(), "node_type": "Brand", "source": "COMPANY"})
    return registry


def duplicate_candidates(rows: list[dict[str, Any]], entity_type: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    name = normalize(record.get("name"))
    matches = []
    for row in rows:
        if record.get("id") and record["id"] == row.get("id"):
            matches.append(row)
        elif record.get("external_id") and record["external_id"] == row.get("external_id"):
            matches.append(row)
        elif entity_type == "Product" and record.get("sku") and record["sku"] == row.get("sku"):
            matches.append(row)
        elif entity_type == "Product" and name and name == normalize(row.get("name")) and normalize(record.get("brand")) == normalize(row.get("brand")):
            matches.append(row)
        elif entity_type in {"Supplier", "Warehouse"} and name == normalize(row.get("name")):
            matches.append(row)
        elif entity_type == "Location" and (normalize(record.get("city")), normalize(record.get("region")), normalize(record.get("country"))) == (normalize(row.get("city")), normalize(row.get("region")), normalize(row.get("country"))):
            matches.append(row)
    return matches


def validate_entity(entity_type: str, payload: dict[str, Any]) -> list[str]:
    errors = []
    if not str(payload.get("name", "")).strip():
        errors.append("name is required")
    if payload.get("status", "ACTIVE") not in {"ACTIVE", "INACTIVE"}:
        errors.append("status must be ACTIVE or INACTIVE")
    if entity_type == "Warehouse" and payload.get("capacity") not in (None, ""):
        try:
            float(payload["capacity"])
        except (TypeError, ValueError):
            errors.append("capacity must be numeric when supplied")
    return errors


def validate_network(connection: Any) -> dict[str, list[dict[str, str]]]:
    issues = {"errors": [], "warnings": [], "info": []}
    active = {}
    for entity_type, table in ENTITY_TABLES.items():
        active[entity_type] = {row["id"]: dict(row) for row in connection.execute(f"SELECT * FROM {table} WHERE status='ACTIVE'")}
    relationships = [dict(row) for row in connection.execute("SELECT * FROM relationships WHERE status='ACTIVE'")]
    brands = brand_registry(connection)
    for row in relationships:
        key = (row["source_type"], row["relationship_type"], row["target_type"])
        if key not in ALLOWED:
            issues["errors"].append({"entity": row["id"], "field": "relationship", "message": "invalid relationship type"})
        if row["source_id"] not in active.get(row["source_type"], {}):
            issues["errors"].append({"entity": row["id"], "field": "source_id", "message": "source endpoint missing or inactive"})
        if row["target_type"] == "Brand" and brand_id(row["target_id"].removeprefix("BRAND:")) not in brands:
            issues["errors"].append({"entity": row["id"], "field": "target_id", "message": "supplier-brand reference is not available"})
        if row["target_type"] != "Brand" and row["target_id"] not in active.get(row["target_type"], {}):
            issues["errors"].append({"entity": row["id"], "field": "target_id", "message": "target endpoint missing or inactive"})
    stored_products = {row["source_id"] for row in relationships if row["relationship_type"] == "STORED_AT"}
    located_warehouses = {row["source_id"] for row in relationships if row["relationship_type"] == "LOCATED_IN"}
    for product in active["Product"].values():
        if product["id"] not in stored_products:
            issues["warnings"].append({"entity": product["id"], "field": "warehouse", "message": "product has no warehouse"})
        if not product.get("brand"):
            issues["warnings"].append({"entity": product["id"], "field": "brand", "message": "product has no brand"})
    for warehouse in active["Warehouse"].values():
        if warehouse["id"] not in located_warehouses:
            issues["warnings"].append({"entity": warehouse["id"], "field": "location_id", "message": "warehouse has no location"})
    for supplier in active["Supplier"].values():
        if not any(row["source_id"] == supplier["id"] and row["relationship_type"] == "SUPPLIES" for row in relationships):
            issues["warnings"].append({"entity": supplier["id"], "field": "relationships", "message": "supplier has no SUPPLIES relationship"})
    return issues
