from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from src.dynamic_risk.current_state import CURRENT_DIR

from .repository import generated_dir
from .validators import brand_id


def build_overlay(connection: Any, validation: dict[str, Any]) -> dict[str, Any]:
    if validation["errors"]:
        raise ValueError("Cannot publish a network with validation errors")
    graph = nx.MultiDiGraph()
    for table, node_type in (("products", "Product"), ("suppliers", "Supplier"), ("warehouses", "Warehouse"), ("locations", "Location")):
        for row in connection.execute(f"SELECT * FROM {table} WHERE status='ACTIVE'"):
            attrs = {key: value for key, value in dict(row).items() if value is not None}
            attrs["node_type"] = node_type
            attrs["provenance"] = "USER_ENTERED"
            graph.add_node(row["id"], **attrs)
    for row in connection.execute("SELECT * FROM relationships WHERE status='ACTIVE'"):
        graph.add_edge(row["source_id"], row["target_id"], relationship=row["relationship_type"], provenance="USER_ENTERED")
    brands = {str(row["brand"]).strip() for row in connection.execute("SELECT brand FROM products WHERE status='ACTIVE'") if row["brand"]}
    for brand in brands:
        normalized_id = brand_id(brand)
        graph.add_node(normalized_id, node_type="Brand", name=brand, provenance="USER_ENTERED / DERIVED_FROM_MASTER_ATTRIBUTE")
        for row in connection.execute("SELECT id FROM products WHERE status='ACTIVE' AND brand=?", (brand,)):
            graph.add_edge(normalized_id, row["id"], relationship="PRODUCES", provenance="USER_ENTERED / DERIVED_FROM_MASTER_ATTRIBUTE")
    base_path = CURRENT_DIR / "current_supply_chain_graph.graphml"
    if base_path.exists():
        base_graph = nx.read_graphml(str(base_path))
        base_nodes = set(base_graph.nodes)
        collisions = sorted(
            node for node in set(graph.nodes).intersection(base_nodes)
            if not (str(node).startswith("BRAND:") and base_graph.nodes[node].get("node_type") == "Brand")
        )
        if collisions:
            raise ValueError(f"Company overlay IDs collide with base graph IDs: {collisions[:5]}")
        for node in set(graph.nodes).intersection(base_nodes):
            if str(node).startswith("BRAND:") and base_graph.nodes[node].get("node_type") != "Brand":
                raise ValueError(f"Company Brand identifier collides with non-Brand base graph node: {node}")
    out = generated_dir()
    out.mkdir(parents=True, exist_ok=True)
    final = out / "company_overlay.graphml"
    with tempfile.NamedTemporaryFile(prefix="company_overlay_", suffix=".graphml", dir=out, delete=False) as handle:
        temp = Path(handle.name)
    try:
        nx.write_graphml(graph, temp)
        content = temp.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        os.replace(temp, final)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {"version": int(connection.execute("SELECT value FROM master_meta WHERE key='published_version'").fetchone()[0]) + 1, "published_at": datetime.now(timezone.utc).isoformat(), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "source_database": str(connection.execute("PRAGMA database_list").fetchone()[2]), "validation": validation, "sha256": digest}
