"""Build a dynamic supply chain graph from supply-chain and event data."""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from graph_schema import (
    EDGE_TYPE_ASSOCIATED_WITH_BRAND,
    EDGE_TYPE_AFFECTS,
    EDGE_TYPE_BELONGS_TO,
    EDGE_TYPE_CATALOG_ASSOCIATED_WITH,
    EDGE_TYPE_LOCATED_IN,
    EDGE_TYPE_OFFERS,
    EDGE_TYPE_PRODUCES,
    EDGE_TYPE_STORED_AT,
    EDGE_TYPE_SUPPLIES,
    NODE_TYPE_BRAND,
    NODE_TYPE_CATEGORY,
    NODE_TYPE_EVENT,
    NODE_TYPE_LOCATION,
    NODE_TYPE_PRODUCT,
    NODE_TYPE_PRODUCT_MODEL,
    NODE_TYPE_SUPPLIER,
    NODE_TYPE_WAREHOUSE,
    GRAPH_OUTPUT_FIELDS,
)
from graph_utils import (
    BRAND_SUPPLIER_CSV,
    BUSINESS_EVENTS_CSV,
    EVENTS_CSV,
    GRAPHML_PATH,
    GRAPH_DIR,
    NODES_CSV_PATH,
    EDGES_CSV_PATH,
    SUMMARY_JSON_PATH,
    SUPPLYCHAIN_CSV,
    build_brand_node_id,
    build_category_node_id,
    build_event_node_id,
    build_location_node_id,
    build_product_node_id,
    build_product_model_node_id,
    build_supplier_node_id,
    build_warehouse_node_id,
    create_directory,
    load_inputs,
    normalize_text,
    to_json_string,
)
from src.data_integration.curated_warehouse_loader import (
    CURATED_WAREHOUSE_CSV,
    CuratedWarehouseData,
    load_curated_warehouse_data,
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def assert_inputs_exist() -> None:
    required_files = [EVENTS_CSV, BUSINESS_EVENTS_CSV, SUPPLYCHAIN_CSV, BRAND_SUPPLIER_CSV, CURATED_WAREHOUSE_CSV]
    missing = [str(path) for path in required_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Required input file(s) missing: {', '.join(missing)}")


def make_safe_edge_key(source: str, target: str, relationship: str) -> Tuple[str, str, str]:
    return source, target, relationship


def add_node(
    graph: nx.MultiDiGraph,
    node_id: str,
    node_type: str,
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    if attributes is None:
        attributes = {}
    if node_id not in graph:
        graph.add_node(node_id, node_type=node_type, name=name, **attributes)
    else:
        existing_attrs = graph.nodes[node_id]
        existing_attrs.update({k: v for k, v in attributes.items() if v not in (None, "")})


def add_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    relationship: str,
    attributes: Optional[Dict[str, Any]] = None,
    edge_index: Optional[set[Tuple[str, str, str]]] = None,
) -> None:
    if attributes is None:
        attributes = {}
    if edge_index is not None:
        key = make_safe_edge_key(source, target, relationship)
        if key in edge_index:
            return
        edge_index.add(key)
    graph.add_edge(source, target, relationship=relationship, **attributes)


def add_supply_chain_nodes(graph: nx.MultiDiGraph, supply_df: pd.DataFrame, edge_index: set[Tuple[str, str, str]]) -> None:
    for _, row in supply_df.iterrows():
        product_id = build_product_node_id(row.get("Product_ID", ""))
        product_name = normalize_text(row.get("Title", ""))
        add_node(
            graph,
            product_id,
            NODE_TYPE_PRODUCT,
            product_name,
            {
                "product_link": normalize_text(row.get("Product Link", "")),
                "category": normalize_text(row.get("Category", "")),
                "brand": normalize_text(row.get("Brand", "")),
                "price": normalize_text(row.get("Price", "")),
                "region": normalize_text(row.get("Region", "")),
            },
        )

        brand_name = normalize_text(row.get("Brand", ""))
        supplier_name = normalize_text(row.get("Supplier", ""))
        warehouse_name = normalize_text(row.get("Warehouse", ""))
        category_name = normalize_text(row.get("Category", ""))
        region_name = normalize_text(row.get("Region", ""))
        city_name = ""
        if warehouse_name.endswith(" Warehouse"):
            city_name = warehouse_name.replace(" Warehouse", "").strip()

        if supplier_name:
            supplier_id = build_supplier_node_id(supplier_name)
            add_node(graph, supplier_id, NODE_TYPE_SUPPLIER, supplier_name, {"region": region_name})
        else:
            supplier_id = ""

        if brand_name:
            brand_node_id = build_brand_node_id(brand_name)
            add_node(graph, brand_node_id, NODE_TYPE_BRAND, brand_name)
            if product_name:
                add_edge(graph, brand_node_id, product_id, EDGE_TYPE_PRODUCES, {}, edge_index)
        else:
            brand_node_id = ""

        if supplier_name and brand_name:
            add_edge(graph, supplier_id, brand_node_id, EDGE_TYPE_SUPPLIES, {}, edge_index)

        if category_name:
            category_id = build_category_node_id(category_name)
            add_node(graph, category_id, NODE_TYPE_CATEGORY, category_name)
            add_edge(graph, product_id, category_id, EDGE_TYPE_BELONGS_TO, {}, edge_index)

        if warehouse_name:
            warehouse_id = build_warehouse_node_id(warehouse_name)
            add_node(graph, warehouse_id, NODE_TYPE_WAREHOUSE, warehouse_name, {"region": region_name})
            add_edge(graph, product_id, warehouse_id, EDGE_TYPE_STORED_AT, {}, edge_index)
            if supplier_name:
                add_edge(graph, supplier_id, warehouse_id, EDGE_TYPE_STORED_AT, {}, edge_index)

            if city_name:
                city_id = build_location_node_id(city_name)
                add_node(graph, city_id, NODE_TYPE_LOCATION, city_name)
                add_edge(graph, warehouse_id, city_id, EDGE_TYPE_LOCATED_IN, {}, edge_index)

        if region_name:
            region_id = build_location_node_id(region_name)
            add_node(graph, region_id, NODE_TYPE_LOCATION, region_name)
            if warehouse_name:
                add_edge(graph, warehouse_id, region_id, EDGE_TYPE_LOCATED_IN, {}, edge_index)
            if supplier_name and not warehouse_name:
                add_edge(graph, supplier_id, region_id, EDGE_TYPE_LOCATED_IN, {}, edge_index)


def add_curated_warehouse_nodes(
    graph: nx.MultiDiGraph,
    curated: CuratedWarehouseData,
    edge_index: set[Tuple[str, str, str]],
) -> None:
    """Add public warehouse references and catalog associations without inventory claims."""
    provenance = "MANUALLY_CURATED_PUBLIC_SOURCE"
    for _, row in curated.warehouses.iterrows():
        warehouse_id = normalize_text(row["Warehouse_ID"])
        city = normalize_text(row["City"])
        add_node(graph, warehouse_id, NODE_TYPE_WAREHOUSE, normalize_text(row["Warehouse_Name"]), {
            "warehouse_id": warehouse_id,
            "warehouse_name": normalize_text(row["Warehouse_Name"]),
            "city": city,
            "state": normalize_text(row["State"]),
            "warehouse_operator": normalize_text(row["Warehouse_Operator"]),
            "third_party_logistics_partner": normalize_text(row["Third_Party_Logistics_Partner"]),
            "warehouse_source_url": normalize_text(row["Warehouse_Source_URL"]),
            "warehouse_verification_status": normalize_text(row["Warehouse_Verification_Status"]),
            "warehouse_evidence_grade": normalize_text(row["Warehouse_Evidence_Grade"]),
            "synthetic": False,
            "provenance": provenance,
            "catalog_only": True,
            "inventory_relationship_available": False,
        })
        location_id = build_location_node_id(city)
        add_node(graph, location_id, NODE_TYPE_LOCATION, city, {
            "city": city, "state": normalize_text(row["State"]), "country": "India",
            "provenance": provenance, "synthetic": False,
        })
        add_edge(graph, warehouse_id, location_id, EDGE_TYPE_LOCATED_IN, {
            "provenance": provenance, "location_match": "EXACT_CURATED_CITY", "synthetic": False,
        }, edge_index)

    for _, row in curated.associations.iterrows():
        warehouse_id = normalize_text(row["Warehouse_ID"])
        brand_name = normalize_text(row["Brand_or_Distributor"])
        brand_id = build_brand_node_id(brand_name)
        model_name = normalize_text(row["Product_Model"])
        product_model_id = build_product_model_node_id(brand_name, model_name)
        add_node(graph, brand_id, NODE_TYPE_BRAND, brand_name)
        add_node(graph, product_model_id, NODE_TYPE_PRODUCT_MODEL, model_name, {
            "brand": brand_name,
            "category": normalize_text(row["Product_Category"]),
            "subcategory": normalize_text(row["Product_Subcategory"]),
            "product_model": model_name,
            "product_source_url": normalize_text(row["Product_Source_URL"]),
            "catalog_as_of": normalize_text(row["Catalog_As_Of"]),
            "association_basis": normalize_text(row["Association_Basis"]),
            "warehouse_stock_verified": False,
            "synthetic": False,
            "provenance": provenance,
        })
        common = {"provenance": provenance, "synthetic": False}
        add_edge(graph, warehouse_id, brand_id, EDGE_TYPE_ASSOCIATED_WITH_BRAND, common, edge_index)
        add_edge(graph, brand_id, product_model_id, EDGE_TYPE_OFFERS, common, edge_index)
        add_edge(graph, warehouse_id, product_model_id, EDGE_TYPE_CATALOG_ASSOCIATED_WITH, {
            **common,
            "catalog_row_id": normalize_text(row["Catalog_Row_ID"]),
            "inventory_verified": False,
            "operational_inventory_relation": False,
            "association_basis": normalize_text(row["Association_Basis"]),
        }, edge_index)


def add_event_nodes(
    graph: nx.MultiDiGraph,
    business_df: pd.DataFrame,
    supply_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    edge_index: set[Tuple[str, str, str]],
) -> None:
    for _, row in business_df.iterrows():
        event_id = build_event_node_id(row.get("Event_ID", ""))
        event_name = normalize_text(row.get("Title", ""))
        add_node(
            graph,
            event_id,
            NODE_TYPE_EVENT,
            event_name,
            {
                "event_type": normalize_text(row.get("Event_Type", "")),
                "severity": normalize_text(row.get("Severity", "")),
                "confidence": normalize_text(row.get("Confidence", "")),
                "risk_score": normalize_text(row.get("Risk_Score", "")),
                "impact_type": normalize_text(row.get("Impact_Type", "")),
                "possible_effect": normalize_text(row.get("Possible_Supply_Chain_Effect", "")),
                "source": normalize_text(row.get("Source", "")),
                "rule_triggered": normalize_text(row.get("Rule_Triggered", "")),
                "date": normalize_text(row.get("Date", "")),
            },
        )

        supplier_name = normalize_text(row.get("Affected_Supplier", ""))
        warehouse_name = normalize_text(row.get("Affected_Warehouse", ""))
        location_name = normalize_text(row.get("Affected_Location", ""))
        affected_entity = normalize_text(row.get("Affected_Entity", ""))
        event_type = normalize_text(row.get("Event_Type", ""))

        if supplier_name and supplier_name.lower() != "unknown":
            supplier_id = build_supplier_node_id(supplier_name)
            if supplier_id in graph:
                add_edge(graph, event_id, supplier_id, EDGE_TYPE_AFFECTS, {"event_type": event_type}, edge_index)

        if warehouse_name and warehouse_name.lower() != "unknown":
            warehouse_id = build_warehouse_node_id(warehouse_name)
            if warehouse_id in graph:
                add_edge(graph, event_id, warehouse_id, EDGE_TYPE_AFFECTS, {"event_type": event_type}, edge_index)

        if location_name and location_name.lower() != "unknown":
            location_id = build_location_node_id(location_name)
            add_node(graph, location_id, NODE_TYPE_LOCATION, location_name)
            add_edge(graph, event_id, location_id, EDGE_TYPE_AFFECTS, {"event_type": event_type}, edge_index)

        if affected_entity and affected_entity.lower() != "unknown":
            brand_id = build_brand_node_id(affected_entity)
            if brand_id in graph:
                add_edge(graph, event_id, brand_id, EDGE_TYPE_AFFECTS, {"event_type": event_type}, edge_index)
                mapping_rows = mapping_df[
                    mapping_df["Brand"].str.lower() == affected_entity.lower()
                ]
                if not mapping_rows.empty:
                    supplier_name = normalize_text(mapping_rows.iloc[0].get("Supplier", ""))
                    warehouse_name = normalize_text(mapping_rows.iloc[0].get("Warehouse", ""))
                    if supplier_name:
                        supplier_id = build_supplier_node_id(supplier_name)
                        if supplier_id in graph:
                            add_edge(graph, event_id, supplier_id, EDGE_TYPE_AFFECTS, {"event_type": event_type}, edge_index)
                    if warehouse_name:
                        warehouse_id = build_warehouse_node_id(warehouse_name)
                        if warehouse_id in graph:
                            add_edge(graph, event_id, warehouse_id, EDGE_TYPE_AFFECTS, {"event_type": event_type}, edge_index)
            else:
                supplier_id = build_supplier_node_id(affected_entity)
                if supplier_id in graph:
                    add_edge(graph, event_id, supplier_id, EDGE_TYPE_AFFECTS, {"event_type": event_type}, edge_index)


def build_graph() -> nx.MultiDiGraph:
    assert_inputs_exist()

    events_df, business_df, supply_df, mapping_df = load_inputs()
    curated = load_curated_warehouse_data()
    graph = nx.MultiDiGraph()
    edge_index: set[Tuple[str, str, str]] = set()

    add_supply_chain_nodes(graph, supply_df, edge_index)
    add_curated_warehouse_nodes(graph, curated, edge_index)
    add_event_nodes(graph, business_df, supply_df, mapping_df, edge_index)

    return graph


def export_graph(graph: nx.MultiDiGraph, *, overwrite: bool = False) -> None:
    create_directory(GRAPH_DIR)
    if not overwrite and (GRAPHML_PATH.exists() or NODES_CSV_PATH.exists() or EDGES_CSV_PATH.exists() or SUMMARY_JSON_PATH.exists()):
        raise FileExistsError("Graph output files already exist. Will not overwrite existing outputs.")

    nx.write_graphml(graph, GRAPHML_PATH)

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "")
        name = attrs.get("name", "")
        attributes = {k: v for k, v in attrs.items() if k not in ("node_type", "name")}
        nodes.append({"node_id": node_id, "node_type": node_type, "name": name, "attributes": to_json_string(attributes)})

    edges = []
    for source, target, attrs in graph.edges(data=True):
        relationship = attrs.get("relationship", "")
        date = attrs.get("date", "")
        risk_score = attrs.get("risk_score", "")
        severity = attrs.get("severity", "")
        event_type = attrs.get("event_type", "")
        source_info = attrs.get("source", "")
        confidence = attrs.get("confidence", "")
        attributes = {k: v for k, v in attrs.items() if k not in ("relationship", "date", "risk_score", "severity", "event_type", "source", "confidence")}
        edges.append(
            {
                "source": source,
                "target": target,
                "relationship": relationship,
                "date": date,
                "risk_score": risk_score,
                "severity": severity,
                "event_type": event_type,
                "source_info": source_info,
                "confidence": confidence,
                "attributes": to_json_string(attributes),
            }
        )

    pd.DataFrame(nodes).to_csv(NODES_CSV_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(edges).to_csv(EDGES_CSV_PATH, index=False, encoding="utf-8-sig")

    summary = build_summary(graph)
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def build_summary(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    node_type_counts = Counter(attrs.get("node_type", "Unknown") for _, attrs in graph.nodes(data=True))
    edge_type_counts = Counter(attrs.get("relationship", "Unknown") for _, _, attrs in graph.edges(data=True))
    event_nodes = node_type_counts.get(NODE_TYPE_EVENT, 0)
    supplier_nodes = node_type_counts.get(NODE_TYPE_SUPPLIER, 0)
    warehouse_nodes = node_type_counts.get(NODE_TYPE_WAREHOUSE, 0)
    product_nodes = node_type_counts.get(NODE_TYPE_PRODUCT, 0)
    affected_relationships = sum(
        1
        for _, _, attrs in graph.edges(data=True)
        if attrs.get("relationship") == EDGE_TYPE_AFFECTS
    )
    dates = []
    for _, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == NODE_TYPE_EVENT:
            date_value = attrs.get("date")
            try:
                parsed = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
                dates.append(parsed)
            except Exception:
                continue
    temporal_range = {
        "start": min(dates).isoformat() if dates else None,
        "end": max(dates).isoformat() if dates else None,
    }
    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "nodes_by_type": dict(node_type_counts),
        "edges_by_relationship": dict(edge_type_counts),
        "event_nodes": event_nodes,
        "supplier_nodes": supplier_nodes,
        "warehouse_nodes": warehouse_nodes,
        "product_nodes": product_nodes,
        "affected_relationships": affected_relationships,
        "temporal_range": temporal_range,
    }


def print_report(summary: Dict[str, Any]) -> None:
    print("Dynamic Supply Chain Graph")
    print("==========================")
    print()
    print("Nodes:")
    print(f"  Total: {summary['total_nodes']}")
    print(f"  Suppliers: {summary['supplier_nodes']}")
    print(f"  Brands: {summary['nodes_by_type'].get('Brand', 0)}")
    print(f"  Products: {summary['product_nodes']}")
    print(f"  Warehouses: {summary['warehouse_nodes']}")
    print(f"  Categories: {summary['nodes_by_type'].get('Category', 0)}")
    print(f"  Locations: {summary['nodes_by_type'].get('Location', 0)}")
    print(f"  Events: {summary['event_nodes']}")
    print()
    print("Edges:")
    print(f"  Total: {summary['total_edges']}")
    for relationship, count in summary["edges_by_relationship"].items():
        print(f"  {relationship}: {count}")
    print()
    print(f"Event Impact Connections: {summary['affected_relationships']}")
    print(f"Temporal date range: {summary['temporal_range']}")


def main() -> None:
    configure_logging()
    try:
        graph = build_graph()
        export_graph(graph)
        summary = build_summary(graph)
        print_report(summary)
    except Exception as exc:
        logging.exception("Graph construction failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
