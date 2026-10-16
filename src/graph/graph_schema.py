"""Schema constants for dynamic supply chain graph construction."""
from __future__ import annotations

NODE_TYPE_SUPPLIER = "Supplier"
NODE_TYPE_BRAND = "Brand"
NODE_TYPE_PRODUCT = "Product"
NODE_TYPE_PRODUCT_MODEL = "ProductModel"
NODE_TYPE_CATEGORY = "Category"
NODE_TYPE_WAREHOUSE = "Warehouse"
NODE_TYPE_LOCATION = "Location"
NODE_TYPE_EVENT = "Event"

EDGE_TYPE_SUPPLIES = "SUPPLIES"
EDGE_TYPE_PRODUCES = "PRODUCES"
EDGE_TYPE_BELONGS_TO = "BELONGS_TO"
EDGE_TYPE_STORED_AT = "STORED_AT"
EDGE_TYPE_LOCATED_IN = "LOCATED_IN"
EDGE_TYPE_AFFECTS = "AFFECTS"
EDGE_TYPE_CAUSES = "CAUSES"
EDGE_TYPE_IMPACTS = "IMPACTS"
EDGE_TYPE_ASSOCIATED_WITH_BRAND = "ASSOCIATED_WITH_BRAND"
EDGE_TYPE_CATALOG_ASSOCIATED_WITH = "CATALOG_ASSOCIATED_WITH"
EDGE_TYPE_OFFERS = "OFFERS"

NODE_ID_PREFIXES = {
    NODE_TYPE_SUPPLIER: "SUPPLIER",
    NODE_TYPE_BRAND: "BRAND",
    NODE_TYPE_PRODUCT: "PRODUCT",
    NODE_TYPE_PRODUCT_MODEL: "PRODUCT_MODEL",
    NODE_TYPE_CATEGORY: "CATEGORY",
    NODE_TYPE_WAREHOUSE: "WAREHOUSE",
    NODE_TYPE_LOCATION: "LOCATION",
    NODE_TYPE_EVENT: "EVENT",
}

GRAPH_OUTPUT_FIELDS = {
    "node": ["node_id", "node_type", "name", "attributes"],
    "edge": [
        "source",
        "target",
        "relationship",
        "date",
        "risk_score",
        "severity",
        "event_type",
        "source_info",
        "confidence",
        "attributes",
    ],
}

SUMMARY_KEYS = [
    "total_nodes",
    "total_edges",
    "nodes_by_type",
    "edges_by_relationship",
    "event_nodes",
    "supplier_nodes",
    "warehouse_nodes",
    "product_nodes",
    "affected_relationships",
    "temporal_range",
]
