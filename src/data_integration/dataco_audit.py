"""Run the additive DataCo inspection, audit, and design-report phase."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .dataco_loader import BASE_DIR, load_main_dataset
    from .dataco_leakage_audit import audit_leakage
    from .dataco_profile import ANALYSIS_DIR, profile_dataframe, write_json, write_missing_values
    from .dataco_target_audit import audit_target
    from .dataco_temporal_audit import audit_temporal_coverage
except ImportError:
    from dataco_loader import BASE_DIR, load_main_dataset
    from dataco_leakage_audit import audit_leakage
    from dataco_profile import ANALYSIS_DIR, profile_dataframe, write_json, write_missing_values
    from dataco_target_audit import audit_target
    from dataco_temporal_audit import audit_temporal_coverage

DOCS_DIR = BASE_DIR / "docs"


def current_field_comparison() -> str:
    return """# DataCo Integration Audit

## Scope and safety

This is a Phase 1 inspection-only integration over 180,519 DataCo rows and 53 columns. DataCo remains under `data/raw/dataco/`; it is not merged with Flipkart, and no production graph, risk output, embedding, dataset, or model was changed. No model or GNN is trained by this audit.

## Temporal and operational evidence

The DataCo order date is `order date (DateOrders)` and the shipping date is `shipping date (DateOrders)`. The order history runs from 2015-01-01 through 2018-01-31, while shipping dates run from 2015-01-03 through 2018-02-06. This is longitudinal operational data, not a single product snapshot. `Days for shipping (real)` is an observed shipment-duration field and is a reasonable candidate historical delay target, but it must not automatically be renamed `Lead_Time_Days`: lead time and realized shipping duration may have different business definitions and prediction timestamps.

`shipping_delay_days = Days for shipping (real) - Days for shipment (scheduled)` is retained as a diagnostic only. It is not currently a model feature or target.

## Target and leakage conclusion

`Late_delivery_risk` is strongly target-adjacent. It agrees exactly with `Delivery Status == "Late delivery"` in the observed rows and is highly associated with realized-versus-scheduled shipment timing. `Delivery Status`, realized shipping days, shipping date, order status, and profit-after-order fields are excluded from a pre-shipment predictor until their availability is proven at the declared prediction point. Exact counts are in `data/analysis/dataco/dataco_target_audit.json`.

## Existing ElectroGraph field provenance

| Variable | Current ElectroGraph status | DataCo capability | Phase 1 recommendation |
|---|---|---|---|
| Supplier | SIMULATED/mapped from brand | **Not support** | No supplier evidence; do not replace, calibrate, or fabricate. |
| Warehouse | SIMULATED/mapped from brand/region | **Not support** | No warehouse evidence; do not replace, calibrate, or fabricate. |
| Inventory | SIMULATED seeded random range by category | **Not support** | No inventory-on-hand or stock-flow evidence. |
| Lead_Time_Days | SIMULATED seeded random range by mapped region | **Calibrate/validate only** | Real shipping duration can test assumptions, but is not automatically equivalent to lead time. |
| Risk_Score | Deterministic event/graph output; source score starts at 0.0 | **Not directly replace** | `Late_delivery_risk` is an operational delivery outcome, not the current external-event graph risk construct. It may support a separate historical risk task. |

DataCo can genuinely improve the evidence basis for shipping-duration assumptions and support a separate supervised historical late-delivery task. It cannot establish current supplier, warehouse, inventory, or graph-event risk semantics.

## Privacy

Exclude customer names, email, password, street, customer identifiers, zipcodes, latitude, longitude, and other direct or quasi-identifying fields from graph nodes, embeddings, RAG documents, and model features unless an explicitly approved privacy-preserving use case requires them. Do not expose personal values in reports.

## Current implementation provenance

- `Supplier` and `Warehouse`: brand-to-supplier mapping plus deterministic fallback rows.
- `Inventory`: seeded random integer ranges by category.
- `Lead_Time_Days`: seeded random integer ranges by mapped region.
- source `Risk_Score`: constant `0.0` in supply-chain enrichment.
- event `Risk_Score`: deterministic business-rule score from event severity, confidence, and base risk.
- graph risk output: deterministic heuristic score from event exposure, severity, confidence, business risk, graph degree, and product connectivity.

## Modeling recommendation

DataCo is suitable for a supervised historical delay/risk experiment only after declaring a prediction timestamp and excluding post-outcome fields. The recommended first target is observed `Days for shipping (real)` as a shipment-duration target, or a separately defined binary late-delivery label whose construction and leakage rules are documented. Do not train that model in Phase 1.
"""


def graph_plan() -> str:
    return """# DataCo Graph Integration Plan

This document is design-only. The production NetworkX graph is unchanged.

## Evidence-supported isolated entities

Potential DataCo nodes are `Order`, `Product`, `Category`, `Location`, `Region`, and `Market`. `Order` should use a scoped/pseudonymous order identifier. Customer identifiers and PII must not become nodes. Supplier and Warehouse nodes must not be created from DataCo because the dataset does not provide evidenced supplier/warehouse relationships.

## Evidence-supported relationships

- `ORDER_CONTAINS_PRODUCT`: order to product name/card identity when the product key policy is defined.
- `PRODUCT_BELONGS_TO_CATEGORY`: product to category name.
- `ORDER_DESTINED_FOR_LOCATION`: order to city/country location.
- `LOCATION_IN_REGION`: city/location to order region.
- `REGION_IN_MARKET`: order region to market.

Attach order/shipping timestamps and operational outcomes as order attributes or separate audit tables, not as unsupported supplier/warehouse edges. Keep DataCo node IDs namespaced from current Flipkart IDs. A later implementation must define deduplication, pseudonymization, temporal cutoff, and cross-source identity rules before graph changes.
"""


def external_plan() -> str:
    return """# External Signal Integration Plan

## Two non-interchangeable layers

**Historical learning layer:** DataCo order, shipping, geography, and outcome records, with dates preserved. Historical model evaluation must use only information available at its declared prediction time.

**Current intelligence layer:** existing NewsAPI and OpenWeather observations collected at runtime. These remain separate from historical DataCo records.

Current 2026 news/weather must never be attached to historical DataCo orders as if those observations caused historical delays. A future scenario workflow may score a current location using a model trained on historical operational records plus current signals, but it must label the result as a present-day scenario/decision-support estimate—not historical causal attribution.

## API evolution

The current collectors are India-focused: NewsAPI uses an India query and OpenWeather iterates a fixed warehouse-city list. Later, add an explicit location/market configuration layer that accepts DataCo-supported countries, regions, and cities, normalizes geographies, records collection timestamps/time zones, and keeps source provenance. Do not replace either collector in Phase 1.
"""


def write_docs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "DATACO_INTEGRATION_AUDIT.md").write_text(current_field_comparison(), encoding="utf-8")
    (DOCS_DIR / "DATACO_GRAPH_INTEGRATION_PLAN.md").write_text(graph_plan(), encoding="utf-8")
    (DOCS_DIR / "EXTERNAL_SIGNAL_INTEGRATION_PLAN.md").write_text(external_plan(), encoding="utf-8")


def main() -> None:
    df, source_path = load_main_dataset()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS_DIR / "dataco_profile.json", profile_dataframe(df, source_path))
    write_missing_values(df, ANALYSIS_DIR / "dataco_missing_values.csv")
    write_json(ANALYSIS_DIR / "dataco_temporal_report.json", audit_temporal_coverage(df))
    write_json(ANALYSIS_DIR / "dataco_target_audit.json", audit_target(df))
    write_json(ANALYSIS_DIR / "dataco_leakage_report.json", audit_leakage(df))
    write_docs()
    print(json.dumps({"status": "available" if source_path else "unavailable", "rows": len(df), "columns": len(df.columns)}, indent=2))


if __name__ == "__main__":
    main()
