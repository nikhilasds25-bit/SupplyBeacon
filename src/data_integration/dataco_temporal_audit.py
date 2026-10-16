"""Audit DataCo date coverage and temporal modeling suitability."""
from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from .dataco_loader import load_main_dataset
    from .dataco_profile import ANALYSIS_DIR, resolve_column, write_json
except ImportError:
    from dataco_loader import load_main_dataset
    from dataco_profile import ANALYSIS_DIR, resolve_column, write_json


def audit_temporal_coverage(df: pd.DataFrame) -> dict[str, Any]:
    order_col = resolve_column(df, ("DateOrders", "Order Date", "order date", "order date (DateOrders)"))
    ship_col = resolve_column(df, ("Shipping date", "shipping date", "Ship Date", "shipping date (DateOrders)"))
    result: dict[str, Any] = {
        "status": "available" if not df.empty else "unavailable",
        "order_date_column": order_col,
        "shipping_date_column": ship_col,
        "order_date_range": None,
        "shipping_date_range": None,
        "historical_span_days": None,
        "order_date_span_days": None,
        "shipping_date_span_days": None,
        "longitudinal_not_snapshot_like": False,
        "temporal_variation": False,
        "recommendations": [],
    }
    parsed = {}
    for label, column in (("order", order_col), ("shipping", ship_col)):
        if column:
            values = pd.to_datetime(df[column], errors="coerce", dayfirst=False).dropna()
            if not values.empty:
                parsed[label] = values
                result[f"{label}_date_range"] = {
                    "min": values.min().isoformat(),
                    "max": values.max().isoformat(),
                    "unique_days": int(values.dt.normalize().nunique()),
                    "valid_count": int(values.count()),
                    "invalid_count": int(df[column].notna().sum() - values.count()),
                }
                result[f"{label}_date_span_days"] = int(
                    (values.max() - values.min()).total_seconds() / 86400
                )
                result[f"{label}_date_range"]["span_days"] = result[f"{label}_date_span_days"]
    if parsed:
        all_dates = pd.concat(parsed.values())
        result["historical_span_days"] = int((all_dates.max() - all_dates.min()).total_seconds() / 86400)
    if parsed:
        result["temporal_variation"] = any(values.dt.normalize().nunique() > 1 for values in parsed.values())
        result["longitudinal_not_snapshot_like"] = result["historical_span_days"] > 1 and result["temporal_variation"]
    if result["temporal_variation"]:
        result["recommendations"] = [
            "Use chronological order-date splits with a validation gap if enough records remain.",
            "Define shipping prediction features using only fields known by the chosen prediction timestamp.",
            "Evaluate dynamic graph snapshots only after deciding an order/shipping cutoff and entity identity policy.",
        ]
    else:
        result["recommendations"] = ["DataCo date coverage is unavailable or has no usable variation; do not claim temporal validation."]
    return result


def main() -> None:
    df, _ = load_main_dataset()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS_DIR / "dataco_temporal_report.json", audit_temporal_coverage(df))


if __name__ == "__main__":
    main()
