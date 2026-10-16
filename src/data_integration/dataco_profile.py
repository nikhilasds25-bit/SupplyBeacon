"""Descriptive profiling for the isolated DataCo operational dataset."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    from .dataco_loader import BASE_DIR, DATACO_DIR, load_main_dataset
except ImportError:
    from dataco_loader import BASE_DIR, DATACO_DIR, load_main_dataset

ANALYSIS_DIR = BASE_DIR / "data" / "analysis" / "dataco"
DATE_COLUMNS = ("DateOrders", "order date", "shipping date")
COLUMN_ALIASES = {
    "order_date": ("DateOrders", "Order Date", "order date", "order date (DateOrders)"),
    "shipping_date": ("Shipping date", "shipping date", "Ship Date", "shipping date (DateOrders)"),
    "real_shipping_days": ("Days for shipping (real)",),
    "scheduled_shipping_days": ("Days for shipment (scheduled)",),
    "delivery_status": ("Delivery Status",),
    "late_delivery_risk": ("Late_delivery_risk", "Late delivery risk"),
}


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-compatible values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Resolve a source column while tolerating harmless case differences."""
    by_lower = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def value_counts(df: pd.DataFrame, column: Optional[str]) -> dict[str, int]:
    if not column:
        return {}
    return {str(json_safe(key)): int(value) for key, value in df[column].value_counts(dropna=False).items()}


def numeric_summary(df: pd.DataFrame, column: Optional[str]) -> dict[str, Any]:
    if not column:
        return {}
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "missing": int(df[column].isna().sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "value_distribution": {
            str(json_safe(key)): int(value)
            for key, value in df[column].value_counts(dropna=False).sort_index().items()
        },
    }


def profile_dataframe(df: pd.DataFrame, source_path: Optional[Path]) -> dict[str, Any]:
    """Build a non-modeling profile and shipping-delay diagnostic."""
    resolved = {key: resolve_column(df, candidates) for key, candidates in COLUMN_ALIASES.items()}
    profile: dict[str, Any] = OrderedDict(
        status="available" if source_path else "unavailable",
        source_file=str(source_path.relative_to(BASE_DIR)) if source_path else None,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        columns=[str(column) for column in df.columns],
        missing_values={str(column): int(value) for column, value in df.isna().sum().items()},
        duplicate_rows=int(df.duplicated().sum()) if not df.empty else 0,
        resolved_columns=resolved,
        unique_orders=None,
        unique_products=None,
        cardinalities={},
        distributions={},
        numeric_distributions={},
    )
    if df.empty:
        return profile

    order_col = resolve_column(df, ("Order Id", "Order ID", "OrderId"))
    product_col = resolve_column(df, ("Product Name", "Product"))
    for label, column in (("unique_orders", order_col), ("unique_products", product_col)):
        profile[label] = int(df[column].nunique(dropna=True)) if column else None

    for label, candidates in {
        "categories": ("Category Name", "Category"),
        "countries": ("Order Country", "Country"),
        "regions": ("Order Region", "Region"),
        "markets": ("Market",),
        "shipping_modes": ("Shipping Mode",),
        "delivery_status": ("Delivery Status",),
        "late_delivery_risk": ("Late_delivery_risk", "Late delivery risk"),
    }.items():
        profile["distributions"][label] = value_counts(df, resolve_column(df, candidates))

    for label, candidates in {
        "days_for_shipping_real": COLUMN_ALIASES["real_shipping_days"],
        "days_for_shipment_scheduled": COLUMN_ALIASES["scheduled_shipping_days"],
    }.items():
        profile["numeric_distributions"][label] = numeric_summary(df, resolve_column(df, candidates))

    real_col = resolved["real_shipping_days"]
    scheduled_col = resolved["scheduled_shipping_days"]
    if real_col and scheduled_col:
        real = pd.to_numeric(df[real_col], errors="coerce")
        scheduled = pd.to_numeric(df[scheduled_col], errors="coerce")
        delay = real - scheduled
        profile["shipping_delay_days"] = {
            "definition": f"{real_col} - {scheduled_col}",
            "count": int(delay.notna().sum()),
            "missing": int(delay.isna().sum()),
            "distribution": {
                "count": int(delay.notna().sum()),
                "min": float(delay.min()),
                "max": float(delay.max()),
                "mean": float(delay.mean()),
                "median": float(delay.median()),
                "std": float(delay.std()) if delay.notna().sum() > 1 else 0.0,
                "value_distribution": {
                    str(json_safe(key)): int(value)
                    for key, value in delay.value_counts(dropna=False).sort_index().items()
                },
                "late_or_equal_count": int((delay >= 0).sum()),
                "positive_delay_count": int((delay > 0).sum()),
            },
        }
    else:
        profile["shipping_delay_days"] = {"status": "not_computable", "reason": "Required shipping columns are absent."}
    return profile


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_safe), encoding="utf-8")


def write_missing_values(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        pd.DataFrame(columns=["column", "missing_count", "missing_fraction"]).to_csv(path, index=False)
        return
    output = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(value) for value in df.isna().sum()],
            "missing_fraction": [float(value / len(df)) for value in df.isna().sum()],
        }
    )
    output.to_csv(path, index=False)


def main() -> None:
    df, source_path = load_main_dataset()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS_DIR / "dataco_profile.json", profile_dataframe(df, source_path))
    write_missing_values(df, ANALYSIS_DIR / "dataco_missing_values.csv")
    print(f"DataCo status: {'available' if source_path else 'unavailable'}")
    print(f"DataCo rows: {len(df)}; columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
