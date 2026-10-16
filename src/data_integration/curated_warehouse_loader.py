"""Validated loader for the manually curated public warehouse catalog."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
CURATED_WAREHOUSE_CSV = (
    BASE_DIR
    / "data"
    / "raw"
    / "warehouse_network"
    / "manually_curated_dataset"
    / "supplybeacon_full_1298.csv"
)
EXPECTED_INPUT_ROWS = 1298

REQUIRED_COLUMNS = {
    "Catalog_Row_ID", "Warehouse_ID", "Warehouse_Name", "City", "State",
    "Warehouse_Operator", "Third_Party_Logistics_Partner", "Brand_or_Distributor",
    "Product_Category", "Product_Subcategory", "Product_Model", "Association_Basis",
    "Warehouse_Stock_Verified", "Product_Source_URL", "Catalog_As_Of", "Synthetic_Flag",
    "Warehouse_Source_URL", "Warehouse_Verification_Status", "Warehouse_Evidence_Grade",
}
IDENTITY_COLUMNS = (
    "Warehouse_Name", "City", "State", "Warehouse_Operator", "Warehouse_Source_URL",
    "Warehouse_Verification_Status", "Warehouse_Evidence_Grade",
)


@dataclass(frozen=True)
class CuratedWarehouseData:
    associations: pd.DataFrame
    warehouses: pd.DataFrame
    conflicts: tuple[dict[str, Any], ...]


def normalize_yes_no(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in {"YES", "NO"}:
        raise ValueError(f"{field} must contain only YES or NO; found {value!r}")
    return normalized


def load_curated_warehouse_data(
    path: Path = CURATED_WAREHOUSE_CSV,
    *,
    expected_rows: int | None = EXPECTED_INPUT_ROWS,
) -> CuratedWarehouseData:
    """Load, validate, and collapse the catalog to one record per warehouse ID."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Curated warehouse dataset is missing required columns: {', '.join(missing)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} curated rows, found {len(frame)}")

    frame = frame.copy()
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    if (frame["Warehouse_ID"] == "").any():
        rows = frame.index[frame["Warehouse_ID"] == ""].tolist()[:10]
        raise ValueError(f"Warehouse_ID is empty in row(s): {rows}")
    frame["Warehouse_Stock_Verified"] = frame["Warehouse_Stock_Verified"].map(
        lambda value: normalize_yes_no(value, field="Warehouse_Stock_Verified")
    )
    frame["Synthetic_Flag"] = frame["Synthetic_Flag"].map(
        lambda value: normalize_yes_no(value, field="Synthetic_Flag")
    )

    conflicts: list[dict[str, Any]] = []
    warehouse_rows: list[dict[str, str]] = []
    for warehouse_id, group in frame.groupby("Warehouse_ID", sort=True):
        record = {"Warehouse_ID": warehouse_id}
        for column in IDENTITY_COLUMNS:
            values = sorted({value for value in group[column] if value})
            if len(values) > 1:
                conflicts.append({"Warehouse_ID": warehouse_id, "column": column, "values": values})
            record[column] = values[0] if values else ""
        partners = sorted({value for value in group["Third_Party_Logistics_Partner"] if value})
        record["Third_Party_Logistics_Partner"] = " | ".join(partners)
        warehouse_rows.append(record)
    if conflicts:
        detail = "; ".join(
            f"{item['Warehouse_ID']}:{item['column']}={item['values']}" for item in conflicts[:10]
        )
        raise ValueError(f"Conflicting curated warehouse identity records: {detail}")

    warehouses = pd.DataFrame(warehouse_rows)
    return CuratedWarehouseData(frame, warehouses, tuple(conflicts))
