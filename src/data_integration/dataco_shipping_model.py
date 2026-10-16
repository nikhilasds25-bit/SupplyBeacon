"""Phase 2 supervised baseline for historical DataCo shipping duration."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .dataco_loader import BASE_DIR, load_main_dataset
except ImportError:
    from dataco_loader import BASE_DIR, load_main_dataset

LOGGER = logging.getLogger(__name__)
ANALYSIS_DIR = BASE_DIR / "data" / "analysis" / "dataco" / "modeling"
MODEL_DIR = BASE_DIR / "models" / "dataco_shipping"
TARGET = "Days for shipping (real)"
DATE_COLUMN = "order date (DateOrders)"

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Product Price",
    "order_year",
    "order_month",
    "order_day_of_week",
    "order_quarter",
]
CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Product Name",
    "Category Name",
    "Market",
    "Order Region",
    "Order Country",
    "Order City",
]

FEATURE_AUDIT = {
    "prediction_point": "ORDER CREATION / BEFORE SHIPMENT COMPLETION",
    "target": {
        "column": TARGET,
        "meaning": "Observed historical shipping duration in days.",
        "not_renamed_to": "Lead_Time_Days",
    },
    "features": {
        "Days for shipment (scheduled)": {
            "meaning": "Promised or planned shipment duration.",
            "availability": "Available at order creation if the schedule is committed then.",
            "included": True,
            "reason": "Operational plan known before realized outcome; verify this assumption for deployment.",
        },
        "Shipping Mode": {
            "meaning": "Selected delivery/shipping service.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Order-time operational choice.",
        },
        "Product Name": {
            "meaning": "Ordered product identity.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Order content known before fulfillment.",
        },
        "Category Name": {
            "meaning": "Product category.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Product metadata known before fulfillment.",
        },
        "Market": {
            "meaning": "Commercial market.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Order geography/business context.",
        },
        "Order Region": {
            "meaning": "Destination region.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Destination context known from the order.",
        },
        "Order Country": {
            "meaning": "Destination country.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Destination context known from the order.",
        },
        "Order City": {
            "meaning": "Destination city.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Destination context known from the order.",
        },
        "Order Item Quantity": {
            "meaning": "Quantity in the order item.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Order quantity is known before fulfillment.",
        },
        "Sales": {
            "meaning": "Recorded sales amount.",
            "availability": "Assumed available at order creation.",
            "included": True,
            "reason": "Retained as an order-value feature, subject to business-system timing validation.",
        },
        "Order Item Total": {
            "meaning": "Recorded order-item total.",
            "availability": "Assumed available at order creation.",
            "included": True,
            "reason": "Retained as an order-value feature, subject to business-system timing validation.",
        },
        "Product Price": {
            "meaning": "Catalog product price.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Product/order metadata known before fulfillment.",
        },
        DATE_COLUMN: {
            "meaning": "Order creation timestamp.",
            "availability": "Available at order creation.",
            "included": True,
            "reason": "Only derived year, month, weekday, and quarter are used; raw timestamp is excluded.",
        },
    },
    "excluded_fields": {
        "Days for shipping (real)": "Primary observed target.",
        "Delivery Status": "Post-outcome status.",
        "Late_delivery_risk": "Outcome label and target-adjacent.",
        "shipping_delay_days": "Derived using realized shipping outcome.",
        "shipping date (DateOrders)": "Known after shipment progression/outcome.",
        "Order Status": "Potentially post-outcome or unavailable at order creation.",
        "Order Profit Per Order": "Outcome-dependent financial field.",
        "Customer PII": "Names, email, password, street, IDs, zipcodes, latitude, and longitude excluded.",
        "Identifiers": "Order/product/customer identifiers excluded to avoid memorization and leakage.",
        "Product Description": "Fully missing in the supplied dataset.",
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features from the order-time schema."""
    required = [TARGET, DATE_COLUMN] + NUMERIC_FEATURES[:5] + CATEGORICAL_FEATURES
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataCo dataset is missing required modeling columns: {missing}")
    output = df.copy()
    dates = pd.to_datetime(output[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("Order date contains invalid values; refusing to create a temporal split.")
    output["order_year"] = dates.dt.year.astype(int)
    output["order_month"] = dates.dt.month.astype(int)
    output["order_day_of_week"] = dates.dt.dayofweek.astype(int)
    output["order_quarter"] = dates.dt.quarter.astype(int)
    output[TARGET] = pd.to_numeric(output[TARGET], errors="coerce")
    output = output.loc[output[TARGET].notna()].copy()
    output["_order_date"] = dates.loc[output.index].dt.normalize()
    return output


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split by unique order dates, ensuring no future date enters an earlier split."""
    dates = sorted(df["_order_date"].unique())
    train_end_index = max(0, int(len(dates) * 0.70) - 1)
    validation_end_index = max(train_end_index + 1, int(len(dates) * 0.85) - 1)
    train_end = pd.Timestamp(dates[train_end_index])
    validation_end = pd.Timestamp(dates[validation_end_index])
    train = df[df["_order_date"] <= train_end].copy()
    validation = df[(df["_order_date"] > train_end) & (df["_order_date"] <= validation_end)].copy()
    test = df[df["_order_date"] > validation_end].copy()
    split = {
        "method": "chronological unique order-date split",
        "train_fraction_by_date": 0.70,
        "validation_fraction_by_date": 0.15,
        "test_fraction_by_date": 0.15,
        "train": {"min_date": str(train["_order_date"].min().date()), "max_date": str(train["_order_date"].max().date()), "rows": len(train)},
        "validation": {"min_date": str(validation["_order_date"].min().date()), "max_date": str(validation["_order_date"].max().date()), "rows": len(validation)},
        "test": {"min_date": str(test["_order_date"].min().date()), "max_date": str(test["_order_date"].max().date()), "rows": len(test)},
        "no_future_overlap": bool(train["_order_date"].max() < validation["_order_date"].min() < test["_order_date"].min()),
    }
    return train, validation, test, split


def make_preprocessor(dense: bool = False) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ]
    )
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])


def make_models() -> dict[str, Pipeline]:
    return {
        "dummy_mean": Pipeline([("preprocess", make_preprocessor()), ("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([("preprocess", make_preprocessor()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", make_preprocessor(dense=True)),
                ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42)),
            ]
        ),
    }


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def group_errors(test: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    scored = test.copy()
    scored["_prediction"] = prediction
    rows = []
    for field in ("Shipping Mode", "Market", "Category Name", "Order Region"):
        for value, group in scored.groupby(field, dropna=False):
            group_metrics = metrics(group[TARGET], group["_prediction"].to_numpy())
            rows.append({"group_field": field, "group_value": str(value), "rows": len(group), **group_metrics})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    df, source_path = load_main_dataset()
    if source_path is None or df.empty:
        raise FileNotFoundError("DataCoSupplyChainDataset.csv is required for Phase 2 modeling.")
    frame = prepare_frame(df)
    train, validation, test, split = chronological_split(frame)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS_DIR / "feature_audit.json", FEATURE_AUDIT)
    write_json(ANALYSIS_DIR / "temporal_split.json", split)
    target_distribution = frame[TARGET].value_counts().sort_index().to_dict()
    write_json(
        ANALYSIS_DIR / "target_distribution.json",
        {"target": TARGET, "min": float(frame[TARGET].min()), "max": float(frame[TARGET].max()), "distribution": {str(k): int(v) for k, v in target_distribution.items()}, "ordinal_values": list(range(7)), "classification_note": "Ordinal classification is plausible for the 0-6 day target, but Phase 2 remains regression."},
    )
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    metrics_rows = []
    predictions = {}
    for name, model in make_models().items():
        LOGGER.info("Training %s", name)
        model.fit(train[features], train[TARGET])
        validation_prediction = model.predict(validation[features])
        test_prediction = model.predict(test[features])
        validation_metrics = metrics(validation[TARGET], validation_prediction)
        test_metrics = metrics(test[TARGET], test_prediction)
        metrics_rows.append({"model": name, "split": "validation", **validation_metrics})
        metrics_rows.append({"model": name, "split": "test", **test_metrics})
        predictions[name] = test_prediction
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(ANALYSIS_DIR / "baseline_metrics.csv", index=False)
    best_name = metrics_df[metrics_df["split"] == "test"].sort_values("mae").iloc[0]["model"]
    best_prediction = predictions[best_name]
    group_errors(test, best_prediction).to_csv(ANALYSIS_DIR / "group_error_analysis.csv", index=False)
    samples = test[["_order_date", TARGET] + CATEGORICAL_FEATURES].copy().head(1000)
    samples["prediction"] = best_prediction[: len(samples)]
    samples.to_csv(ANALYSIS_DIR / "prediction_samples.csv", index=False)
    write_json(ANALYSIS_DIR / "run_summary.json", {"source": str(source_path.relative_to(BASE_DIR)), "target": TARGET, "best_model_by_test_mae": best_name, "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)})
    print(json.dumps({"target": TARGET, "best_model_by_test_mae": best_name, "split": split}, indent=2))


if __name__ == "__main__":
    main()
