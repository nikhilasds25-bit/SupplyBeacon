"""Audit DataCo outcome variables and direct/indirect target construction."""
from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from .dataco_loader import load_main_dataset
    from .dataco_profile import ANALYSIS_DIR, resolve_column, write_json
except ImportError:
    from dataco_loader import load_main_dataset
    from dataco_profile import ANALYSIS_DIR, resolve_column, write_json


def audit_target(df: pd.DataFrame) -> dict[str, Any]:
    target = resolve_column(df, ("Late_delivery_risk", "Late delivery risk"))
    real = resolve_column(df, ("Days for shipping (real)",))
    scheduled = resolve_column(df, ("Days for shipment (scheduled)",))
    status = resolve_column(df, ("Delivery Status",))
    result: dict[str, Any] = {
        "status": "available" if not df.empty else "unavailable",
        "target": target,
        "candidate_direct_inputs": [column for column in (real, scheduled, status) if column],
        "observed_target_distribution": {},
        "diagnostics": [],
        "derivation_checks": [],
        "exact_agreement_checks": {},
        "prediction_time_warning": "Late_delivery_risk is an outcome label; it must not be used as an input feature.",
    }
    if target:
        result["observed_target_distribution"] = {
            str(key): int(value) for key, value in df[target].value_counts(dropna=False).items()
        }
    if target and real and scheduled:
        target_values = pd.to_numeric(df[target], errors="coerce")
        real_values = pd.to_numeric(df[real], errors="coerce")
        scheduled_values = pd.to_numeric(df[scheduled], errors="coerce")
        delay = real_values - scheduled_values
        valid = target_values.notna() & delay.notna()
        if valid.any():
            binary_target = target_values[valid].isin([0, 1])
            if binary_target.all():
                target_binary = target_values[valid].astype(int)
                for name, rule in (
                    ("target_equals_positive_shipping_delay", (delay[valid] > 0).astype(int)),
                    ("target_equals_shipping_at_or_after_schedule", (delay[valid] >= 0).astype(int)),
                ):
                    result["derivation_checks"].append(
                        {
                            "name": name,
                            "matching_rows": int((target_binary == rule).sum()),
                            "evaluated_rows": int(len(rule)),
                            "match_fraction": float((target_binary == rule).mean()),
                            "interpretation": "Strong evidence of a row-level derivation if exact agreement is observed, but source documentation is still required.",
                        }
                    )
                checks = {
                    "target_equals_positive_shipping_delay": (delay[valid] > 0).astype(int),
                    "target_equals_shipping_at_or_after_schedule": (delay[valid] >= 0).astype(int),
                }
                for name, rule in checks.items():
                    matches = target_binary == rule
                    result["exact_agreement_checks"][name] = {
                        "agreement_count": int(matches.sum()),
                        "mismatch_count": int((~matches).sum()),
                        "evaluated_count": int(len(matches)),
                    }
            for name, values in (
                ("real_shipping_days", real_values[valid]),
                ("scheduled_shipping_days", scheduled_values[valid]),
                ("shipping_delay_days", delay[valid]),
            ):
                grouped = pd.DataFrame({"value": values, "target": target_values[valid]}).groupby("value")["target"]
                mixed_values = [str(key) for key, group in grouped if group.nunique(dropna=True) > 1]
                result["exact_agreement_checks"][f"target_consistency_by_{name}"] = {
                    "unique_values": int(values.nunique()),
                    "values_with_both_target_classes": mixed_values,
                    "deterministic_by_value": not mixed_values,
                }
            result["diagnostics"].append(
                {
                    "name": "target_vs_shipping_delay",
                    "description": "Cross-tabulation of Late_delivery_risk and real-minus-scheduled shipping days; association is diagnostic, not proof of formula.",
                    "cross_tab": pd.crosstab(target_values[valid], delay[valid]).to_dict(),
                }
            )
    if target and status:
        status_rule = df.loc[df[target].notna(), status].eq("Late delivery").astype(int)
        target_for_status = pd.to_numeric(df.loc[df[target].notna(), target], errors="coerce")
        valid_status = target_for_status.notna()
        matches = target_for_status[valid_status].astype(int) == status_rule[valid_status]
        result["exact_agreement_checks"]["target_equals_late_delivery_status"] = {
            "agreement_count": int(matches.sum()),
            "mismatch_count": int((~matches).sum()),
            "evaluated_count": int(len(matches)),
        }
        result["diagnostics"].append(
            {
                "name": "target_vs_delivery_status",
                "description": "Cross-tabulation of Late_delivery_risk and Delivery Status; status is generally post-outcome and should be excluded for pre-shipment prediction.",
                "cross_tab": pd.crosstab(df[target], df[status], dropna=False).to_dict(),
            }
        )
    result["conclusion"] = (
        "The supplied columns can reveal direct or indirect target construction, but source documentation or row-level formula confirmation is required before asserting exact derivation."
        if target
        else "DataCo target column is unavailable; no derivation claim can be made."
    )
    return result


def main() -> None:
    df, _ = load_main_dataset()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS_DIR / "dataco_target_audit.json", audit_target(df))


if __name__ == "__main__":
    main()
