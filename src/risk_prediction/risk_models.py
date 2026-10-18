"""Graph-based risk scoring formulas and thresholds."""

from __future__ import annotations

from typing import Dict

SEVERITY_SCORES: Dict[str, float] = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.5,
    "LOW": 0.25,
    "UNKNOWN": 0.0,
}

RISK_THRESHOLDS: Dict[str, float] = {
    "Critical": 0.85,
    "High": 0.60,
    "Medium": 0.30,
    "Low": 0.0,
}

# Final score weights for the baseline risk prediction formula.
RISK_WEIGHTS: Dict[str, float] = {
    "severity": 0.30,
    "confidence": 0.20,
    "business_risk": 0.20,
    "event_exposure": 0.15,
    "degree": 0.10,
    "product_connectivity": 0.05,
}


def severity_value(label: str) -> float:
    """Convert a severity label into a numeric score between 0 and 1."""
    if not label:
        return SEVERITY_SCORES["UNKNOWN"]
    return SEVERITY_SCORES.get(label.strip().upper(), SEVERITY_SCORES["UNKNOWN"])


def normalize_ratio(value: float, max_value: float) -> float:
    """Normalize a non-negative ratio into a bounded [0.0, 1.0] value."""
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def clamp_score(score: float) -> float:
    """Clamp a risk score to the valid range [0.0, 1.0]."""
    return max(0.0, min(1.0, score))


def risk_level(score: float) -> str:
    """Map a numeric risk score into a labeled risk bucket."""
    for label, threshold in sorted(RISK_THRESHOLDS.items(), key=lambda item: -item[1]):
        if score >= threshold:
            return label
    return "Low"


def compute_risk_score(features: Dict[str, float]) -> float:
    """Compute a deterministic risk score from graph and event features."""
    severity = clamp_score(features.get("max_severity", 0.0))
    confidence = clamp_score(features.get("average_confidence", 0.0))
    business_risk = clamp_score(features.get("business_risk", 0.0))
    event_exposure = clamp_score(features.get("event_exposure", 0.0))
    degree = clamp_score(features.get("degree", 0.0))
    product_connectivity = clamp_score(features.get("product_connectivity", 0.0))

    score = (
        severity * RISK_WEIGHTS["severity"]
        + confidence * RISK_WEIGHTS["confidence"]
        + business_risk * RISK_WEIGHTS["business_risk"]
        + event_exposure * RISK_WEIGHTS["event_exposure"]
        + degree * RISK_WEIGHTS["degree"]
        + product_connectivity * RISK_WEIGHTS["product_connectivity"]
    )
    return clamp_score(score)


def build_risk_reason(features: Dict[str, float]) -> str:
    """Create a human-readable explanation for the computed risk score."""
    event_count = int(features.get("connected_events", 0))
    direct_events = int(features.get("direct_event_count", 0))
    avg_confidence = round(features.get("average_confidence", 0.0), 2)
    max_severity = features.get("max_severity_label", "Unknown")
    business_risk = round(features.get("business_risk", 0.0), 2)
    warehouses = int(features.get("connected_warehouses", 0))
    locations = int(features.get("connected_locations", 0))
    products = int(features.get("connected_products", 0))
    suppliers = int(features.get("connected_suppliers", 0))
    exposure = round(features.get("event_exposure", 0.0), 2)

    if event_count > 0:
        if direct_events > 0:
            reason = (
                f"Direct event exposure from {direct_events} event(s) with max severity {max_severity} "
                f"and average confidence {avg_confidence}."
            )
        else:
            reason = (
                f"Indirect exposure to {event_count} event(s) through connected supplies and locations "
                f"(nearest distance {features.get('min_event_distance', 'Unknown')})."
            )
        if business_risk > 0:
            reason += f" Business impact risk score is {business_risk}."
        if products > 0 or suppliers > 0 or warehouses > 0:
            reason += f" Connected to {products} product(s), {suppliers} supplier(s), and {warehouses} warehouse(s)."
        return reason

    if products > 0 or suppliers > 0 or warehouses > 0:
        return (
            f"No event exposure detected. The node remains connected to {products} product(s), "
            f"{suppliers} supplier(s), and {warehouses} warehouse(s), but no current event risk is present."
        )

    return "No event exposure detected and no connected supply chain elements were found."
