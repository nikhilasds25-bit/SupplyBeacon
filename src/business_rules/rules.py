"""Business rules for supply chain event impact assessment."""
from __future__ import annotations

from typing import Dict, Optional

# A rule defines the likely impact type, supply chain effect, and a base risk
# factor for a known event type.
RULE_CONFIGS: Dict[str, Dict[str, object]] = {
    "Heavy Rain": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Delivery Delay",
        "Default_Category": "Logistics",
        "Base_Risk": 0.75,
        "Rule_Triggered": "Heavy Rain can interrupt transport and last-mile delivery.",
    },
    "Moderate Rain": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Possible Delivery Delay",
        "Default_Category": "Logistics",
        "Base_Risk": 0.55,
        "Rule_Triggered": "Moderate rain may affect transport conditions.",
    },
    "Light Rain": {
        "Impact_Type": "Weather Condition",
        "Possible_Supply_Chain_Effect": "Limited Operational Impact",
        "Default_Category": "Logistics",
        "Base_Risk": 0.25,
        "Rule_Triggered": "Light rain is monitored but does not by itself establish disruption.",
    },
    "Thunderstorm": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Delivery Delay",
        "Default_Category": "Logistics",
        "Base_Risk": 0.8,
        "Rule_Triggered": "Thunderstorms may disrupt transport and last-mile delivery.",
    },
    "Storm": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Delivery Delay",
        "Default_Category": "Logistics",
        "Base_Risk": 0.75,
        "Rule_Triggered": "Storms often increase logistics risk and transit delays.",
    },
    "Port Congestion": {
        "Impact_Type": "Logistics Disruption",
        "Possible_Supply_Chain_Effect": "Increased Lead Time",
        "Default_Category": "Logistics",
        "Base_Risk": 0.85,
        "Rule_Triggered": "Port congestion directly delays incoming and outgoing shipments.",
    },
    "Flood": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Transport Disruption",
        "Default_Category": "Logistics",
        "Base_Risk": 0.85,
        "Rule_Triggered": "Flooding can disrupt roads, warehouses, and transport networks.",
    },
    "Extreme Heat": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Logistics Disruption",
        "Default_Category": "Logistics",
        "Base_Risk": 0.7,
        "Rule_Triggered": "Extreme heat can slow logistics and affect warehouse handling.",
    },
    "Extreme Weather": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Logistics Disruption",
        "Default_Category": "Logistics",
        "Base_Risk": 0.8,
        "Rule_Triggered": "Extreme weather conditions can disrupt transport and inventory handling.",
    },
    "Drought": {
        "Impact_Type": "Weather Disruption",
        "Possible_Supply_Chain_Effect": "Logistics Disruption",
        "Default_Category": "Logistics",
        "Base_Risk": 0.7,
        "Rule_Triggered": "Drought may affect supply chain operations through transport constraints.",
    },
    "Supply Shortage": {
        "Impact_Type": "Supply Disruption",
        "Possible_Supply_Chain_Effect": "Component Availability Risk",
        "Default_Category": "Supply",
        "Base_Risk": 0.85,
        "Rule_Triggered": "Supply shortages threaten production and component availability.",
    },
    "Semiconductor Shortage": {
        "Impact_Type": "Supply Disruption",
        "Possible_Supply_Chain_Effect": "Production Delay",
        "Default_Category": "Supply",
        "Base_Risk": 0.9,
        "Rule_Triggered": "Semiconductor shortages have a high impact on electronics production.",
    },
    "Factory Shutdown": {
        "Impact_Type": "Production Disruption",
        "Possible_Supply_Chain_Effect": "Production Halt",
        "Default_Category": "Manufacturing",
        "Base_Risk": 0.9,
        "Rule_Triggered": "Factory shutdowns halt assembly and manufacturing operations.",
    },
    "Logistics Disruption": {
        "Impact_Type": "Logistics Disruption",
        "Possible_Supply_Chain_Effect": "Delivery Delay",
        "Default_Category": "Logistics",
        "Base_Risk": 0.8,
        "Rule_Triggered": "Logistics disruption typically delays deliveries and shipments.",
    },
    # Additional related event types for consistent handling.
    "Production Delay": {
        "Impact_Type": "Production Disruption",
        "Possible_Supply_Chain_Effect": "Production Slowdown",
        "Default_Category": "Manufacturing",
        "Base_Risk": 0.8,
        "Rule_Triggered": "Production delays reduce throughput and increase cycle time.",
    },
    "Warehouse Disruption": {
        "Impact_Type": "Logistics Disruption",
        "Possible_Supply_Chain_Effect": "Inventory Handling Risk",
        "Default_Category": "Logistics",
        "Base_Risk": 0.8,
        "Rule_Triggered": "Warehouse disruption can affect inventory movement and order fulfilment.",
    },
}

DEFAULT_RULE: Dict[str, object] = {
    "Impact_Type": "Supply Chain Impact",
    "Possible_Supply_Chain_Effect": "Operational Uncertainty",
    "Default_Category": "Electronics",
    "Base_Risk": 0.6,
    "Rule_Triggered": "Fallback rule used for an event type without a predefined business rule.",
}

SEVERITY_WEIGHTS: Dict[str, float] = {
    "Critical": 1.0,
    "High": 0.9,
    "Medium": 0.7,
    "Low": 0.4,
}


def get_rule_config(event_type: str) -> Dict[str, object]:
    """Return the configured rule for a known event type."""
    return RULE_CONFIGS.get(event_type, DEFAULT_RULE)


def compute_risk_score(severity: str, confidence: float, base_risk: float) -> float:
    """Compute a deterministic risk score from severity, confidence and the event rule.

    The score combines a rule-specific base risk with the event severity and the
    event confidence score. The calculation uses a fixed formula so decisions
    remain explainable and repeatable.
    """
    severity_weight = SEVERITY_WEIGHTS.get(severity, 0.5)
    score = base_risk * 0.5 + severity_weight * 0.35 + float(confidence) * 0.15
    return round(min(1.0, max(0.0, score)), 2)


def impact_level_from_score(score: float) -> str:
    """Convert a risk score into a simple impact level."""
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def derive_affected_category(event_type: str, default_category: str) -> str:
    """Return a default affected category for an event type if no brand is matched."""
    rule = get_rule_config(event_type)
    return str(rule.get("Default_Category", default_category))


def derive_impact_type(event_type: str) -> str:
    """Return the impact type for an event type."""
    return str(get_rule_config(event_type).get("Impact_Type", DEFAULT_RULE["Impact_Type"]))


def derive_possible_effect(event_type: str) -> str:
    """Return the likely supply chain effect for an event type."""
    return str(get_rule_config(event_type).get("Possible_Supply_Chain_Effect", DEFAULT_RULE["Possible_Supply_Chain_Effect"]))


def derive_rule_triggered(event_type: str) -> str:
    """Return the human-readable rule triggered by the event type."""
    return str(get_rule_config(event_type).get("Rule_Triggered", DEFAULT_RULE["Rule_Triggered"]))
