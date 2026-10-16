"""Deterministic business-disruption taxonomy and evidence classification."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


INFORMATIONAL_SIGNAL = "INFORMATIONAL_SIGNAL"
POTENTIAL_DISRUPTION = "POTENTIAL_DISRUPTION"
OPERATIONAL_DISRUPTION = "OPERATIONAL_DISRUPTION"


@dataclass(frozen=True)
class DisruptionRule:
    family: str
    event_type: str
    topic_pattern: re.Pattern[str]
    impact_type: str
    base_severity: str


def _rule(
    family: str,
    event_type: str,
    topic_pattern: str,
    impact_type: str,
    base_severity: str = "Medium",
) -> DisruptionRule:
    return DisruptionRule(
        family=family,
        event_type=event_type,
        topic_pattern=re.compile(topic_pattern, re.IGNORECASE),
        impact_type=impact_type,
        base_severity=base_severity,
    )


BUSINESS_DISRUPTION_RULES = (
    _rule("WAREHOUSE_FIRE", "Warehouse Fire", r"\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b.{0,60}\bfire\b|\bfire\b.{0,60}\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b", "WAREHOUSE_OPERATION", "High"),
    _rule("WAREHOUSE_CLOSURE", "Warehouse Closure", r"\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b.{0,60}\b(?:clos(?:ed|ure)|shut(?:s|ting)? down|shutdown)\b|\b(?:clos(?:ed|ure)|shut(?:s|ting)? down|shutdown)\b.{0,60}\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b", "WAREHOUSE_OPERATION", "High"),
    _rule("WAREHOUSE_FLOODING", "Warehouse Flooding", r"\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b.{0,60}\bflood(?:ed|ing)?\b|\bflood(?:ed|ing)?\b.{0,60}\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b", "WAREHOUSE_OPERATION", "High"),
    _rule("WAREHOUSE_EVACUATION", "Warehouse Evacuation", r"\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b.{0,60}\bevacuat(?:ed|ion)\b|\bevacuat(?:ed|ion)\b.{0,60}\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b", "WAREHOUSE_OPERATION", "High"),
    _rule("WAREHOUSE_EQUIPMENT_FAILURE", "Warehouse Equipment Failure", r"\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b.{0,60}\b(?:equipment|conveyor|sorting system|forklift|automation)\b.{0,30}\b(?:fail(?:ure|ed)?|breakdown|outage|damaged)\b|\b(?:equipment|conveyor|sorting system|forklift|automation)\b.{0,30}\b(?:fail(?:ure|ed)?|breakdown|outage|damaged)\b.{0,60}\b(?:warehouse|distribution center|fulfilment center|fulfillment center|depot)\b", "WAREHOUSE_OPERATION", "Medium"),
    _rule("ROAD_CLOSURE", "Road Closure", r"\b(?:roads?|highways?|motorways?|expressways?|flyovers?)\b.{0,50}\b(?:clos(?:ed|ure)|block(?:ed|age)|shut(?:s|ting)? down|restrict(?:ed|ion)|ban(?:ned)?)\b|\b(?:clos(?:ed|ure)|block(?:ed|age)|shut(?:s|ting)? down|restrict(?:ed|ion)|ban(?:ned)?)\b.{0,50}\b(?:roads?|highways?|motorways?|expressways?|flyovers?)\b", "ROAD_LOGISTICS", "Medium"),
    _rule("BRIDGE_CLOSURE", "Bridge Closure", r"\bbridge\b.{0,50}\b(?:clos(?:ed|ure)|block(?:ed|age)|damaged|unsafe|shut(?:s|ting)? down)\b|\b(?:clos(?:ed|ure)|block(?:ed|age)|damaged|unsafe|shut(?:s|ting)? down)\b.{0,50}\bbridge\b", "ROAD_LOGISTICS", "High"),
    _rule("TRAFFIC_DISRUPTION", "Traffic Disruption", r"\b(?:traffic|motorists?|commuters?|passenger vehicles?|passenger buses?)\b.{0,45}\b(?:snarl(?:s|ed)?|congestion|delays?|disrupt(?:ed|ion))\b|\b(?:snarl(?:s|ed)?|congestion|delays?|disrupt(?:ed|ion))\b.{0,45}\b(?:traffic|motorists?|commuters?|passenger vehicles?|passenger buses?)\b", "ROAD_LOGISTICS", "Low"),
    _rule("RAIL_DISRUPTION", "Rail Disruption", r"\b(?:rail|railway|train|freight train)\b.{0,50}\b(?:disrupt(?:ed|ion)|suspend(?:ed|sion)|cancel(?:led|ed)|halt(?:ed)?|derail(?:ed|ment))\b|\b(?:disrupt(?:ed|ion)|suspend(?:ed|sion)|cancel(?:led|ed)|halt(?:ed)?|derail(?:ed|ment))\b.{0,50}\b(?:rail|railway|train|freight train)\b", "RAIL_LOGISTICS", "High"),
    _rule("PORT_DISRUPTION", "Port Disruption", r"\bport\b.{0,50}\b(?:clos(?:ed|ure)|shutdown|shut down|suspend(?:ed|sion)|halt(?:ed)?|disrupt(?:ed|ion)|blocked|congestion)\b|\b(?:clos(?:ed|ure)|shutdown|shut down|suspend(?:ed|sion)|halt(?:ed)?|disrupt(?:ed|ion)|blocked|congestion)\b.{0,50}\bport\b", "PORT_LOGISTICS", "High"),
    _rule("AIR_CARGO_DISRUPTION", "Air Cargo Disruption", r"\b(?:airport|air cargo|air freight|cargo flight)\b.{0,50}\b(?:clos(?:ed|ure)|suspend(?:ed|sion)|cancel(?:led|ed)|halt(?:ed)?|ground(?:ed|ing)|disrupt(?:ed|ion))\b|\b(?:clos(?:ed|ure)|suspend(?:ed|sion)|cancel(?:led|ed)|halt(?:ed)?|ground(?:ed|ing)|disrupt(?:ed|ion))\b.{0,50}\b(?:airport|air cargo|air freight|cargo flight)\b", "AIR_CARGO", "High"),
    _rule("TRANSPORT_STRIKE", "Transport Strike", r"\b(?:transport|transportation|logistics|freight|delivery)\b.{0,40}\bstrike\b|\bstrike\b.{0,40}\b(?:transport|transportation|logistics|freight|delivery)\b", "LABOR_AVAILABILITY", "High"),
    _rule("TRUCKER_STRIKE", "Trucker Strike", r"\b(?:truck(?:er|ing)?s?|lorry|road haulage)\b.{0,40}\bstrike\b|\bstrike\b.{0,40}\b(?:truck(?:er|ing)?s?|lorry|road haulage)\b", "LABOR_AVAILABILITY", "High"),
    _rule("WAREHOUSE_STRIKE", "Warehouse Strike", r"\bwarehouse\b.{0,40}\bstrike\b|\bstrike\b.{0,40}\bwarehouse\b", "LABOR_AVAILABILITY", "High"),
    _rule("POWER_OUTAGE", "Power Outage", r"\b(?:power|electricity|grid)\b.{0,40}\b(?:outage|blackout|cut|failure)\b|\b(?:outage|blackout)\b.{0,40}\b(?:power|electricity|grid)\b", "POWER_AVAILABILITY", "High"),
    _rule("FUEL_SHORTAGE", "Fuel Shortage", r"\b(?:fuel|diesel|petrol)\b.{0,40}\b(?:shortage|ration(?:ed|ing)|running out|supply cut)\b|\b(?:shortage|ration(?:ed|ing))\b.{0,40}\b(?:fuel|diesel|petrol)\b", "FUEL_AVAILABILITY", "High"),
    _rule("CYCLONE", "Cyclone", r"\b(?:cyclone|hurricane|typhoon)\b", "ROAD_LOGISTICS", "High"),
    _rule("FLOOD", "Flood", r"\b(?:flood(?:s|ed|ing)?|flash flood(?:s|ed|ing)?)\b", "ROAD_LOGISTICS", "High"),
    _rule("SEVERE_STORM", "Severe Storm", r"\b(?:severe|extreme|damaging)\s+(?:storm|winds?)\b|\b(?:storm|winds?)\b.{0,30}\b(?:severe|extreme|damaging)\b", "ROAD_LOGISTICS", "High"),
    _rule("EXTREME_RAINFALL", "Extreme Rainfall", r"\b(?:extreme|exceptional|record|very heavy|severe)\s+rain(?:fall)?\b", "ROAD_LOGISTICS", "Medium"),
    _rule("LANDSLIDE", "Landslide", r"\blandslid(?:e|es|ing)\b|\bmajor landslip\b", "ROAD_LOGISTICS", "High"),
    _rule("FACTORY_SHUTDOWN", "Factory Shutdown", r"\b(?:factory|plant|manufacturing site)\b.{0,50}\b(?:shutdown|shut down|closed|closure)\b|\b(?:shutdown|shut down|closed|closure)\b.{0,50}\b(?:factory|plant|manufacturing site)\b", "MANUFACTURING", "High"),
    _rule("PRODUCTION_HALT", "Production Halt", r"\bproduction\b.{0,50}\b(?:halt(?:ed)?|stopp(?:ed|age)|suspend(?:ed|sion))\b|\b(?:halt(?:ed)?|stopp(?:ed|age)|suspend(?:ed|sion))\b.{0,50}\bproduction\b", "MANUFACTURING", "High"),
    _rule("FACTORY_FIRE", "Factory Fire", r"\b(?:factory|plant|manufacturing site)\b.{0,60}\bfire\b|\bfire\b.{0,60}\b(?:factory|plant|manufacturing site)\b", "MANUFACTURING", "High"),
    _rule("CYBER_LOGISTICS_DISRUPTION", "Cyber Logistics Disruption", r"\b(?:cyberattack|cyber attack|ransomware|malware|systems? hack(?:ed)?)\b.{0,100}\b(?:logistics|warehouse|freight|cargo|transport|shipping|supply chain)\b|\b(?:logistics|warehouse|freight|cargo|transport|shipping|supply chain)\b.{0,100}\b(?:cyberattack|cyber attack|ransomware|malware|systems? hack(?:ed)?)\b", "CYBER_OPERATION", "High"),
    _rule("CURFEW", "Curfew", r"\bcurfew\b", "REGULATORY", "Medium"),
    _rule("BORDER_RESTRICTION", "Border Restriction", r"\b(?:border|cross-border)\b.{0,50}\b(?:clos(?:ed|ure)|restrict(?:ed|ion)|suspend(?:ed|sion))\b|\b(?:clos(?:ed|ure)|restrict(?:ed|ion)|suspend(?:ed|sion))\b.{0,50}\b(?:border|cross-border)\b", "REGULATORY", "High"),
    _rule("CUSTOMS_DISRUPTION", "Customs Disruption", r"\bcustoms\b.{0,50}\b(?:disrupt(?:ed|ion)|suspend(?:ed|sion)|halt(?:ed)?|closed|backlog|blocked)\b|\b(?:disrupt(?:ed|ion)|suspend(?:ed|sion)|halt(?:ed)?|closed|backlog|blocked)\b.{0,50}\bcustoms\b", "REGULATORY", "Medium"),
    _rule("FREIGHT_RESTRICTION", "Freight Restriction", r"\bfreight\b.{0,50}\b(?:restrict(?:ed|ion)|suspend(?:ed|sion)|ban(?:ned)?|halt(?:ed)?|block(?:ed|age))\b|\b(?:restrict(?:ed|ion)|suspend(?:ed|sion)|ban(?:ned)?|halt(?:ed)?|block(?:ed|age))\b.{0,50}\bfreight\b", "REGULATORY", "High"),
    _rule("PRODUCT_RECALL", "Product Recall", r"\bproduct recall\b|\brecall(?:ed|s)?\b.{0,40}\b(?:product|component|device|electronics)\b", "PRODUCT_AVAILABILITY", "Medium"),
    _rule("PRODUCTION_SUSPENSION", "Production Suspension", r"\bproduction\b.{0,50}\b(?:suspend(?:ed|sion)|pause[ds]?)\b|\b(?:suspend(?:ed|sion)|pause[ds]?)\b.{0,50}\bproduction\b", "MANUFACTURING", "High"),
    _rule("COMPONENT_SHORTAGE", "Component Shortage", r"\b(?:component|parts?) shortage\b", "PRODUCT_AVAILABILITY", "High"),
    _rule("SEMICONDUCTOR_SHORTAGE", "Semiconductor Shortage", r"\b(?:semiconductor|chip) shortage\b", "PRODUCT_AVAILABILITY", "High"),
    _rule("SUPPLY_SHORTAGE", "Supply Shortage", r"\b(?:supply|raw material) shortage\b", "PRODUCT_AVAILABILITY", "High"),
    _rule("SHIPPING_DELAY", "Shipping Delay", r"\b(?:shipping|shipment|delivery) delays?\b", "ROAD_LOGISTICS", "Medium"),
)

RULE_BY_FAMILY = {rule.family: rule for rule in BUSINESS_DISRUPTION_RULES}
RULE_BY_EVENT_TYPE = {rule.event_type: rule for rule in BUSINESS_DISRUPTION_RULES}
LEGACY_EVENT_FAMILY_ALIASES = {
    "Warehouse Disruption": {"WAREHOUSE_FIRE", "WAREHOUSE_CLOSURE", "WAREHOUSE_FLOODING", "WAREHOUSE_EVACUATION", "WAREHOUSE_EQUIPMENT_FAILURE", "WAREHOUSE_STRIKE"},
    "Factory Fire": {"FACTORY_FIRE"},
    "Factory Shutdown": {"FACTORY_SHUTDOWN"},
    "Production Halt": {"PRODUCTION_HALT"},
    "Power Outage": {"POWER_OUTAGE"},
    "Road Closure": {"ROAD_CLOSURE"},
    "Port Congestion": {"PORT_DISRUPTION"},
    "Heavy Rain": {"EXTREME_RAINFALL"},
    "Flood": {"FLOOD", "WAREHOUSE_FLOODING"},
    "Cyclone": {"CYCLONE"},
    "Strike": {"TRANSPORT_STRIKE", "TRUCKER_STRIKE", "WAREHOUSE_STRIKE"},
    "Supply Shortage": {"SUPPLY_SHORTAGE"},
    "Component Shortage": {"COMPONENT_SHORTAGE"},
    "Semiconductor Shortage": {"SEMICONDUCTOR_SHORTAGE"},
    "Shipping Delay": {"SHIPPING_DELAY"},
}

DIRECT_EFFECT_PATTERN = re.compile(
    r"\b(?:closed|closure|shutdown|shut down|suspended|suspension|blocked|blockage|"
    r"halted|stopped|failure|breakdown|cancelled|canceled|evacuated|flooded|damaged|disrupted|disruption|"
    r"operations stopped|freight suspended|cargo suspended|transport suspended|"
    r"production stopped|outage|blackout|shortage|strike|restriction|restricted|"
    r"recall(?:ed)?|derailed|grounded|backlog|delays?|banned|prohibited|"
    r"disrupts|disrupting|halts|stops)\b",
    re.IGNORECASE,
)
TRANSPORT_CONTEXT_PATTERN = re.compile(
    r"\b(?:freight|cargo|trucks?|trucking|goods\s+movement|goods\s+vehicles?|"
    r"commercial\s+vehicles?|deliver(?:y|ies)|shipments?|logistics|supply|"
    r"warehouse\s+access|distribution|transport\s+of\s+goods|haulage|"
    r"port\s+access|industrial\s+transport|freight\s+corridor)\b",
    re.IGNORECASE,
)
TRANSPORT_CONTEXT_REQUIRED_FAMILIES = {
    "ROAD_CLOSURE",
    "BRIDGE_CLOSURE",
    "TRAFFIC_DISRUPTION",
    "LANDSLIDE",
}
UNIMPACTED_TRANSPORT_PATTERN = re.compile(
    r"\b(?:freight|cargo|truck(?:ing)?|goods movement|deliver(?:y|ies)|"
    r"shipment|logistics|supply chain)\b.{0,45}\b(?:unaffected|unimpeded|"
    r"continued normally|no impact|not disrupted|not delayed|not affected)\b|"
    r"\b(?:unaffected|unimpeded|continued normally|no impact|not disrupted|"
    r"not delayed|not affected)\b.{0,45}\b(?:freight|cargo|truck(?:ing)?|"
    r"goods movement|deliver(?:y|ies)|shipment|logistics|supply chain)\b",
    re.IGNORECASE,
)
POTENTIAL_CUE_PATTERN = re.compile(
    r"\b(?:may|might|could|possible|possibly|potential|expected|forecast|warning|"
    r"alert|threat|risk|would|if)\b",
    re.IGNORECASE,
)
NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|without|denies?|prevented|avoided|unlikely|did not|has not|"
    r"have not|was not|were not)\b",
    re.IGNORECASE,
)
SEVERE_IMPACT_PATTERN = re.compile(
    r"\b(?:nationwide|widespread|multi[- ]?(?:day|week|month|mode)|"
    r"all operations|entire region|prolonged|indefinite|multiple modes)\b",
    re.IGNORECASE,
)
CRITICAL_IMPACT_PATTERN = re.compile(
    r"\b(?:nationwide|widespread|indefinite|all operations|entire region|"
    r"prolonged|multi[- ]?(?:day|week|month)|multiple modes)\b",
    re.IGNORECASE,
)
MINOR_IMPACT_PATTERN = re.compile(
    r"\b(?:minor|localized|brief|short[- ]lived|limited|partial|few hours)\b",
    re.IGNORECASE,
)
COMMENTARY_PATTERN = re.compile(
    r"\b(?:earnings|share price|stock market|invest(?:ment|or)|product launch|"
    r"sports?|celebrity|political debate|industry commentary|analysts? say|"
    r"generic supply.chain concerns?)\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if part.strip()]


def matching_disruption_rules(text: str) -> list[tuple[DisruptionRule, str]]:
    """Return each family with the sentence that explicitly names its topic."""
    matches: dict[str, tuple[DisruptionRule, str]] = {}
    for sentence in _sentences(text):
        for rule in BUSINESS_DISRUPTION_RULES:
            topic_match = rule.topic_pattern.search(sentence)
            if topic_match and not _negated(sentence, topic_match) and rule.family not in matches:
                matches[rule.family] = (rule, sentence)
    return list(matches.values())


def _negated(sentence: str, match: re.Match[str]) -> bool:
    return bool(NEGATION_PATTERN.search(sentence[max(0, match.start() - 45):match.start()]))


def _direct_effect(sentence: str) -> re.Match[str] | None:
    for match in DIRECT_EFFECT_PATTERN.finditer(sentence):
        if _negated(sentence, match):
            continue
        prefix = sentence[max(0, match.start() - 55):match.start()]
        suffix = sentence[match.end():match.end() + 45]
        if POTENTIAL_CUE_PATTERN.search(prefix) or POTENTIAL_CUE_PATTERN.search(suffix):
            continue
        return match
    return None


def _has_related_transport_context(sentence: str, effect: re.Match[str]) -> bool:
    """Require goods/logistics context close to the direct infrastructure effect."""
    start = max(0, effect.start() - 120)
    end = min(len(sentence), effect.end() + 120)
    context = sentence[start:end]
    if UNIMPACTED_TRANSPORT_PATTERN.search(context):
        return False
    return bool(TRANSPORT_CONTEXT_PATTERN.search(context))


def explicit_statewide_scope(text: str, state: str) -> bool:
    """Require explicit state-wide scope rather than a passing state mention."""
    state_pattern = re.escape(" ".join((state or "").casefold().split()))
    if not state_pattern:
        return False
    scoped_disruption = re.compile(
        r"\b(?:highways?|roads?|freight movement|truck routes?|transport|"
        r"truck(?:er|ing)?s?|strike|curfew|border|customs|freight|cargo|"
        r"flood(?:s|ed|ing)?|cyclone|factory|warehouse|production|power|"
        r"fuel|evacuat(?:ed|ion)|disruption|restrict(?:ed|ion)|outage|shutdown)\b",
        re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text or ""):
        normalized = " ".join(sentence.casefold().split())
        if (
            not re.search(rf"\b{state_pattern}\b", normalized)
            or not scoped_disruption.search(normalized)
        ):
            continue
        if (
            re.search(r"\bstatewide\b", normalized)
            or re.search(rf"\bthroughout\s+(?:the\s+)?{state_pattern}\b", normalized)
            or re.search(
                rf"\b(?:all|every|major)\b.{{0,50}}\b(?:highways|roads|freight movement|truck routes)\b"
                rf".{{0,50}}\b(?:across|throughout|in)\s+{state_pattern}\b",
                normalized,
            )
            or re.search(
                rf"\b(?:across|throughout)\s+{state_pattern}\b.{{0,60}}"
                r"\b(?:highways|roads|freight movement|truck routes)\b",
                normalized,
            )
        ):
            return True
    return False


def classify_news_evidence(title: str, description: str, family: str | None = None) -> dict[str, Any]:
    """Classify news evidence with a direct, sentence-scoped deterministic rule."""
    text = f"{title or ''}. {description or ''}"
    matches = matching_disruption_rules(text)
    if family is not None:
        matches = [item for item in matches if item[0].family == family]

    if not matches:
        commentary = bool(COMMENTARY_PATTERN.search(text))
        return {
            "evidence_level": INFORMATIONAL_SIGNAL,
            "event_class": "INDUSTRY_COMMENTARY" if commentary else "IRRELEVANT",
            "operationally_relevant": False,
            "event_family": family or "",
            "impact_type": "",
            "severity": "Low",
            "directness": "GENERAL_RISK_SIGNAL",
            "classification_reason": (
                "Classified as a signal only because the article contains commentary or a broad topic "
                "without a specific supply-chain disruption family."
            ),
            "matched_evidence": "",
        }

    rule, topic_sentence = matches[0]
    topic_match = rule.topic_pattern.search(topic_sentence)
    if topic_match and _negated(topic_sentence, topic_match):
        return {
            "evidence_level": INFORMATIONAL_SIGNAL,
            "event_class": "IRRELEVANT",
            "operationally_relevant": False,
            "event_family": rule.family,
            "impact_type": rule.impact_type,
            "severity": "Low",
            "directness": "GENERAL_RISK_SIGNAL",
            "classification_reason": (
                f"Classified as a signal only because the source negates or rules out "
                f"{rule.family.replace('_', ' ').lower()}."
            ),
            "matched_evidence": topic_sentence,
        }
    effect = _direct_effect(topic_sentence)
    if topic_match and effect:
        transport_context_required = rule.family in TRANSPORT_CONTEXT_REQUIRED_FAMILIES
        if transport_context_required and not _has_related_transport_context(topic_sentence, effect):
            level = POTENTIAL_DISRUPTION
            reason = (
                "Road infrastructure disruption is reported, but no direct freight, cargo, delivery, "
                "or supply-chain impact is established."
            )
            return {
                "evidence_level": level,
                "event_class": "RISK_WARNING",
                "operationally_relevant": False,
                "event_family": rule.family,
                "impact_type": rule.impact_type,
                "severity": "Low",
                "directness": "GENERAL_RISK_SIGNAL",
                "classification_reason": reason,
                "matched_evidence": topic_sentence,
            }
        level = OPERATIONAL_DISRUPTION
        severity = rule.base_severity
        if CRITICAL_IMPACT_PATTERN.search(topic_sentence):
            severity = "Critical"
        elif MINOR_IMPACT_PATTERN.search(topic_sentence):
            severity = "Low"
        elif SEVERE_IMPACT_PATTERN.search(topic_sentence) and severity != "Critical":
            severity = "High"
        reason = (
            f"Classified as an operational disruption because the source explicitly reports "
            f"{effect.group(0).lower()} for {rule.family.replace('_', ' ').lower()}."
        )
        directness = "DIRECT_OPERATIONAL"
    else:
        level = POTENTIAL_DISRUPTION
        severity = "Medium" if re.search(r"\b(?:severe|extreme|red alert|high alert)\b", text, re.I) else "Low"
        reason = (
            f"Classified as a potential disruption because the source names {rule.family.replace('_', ' ').lower()} "
            "but does not report an unqualified operational effect in the same sentence."
        )
        directness = "EXPECTED_OPERATIONAL_IMPACT" if POTENTIAL_CUE_PATTERN.search(text) else "GENERAL_RISK_SIGNAL"

    return {
        "evidence_level": level,
        "event_class": "ACTUAL_DISRUPTION" if level == OPERATIONAL_DISRUPTION else "RISK_WARNING",
        "operationally_relevant": level == OPERATIONAL_DISRUPTION,
        "event_family": rule.family,
        "impact_type": rule.impact_type,
        "severity": severity,
        "directness": directness,
        "classification_reason": reason,
        "matched_evidence": topic_sentence,
    }


def classify_weather_evidence(
    event_type: str,
    description: str,
    alert_text: str = "",
) -> dict[str, Any]:
    """Keep observations informational; alerts need direct logistics impact for operations."""
    combined = f"{alert_text}. {description or ''}"
    rule_matches = matching_disruption_rules(combined)
    weather_families = {
        "CYCLONE", "FLOOD", "SEVERE_STORM", "EXTREME_RAINFALL", "LANDSLIDE",
        "WAREHOUSE_FLOODING", "ROAD_CLOSURE", "BRIDGE_CLOSURE", "TRAFFIC_DISRUPTION", "RAIL_DISRUPTION",
        "PORT_DISRUPTION", "AIR_CARGO_DISRUPTION",
    }
    matches = [item for item in rule_matches if item[0].family in weather_families]
    matched = matches[0] if matches else None
    effect = _direct_effect(matched[1]) if matched else None
    transport_context_ok = bool(
        matched
        and effect
        and (
            matched[0].family == "WAREHOUSE_FLOODING"
            or (
                matched[0].family not in weather_families
                and matched[0].family not in TRANSPORT_CONTEXT_REQUIRED_FAMILIES
            )
            or _has_related_transport_context(matched[1], effect)
        )
    )
    if alert_text and matched and effect and transport_context_ok:
        rule = matched[0]
        severity = "Critical" if CRITICAL_IMPACT_PATTERN.search(matched[1]) else rule.base_severity
        if MINOR_IMPACT_PATTERN.search(matched[1]):
            severity = "Low"
        return {
            "evidence_level": OPERATIONAL_DISRUPTION,
            "event_class": "ACTUAL_DISRUPTION",
            "operationally_relevant": True,
            "event_family": rule.family,
            "impact_type": rule.impact_type,
            "severity": severity,
            "directness": "DIRECT_OPERATIONAL",
            "classification_reason": (
                f"Classified as an operational disruption because the validated weather alert explicitly "
                f"reports {effect.group(0).lower()} affecting {rule.family.replace('_', ' ').lower()}."
            ),
            "matched_evidence": matched[1],
        }
    if alert_text and event_type.casefold() not in {"light rain", "moderate rain"}:
        family = matched[0].family if matched else ""
        impact_type = matched[0].impact_type if matched else "ROAD_LOGISTICS"
        return {
            "evidence_level": POTENTIAL_DISRUPTION,
            "event_class": "RISK_WARNING",
            "operationally_relevant": False,
            "event_family": family,
            "impact_type": impact_type,
            "severity": "Medium",
            "directness": "EXPECTED_OPERATIONAL_IMPACT",
            "classification_reason": (
                "Classified as a potential disruption because a validated severe-weather alert exists, "
                "but it does not report a direct logistics or facility impact."
            ),
            "matched_evidence": alert_text,
        }
    return {
        "evidence_level": INFORMATIONAL_SIGNAL,
        "event_class": "RISK_WARNING",
        "operationally_relevant": False,
        "event_family": "",
        "impact_type": "ROAD_LOGISTICS" if event_type.casefold() not in {"light rain", "moderate rain"} else "",
        "severity": "Low",
        "directness": "GENERAL_RISK_SIGNAL",
        "classification_reason": (
            "Classified as a signal only because the weather observation or ordinary forecast does not "
            "contain a validated alert with explicit operational impact."
        ),
        "matched_evidence": description or "",
    }
