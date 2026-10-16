"""Read-only deterministic routing for safety-critical Ask SupplyBeacon domains.

This layer never caches current state. The scenario branch calls the same API
simulation function used by Scenario Lab; it does not reimplement exposure.
"""
from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from src.dynamic_risk.current_state import load_current_state
from src.dynamic_risk.risk_views import COMPONENTS, numeric, risk_rows
from src.dynamic_risk.risk_views import operational_event_ids


def _contains(query: str, value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and re.search(r"(?<![\w])" + re.escape(text) + r"(?![\w])", query, re.IGNORECASE))


def _warehouse_candidates(graph: Any, query: str) -> list[str]:
    matches = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Warehouse":
            continue
        if any(_contains(query, value) for value in (node_id, attrs.get("warehouse_id"), attrs.get("name"), attrs.get("warehouse_name"))):
            matches.append(str(node_id))
    if not matches:
        # City is a location lookup, never an alias for a different city.
        # It resolves a warehouse only when exactly one warehouse has that
        # exact city name; otherwise the caller receives clarification.
        matches = [str(node_id) for node_id, attrs in graph.nodes(data=True)
                   if attrs.get("node_type") == "Warehouse" and _contains(query, attrs.get("city"))]
    return sorted(set(matches))


def _related(graph: Any, source: str, relation: str, node_type: str) -> list[str]:
    if source not in graph:
        return []
    return sorted({str(target) for target in graph.successors(source)
                   if (graph.get_edge_data(source, target) or {}).get("relationship") == relation
                   and graph.nodes[target].get("node_type") == node_type})


def _evidence(source_type: str, entity_id: str | None, relation: str | None, value: Any,
              *, provenance: str, timestamp: str | None = None, temporal_status: str | None = None) -> dict[str, Any]:
    return {"source_type": source_type, "entity_id": entity_id, "relation": relation,
            "value": value, "timestamp": timestamp, "temporal_status": temporal_status,
            "provenance": provenance}


def _response(query: str, answer: str, intent: str, *, entities: list[dict[str, Any]] | None = None,
              evidence: list[dict[str, Any]] | None = None, limitations: list[str] | None = None,
              warnings: list[str] | None = None, extra: dict[str, Any] | None = None,
              secondary: list[str] | None = None, started: float = 0.0) -> dict[str, Any]:
    evidence = evidence or []
    entities = entities or []
    limitations = limitations or []
    warnings = warnings or []
    plan = {"primary_intent": intent, "secondary_intents": secondary or [],
            "entity_mentions": [item.get("name", "") for item in entities],
            "resolved_entity_ids": [item["node_id"] for item in entities if item.get("node_id")],
            "requested_entity_types": sorted({item.get("node_type", "") for item in entities}),
            "clarification_needed": not entities and intent in {"SCENARIO", "CATALOG_LOOKUP", "WAREHOUSE_LOOKUP"}}
    elapsed = (perf_counter() - started) * 1000 if started else 0.0
    return {
        "query": query, "answer": answer, "intent": intent, "query_intent": intent, "query_plan": plan,
        "entities": entities, "resolved_entities": entities, "informational_entities": [],
        "evidence": evidence, "evidence_count": len(evidence),
        "evidence_used": [f"domain-{index}" for index in range(len(evidence))],
        "graph_paths": [], "events": [], "weather": [], "risk_evidence": [],
        "delivery_risk": None, "historical_evidence": {}, "current_metadata": {},
        "scenario_result": None, "catalog_context": None,
        "limitations": limitations, "warnings": warnings,
        "provenance_summary": sorted({item["source_type"] for item in evidence}),
        "current_data_fresh": None, "llm_used": False, "fallback_used": False,
        "tools_used": ["deterministic_domain_router"], "workflow_trace": ["deterministic_domain_router"],
        "timings": {"total_ms": elapsed},
        "structured_answer": {"answer": answer, "primary_intent": intent,
                              "secondary_intents": secondary or [], "entities": entities,
                              "evidence_used": [f"domain-{index}" for index in range(len(evidence))],
                              "warnings": warnings, "llm_used": False, "fallback_used": False},
        **(extra or {}),
    }


def answer_grounded_domain(query: str) -> dict[str, Any] | None:
    """Return a complete Ask response for explicit authoritative domains.

    Unsupported and context-free follow-ups return clarification, never an
    inferred first graph match. Other questions remain in the existing flow.
    """
    started = perf_counter()
    lower = query.casefold()
    scenario = any(term in lower for term in ("what if", "unavailable", "closes", "closed", "what happens to", "in that scenario", "reroute", "alternatives"))
    catalog = any(term in lower for term in ("catalog", "associated", "actually stored", "stored there", "stock of", "stocked there", "store product", "stores product", "stored at", "linked to this warehouse", "product impact"))
    risk = any(term in lower for term in ("risk", "risky", "at risk"))
    historical = any(term in lower for term in ("dataco", "historical shipping model", "historical model"))
    warehouse_events = "event" in lower and "warehouse" in lower and any(term in lower for term in ("current", "currently", "affecting", "active"))
    graph_structure = any(term in lower for term in ("graph structure", "graph neighbors", "graph neighbours", "structural connection"))
    system_provenance = any(term in lower for term in ("is this current risk a probability", "is current risk a probability", "supplybeacon provenance", "evidence behind this answer", "how does supplybeacon know"))
    lookup = any(term in lower for term in ("tell me about", "look up", "lookup", "details for", "where is"))
    if system_provenance:
        if "this answer" in lower:
            answer = "Name the answer or entity whose evidence you want to inspect. This stateless request has no verified prior-answer context."
        else:
            answer = ("SupplyBeacon's current risk score is a deterministic rule-based value, not a calibrated disruption probability. "
                      "Operational conclusions require typed current-graph paths and runtime-valid events; scenario results are modeled separately; "
                      "DataCo metrics describe a historical shipping model. Ask about a named entity to see its specific evidence.")
        return _response(query, answer, "SYSTEM_PROVENANCE", limitations=["No unstated conversation context was assumed."], started=started)
    if historical and risk:
        # Preserve the existing mixed network-risk/DataCo workflow; these
        # evidence domains must not be numerically combined here.
        return None
    if not (scenario or catalog or (risk and "wh_real_" in lower) or warehouse_events or graph_structure or lookup):
        # Existing LangGraph current-risk, weather, event, and historical
        # workflows retain their established response and tool contracts.
        return None
    state = load_current_state()
    graph = state.get("graph")
    if graph is None:
        return _response(query, "The current graph is unavailable, so SupplyBeacon cannot verify this answer.",
                         "SCENARIO" if scenario else "CATALOG_LOOKUP" if catalog else "CURRENT_RISK",
                         limitations=["Current graph unavailable."], started=started)
    if graph_structure or (lookup and not (scenario or catalog)):
        matches = [str(node_id) for node_id, attrs in graph.nodes(data=True)
                   if any(_contains(query, value) for value in (node_id, attrs.get("name"), attrs.get("warehouse_id")))]
        if matches:
            longest = max(len(str(graph.nodes[node_id].get("name", ""))) for node_id in matches)
            matches = [node_id for node_id in matches if len(str(graph.nodes[node_id].get("name", ""))) == longest]
        if len(matches) != 1:
            return _response(query, "Specify one unambiguous graph entity name or ID for this lookup.",
                             "GRAPH_STRUCTURE" if graph_structure else "WAREHOUSE_LOOKUP" if "warehouse" in lower else "PRODUCT_LOOKUP" if "product" in lower else "SUPPLIER_LOOKUP",
                             limitations=["No entity was selected from fuzzy or ambiguous matches."], started=started)
        node_id = matches[0]
        attrs = graph.nodes[node_id]
        entity_type = str(attrs.get("node_type", "Unknown"))
        entity = {"node_id": node_id, "node_type": entity_type, "name": attrs.get("name", node_id)}
        if graph_structure:
            edges = sorted({(str(source), str(target), str(data.get("relationship", "UNKNOWN")))
                            for source, target, data in graph.in_edges(node_id, data=True)} |
                           {(str(source), str(target), str(data.get("relationship", "UNKNOWN")))
                            for source, target, data in graph.out_edges(node_id, data=True)})
            answer = (f"Structural graph connections for {entity['name']}: "
                      + "; ".join(f"{source} --{relation}--> {target}" for source, target, relation in edges[:20])
                      + ". These topology links alone do not establish operational exposure or inventory.")
            evidence = [_evidence("CURRENT_GRAPH", node_id, relation, {"source": source, "target": target}, provenance="STRUCTURAL_ONLY")
                        for source, target, relation in edges[:20]]
            return _response(query, answer, "GRAPH_STRUCTURE", entities=[entity], evidence=evidence,
                             limitations=["Graph topology is structural; operational conclusions require relationship-specific validation."], started=started)
        answer = (f"{entity['name']} is a {entity_type} in the current graph (ID {node_id}). "
                  + (f"Location: {attrs.get('city')}, {attrs.get('state')}. " if attrs.get("city") and attrs.get("state") else "")
                  + "This lookup alone does not establish current risk, physical stock, or supplier exposure.")
        evidence = [_evidence("CURRENT_GRAPH", node_id, "node_attributes",
                              {key: attrs.get(key) for key in ("name", "node_type", "city", "state", "catalog_only") if key in attrs},
                              provenance="OBSERVED")]
        return _response(query, answer, f"{entity_type.upper()}_LOOKUP", entities=[entity], evidence=evidence, started=started)
    if warehouse_events and not (scenario or catalog):
        named = _warehouse_candidates(graph, query)
        paths = sorted({(str(event_id), str(warehouse_id))
                        for warehouse_id, attrs in graph.nodes(data=True) if attrs.get("node_type") == "Warehouse"
                        and (not named or str(warehouse_id) in named)
                        for event_id in operational_event_ids(graph, str(warehouse_id))})
        answer = ("Current operational events with validated AFFECTS paths to warehouses: "
                  + "; ".join(f"{graph.nodes[event_id].get('name', event_id)} affects {graph.nodes[warehouse_id].get('name', warehouse_id)}"
                                  for event_id, warehouse_id in paths) + ".") if paths else \
                 "No ACTIVE operational event has a validated AFFECTS path to a warehouse in the current runtime graph. Informational, upcoming, expired, and unknown-timing signals do not establish current operational exposure."
        evidence = [_evidence("CURRENT_EVENT", event_id, "AFFECTS", {"warehouse_id": warehouse_id},
                              provenance="VALIDATED_CURRENT_GRAPH", temporal_status="ACTIVE")
                    for event_id, warehouse_id in paths]
        response = _response(query, answer, "CURRENT_EVENTS", evidence=evidence,
                             limitations=["Only ACTIVE operational events on typed AFFECTS paths are counted."], started=started)
        response["events"] = [{"node_id": event_id, "name": graph.nodes[event_id].get("name", event_id),
                               "event_status": "ACTIVE", "operationally_relevant": True,
                               "warehouse_id": warehouse_id}
                              for event_id, warehouse_id in paths]
        response["tools_used"] = ["get_current_operational_event_paths"]
        return response
    matches = _warehouse_candidates(graph, query)
    if len(matches) != 1:
        if not matches and not ("warehouse" in lower or "wh_real_" in lower) and not (scenario or catalog):
            return None
        candidates = [{"node_id": node_id, "node_type": "Warehouse", "name": graph.nodes[node_id].get("name", node_id)} for node_id in matches]
        answer = ("More than one warehouse matches. Specify one exact warehouse ID or name: "
                  + "; ".join(f"{item['name']} ({item['node_id']})" for item in candidates) + ".") if matches else \
                 "Specify one exact warehouse name or ID. SupplyBeacon will not assume a scenario target or reuse hidden conversation context."
        return _response(query, answer, "SCENARIO" if scenario else "CATALOG_LOOKUP" if catalog else "CURRENT_RISK",
                         entities=candidates, limitations=["No operational conclusion was drawn without an unambiguous warehouse."], started=started)

    warehouse_id = matches[0]
    attrs = graph.nodes[warehouse_id]
    entity = {"node_id": warehouse_id, "node_type": "Warehouse", "name": attrs.get("name", warehouse_id),
              "city": attrs.get("city"), "state": attrs.get("state")}
    evidence = [_evidence("CURRENT_GRAPH", warehouse_id, "node_type", "Warehouse", provenance="OBSERVED")]
    answer_parts = []
    extra: dict[str, Any] = {}
    secondary: list[str] = []
    intent = "MULTI_INTENT" if sum((scenario, catalog, risk)) > 1 else "SCENARIO" if scenario else "CATALOG_LOOKUP" if catalog else "CURRENT_RISK"

    if risk:
        row = risk_rows(state).get(warehouse_id)
        score = numeric(row.get("Risk_Score")) if row else None
        level = str(row.get("Risk_Level", "Unavailable")) if score is not None else "Unavailable"
        components = {field: numeric(row.get(field)) if row else None for field in COMPONENTS}
        risk_record = {"node_id": warehouse_id, "risk_score": score, "risk_level": level,
                       "risk_components": components, "risk_reason": row.get("Risk_Reason") if row else None,
                       "source_type": "CURRENT_RISK_ENGINE"}
        answer_parts.append(f"{entity['name']} has derived current risk {level}" +
                            (f" (score {score}; rule-based, not a disruption probability)." if score is not None else
                             "; a score is unavailable because applicable evidence is missing."))
        evidence.append(_evidence("CURRENT_RISK_ENGINE", warehouse_id, "risk", risk_record, provenance="DERIVED / RULE_BASED"))
        extra["risk_evidence"] = [{"subject": warehouse_id, "predicate": "Risk_Level", "value": level, "provenance": "DERIVED / RULE_BASED"},
                                  {"subject": warehouse_id, "predicate": "Risk_Score", "value": score, "provenance": "DERIVED / RULE_BASED"}]
        secondary.append("CURRENT_RISK")

    if catalog or (scenario and attrs.get("catalog_only") in {True, "true", "True", 1, "1"}):
        models = _related(graph, warehouse_id, "CATALOG_ASSOCIATED_WITH", "ProductModel")
        brands = _related(graph, warehouse_id, "ASSOCIATED_WITH_BRAND", "Brand")
        stored_products = sorted({str(product_id) for product_id in graph.predecessors(warehouse_id)
                                  if graph.nodes[product_id].get("node_type") == "Product"
                                  and (graph.get_edge_data(product_id, warehouse_id) or {}).get("relationship") == "STORED_AT"})
        names = [str(graph.nodes[node_id].get("name", node_id)) for node_id in models]
        context = {"warehouse_id": warehouse_id, "product_model_ids": models, "product_model_names": names,
                   "brand_ids": brands, "catalog_association_count": len(models),
                   "verified_stored_product_ids": stored_products,
                   "inventory_relationship_available": bool(stored_products)}
        extra["catalog_context"] = context
        evidence.append(_evidence("CURATED_REFERENCE_DATA", warehouse_id, "CATALOG_ASSOCIATED_WITH", models,
                                  provenance="REFERENCE_ONLY"))
        evidence.append(_evidence("CURATED_REFERENCE_DATA", warehouse_id, "ASSOCIATED_WITH_BRAND", brands,
                                  provenance="REFERENCE_ONLY"))
        if models:
            answer_parts.append(f"The curated reference dataset associates {entity['name']} with {len(models)} ProductModel catalog entries"
                                + (f", including {', '.join(names[:5])}" if names else "") + ".")
        else:
            answer_parts.append(f"No ProductModel catalog association is recorded for {entity['name']}.")
        if brands:
            answer_parts.append("Reference-associated brands: " + ", ".join(str(graph.nodes[brand_id].get("name", brand_id)) for brand_id in brands[:10]) + ".")
        if stored_products:
            answer_parts.append(f"Separately, {len(stored_products)} operational Product nodes have a verified graph STORED_AT relation to this warehouse.")
        else:
            answer_parts.append("No verified STORED_AT operational inventory relation is available here. Catalog associations do not verify physical stock or prove no business impact.")
        secondary.append("CATALOG_LOOKUP")

    if scenario:
        # Import lazily: app.py owns the accepted Phase 3 simulator and imports
        # run_agentic_query, so a module-level import would be circular.
        from src.api.app import ScenarioRequest, simulate_scenario
        duration_match = re.search(r"\b(12|24|48|72)\s*(?:hours?|hrs?)\b", lower)
        severity_match = re.search(r"\b(medium|high|critical)\b", lower)
        result = simulate_scenario(ScenarioRequest(
            scenario_type="WAREHOUSE_UNAVAILABLE", target_node_id=warehouse_id,
            duration_hours=int(duration_match.group(1)) if duration_match else 48,
            scenario_severity=severity_match.group(1).capitalize() if severity_match else "Medium",
        ))
        extra["scenario_result"] = result.model_dump(mode="json")
        evidence.append(_evidence("SCENARIO_ENGINE", warehouse_id, "WAREHOUSE_UNAVAILABLE",
                                  {"affected_product_ids": result.affected_product_ids,
                                   "affected_supplier_ids": result.affected_supplier_ids,
                                   "alternative_warehouse_candidates": result.alternative_warehouse_candidates,
                                   "inventory_relationship_available": result.inventory_relationship_available},
                                  provenance="DERIVED / RULE_BASED"))
        answer_parts.append(result.explanation)
        secondary.append("SCENARIO")

    answer = " ".join(answer_parts)
    limitations = list(dict.fromkeys((extra.get("scenario_result") or {}).get("limitations", [])))
    if catalog:
        limitations.append("Catalog association is not verified physical inventory.")
    response = _response(query, answer, intent, entities=[entity], evidence=evidence,
                         limitations=limitations, extra=extra,
                         secondary=secondary if intent == "MULTI_INTENT" else [], started=started)
    response["tools_used"] = ["resolve_exact_warehouse", *( ["get_current_risk"] if risk else [] ),
                              *( ["get_catalog_associations"] if catalog else [] ),
                              *( ["simulate_warehouse_unavailable"] if scenario else [] )]
    return response
