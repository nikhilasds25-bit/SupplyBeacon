"""Capture secret-free current-risk diagnostics; optionally run the configured refresh."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--live-base", default="")
    args = parser.parse_args()
    # Some provider exception messages contain credential-bearing request URLs.
    logging.disable(logging.CRITICAL)
    from src.dynamic_risk.current_state import CURRENT_DIR, load_current_state
    from src.event_extraction.temporal_validity import is_currently_eligible
    from src.api import app as api
    from src.external_data.unified_sources import _configured_key
    refresh_method = "not requested"
    if args.refresh:
        key = _configured_key("SUPPLYBEACON_ADMIN_API_KEY")
        if key:
            import os
            os.environ["SUPPLYBEACON_ADMIN_API_KEY"] = key
            api.current_refresh(x_supplybeacon_admin_key=key)
            refresh_method = "protected endpoint handler"
        else:
            # Same pipeline, without creating or bypassing an API credential.
            from src.dynamic_risk.refresh_current_state import refresh_current_state
            refresh_current_state()
            from src.agentic.tools import invalidate_retriever_cache
            invalidate_retriever_cache()
            refresh_method = "existing pipeline directly (no configured admin key)"
    state = load_current_state()
    events, graph = state["events"], state["graph"]
    traces = api.current_impact()
    diagnostics = []
    for event in events:
        eligible = is_currently_eligible(event)
        node = api._resolve_event_node(graph, event)
        targets = sorted({target for _, target, edge in graph.out_edges(node, data=True)
                          if edge.get("relationship") == "AFFECTS"}) if node else []
        reasons = []
        if event.get("Operationally_Relevant") is not True:
            reasons.append("non-operational: " + str(event.get("Event_Class")))
        if event["event_status"] != "ACTIVE":
            reasons.append(event["event_status"] + ": " + str(event.get("temporal_basis")))
        if eligible and not targets:
            reasons.append("no explicit warehouse/location match")
        diagnostics.append(dict(
            Event_ID=event.get("Event_ID"), title=event.get("Title"), type=event.get("Event_Type"),
            source=event.get("source_channel"), location=event.get("Location"),
            start_time=event.get("event_start"), end_time=event.get("event_end"),
            temporal_status=event["event_status"], operational_relevant=event.get("Operationally_Relevant"),
            operational_eligible=eligible, resolution_status="RESOLVED" if targets else "UNRESOLVED" if eligible else "NOT_ELIGIBLE",
            excluded_reason="; ".join(reasons) or "none", affected_graph_entity_count=len(targets),
            affected_graph_entities=targets,
        ))
    persisted_events = json.loads((CURRENT_DIR / "current_events.json").read_text(encoding="utf-8"))
    import pandas as pd
    persisted_risk = pd.read_csv(CURRENT_DIR / "current_risk_scores.csv")
    metadata = state["metadata"]
    summary = api.current_summary()
    report = dict(
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        refresh_method=refresh_method,
        files={p.name: datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
               for p in CURRENT_DIR.iterdir() if p.is_file()},
        news_fetch_timestamp=metadata.get("news_fetch_timestamp"), weather_fetch_timestamp=metadata.get("weather_fetch_timestamp"),
        article_count=len(state["news"].get("articles", [])), weather_record_count=len(state["weather"].get("cities", {})),
        raw_events=len(events), persisted_status_counts=dict(Counter(e.get("event_status") for e in persisted_events)),
        runtime_status_counts=dict(Counter(e["event_status"] for e in events)),
        persisted_operational_events=metadata.get("operational_event_count"),
        operational_events=sum(is_currently_eligible(e) for e in events),
        excluded_events=sum(not is_currently_eligible(e) for e in events),
        resolved_events=sum(d["resolution_status"] == "RESOLVED" for d in diagnostics),
        affected_warehouses=len({w["node_id"] for t in traces for w in t["warehouses"]}),
        affected_products=len({p for t in traces for p in t["product_ids"]}),
        affected_suppliers=len({s for t in traces for s in t["supplier_ids"]}),
        persisted_risk_counts=persisted_risk.Risk_Level.value_counts().to_dict(),
        runtime_risk_counts=state["risk"].Risk_Level.value_counts().to_dict(),
        summary=summary, events=diagnostics,
        collectors={category: {name: {k: v for k, v in details.items() if k != "error"}
                    | {"error_present": bool(details.get("error"))} for name, details in sources.items()}
                    for category, sources in metadata.get("collector_sources", {}).items()},
        articles=[{k: a.get(k) for k in ("title", "description", "published_at", "source_channel")}
                  for a in state["news"].get("articles", [])],
        weather=[dict(city=city, observed_at=record.get("dt"), condition=record.get("weather"),
                      rain=record.get("rain"), alerts=record.get("alerts", []))
                 for city, record in state["weather"].get("cities", {}).items()],
    )
    if args.live_base:
        import requests
        responses = {name: requests.get(args.live_base + path, timeout=60)
                     for name, path in {"summary": "/current/summary", "events": "/current/events?operational_only=true",
                                        "products": "/current/products", "warehouses": "/current/warehouses",
                                        "suppliers": "/current/suppliers", "metadata": "/current/metadata"}.items()}
        for response in responses.values():
            response.raise_for_status()
        live = {name: response.json() for name, response in responses.items()}
        report["live_api"] = dict(
            base=args.live_base, summary=live["summary"], metadata=live["metadata"],
            event_count=len(live["events"]),
            entity_risk_counts=dict(Counter(row["risk_level"] for name in ("products", "warehouses", "suppliers") for row in live[name])),
            matches_local_state=live["summary"] == summary,
            summary_matches_entity_endpoints=(
                live["summary"]["operational_events"] == len(live["events"])
                and live["summary"]["affected_warehouses"] == sum(row["connected_events"] > 0 for row in live["warehouses"])
                and all(live["summary"]["risky_" + name] == sum(row["risk_level"] in {"Medium", "High", "Critical"} for row in live[name])
                        for name in ("products", "suppliers"))),
            cache_headers={name: {header: response.headers.get(header) for header in ("Cache-Control", "ETag", "Last-Modified")}
                           for name, response in responses.items()},
        )
    out = ROOT / "data" / "analysis" / "current-risk-debug"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    fields = ("Event_ID", "title", "type", "source", "location", "start_time", "end_time", "temporal_status",
              "operational_eligible", "resolution_status", "excluded_reason", "affected_graph_entity_count")
    def cell(value):
        return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")
    lines = [f"# Current-risk diagnostic: {args.label}", "", f"Evaluated at {report['evaluated_at']}.", "",
             "All event windows below retain their source timezone. Counts are evaluated at runtime.", "",
             "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(cell(event[field]) for field in fields) + " |" for event in diagnostics)
    lines += ["", "Affected graph entity count means direct AFFECTS targets (warehouse/location); excluded events have no current exposure.",
              "", "## Artifact modification times (UTC)", "", "| File | Modified |", "| --- | --- |"]
    lines.extend(f"| {name} | {stamp} |" for name, stamp in report["files"].items())
    lines += ["", "## Counts", "", "```json", json.dumps({key: value for key, value in report.items()
                 if key not in {"events", "articles", "weather", "files", "live_api"}}, indent=2, default=str), "```", ""]
    (out / f"{args.label}.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"articles", "weather"}}, indent=2, default=str))


if __name__ == "__main__":
    main()
