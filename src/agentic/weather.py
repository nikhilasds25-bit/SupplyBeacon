"""Read-only weather evidence from the current snapshot, without risk inference."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any

from src.dynamic_risk import current_state


def current_weather_tool(query: str = "") -> dict[str, Any]:
    path = current_state.CURRENT_DIR / "current_weather.json"
    if not path.exists():
        return {"records": [], "retrieval_ok": False, "limitations": ["Current weather artifact is unavailable."]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    fetched = payload.get("collected_at")
    freshness = current_state.freshness_status({"weather_fetch_timestamp": fetched})
    cities = payload.get("cities", {})
    locations = sorted({str(record.get("city") or key.split(":")[0]) for key, record in cities.items()})
    text = query.lower()
    aliases = {location.lower(): location for location in locations}
    for alias, canonical in (("new delhi", "Delhi"), ("bangalore", "Bengaluru")):
        if canonical in locations:
            aliases[alias] = canonical
    selected = {location for alias, location in aliases.items() if re.search(rf"\b{re.escape(alias)}\b", text)}
    # Explicit unsupported locations must not silently return every location.
    requested = re.search(r"\b(?:in|at|for)\s+(.+?)(?:[?.!]|$)", text)
    unknown_location = bool(requested and not selected and not any(
        term in requested.group(1) for term in ("location", "warehouse", "monitored", "all", "today", "now")
    ))
    rain_only = bool(re.search(r"\b(?:rain|raining|rainfall)\b", text))
    adverse_only = bool(re.search(r"\b(?:bad|severe|adverse|harsh|storm|disruptive)\b", text))
    records = []
    for key, raw in cities.items():
        location = str(raw.get("city") or key.split(":")[0])
        if unknown_location or (selected and location not in selected):
            continue
        provider = raw.get("source_channel") or (key.split(":", 1)[1] if ":" in key else "OPENWEATHER")
        validation_status = str(raw.get("location_validation_status") or "").upper()
        if validation_status.startswith("REJECTED"):
            continue
        if provider == "WEATHERAPI" and validation_status != "VALIDATED":
            continue
        is_observation = provider == "OPENWEATHER"
        conditions = raw.get("weather") or []
        condition = "; ".join(str(item.get("description") or item.get("main")) for item in conditions
                              if item.get("description") or item.get("main")) or None
        rain = raw.get("rain") or {}
        rain_evidence = {period: value for period, value in rain.items() if isinstance(value, (int, float))}
        rain_reported = bool(re.search(r"\b(?:rain|rainfall|drizzle|showers?)\b", condition or "", re.I)) or any(value > 0 for value in rain_evidence.values())
        has_adverse = rain_reported or bool(re.search(r"\b(?:storm|thunderstorm|flood|snow|squall|tornado|cyclone)\b", condition or "", re.I))
        # A forecast of rain or bad weather is not proof that a location currently has adverse weather.
        if rain_only and not (is_observation and rain_reported):
            continue
        if adverse_only and not (is_observation and has_adverse):
            continue
        observed = None
        if is_observation and isinstance(raw.get("dt"), (int, float)):
            observed = datetime.fromtimestamp(raw["dt"], timezone.utc).isoformat()
        records.append({
            "location": location, "condition": condition,
            "temperature_c": (raw.get("main") or {}).get("temp") if is_observation else None,
            "rain_mm": rain_evidence, "rain_reported": rain_reported,
            "source_channel": provider,
            "record_kind": "OBSERVATION" if is_observation else "DAILY_FORECAST_OR_ALERT",
            "observed_at": observed, "fetch_timestamp": fetched,
            "fresh": freshness["weather_fresh"], "age_hours": freshness["weather_age_hours"],
            "source_artifact": "data/current/current_weather.json", "provenance": "EXTERNAL_API",
        })
    return {
        "records": records, "retrieval_ok": True, "fetch_timestamp": fetched,
        "weather_fresh": freshness["weather_fresh"], "weather_age_hours": freshness["weather_age_hours"],
        "rain_only": rain_only, "adverse_only": adverse_only, "location_not_found": unknown_location,
        "limitations": ["No matching monitored location in the current weather snapshot."] if unknown_location else [],
        "notes": ["Provider records are separate; no consensus values are inferred.",
                  "WeatherAPI rows retain daily forecast/alert descriptions, not current observations. Missing measurements remain unavailable.",
                  "Rain-location queries use observation evidence only; forecasts do not establish current rain."],
    }
