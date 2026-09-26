"""Event Extraction Engine

Reads raw news and weather JSON files and extracts structured events suitable
for downstream business rule engines. Generates data/external/events/events.csv
and events.json with a normalized schema.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import sys
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from event_models import (
    AFFECTED_ENTITY_KEYWORDS,
    EVENT_KEYWORDS,
    LOCATION_KEYWORDS,
    LOCATIONS,
    SEVERITY_DEFAULT,
)
from utils import (
    EVENTS_DIR,
    RAW_NEWS_DIR,
    RAW_WEATHER_DIR,
    extract_articles_from_news_payload,
    normalize_article,
    normalize_weather_record,
    read_json_files,
)
from temporal_validity import extract_news_validity


# Output paths
EVENTS_CSV = EVENTS_DIR / "events.csv"
EVENTS_JSON = EVENTS_DIR / "events.json"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def find_keywords_in_text(text: str, keywords: Dict[str, str]) -> List[str]:
    """Return list of matched canonical event types found in text.

    keywords keys are lowercase phrases mapped to event types.
    Matches are case-insensitive and word-boundary aware.
    """
    found: List[str] = []
    if not text:
        return found
    lowered = text.lower()
    for kw, ev in keywords.items():
        # use simple substring match, but ensure word boundaries where applicable
        pattern = re.escape(kw)
        if re.search(rf"\b{pattern}\b", lowered):
            found.append(ev)
    return found


def detect_locations(text: str) -> List[str]:
    """Detect known locations and regions in the provided text."""
    found: List[str] = []
    if not text:
        return found
    for canonical, aliases in LOCATION_KEYWORDS.items():
        for alias in aliases:
            pattern = re.escape(alias)
            if re.search(rf"\b{pattern}\b", text, flags=re.I):
                found.append(canonical)
                break
    return list(dict.fromkeys(found))


def detect_entities(text: str) -> List[str]:
    """Detect affected entities using known brand and company aliases."""
    found: List[str] = []
    if not text:
        return found
    for canonical, aliases in AFFECTED_ENTITY_KEYWORDS.items():
        for alias in aliases:
            pattern = re.escape(alias)
            if re.search(rf"\b{pattern}\b", text, flags=re.I):
                found.append(canonical)
                break
    return list(dict.fromkeys(found))


def event_phrases_for_type(event_type: str) -> List[str]:
    """Return the event keyword phrases that map to a canonical event type."""
    return [kw for kw, ev in EVENT_KEYWORDS.items() if ev == event_type]


def compute_confidence(
    event_type: str,
    title: str,
    description: str,
    location_found: bool,
    entity_found: bool,
    keyword_matches: int,
) -> float:
    """Compute a rule-based confidence score in [0.0, 1.0].

    Confidence is based on event keyword strength, presence in title,
    support from description, and whether a location or affected entity was
    detected. This is a deterministic rule-based score, not a learned
    probability.
    """
    score = 0.2
    if keyword_matches > 1:
        score += min(0.2, 0.05 * (keyword_matches - 1))

    phrases = event_phrases_for_type(event_type)
    if title:
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", title, flags=re.I):
                score += 0.25
                break
    if description:
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", description, flags=re.I):
                score += 0.15
                break
    if location_found:
        score += 0.15
    if entity_found:
        score += 0.15

    final = min(score, 1.0)
    return round(final, 2)


def choose_severity(event_type: Optional[str]) -> str:
    if not event_type:
        return "Low"
    return SEVERITY_DEFAULT.get(event_type, "Medium")


OPERATIONAL_EVENT_TERMS = (
    "factory shutdown",
    "port congestion",
    "transport strike",
    "road closure",
    "flood",
    "cyclone",
    "severe rainfall",
    "heavy rain",
    "production halt",
    "supply shortage",
    "component shortage",
    "semiconductor shortage",
    "chip shortage",
    "logistics disruption",
    "warehouse closure",
    "power outage",
    "production disruption",
)
RISK_WARNING_TERMS = (
    "warning",
    "alert",
    "forecast",
    "expected",
    "threat",
    "risk",
    "may disrupt",
    "could disrupt",
)
COMMENTARY_TERMS = (
    "experts",
    "industry leaders",
    "should improve",
    "must move",
    "strategy",
    "market expected to grow",
    "call for",
    "panel discussion",
)
NEGATED_EVENT_PHRASES = (
    "no rain", "no heavy rain", "largely dry", "rain unlikely",
    "no disruption", "no shortage", "no strike", "no flooding",
)
DOMAIN_RELEVANCE_TERMS = (
    "electronics", "semiconductor", "component", "logistics", "transport",
    "port", "warehouse", "manufactur", "supply chain", "shipping",
)


def _is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    if phrase in lowered and any(negation in lowered for negation in NEGATED_EVENT_PHRASES):
        return True
    match = re.search(rf"\b{re.escape(phrase)}\b", lowered)
    if not match:
        return False
    prefix = lowered[max(0, match.start() - 45):match.start()]
    return bool(re.search(r"\b(?:no|without|unlikely|unlikely to|largely dry)\b", prefix))


def classify_news_event(title: str, description: str) -> tuple[str, bool]:
    """Classify news deterministically without treating commentary as disruption."""
    text = f"{title} {description}".lower()
    matched_terms = [
        term for term in OPERATIONAL_EVENT_TERMS
        if term in text and not _is_negated(text, term)
    ]
    if not matched_terms:
        if any(term in text for term in COMMENTARY_TERMS):
            return "INDUSTRY_COMMENTARY", False
        return "IRRELEVANT", False
    if not any(term in text for term in DOMAIN_RELEVANCE_TERMS):
        return "ACTUAL_DISRUPTION", False
    if any(term in text for term in RISK_WARNING_TERMS):
        return "RISK_WARNING", True
    return "ACTUAL_DISRUPTION", True


def _weather_event_type(record: Dict[str, Any], description: str) -> Optional[str]:
    """Map OpenWeather evidence to a conservative event type."""
    weather = (record.get("weather") or [{}])[0] if isinstance(record, dict) else {}
    main = str(weather.get("main", "")).strip().lower()
    alerts = record.get("alerts") or record.get("alert") or []
    alert_text = " ".join(
        str(item.get("event") or item.get("headline") or "")
        for item in alerts
        if isinstance(item, dict)
    ).lower()
    if alert_text:
        if "flood" in alert_text:
            return "Flood"
        if "cyclone" in alert_text or "hurricane" in alert_text:
            return "Cyclone"
        if "thunderstorm" in alert_text:
            return "Thunderstorm"
        if any(term in alert_text for term in ("heavy rain", "very heavy rain", "extreme rain", "heavy rainfall", "very heavy rainfall", "extreme rainfall")):
            return "Heavy Rain"
        if any(term in alert_text for term in ("strong wind", "high wind", "damaging wind", "wind warning")):
            return "Storm"
    code = int(weather.get("id", 0) or 0)
    rain_1h = float((record.get("rain") or {}).get("1h", 0) or 0) if isinstance(record, dict) else 0.0
    if 200 <= code < 300 or main == "thunderstorm":
        return "Thunderstorm"
    if main in {"squall", "tornado"}:
        return "Storm"
    if main == "rain" or 300 <= code < 600:
        if code in {500} or "light rain" in description.lower():
            return "Light Rain"
        if code in {501, 520, 521} or "moderate rain" in description.lower():
            return "Moderate Rain"
        if rain_1h and rain_1h < 2:
            return "Light Rain"
        if rain_1h and rain_1h < 7.6:
            return "Moderate Rain"
        return "Heavy Rain"
    if main == "extreme heat" or "extreme heat" in description.lower():
        return "Extreme Heat"
    return None


def _weather_is_operationally_relevant(record: Dict[str, Any], event_type: str) -> bool:
    """Use only explicit API warning/severity evidence for operational impact."""
    if event_type in {"Heavy Rain", "Thunderstorm", "Storm", "Flood", "Cyclone", "Extreme Heat"}:
        return True
    if event_type != "Moderate Rain" or not isinstance(record, dict):
        return False
    raw_alert = record.get("alerts") or record.get("alert")
    return bool(raw_alert)


def _weather_temporal_fields(record: Dict[str, Any], date: str) -> dict[str, Any]:
    raw_alerts = (record.get("alerts") or record.get("alert") or []) if isinstance(record, dict) else []
    alert = raw_alerts[0] if raw_alerts and isinstance(raw_alerts[0], dict) else {}
    start = normalize_date(alert.get("effective") or alert.get("onset"))
    end = normalize_date(alert.get("expires") or alert.get("ends"))
    if start or end:
        return {
            "event_start": start,
            "event_end": end,
            "temporal_confidence": "HIGH",
            "temporal_basis": "WeatherAPI alert effective/expires timestamps",
        }
    return {
        "event_start": normalize_date(date),
        "event_end": normalize_date(
            (datetime.fromisoformat(date.replace("Z", "+00:00")) + pd.Timedelta(hours=6)).isoformat()
        ),
        "temporal_confidence": "MEDIUM",
        "temporal_basis": "PROJECT OPERATIONAL VALIDITY WINDOW",
    }


def normalize_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    # Try common ISO formats and fall back to pandas
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.isoformat()
    except Exception:
        try:
            pdts = pd.to_datetime(value, errors="coerce")
            if pd.isna(pdts):
                return None
            return pdts.isoformat()
        except Exception:
            return None


def article_to_events(article: Dict[str, Any], source_file: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    norm = normalize_article(article)
    title = norm.get("title", "")
    description = norm.get("description", "")
    date_raw = norm.get("publishedAt")
    # Missing publication time cannot be replaced by collection time: relative
    # phrases such as "today" would otherwise acquire an invented validity date.
    published = normalize_date(date_raw)

    text_blob = " ".join([title or "", description or ""])

    matched = sorted(
        set(
            event_type
            for phrase, event_type in EVENT_KEYWORDS.items()
            if re.search(rf"\b{re.escape(phrase)}\b", text_blob, flags=re.I)
            and not _is_negated(text_blob, phrase)
        )
    )
    locations = detect_locations(text_blob)
    entities = detect_entities(text_blob)

    if not matched:
        return []

    # Create one event per matched event type
    for ev in matched:
        keyword_matches = sum(
            1
            for phrase in event_phrases_for_type(ev)
            if re.search(rf"\b{re.escape(phrase)}\b", text_blob, flags=re.I)
        )
        severity = choose_severity(ev)
        confidence = compute_confidence(
            event_type=ev,
            title=title,
            description=description,
            location_found=bool(locations),
            entity_found=bool(entities),
            keyword_matches=keyword_matches,
        )
        event = {
            "Event_ID": None,
            "Date": published,
            "Event_Type": ev,
            "Title": title or (ev + " reported"),
            "Description": description or "",
            "Location": ", ".join(locations) if locations else "Unknown",
            "Affected_Entity": ", ".join(entities) if entities else "Unknown",
            "Category": "News",
            "Severity": severity,
            "Source": source_file.name,
            "Confidence": confidence,
            "Event_Class": "",
            "Operationally_Relevant": False,
            "source_channel": norm.get("source_channel", "NEWS_API"),
            "source_references": norm.get("source_references", []),
        }
        event_class, operational = classify_news_event(title, description)
        event["Event_Class"] = event_class
        event["Operationally_Relevant"] = operational
        event.update(extract_news_validity(title, description, published))
        if not operational:
            event["Event_Type"] = "Industry Commentary" if event_class == "INDUSTRY_COMMENTARY" else "Irrelevant"
            event["Severity"] = "Low"
            event["Confidence"] = min(confidence, 0.2)
        events.append(event)
    return events


def weather_to_events(city: str, record: Dict[str, Any], source_file: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    norm = normalize_weather_record(city, record)
    desc = norm.get("description") or ""
    ts = norm.get("timestamp")
    date = normalize_date(ts) or datetime.utcnow().isoformat()

    event_type = _weather_event_type(norm.get("raw", record), desc)
    matched = [event_type] if event_type else []
    location = city
    if city not in LOCATIONS:
        detected = detect_locations(city)
        location = detected[0] if detected else city

    if matched:
        for ev in matched:
            keyword_matches = sum(
                1
                for phrase in event_phrases_for_type(ev)
                if re.search(rf"\b{re.escape(phrase)}\b", desc, flags=re.I)
            )
            severity = choose_severity(ev)
            confidence = compute_confidence(
                event_type=ev,
                title="",
                description=desc,
                location_found=bool(location and location != "Unknown"),
                entity_found=False,
                keyword_matches=keyword_matches,
            )
            operationally_relevant = _weather_is_operationally_relevant(norm.get("raw", record), ev)
            event = {
                "Event_ID": None,
                "Date": date,
                "Event_Type": ev,
                "Title": f"{ev} in {location}",
                "Description": desc,
                "Location": location,
                "Affected_Entity": "Unknown",
                "Category": "Weather",
                "Severity": severity,
                "Source": source_file.name,
                "Confidence": confidence,
                "Event_Class": "ACTUAL_DISRUPTION" if operationally_relevant else "RISK_WARNING",
                "Operationally_Relevant": operationally_relevant,
                "source_channel": (norm.get("raw") or {}).get("source_channel", "OPENWEATHER"),
            }
            raw_alerts = (norm.get("raw") or {}).get("alerts") or []
            if raw_alerts and isinstance(raw_alerts[0], dict):
                alert = raw_alerts[0]
                event.update({
                    "alert_event": alert.get("event"),
                    "alert_headline": alert.get("headline"),
                    "alert_effective": alert.get("effective") or alert.get("onset"),
                    "alert_expires": alert.get("expires") or alert.get("ends"),
                    "alert_created_event": True,
                })
            event.update(_weather_temporal_fields(norm.get("raw", record), date))
            events.append(event)
    else:
        if desc and re.search(r"heavy .*rain|heavy rain", desc, flags=re.I):
            ev = "Heavy Rain"
            severity = choose_severity(ev)
            confidence = compute_confidence(
                event_type=ev,
                title="",
                description=desc,
                location_found=bool(location and location != "Unknown"),
                entity_found=False,
                keyword_matches=1,
            )
            event = {
                    "Event_ID": None,
                    "Date": date,
                    "Event_Type": ev,
                    "Title": f"Heavy Rain in {location}",
                    "Description": desc,
                    "Location": location,
                    "Affected_Entity": "Unknown",
                    "Category": "Weather",
                    "Severity": severity,
                    "Source": source_file.name,
                    "Confidence": confidence,
                    "source_channel": (norm.get("raw") or {}).get("source_channel", "OPENWEATHER"),
                }
            event.update(_weather_temporal_fields(norm.get("raw", record), date))
            events.append(event)

    return events


def deduplicate_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for ev in events:
        key = (
            ev.get("Event_Type", ""),
            ev.get("Date", ""),
            ev.get("Location", ""),
            ev.get("Title", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(ev)
    return unique


def assign_event_ids(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i, ev in enumerate(events, start=1):
        ev["Event_ID"] = i
    return events


def generate_report(events: List[Dict[str, Any]], news_count: int, weather_count: int) -> str:
    total_events = len(events)
    by_type = Counter(ev.get("Event_Type") for ev in events)
    by_location = Counter(ev.get("Location") for ev in events)
    by_severity = Counter(ev.get("Severity") for ev in events)

    lines = []
    lines.append("====================================")
    lines.append("Event Extraction Report")
    lines.append("====================================")
    lines.append(f"Total News Articles: {news_count}")
    lines.append(f"Total Weather Records: {weather_count}")
    lines.append(f"Total Events Extracted: {total_events}")
    lines.append("")
    lines.append("Events by Type:")
    for k, v in by_type.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Events by Location:")
    for k, v in by_location.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")
    unknown_location_count = sum(1 for ev in events if ev.get("Location") == "Unknown")
    unknown_entity_count = sum(
        1 for ev in events if ev.get("Affected_Entity") in ("Unknown", "")
    )
    average_confidence = (
        round(sum(ev.get("Confidence", 0.0) for ev in events) / total_events, 2)
        if total_events
        else 0.0
    )

    lines.append("Events by Severity:")
    for k, v in by_severity.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"Events with Unknown Location: {unknown_location_count}")
    lines.append(f"Events with Unknown Affected Entity: {unknown_entity_count}")
    lines.append(f"Average Confidence Score: {average_confidence}")
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    logging.info("Starting event extraction")

    # Read raw files
    news_files = read_json_files(RAW_NEWS_DIR)
    weather_files = read_json_files(RAW_WEATHER_DIR)

    total_news_articles = 0
    total_weather_records = 0
    extracted_events: List[Dict[str, Any]] = []

    # Process news
    for file_path, payload in news_files:
        articles = extract_articles_from_news_payload(payload)
        total_news_articles += len(articles)
        for art in articles:
            events = article_to_events(art, file_path)
            extracted_events.extend(events)

    # Process weather
    for file_path, payload in weather_files:
        # payload expected to have 'cities' map from earlier collector
        cities = payload.get("cities") if isinstance(payload, dict) else None
        if not cities:
            # fallback: if payload itself looks like a single record
            logging.debug("Weather payload missing 'cities' key in %s", file_path)
            continue
        for city, rec in cities.items():
            total_weather_records += 1
            events = weather_to_events(city, rec, file_path)
            extracted_events.extend(events)

    logging.info("Raw extracted events before dedup: %d", len(extracted_events))

    # Clean and deduplicate
    cleaned = [ev for ev in extracted_events if ev.get("Description")]
    deduped = deduplicate_events(cleaned)
    assigned = assign_event_ids(deduped)

    # Save outputs
    if assigned:
        df = pd.DataFrame(assigned)
        df.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
        df.to_json(EVENTS_JSON, orient="records", force_ascii=False, indent=2)
        logging.info("Saved events to %s and %s", EVENTS_CSV, EVENTS_JSON)
    else:
        logging.info("No events extracted; nothing to save.")

    report = generate_report(assigned, total_news_articles, total_weather_records)
    logging.info("\n%s", report)


if __name__ == "__main__":
    main()
