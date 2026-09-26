"""Deterministic temporal validity for extracted external events."""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

TEMPORAL_STATUSES = {"ACTIVE", "UPCOMING", "EXPIRED", "UNKNOWN"}
TEMPORAL_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
_MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"),
        start=1,
    )
}
_DATE_PATTERN = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+"
    r"(?P<day>\d{1,2})(?:\s*[-–]\s*(?P<end_day>\d{1,2}))?"
    r"(?:,?\s*(?P<year>20\d{2}))?\b",
    re.IGNORECASE,
)

def utc_now() -> datetime:
    """Return the application clock used for runtime temporal evaluation."""
    return datetime.now(timezone.utc)


def _as_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _day_window(day: date, tzinfo: Any) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=tzinfo)
    end = datetime.combine(day, time.max, tzinfo=tzinfo)
    return start, end


def _date_window(text: str, publication: datetime) -> tuple[datetime, datetime, str] | None:
    matches = list(_DATE_PATTERN.finditer(text or ""))
    signatures = {
        (
            match.group("month").lower(),
            int(match.group("day")),
            int(match.group("end_day") or match.group("day")),
            int(match.group("year") or publication.year),
        )
        for match in matches
    }
    if len(signatures) != 1:
        return None
    month_name, start_day_number, end_day_number, year = next(iter(signatures))
    month = _MONTHS[month_name]
    try:
        start_day = date(year, month, start_day_number)
        end_day = date(year, month, end_day_number)
        start, _ = _day_window(start_day, publication.tzinfo)
        _, end = _day_window(end_day, publication.tzinfo)
    except ValueError:
        return None
    basis = "Explicit forecast date in article metadata"
    return start, end, basis


def extract_news_validity(
    title: str,
    description: str,
    published_at: Any,
) -> dict[str, Any]:
    """Extract only explicit, deterministic article validity dates.

    Relative phrases are resolved against publication time, never collection
    time. Ambiguous multi-date articles are intentionally left unknown.
    """
    publication = _as_datetime(published_at)
    if publication is None:
        return {
            "event_start": None,
            "event_end": None,
            "temporal_confidence": "LOW",
            "temporal_basis": "No event-validity date available",
        }
    text = f"{title or ''} {description or ''}"
    window = _date_window(text, publication)
    if window:
        start, end, basis = window
    else:
        relative = re.search(r"\b(today|tomorrow|yesterday)\b", text, re.IGNORECASE)
        if not relative:
            return {
                "event_start": None,
                "event_end": None,
                "temporal_confidence": "LOW",
                "temporal_basis": "No event-validity date available",
            }
        offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[relative.group(1).lower()]
        start, end = _day_window((publication + timedelta(days=offset)).date(), publication.tzinfo)
        basis = "Relative date resolved against article publication timestamp"
    return {
        "event_start": start.isoformat(),
        "event_end": end.isoformat(),
        "temporal_confidence": "HIGH",
        "temporal_basis": basis,
    }


def temporal_status(
    event_start: Any,
    event_end: Any,
    now: datetime | None = None,
) -> str:
    """Evaluate a validity window without treating source freshness as validity."""
    start = _as_datetime(event_start)
    end = _as_datetime(event_end)
    if start is None and end is None:
        return "UNKNOWN"
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if end is not None and current > end:
        return "EXPIRED"
    if start is not None and current < start:
        return "UPCOMING"
    return "ACTIVE"


def apply_runtime_temporal_status(event: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Return an event with status reevaluated against the current clock."""
    result = dict(event)
    if not result.get("event_start") and not result.get("event_end"):
        source = str(result.get("Source", "")).lower()
        if "news" in source:
            result.update(extract_news_validity(
                str(result.get("Title", "")),
                str(result.get("Description", "")),
                result.get("Date"),
            ))
        elif "weather" in source:
            observed = _as_datetime(result.get("Date"))
            if observed is not None:
                result["event_start"] = observed.isoformat()
                # PROJECT OPERATIONAL VALIDITY WINDOW; not a meteorological forecast guarantee.
                result["event_end"] = (observed + timedelta(hours=6)).isoformat()
                result["temporal_confidence"] = "MEDIUM"
                result["temporal_basis"] = "OpenWeather observation timestamp with short observation window"
    result["event_status"] = temporal_status(
        result.get("event_start"),
        result.get("event_end"),
        now=now,
    )
    return result


def is_currently_eligible(event: dict[str, Any], now: datetime | None = None) -> bool:
    """Only active operational events contribute to current operational risk."""
    return bool(event.get("Operationally_Relevant") is True) and (
        apply_runtime_temporal_status(event, now=now)["event_status"] == "ACTIVE"
    )
