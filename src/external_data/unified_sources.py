"""Unified external-source adapters for the current refresh workflow."""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any

import requests

from .news_collector import collect_news
from .weather_collector import collect_weather
from .utils import load_env, sanitize_external_error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _configured_key(name: str) -> str | None:
    return os.getenv(name) or (
        load_env(BASE_DIR / ".env").get(name)
        if (BASE_DIR / ".env").exists()
        else None
    )

NEWS_DATA_URL = "https://newsdata.io/api/1/news"
CURRENTS_URL = "https://api.currentsapi.services/v1/search"
WEATHER_API_URL = "https://api.weatherapi.com/v1/forecast.json"
SUPPLY_CHAIN_QUERY_TERMS = (
    "supply chain", "logistics disruption", "port disruption", "port congestion",
    "factory shutdown", "manufacturing disruption", "semiconductor shortage",
    "component shortage", "transport strike", "road closure", "rail disruption",
    "heavy rain logistics", "flood logistics",
)
SUPPLY_CHAIN_QUERY = " OR ".join(f'"{term}"' for term in SUPPLY_CHAIN_QUERY_TERMS)
NEWS_DATA_QUERY = " OR ".join(f'"{term}"' for term in SUPPLY_CHAIN_QUERY_TERMS[:4])


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _article(
    raw: dict[str, Any],
    *,
    source_channel: str,
    title_keys: tuple[str, ...] = ("title",),
    description_keys: tuple[str, ...] = ("description", "content"),
    date_keys: tuple[str, ...] = ("publishedAt", "published_at", "pubDate", "published"),
) -> dict[str, Any]:
    def first(keys: tuple[str, ...]) -> Any:
        return next((raw.get(key) for key in keys if raw.get(key)), None)

    source = raw.get("source")
    if isinstance(source, dict):
        source = source.get("name") or source.get("url")
    return {
        "title": _text(first(title_keys)),
        "description": _text(first(description_keys)),
        "content": _text(raw.get("content") or raw.get("description")),
        "url": _text(raw.get("url") or raw.get("link")),
        "published_at": first(date_keys),
        "source": _text(source),
        "source_channel": source_channel,
        "raw": raw,
    }


def normalize_news_articles(payload: dict[str, Any], source_channel: str) -> list[dict[str, Any]]:
    raw_articles = payload.get("articles") or payload.get("results") or payload.get("news") or []
    return [_article(item, source_channel=source_channel) for item in raw_articles if isinstance(item, dict)]


def _dedup_key(article: dict[str, Any]) -> str:
    url = _text(article.get("url")).lower().rstrip("/")
    if url:
        return f"url:{url}"
    title = re.sub(r"[^a-z0-9]+", " ", _text(article.get("title")).lower()).strip()
    published = _text(article.get("published_at"))[:10]
    return f"title:{title}|date:{published}"


def deduplicate_news_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate cross-source coverage while retaining source provenance."""
    grouped: dict[str, dict[str, Any]] = {}
    for article in articles:
        key = _dedup_key(article)
        if key not in grouped:
            item = dict(article)
            item["source_channels"] = [article.get("source_channel")]
            item["source_references"] = [{
                "source_channel": article.get("source_channel"),
                "source": article.get("source"),
                "url": article.get("url"),
            }]
            grouped[key] = item
            continue
        existing = grouped[key]
        channel = article.get("source_channel")
        if channel not in existing["source_channels"]:
            existing["source_channels"].append(channel)
        existing["source_references"].append({
            "source_channel": channel,
            "source": article.get("source"),
            "url": article.get("url"),
        })
    return list(grouped.values())


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("external source returned a non-object payload")
    return payload


def collect_unified_news(news_api_key: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    articles: list[dict[str, Any]] = []
    configured = [
        ("NEWS_API", news_api_key, lambda: collect_news(news_api_key or {}).get("response", {})),
        ("NEWSDATA_IO", _configured_key("NEWSDATA_API_KEY"), lambda: _request_json(
            NEWS_DATA_URL, {"apikey": _configured_key("NEWSDATA_API_KEY"), "q": NEWS_DATA_QUERY, "country": "in", "language": "en", "size": 10}
        )),
        ("CURRENTS_API", _configured_key("CURRENTS_API_KEY"), lambda: _request_json(
            CURRENTS_URL, {"apiKey": _configured_key("CURRENTS_API_KEY"), "keywords": SUPPLY_CHAIN_QUERY, "country": "IN", "language": "en", "page_size": 10}
        )),
    ]
    for channel, key, fetch in configured:
        if not key:
            sources[channel] = {"success": False, "fetched_count": 0, "accepted_count": 0, "error": "API key not configured"}
            continue
        try:
            payload = fetch()
            normalized = normalize_news_articles(payload, channel)
            articles.extend(normalized)
            sources[channel] = {"success": True, "fetched_count": len(normalized), "accepted_count": len(normalized), "error": None}
        except Exception as exc:
            sources[channel] = {"success": False, "fetched_count": 0, "accepted_count": 0, "error": sanitize_external_error(exc)}
    deduplicated = deduplicate_news_articles(articles)
    for source in sources.values():
        if source["success"]:
            source["accepted_count"] = sum(
                1 for article in deduplicated
                if any(
                    reference.get("source_channel") == next(
                        channel for channel, details in sources.items() if details is source
                    )
                    for reference in article.get("source_references", [])
                )
            )
    return deduplicated, sources


def _weatherapi_record(city: str, payload: dict[str, Any]) -> dict[str, Any]:
    forecast = (payload.get("forecast", {}).get("forecastday") or [{}])[0]
    day = forecast.get("day", {})
    alerts = payload.get("alerts", {}).get("alert", []) or []
    alert_text = _text((alerts[0] if alerts else {}).get("event") or (alerts[0] if alerts else {}).get("headline"))
    condition = alert_text or day.get("condition", {}).get("text", "")
    return {
        "weather": [{"main": condition, "description": condition}],
        "dt": int(datetime.now(timezone.utc).timestamp()),
        "alerts": alerts,
        "source_channel": "WEATHERAPI",
        "city": city,
        "record_kind": "ALERT" if alerts else "DAILY_FORECAST",
        "forecast_date": forecast.get("date"),
        "daily_precipitation_mm": day.get("totalprecip_mm"),
        "max_wind_kph": day.get("maxwind_kph"),
    }


def collect_unified_weather(openweather_key: str | None, cities: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    result: dict[str, Any] = {"collected_at": datetime.now(timezone.utc).isoformat(), "cities": {}}
    sources: dict[str, dict[str, Any]] = {}
    if openweather_key:
        try:
            payload = collect_weather(openweather_key, cities)
            result["cities"].update(payload.get("cities", {}))
            accepted = len(payload.get("cities", {}))
            sources["OPENWEATHER"] = {
                "success": accepted > 0,
                "fetched_count": len(payload.get("cities", {})),
                "accepted_count": accepted,
                "error": None if accepted > 0 else "No usable weather observations returned",
            }
        except Exception as exc:
            sources["OPENWEATHER"] = {"success": False, "fetched_count": 0, "accepted_count": 0, "error": sanitize_external_error(exc)}
    else:
        sources["OPENWEATHER"] = {"success": False, "fetched_count": 0, "accepted_count": 0, "error": "API key not configured"}
    key = _configured_key("WEATHERAPI_KEY")
    if key:
        accepted = 0
        try:
            for city in cities:
                result["cities"][f"{city}:WEATHERAPI"] = _weatherapi_record(
                    city, _request_json(WEATHER_API_URL, {"key": key, "q": f"{city}, India", "alerts": "yes"})
                )
                accepted += 1
            sources["WEATHERAPI"] = {"success": True, "fetched_count": accepted, "accepted_count": accepted, "error": None}
        except Exception as exc:
            sources["WEATHERAPI"] = {"success": False, "fetched_count": accepted, "accepted_count": accepted, "error": sanitize_external_error(exc)}
    else:
        sources["WEATHERAPI"] = {"success": False, "fetched_count": 0, "accepted_count": 0, "error": "API key not configured"}
    return result, sources
