"""Utilities for event extraction: reading raw JSON files and extracting text."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

RAW_NEWS_DIR = Path("data") / "external" / "news"
RAW_WEATHER_DIR = Path("data") / "external" / "weather"
EVENTS_DIR = Path("data") / "external" / "events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def read_json_files(directory: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Read all JSON files in a directory and return list of (path, data)."""
    files: List[Tuple[Path, Dict[str, Any]]] = []
    if not directory.exists():
        logging.warning("Directory does not exist: %s", directory)
        return files
    for file in sorted(directory.glob("*_*.json")):
        try:
            with file.open("r", encoding="utf-8") as reader:
                data = json.load(reader)
            files.append((file, data))
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to read JSON file %s: %s", file, exc)
    return files


def extract_articles_from_news_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """From a news collector payload extract list of article dicts.

    The collector may store the API response under 'response' or at top-level.
    This function normalizes it into a list of articles with keys we expect.
    """
    articles: List[Dict[str, Any]] = []

    # Payload may wrap the API JSON under 'response'
    if "response" in payload and isinstance(payload["response"], dict):
        api_json = payload["response"]
    else:
        api_json = payload

    # Articles may be in 'articles'
    if isinstance(api_json, dict) and "articles" in api_json:
        for art in api_json.get("articles", []):
            articles.append(art)
    # Otherwise, if payload has 'articles' at top
    elif "articles" in payload and isinstance(payload["articles"], list):
        for art in payload.get("articles", []):
            articles.append(art)

    return articles


def normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a NewsAPI article dict into a safe structure."""
    title = article.get("title") or ""
    description = article.get("description") or article.get("content") or ""
    published_at = article.get("publishedAt") or article.get("published_at")
    source = article.get("source", {}).get("name") if isinstance(article.get("source"), dict) else article.get("source")
    return {
        "title": title,
        "description": description,
        "publishedAt": published_at,
        "source_channel": article.get("source_channel", "NEWS_API"),
        "source_references": article.get("source_references", []),
        "source": source,
        "raw": article,
    }


def normalize_weather_record(city: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a weather API record per city into a standard dict."""
    # The record may include an 'error' key if collection failed
    if isinstance(record, dict) and "error" in record:
        return {
            "city": city,
            "description": f"error: {record.get('error')}",
            "timestamp": None,
            "raw": record,
        }

    # OpenWeather uses 'dt' as unix timestamp and 'weather' list for description
    description = ""
    if isinstance(record, dict):
        weather_list = record.get("weather") or []
        if weather_list and isinstance(weather_list, list):
            desc = weather_list[0].get("description")
            if desc:
                description = desc
        # fallback to message
        description = description or record.get("message") or ""
        ts = None
        if record.get("dt"):
            try:
                ts = datetime.utcfromtimestamp(int(record.get("dt"))).isoformat() + "Z"
            except Exception:
                ts = None
        return {"city": city, "description": description, "timestamp": ts, "raw": record}

    return {"city": city, "description": "", "timestamp": None, "raw": record}
