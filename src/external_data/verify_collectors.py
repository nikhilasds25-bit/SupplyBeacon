"""Verify external news and weather collections.

This utility checks that raw JSON files exist under data/external/news and
data/external/weather, validates their structure, and logs a summary report.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import utils

load_env = utils.load_env

BASE_DIR = Path(__file__).resolve().parents[2]
NEWS_DIR = BASE_DIR / "data" / "external" / "news"
WEATHER_DIR = BASE_DIR / "data" / "external" / "weather"
ENV_PATH = BASE_DIR / ".env"

NEWS_REQUIRED_KEYS = {"collected_at", "source", "query", "article_count", "response"}
WEATHER_REQUIRED_KEYS = {"collected_at", "source", "cities"}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def list_json_files(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.json")) if directory.exists() else []


def validate_news_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    missing = NEWS_REQUIRED_KEYS - set(payload.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    if not isinstance(payload.get("response"), dict):
        return False, "response field must be a JSON object"
    if "articles" not in payload["response"]:
        return False, "response object must contain articles"
    return True, "ok"


def validate_weather_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    missing = WEATHER_REQUIRED_KEYS - set(payload.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    if not isinstance(payload.get("cities"), dict):
        return False, "cities field must be a JSON object"
    return True, "ok"


def summarize_news_files(news_files: List[Path]) -> Dict[str, Any]:
    summary = {
        "file_count": len(news_files),
        "valid_count": 0,
        "invalid_count": 0,
        "article_count": 0,
    }
    for path in news_files:
        try:
            payload = read_json_file(path)
            is_valid, message = validate_news_payload(payload)
            if not is_valid:
                logging.warning("Invalid news file %s: %s", path.name, message)
                summary["invalid_count"] += 1
                continue
            summary["valid_count"] += 1
            summary["article_count"] += int(payload.get("article_count", 0))
        except Exception as exc:
            logging.exception("Failed to validate news file %s: %s", path.name, exc)
            summary["invalid_count"] += 1
    return summary


def summarize_weather_files(weather_files: List[Path]) -> Dict[str, Any]:
    summary = {
        "file_count": len(weather_files),
        "valid_count": 0,
        "invalid_count": 0,
        "record_count": 0,
    }
    for path in weather_files:
        try:
            payload = read_json_file(path)
            is_valid, message = validate_weather_payload(payload)
            if not is_valid:
                logging.warning("Invalid weather file %s: %s", path.name, message)
                summary["invalid_count"] += 1
                continue
            summary["valid_count"] += 1
            summary["record_count"] += len(payload.get("cities", {}))
        except Exception as exc:
            logging.exception("Failed to validate weather file %s: %s", path.name, exc)
            summary["invalid_count"] += 1
    return summary


def verify_env_file() -> None:
    if not ENV_PATH.exists():
        logging.warning("Missing environment file: %s", ENV_PATH)
        return
    env_vars = load_env(ENV_PATH)
    news_key = env_vars.get("NEWS_API_KEY")
    weather_key = env_vars.get("OPENWEATHER_API_KEY")
    logging.info("Found .env file at %s", ENV_PATH)
    logging.info("NEWS_API_KEY present: %s", bool(news_key and news_key != "YOUR_NEWSAPI_KEY"))
    logging.info("OPENWEATHER_API_KEY present: %s", bool(weather_key and weather_key != "YOUR_OPENWEATHER_API_KEY"))


def main() -> None:
    configure_logging()
    logging.info("Verifying external collector configuration and data files")

    verify_env_file()

    news_files = list_json_files(NEWS_DIR)
    weather_files = list_json_files(WEATHER_DIR)

    logging.info("Found %s news JSON files in %s", len(news_files), NEWS_DIR)
    logging.info("Found %s weather JSON files in %s", len(weather_files), WEATHER_DIR)

    news_summary = summarize_news_files(news_files)
    weather_summary = summarize_weather_files(weather_files)

    logging.info("News files: %s total, %s valid, %s invalid, %s articles", news_summary["file_count"], news_summary["valid_count"], news_summary["invalid_count"], news_summary["article_count"])
    logging.info("Weather files: %s total, %s valid, %s invalid, %s records", weather_summary["file_count"], weather_summary["valid_count"], weather_summary["invalid_count"], weather_summary["record_count"])

    if news_summary["file_count"] == 0 and weather_summary["file_count"] == 0:
        logging.warning("No external JSON files found. Run the collectors first.")


if __name__ == "__main__":
    main()
