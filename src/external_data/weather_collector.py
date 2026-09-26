"""Collect daily weather information for warehouse cities."""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import requests
import os
import string

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from . import utils

create_directory = utils.create_directory
load_env = utils.load_env
normalize_city_name = utils.normalize_city_name
requests_with_retry = utils.requests_with_retry
sanitize_external_error = utils.sanitize_external_error
save_json = utils.save_json

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "external"
WEATHER_DIR = DATA_DIR / "weather"
ENV_PATH = BASE_DIR / ".env"

WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_ENV_KEY = "OPENWEATHER_API_KEY"

REGIONAL_LOCATION_NAMES = {
    "india", "pan-india", "north india", "south india", "east india", "west india",
    "central india", "northeast india", "north-east india",
}


def warehouse_weather_locations(graph: Any) -> list[str]:
    """Return unique, provider-queryable locations linked to runtime warehouses.

    Base data links a warehouse to both its city and a regional roll-up. A city
    observation must not be fabricated for a broad region, so exact city/name
    relationships are preferred and regional-only warehouses are left
    unmonitored until a precise location is published.
    """
    locations: list[str] = []
    for warehouse_id, warehouse in graph.nodes(data=True):
        if warehouse.get("node_type") != "Warehouse":
            continue
        candidates: list[tuple[str, dict[str, Any]]] = []
        neighbors = list(graph.successors(warehouse_id)) + list(graph.predecessors(warehouse_id))
        for location_id in dict.fromkeys(neighbors):
            edge = graph.get_edge_data(warehouse_id, location_id) or graph.get_edge_data(location_id, warehouse_id) or {}
            location = graph.nodes[location_id]
            if edge.get("relationship") == "LOCATED_IN" and location.get("node_type") == "Location":
                candidates.append((str(location.get("name") or location.get("city") or "").strip(), location))

        warehouse_name = str(warehouse.get("name") or "").strip()
        expected = warehouse_name.removesuffix(" Warehouse").strip().casefold()
        precise = [
            str(attrs.get("city") or name).strip()
            for name, attrs in candidates
            if str(attrs.get("city") or name).strip().casefold() not in REGIONAL_LOCATION_NAMES
        ]
        exact = [name for name in precise if name.casefold() == expected]
        selected = exact or (precise if len(precise) == 1 else [])
        for name in selected:
            if name and name.casefold() not in {item.casefold() for item in locations}:
                locations.append(name)
    return sorted(locations, key=str.casefold)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_api_key() -> str:
    env_vars = load_env(ENV_PATH)
    api_key = env_vars.get(WEATHER_ENV_KEY)
    if not api_key:
        raise ValueError(
            f"Missing {WEATHER_ENV_KEY} in {ENV_PATH}. Please add your OpenWeather API key."
        )
    return api_key


def _safe_key_diagnostics(api_key: str) -> Dict[str, object]:
    """Return masked diagnostics about the API key without revealing it.

    - key_length: length of key
    - first4: first 4 characters (masked)
    - last4: last 4 characters (masked)
    - has_whitespace: whether key contains whitespace
    - has_quotes: whether key contains quote characters
    - has_nonprintable: whether key contains non-printable characters
    """
    length = len(api_key)
    first4 = api_key[:4] if length >= 4 else api_key
    last4 = api_key[-4:] if length >= 4 else api_key
    has_whitespace = any(ch.isspace() for ch in api_key)
    has_quotes = '"' in api_key or "'" in api_key
    printable = set(string.printable)
    has_nonprintable = any(ch not in printable for ch in api_key)

    # Mask first/last4 for logging (show literals but avoid full key)
    masked = {
        "key_length": length,
        "first4": (first4 if length >= 4 else "<short>"),
        "last4": (last4 if length >= 4 else "<short>"),
        "has_whitespace": has_whitespace,
        "has_quotes": has_quotes,
        "has_nonprintable": has_nonprintable,
    }
    return masked


def fetch_weather_for_city(session: requests.Session, api_key: str, city: str) -> Dict[str, Any]:
    """Fetch current weather for a city using OpenWeatherMap."""
    city_name = normalize_city_name(city)
    params = {
        "q": f"{city_name},IN",
        "appid": api_key,
        "units": "metric",
    }
    response = requests_with_retry(
        session,
        method="GET",
        url=WEATHER_API_URL,
        params=params,
    )
    payload = response.json()
    if response.status_code != 200:
        raise RuntimeError(
            f"Weather API failed for {city}: {payload.get('message', 'unknown error')}"
        )
    return payload


def build_output_path() -> Path:
    create_directory(WEATHER_DIR)
    today = date.today().isoformat()
    return WEATHER_DIR / f"{today}_weather.json"


import sys

def collect_weather(api_key: str, cities: Iterable[str]) -> Dict[str, Any]:
    monitored_cities = list(dict.fromkeys(str(city).strip() for city in cities if str(city).strip()))
    logging.info("Starting weather data collection for %s cities", len(monitored_cities))
    session = requests.Session()
    successes: Dict[str, Any] = {}
    failures: Dict[str, str] = {}

    for city in monitored_cities:
        try:
            payload = fetch_weather_for_city(session, api_key, city)
            # Basic validation: OpenWeather returns a 'weather' key for valid responses
            if isinstance(payload, dict) and payload.get("weather"):
                successes[city] = payload
                logging.info("Collected weather for %s", city)
            else:
                failures[city] = "invalid response structure"
                logging.warning("Invalid weather payload structure for %s", city)
        except Exception as exc:
            # Log a concise failure message without exposing the API key or sensitive params
            safe_error = sanitize_external_error(exc)
            logging.warning("Failed to collect weather for %s: %s", city, safe_error)
            failures[city] = safe_error

    results: Dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source": "openweathermap.org",
        "cities": successes,  # only successful records
        "successful_count": len(successes),
        "failed_count": len(failures),
        "failed_cities": list(failures.keys()),
        "failure_reasons": failures,
    }

    # Logging summary
    logging.info("Weather collection summary: Successful cities: %s Failed cities: %s", len(successes), len(failures))
    if failures:
        logging.info("Failed city names: %s", ", ".join(failures.keys()))

    return results


def main() -> None:
    configure_logging()
    try:
        api_key = load_api_key()
        logging.info("Loaded OpenWeather API key from %s", ENV_PATH)

        # Safe diagnostics: do not log the key itself, only masked info
        try:
            diag = _safe_key_diagnostics(api_key)
            logging.info(
                "OPENWEATHER_API_KEY diagnostics: path=%s exists=True length=%s first4=%s last4=%s has_whitespace=%s has_quotes=%s has_nonprintable=%s",
                ENV_PATH,
                diag["key_length"],
                diag["first4"],
                diag["last4"],
                diag["has_whitespace"],
                diag["has_quotes"],
                diag["has_nonprintable"],
            )
        except Exception:
            logging.warning("Failed to compute safe diagnostics for OPENWEATHER_API_KEY")

        # Also report whether an environment variable with the same name exists
        env_present = os.environ.get(WEATHER_ENV_KEY) is not None
        if env_present:
            env_len = len(os.environ.get(WEATHER_ENV_KEY, ""))
            logging.info("Environment variable %s is set (length=%s). Using .env value over env var.", WEATHER_ENV_KEY, env_len)

        from src.dynamic_risk.current_state import load_current_state

        graph = load_current_state().get("graph")
        cities = warehouse_weather_locations(graph) if graph is not None else []
        if not cities:
            raise RuntimeError("No precise warehouse locations are available in the current runtime graph")
        payload = collect_weather(api_key, cities)

        success_count = int(payload.get("successful_count", 0))
        failed_count = int(payload.get("failed_count", 0))

        # If all cities failed, treat as a failure and exit with non-zero status
        if success_count == 0:
            logging.error(
                "Weather data collection failed for all %s cities. Failed cities: %s",
                len(cities),
                ", ".join(payload.get("failed_cities", [])),
            )
            # Do not save a misleading successful dataset
            sys.exit(1)

        # Save only successful records, but include failure summary for transparency
        output_file = build_output_path()
        save_json(payload, output_file)
        logging.info("Saved weather collection to %s", output_file)
        logging.info("Weather collection summary: Successful cities: %s, Failed cities: %s", success_count, failed_count)

    except Exception as exc:
        logging.exception("Weather collection failed: %s", exc)
        # Ensure a non-zero exit code for unexpected failures
        sys.exit(1)


if __name__ == "__main__":
    main()
