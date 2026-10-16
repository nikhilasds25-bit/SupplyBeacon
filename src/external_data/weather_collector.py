"""Collect daily weather information for warehouse cities."""
from __future__ import annotations

import logging
import sys
import json
from datetime import date, datetime, timezone
from math import asin, cos, radians, sin, sqrt, isfinite
import unicodedata
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping

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
LOCATION_CACHE_PATH = Path(
    os.getenv("SUPPLYBEACON_CURRENT_DATA_DIR") or BASE_DIR / "data" / "current"
) / "weather_location_resolution_cache.json"

REGIONAL_LOCATION_NAMES = {
    "india", "pan-india", "north india", "south india", "east india", "west india",
    "central india", "northeast india", "north-east india",
}

STATE_ALIASES = {
    "national capital territory of delhi": "delhi",
    "nct of delhi": "delhi",
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
}

CITY_ALIAS_GROUPS = (
    frozenset({"bengaluru", "bangalore"}),
    frozenset({"gurugram", "gurgaon"}),
    frozenset({"ernakulam", "ernacolum"}),
)

# A different provider locality must be close enough to represent the same
# urban area, not merely somewhere in the same state.
MAX_CROSS_PROVIDER_DISTANCE_KM = 25.0


class GeographicResolutionError(RuntimeError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


def _geo_token(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("&", " and ")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", text).replace("_", " ").split())


def _state_token(value: Any) -> str:
    token = _geo_token(value)
    return STATE_ALIASES.get(token, token)


def _city_match_kind(requested_city: Any, resolved_city: Any) -> str | None:
    requested = _geo_token(requested_city)
    resolved = _geo_token(resolved_city)
    if not requested or not resolved:
        return None
    if requested == resolved:
        return "EXACT_CITY"
    if any({requested, resolved}.issubset(group) for group in CITY_ALIAS_GROUPS):
        return "CONTROLLED_CITY_ALIAS"
    return None


def _coordinates(metadata: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        lat, lon = float(metadata["latitude"]), float(metadata["longitude"])
        if isfinite(lat) and isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except (KeyError, TypeError, ValueError):
        return None


def _distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, first)
    lat2, lon2 = map(radians, second)
    lat_delta = lat2 - lat1
    lon_delta = lon2 - lon1
    haversine = sin(lat_delta / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lon_delta / 2) ** 2
    haversine = min(1.0, max(0.0, haversine))
    return 2 * 6371.0088 * asin(sqrt(haversine))


def provider_request_failure(
    provider: str,
    location: str | Mapping[str, Any],
    query: str,
    failure_type: str,
    reason: object,
    *,
    exception_type: str | None = None,
) -> dict[str, Any]:
    """Build a secret-safe, persistable provider-request failure record."""
    spec = normalize_weather_location(location)
    failure = {
        "provider": provider.upper(),
        "requested_city": spec["city"],
        "requested_state": spec["state"],
        "requested_country": spec["country"],
        "query": sanitize_external_error(query),
        "failure_type": failure_type,
        "reason": sanitize_external_error(reason),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if exception_type:
        failure["exception_type"] = exception_type
    return failure


def normalize_weather_location(value: str | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(value, Mapping):
        city = str(value.get("city") or value.get("name") or "").strip()
        state = str(value.get("state") or value.get("region") or "").strip()
        country = str(value.get("country") or "India").strip()
    else:
        city, state, country = str(value).strip(), "", "India"
    if not city:
        raise ValueError("Weather location city must not be empty")
    return {"city": city, "state": state, "country": country or "India"}


def weather_location_cache_key(location: str | Mapping[str, Any]) -> str:
    spec = normalize_weather_location(location)
    return "|".join((_geo_token(spec["city"]), _state_token(spec["state"]), _geo_token(spec["country"])))


def weather_location_query(location: str | Mapping[str, Any]) -> str:
    spec = normalize_weather_location(location)
    return ", ".join(part for part in (spec["city"], spec["state"], spec["country"]) if part)


def load_location_resolution_cache(path: Path = LOCATION_CACHE_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_location_resolution_cache(cache: dict[str, dict[str, Any]], path: Path = LOCATION_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def trusted_location_reference(entry: Mapping[str, Any], location: Mapping[str, Any]) -> dict[str, Any] | None:
    """Only exact/alias anchors can establish identity; proximity cannot bootstrap it."""
    spec = normalize_weather_location(location)
    anchor = entry.get("trusted_reference") or entry
    if not isinstance(anchor, Mapping):
        return None
    region = _state_token(anchor.get("resolved_region"))
    state = _state_token(spec["state"])
    source = str(anchor.get("geocode_source") or "").upper()
    if (
        anchor.get("location_validation_status") == "VALIDATED"
        and source
        and _geo_token(anchor.get("requested_city")) == _geo_token(spec["city"])
        and _state_token(anchor.get("requested_state")) == state
        and _geo_token(anchor.get("requested_country")) in {"india", "in"}
        and _geo_token(anchor.get("resolved_country")) in {"india", "in"}
        and (not state or region == state or (not region and source == "OPENWEATHER"))
        and _city_match_kind(spec["city"], anchor.get("resolved_city"))
        and _coordinates(anchor)
    ):
        fields = ("requested_city", "requested_state", "requested_country", "resolved_city",
                  "resolved_region", "resolved_country", "latitude", "longitude", "geocode_source",
                  "geocode_query", "location_validation_status", "city_validation_basis")
        return {key: anchor.get(key) for key in fields}
    return None


def _cached_coordinates(cache: dict[str, dict[str, Any]], location: Mapping[str, Any]) -> tuple[float, float] | None:
    anchor = trusted_location_reference(cache.get(weather_location_cache_key(location), {}), location)
    return _coordinates(anchor) if anchor else None


def cache_validated_resolution(
    cache: dict[str, dict[str, Any]],
    location: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if metadata.get("location_validation_status") != "VALIDATED":
        return
    if metadata.get("city_validation_basis") == "DISTANCE_CORROBORATED":
        anchor = trusted_location_reference(metadata, location)
        if anchor:
            cache[weather_location_cache_key(location)] = anchor
        return
    if metadata.get("latitude") in (None, "") or metadata.get("longitude") in (None, ""):
        return
    cache[weather_location_cache_key(location)] = {
        key: metadata.get(key)
        for key in (
            "requested_city", "requested_state", "requested_country", "resolved_city",
            "resolved_region", "resolved_country", "latitude", "longitude",
            "geocode_source", "geocode_query", "geocode_country", "geocode_region",
            "geocode_confidence", "location_validation_status", "city_validation_basis",
            "reference_source", "reference_distance_km", "distance_threshold_km",
        )
    }


def validate_provider_location(
    location: str | Mapping[str, Any],
    payload: Mapping[str, Any],
    provider: str,
    query: str,
    *,
    reference_resolution: Mapping[str, Any] | None = None,
    max_reference_distance_km: float = MAX_CROSS_PROVIDER_DISTANCE_KM,
) -> tuple[bool, dict[str, Any], str | None]:
    spec = normalize_weather_location(location)
    provider_name = provider.upper()
    if provider_name == "WEATHERAPI":
        resolved = payload.get("location") if isinstance(payload.get("location"), Mapping) else {}
        resolved_city = str(resolved.get("name") or "").strip()
        resolved_region = str(resolved.get("region") or "").strip()
        resolved_country = str(resolved.get("country") or "").strip()
        latitude, longitude = resolved.get("lat"), resolved.get("lon")
    else:
        system = payload.get("sys") if isinstance(payload.get("sys"), Mapping) else {}
        coordinates = payload.get("coord") if isinstance(payload.get("coord"), Mapping) else {}
        resolved_city = str(payload.get("name") or "").strip()
        resolved_region = str(payload.get("state") or payload.get("region") or "").strip()
        resolved_country = str(system.get("country") or "").strip()
        latitude, longitude = coordinates.get("lat"), coordinates.get("lon")

    country_valid = _geo_token(resolved_country) in {"india", "in"}
    expected_state = _state_token(spec["state"])
    returned_state = _state_token(resolved_region)
    state_valid = (
        not expected_state
        or (bool(returned_state) and expected_state == returned_state)
        or (provider_name == "OPENWEATHER" and not returned_state)
    )
    city_validation_basis = _city_match_kind(spec["city"], resolved_city)
    reference_source = None
    reference_distance_km = None

    anchor = trusted_location_reference(reference_resolution or {}, spec)
    query_coordinates = None
    try:
        query_lat, query_lon = query.split(",")
        query_coordinates = _coordinates({"latitude": query_lat, "longitude": query_lon})
    except ValueError:
        pass
    coordinate_request = query_coordinates is not None
    coordinate_request_valid = False
    if anchor:
        reference_resolution = anchor
        reference_source = str(reference_resolution.get("geocode_source") or "").upper() or None
        reference_is_independent = bool(reference_source and reference_source != provider_name)
        reference_is_validated = reference_resolution.get("location_validation_status") == "VALIDATED"
        reference_request_matches = (
            _geo_token(reference_resolution.get("requested_city")) == _geo_token(spec["city"])
            and _state_token(reference_resolution.get("requested_state")) == expected_state
        )
        reference_matches_request = bool(
            _city_match_kind(spec["city"], reference_resolution.get("resolved_city"))
            and _geo_token(reference_resolution.get("resolved_country")) in {"india", "in"}
        )
        reference_state = _state_token(reference_resolution.get("resolved_region"))
        reference_state_valid = not expected_state or not reference_state or reference_state == expected_state
        reference_coordinates = _coordinates(reference_resolution)
        resolved_coordinates = _coordinates({"latitude": latitude, "longitude": longitude})
        query_uses_anchor = coordinate_request and _distance_km(query_coordinates, reference_coordinates) < 0.001
        if (
            (reference_is_independent or query_uses_anchor)
            and reference_is_validated
            and reference_request_matches
            and reference_matches_request
            and reference_state_valid
            and reference_coordinates
            and resolved_coordinates
        ):
            distance = _distance_km(reference_coordinates, resolved_coordinates)
            reference_distance_km = round(distance, 3)
            coordinate_request_valid = bool(query_uses_anchor and distance <= max_reference_distance_km)
            if distance <= max_reference_distance_km and (not city_validation_basis or coordinate_request):
                city_validation_basis = "DISTANCE_CORROBORATED"

    status = "VALIDATED"
    reason = None
    if not country_valid:
        status = "REJECTED_COUNTRY"
        reason = f"Provider resolved {spec['city']} to non-India country {resolved_country or 'Unknown'}"
    elif not state_valid:
        status = "REJECTED_REGION"
        reason = f"Provider region {resolved_region or 'Unknown'} does not match expected state {spec['state']}"
    elif not city_validation_basis or (coordinate_request and not coordinate_request_valid):
        if reference_distance_km is not None:
            status = "REJECTED_CITY_DISTANCE"
            reason = (
                f"Provider city {resolved_city or 'Unknown'} is {reference_distance_km:.3f} km from "
                f"the independently validated {spec['city']} reference; maximum allowed is "
                f"{max_reference_distance_km:.1f} km"
            )
        else:
            status = "REJECTED_CITY"
            reason = (
                f"Provider city {resolved_city or 'Unknown'} does not match expected city "
                f"{spec['city']} and has no qualifying independent distance validation"
            )

    confidence = "MEDIUM" if city_validation_basis == "DISTANCE_CORROBORATED" else "HIGH"
    metadata = {
        "requested_city": spec["city"],
        "requested_state": spec["state"],
        "requested_country": spec["country"],
        "resolved_city": resolved_city,
        "resolved_region": resolved_region,
        "resolved_country": resolved_country,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": (payload.get("location") or {}).get("tz_id") if provider_name == "WEATHERAPI" else payload.get("timezone"),
        "geocode_source": provider_name,
        "geocode_query": query,
        "geocode_country": resolved_country,
        "geocode_region": resolved_region,
        "geocode_confidence": confidence if status == "VALIDATED" else "REJECTED",
        "location_validation_status": status,
        "city_validation_basis": city_validation_basis if status == "VALIDATED" else "REJECTED",
        "reference_source": reference_source,
        "reference_distance_km": reference_distance_km,
        "distance_threshold_km": max_reference_distance_km,
    }
    if status == "VALIDATED" and city_validation_basis == "DISTANCE_CORROBORATED":
        metadata["trusted_reference"] = anchor
    return status == "VALIDATED", metadata, reason


def warehouse_weather_location_specs(graph: Any) -> list[dict[str, str]]:
    """Discover distinct city/state/country provider locations from graph relationships."""
    locations: dict[str, dict[str, str]] = {}
    for warehouse_id, warehouse in graph.nodes(data=True):
        if warehouse.get("node_type") != "Warehouse":
            continue
        candidates: list[tuple[str, dict[str, Any]]] = []
        neighbors = list(graph.successors(warehouse_id)) + list(graph.predecessors(warehouse_id))
        for location_id in dict.fromkeys(neighbors):
            edge = graph.get_edge_data(warehouse_id, location_id) or graph.get_edge_data(location_id, warehouse_id) or {}
            attrs = graph.nodes[location_id]
            if edge.get("relationship") == "LOCATED_IN" and attrs.get("node_type") == "Location":
                candidates.append((str(attrs.get("name") or attrs.get("city") or "").strip(), attrs))

        warehouse_name = str(warehouse.get("name") or "").strip()
        expected = warehouse_name.removesuffix(" Warehouse").strip().casefold()
        precise = [
            (str(attrs.get("city") or name).strip(), attrs)
            for name, attrs in candidates
            if str(attrs.get("city") or name).strip().casefold() not in REGIONAL_LOCATION_NAMES
        ]
        exact = [(city, attrs) for city, attrs in precise if city.casefold() == expected]
        for city, attrs in (exact or (precise if len(precise) == 1 else [])):
            spec = normalize_weather_location({
                "city": city,
                "state": attrs.get("state") or warehouse.get("state") or "",
                "country": attrs.get("country") or warehouse.get("country") or "India",
            })
            locations.setdefault(weather_location_cache_key(spec), spec)
    return sorted(locations.values(), key=lambda item: (_geo_token(item["city"]), _state_token(item["state"])))


def warehouse_weather_locations(graph: Any) -> list[str]:
    """Return unique, provider-queryable locations linked to runtime warehouses.

    Base data links a warehouse to both its city and a regional roll-up. A city
    observation must not be fabricated for a broad region, so exact city/name
    relationships are preferred and regional-only warehouses are left
    unmonitored until a precise location is published.
    """
    cities: dict[str, str] = {}
    for spec in warehouse_weather_location_specs(graph):
        cities.setdefault(_geo_token(spec["city"]), spec["city"])
    return sorted(cities.values(), key=str.casefold)


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


def fetch_weather_for_city(
    session: requests.Session,
    api_key: str,
    city: str | Mapping[str, Any],
    resolution_cache: dict[str, dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Fetch and geographically validate an OpenWeather observation."""
    spec = normalize_weather_location(city)
    cache = resolution_cache if resolution_cache is not None else {}
    coordinates = _cached_coordinates(cache, spec)
    query = weather_location_query(spec)
    params: dict[str, Any] = {"appid": api_key, "units": "metric"}
    if coordinates:
        params.update({"lat": coordinates[0], "lon": coordinates[1]})
        query = f"{coordinates[0]},{coordinates[1]}"
    else:
        params["q"] = query
    response = requests_with_retry(
        session,
        method="GET",
        url=WEATHER_API_URL,
        params=params,
    )
    payload = response.json()
    if response.status_code != 200:
        raise RuntimeError(
            f"Weather API failed for {spec['city']}: {payload.get('message', 'unknown error')}"
        )
    accepted, metadata, reason = validate_provider_location(
        spec, payload, "OPENWEATHER", query,
        reference_resolution=cache.get(weather_location_cache_key(spec)),
    )
    if not accepted:
        error = GeographicResolutionError(reason or "OpenWeather location validation failed", metadata)
        error.payload = payload
        raise error
    cache_validated_resolution(cache, spec, metadata)
    return {**payload, **metadata, "city": spec["city"], "state": spec["state"], "country": spec["country"]}


def build_output_path() -> Path:
    create_directory(WEATHER_DIR)
    today = date.today().isoformat()
    return WEATHER_DIR / f"{today}_weather.json"


import sys

def collect_weather(
    api_key: str,
    cities: Iterable[str | Mapping[str, Any]],
    resolution_cache: dict[str, dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    cache = resolution_cache if resolution_cache is not None else {}
    monitored = {
        weather_location_cache_key(spec): spec
        for value in cities
        if (spec := normalize_weather_location(value)).get("city")
    }
    monitored_locations = list(monitored.values())
    city_counts: dict[str, int] = {}
    for spec in monitored_locations:
        city_counts[_geo_token(spec["city"])] = city_counts.get(_geo_token(spec["city"]), 0) + 1
    logging.info("Starting weather data collection for %s locations", len(monitored_locations))
    session = requests.Session()
    successes: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    rejected_locations: list[dict[str, Any]] = []
    request_failures: list[dict[str, Any]] = []
    pending_locations: list[dict[str, Any]] = []

    for spec in monitored_locations:
        city = spec["city"]
        storage_key = city if city_counts[_geo_token(city)] == 1 else f"{city}, {spec['state']}"
        coordinates = _cached_coordinates(cache, spec)
        query = f"{coordinates[0]},{coordinates[1]}" if coordinates else weather_location_query(spec)
        try:
            payload = fetch_weather_for_city(session, api_key, spec, cache)
            # Basic validation: OpenWeather returns a 'weather' key for valid responses
            if isinstance(payload, dict) and payload.get("weather"):
                successes[storage_key] = payload
                logging.info("Collected weather for %s", weather_location_query(spec))
            else:
                failures[storage_key] = "invalid response structure"
                request_failures.append(provider_request_failure(
                    "OPENWEATHER", spec, query, "INVALID_RESPONSE", "invalid response structure"
                ))
                logging.warning("Invalid weather payload structure for %s", weather_location_query(spec))
        except GeographicResolutionError as exc:
            failures[storage_key] = str(exc)
            rejected_locations.append({"provider": "OPENWEATHER", **exc.metadata, "reason": str(exc)})
            if exc.metadata.get("location_validation_status") == "REJECTED_CITY" and getattr(exc, "payload", None):
                pending_locations.append({"location": spec, "query": query, "payload": exc.payload, "storage_key": storage_key})
            logging.warning("Rejected OpenWeather location for %s: %s", weather_location_query(spec), exc)
        except Exception as exc:
            # Log a concise failure message without exposing the API key or sensitive params
            safe_error = sanitize_external_error(exc)
            logging.warning("Failed to collect weather for %s: %s", city, safe_error)
            failures[storage_key] = safe_error
            request_failures.append(provider_request_failure(
                "OPENWEATHER", spec, query, "REQUEST_ERROR", safe_error,
                exception_type=type(exc).__name__,
            ))

    results: Dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source": "openweathermap.org",
        "cities": successes,  # only successful records
        "successful_count": len(successes),
        "failed_count": len(failures),
        "failed_cities": list(failures.keys()),
        "failure_reasons": failures,
        "rejected_locations": rejected_locations,
        "provider_request_failures": request_failures,
        "_pending_locations": pending_locations,
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
        locations = warehouse_weather_location_specs(graph) if graph is not None else []
        if not locations:
            raise RuntimeError("No precise warehouse locations are available in the current runtime graph")
        cache = load_location_resolution_cache()
        payload = collect_weather(api_key, locations, cache)
        save_location_resolution_cache(cache)

        success_count = int(payload.get("successful_count", 0))
        failed_count = int(payload.get("failed_count", 0))

        # If all cities failed, treat as a failure and exit with non-zero status
        if success_count == 0:
            logging.error(
                "Weather data collection failed for all %s cities. Failed cities: %s",
                len(locations),
                ", ".join(payload.get("failed_cities", [])),
            )
            # Do not save a misleading successful dataset
            sys.exit(1)

        # Save only successful records, but include failure summary for transparency
        output_file = build_output_path()
        payload.pop("_pending_locations", None)
        save_json(payload, output_file)
        logging.info("Saved weather collection to %s", output_file)
        logging.info("Weather collection summary: Successful cities: %s, Failed cities: %s", success_count, failed_count)

    except Exception as exc:
        logging.exception("Weather collection failed: %s", exc)
        # Ensure a non-zero exit code for unexpected failures
        sys.exit(1)


if __name__ == "__main__":
    main()
