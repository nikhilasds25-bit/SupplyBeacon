"""Utility helpers for external data collectors.

This module provides helper functions for reading environment configuration,
performing retried HTTP requests, creating required directories, and saving JSON
output in a production-quality way.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


_SECRET_QUERY_PARAM = re.compile(
    r"(?i)([?&](?:api[_-]?key|apikey|appid|key|token|access[_-]?token)=)[^&\s]+"
)


def sanitize_external_error(error: object) -> str:
    """Return provider diagnostics without persisting credentials from URLs."""
    return _SECRET_QUERY_PARAM.sub(r"\1<redacted>", str(error))


def load_env(env_path: Path) -> Dict[str, str]:
    """Load environment variables from a .env file.

    The .env file must contain key=value pairs. Comments and blank lines are
    ignored.
    """
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    env_vars: Dict[str, str] = {}
    with env_path.open(encoding="utf-8") as reader:
        for line in reader:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env_vars[key] = value
    return env_vars


def create_directory(path: Path) -> None:
    """Create a directory and any missing parent directories."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Save a Python object to JSON with indentation."""
    with path.open("w", encoding="utf-8") as writer:
        json.dump(data, writer, ensure_ascii=False, indent=2)


def requests_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Optional[tuple[int, ...]] = (429, 500, 502, 503, 504),
    **kwargs: Any,
) -> requests.Response:
    """Send an HTTP request with retries for transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            response = session.request(method, url, timeout=15, **kwargs)
            if response.status_code in status_forcelist:
                logging.warning(
                    "Request to %s returned status %s on attempt %s/%s",
                    url,
                    response.status_code,
                    attempt,
                    max_retries,
                )
                raise requests.HTTPError(f"Status code {response.status_code}")
            return response
        except (requests.RequestException, requests.HTTPError) as exc:
            logging.warning(
                "Request attempt %s/%s failed for URL %s: %s",
                attempt,
                max_retries,
                url,
                    sanitize_external_error(exc),
            )
            if attempt == max_retries:
                logging.error("Exceeded max retries for URL %s", url)
                raise
            sleep_seconds = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)


def normalize_city_name(city: str) -> str:
    """Normalize a city name into a query-friendly value."""
    normalized = re.sub(r"[^A-Za-z0-9 ]+", "", city).strip()
    return normalized
