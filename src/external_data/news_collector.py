"""Collect external news for Indian electronics supply chain topics.

This script reads API keys from a .env file and saves one JSON file per day
under data/external/news/. The file name uses the current date and includes
collection metadata.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from . import utils

create_directory = utils.create_directory
load_env = utils.load_env
requests_with_retry = utils.requests_with_retry
save_json = utils.save_json

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "external"
NEWS_DIR = DATA_DIR / "news"
ENV_PATH = BASE_DIR / ".env"

NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_ENV_KEY = "NEWS_API_KEY"

QUERY_TERMS = [
    "electronics",
    "semiconductors",
    "mobile phones",
    "laptops",
    "consumer electronics",
    "Flipkart",
    "Amazon India",
    "manufacturing",
    "logistics",
    "shipping",
    "warehouses",
    "Tamil Nadu",
    "Karnataka",
    "India electronics",
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def build_news_query(terms: list[str]) -> str:
    """Build a NewsAPI-compatible query string from a list of terms."""
    quoted = [f'"{term}"' if " " in term else term for term in terms]
    joined = " OR ".join(quoted)
    return f"({joined}) AND (India OR Indian)"


def load_api_key() -> str:
    """Load the News API key from the .env file."""
    env_vars = load_env(ENV_PATH)
    api_key = env_vars.get(NEWS_ENV_KEY)
    if not api_key:
        raise ValueError(
            f"Missing {NEWS_ENV_KEY} in {ENV_PATH}. Please add your News API key."
        )
    return api_key


def collect_news(api_key: str) -> Dict[str, Any]:
    """Fetch news articles from the configured news provider."""
    query = build_news_query(QUERY_TERMS)
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 25,
        "apiKey": api_key,
    }

    logging.info("Starting News API request to %s", NEWS_API_URL)
    session = requests.Session()
    response = requests_with_retry(
        session,
        method="GET",
        url=NEWS_API_URL,
        params=params,
    )
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(
            f"News API request failed: {payload.get('message', 'unknown error')}"
        )
    article_count = payload.get("totalResults", 0)
    logging.info("News API request succeeded, fetched %s articles.", article_count)
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source": "newsapi.org",
        "query": query,
        "article_count": article_count,
        "response": payload,
    }


def build_output_path() -> Path:
    """Create the news output directory and return today's file path."""
    create_directory(NEWS_DIR)
    today = date.today().isoformat()
    return NEWS_DIR / f"{today}_news.json"


def main() -> None:
    configure_logging()
    try:
        api_key = load_api_key()
        logging.info("Loaded News API key from %s", ENV_PATH)

        payload = collect_news(api_key)
        output_file = build_output_path()
        save_json(payload, output_file)

        logging.info("Saved news collection to %s", output_file)
        logging.info("Collected %s news articles.", payload["article_count"])
    except Exception as exc:
        logging.exception("News collection failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
