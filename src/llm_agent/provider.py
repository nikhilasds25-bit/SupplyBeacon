"""Provider abstraction for grounded ElectroGraph LLM calls."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import requests


class ProviderError(RuntimeError):
    """Raised when an LLM provider cannot produce a response."""


class LLMProvider(ABC):
    """Provider contract independent of a specific LLM vendor."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_query: str,
        tool_evidence: dict[str, Any],
    ) -> str:
        """Generate text from a grounded prompt and structured evidence."""


class OpenAICompatibleProvider(LLMProvider):
    """Small adapter for providers exposing an OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ELECTROGRAPH_LLM_API_KEY")
        self.base_url = (base_url or os.getenv("ELECTROGRAPH_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("ELECTROGRAPH_LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_query: str,
        tool_evidence: dict[str, Any],
    ) -> str:
        if not self.configured:
            raise ProviderError("ELECTROGRAPH_LLM_API_KEY is not configured.")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": user_query, "tool_evidence": tool_evidence},
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"LLM provider request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("LLM provider returned empty content.")
        return content.strip()
