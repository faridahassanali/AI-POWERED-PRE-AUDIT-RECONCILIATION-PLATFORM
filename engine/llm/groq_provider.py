"""
Groq Provider.

Calls Groq's OpenAI-compatible chat completions endpoint.

Model: openai/gpt-oss-120b (free tier as of Aug 2026: 30 req/min,
200K tokens/day, 1,000 req/day -- comfortably above this project's
volume; see the model-selection discussion for the reasoning).

Uses plain HTTP (requests) rather than the groq SDK, so this package
has one fewer hard dependency and the same pattern works identically
for both providers in this file/gemini_provider.py.
"""

import json
import os
import time
from typing import Any

import requests

from engine.llm.base import (
    LLMConfigError,
    LLMExplanation,
    LLMOutputError,
    LLMTransientError,
    validate_citations,
)
from engine.llm.prompts import (
    RESPONSE_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
)


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT_SECONDS = 30


class GroqProvider:
    """
    LLMProvider implementation for Groq.

    Reads the API key from the GROQ_API_KEY environment variable at
    call time (not at import time), so tests can monkeypatch the
    environment without reloading the module.
    """

    name = "groq"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.timeout = timeout

    def _api_key(self) -> str:

        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise LLMConfigError(
                "GROQ_API_KEY is not set. This is a configuration "
                "problem, not a transient one -- the router will "
                "not silently fail over to Gemini for this; fix the "
                "key and retry."
            )

        return api_key

    def generate(self, ai_input: dict[str, Any]) -> LLMExplanation:

        api_key = self._api_key()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ai_input)},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )

        except requests.exceptions.Timeout as exc:
            raise LLMTransientError(
                f"Groq request timed out after {self.timeout}s."
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            raise LLMTransientError(
                f"Could not connect to Groq: {exc}"
            ) from exc

        # -----------------------------------------------------------
        # AUTH / CONFIG ERRORS -- not retryable, not fail-over-able.
        # -----------------------------------------------------------

        if response.status_code in (401, 403):
            raise LLMConfigError(
                f"Groq rejected the API key (HTTP {response.status_code}). "
                "Check GROQ_API_KEY."
            )

        # -----------------------------------------------------------
        # RATE LIMIT / SERVER ERRORS -- transient, router will retry
        # then fail over to Gemini.
        # -----------------------------------------------------------

        if response.status_code == 429:
            retry_after_header = response.headers.get("retry-after")
            retry_after: float | None = None
            if retry_after_header is not None:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = None
            raise LLMTransientError(
                f"Groq rate limit hit (retry-after: {retry_after_header}).",
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            raise LLMTransientError(
                f"Groq server error (HTTP {response.status_code})."
            )

        if response.status_code != 200:
            raise LLMTransientError(
                f"Unexpected Groq response (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )

        return self._parse_response(response.json(), ai_input)

    def _parse_response(
        self,
        body: dict[str, Any],
        ai_input: dict[str, Any],
    ) -> LLMExplanation:

        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            explanation = parsed["explanation"]
            recommendation = parsed["recommendation"]
            cited_policy_ids = parsed["cited_policy_ids"]

        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise LLMOutputError(
                f"Could not parse Groq response into the expected "
                f"JSON shape: {exc}"
            ) from exc

        validate_citations(
            cited_policy_ids,
            ai_input.get("policy_context", []),
        )

        return LLMExplanation(
            finding_id=ai_input["finding_id"],
            audit_run_id=ai_input["audit_run_id"],
            ai_explanation=explanation,
            ai_recommendation=recommendation,
            cited_policy_ids=cited_policy_ids,
            provider_used=self.name,
            model_used=self.model,
        )