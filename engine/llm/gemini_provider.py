"""
Gemini Provider.

Calls the Gemini API (Google AI Studio) generateContent endpoint.

Model: gemini-3.5-flash (confirmed free tier as of Aug 2026 -- see
the model-selection discussion; gemini-3.7-flash is newer but its
free-tier status wasn't confirmed at time of writing, so it is not
the default here).

This provider exists ONLY as the fail-over path for GroqProvider --
see engine/llm/router.py. Same prompt, same output shape, same
validation; only the transport differs.
"""

import json
import os
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


GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_TIMEOUT_SECONDS = 30


class GeminiProvider:
    """
    LLMProvider implementation for Gemini / Google AI Studio.

    Reads the API key from the GEMINI_API_KEY environment variable
    at call time, same pattern as GroqProvider.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.timeout = timeout

    def _api_key(self) -> str:

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. This is a configuration "
                "problem, not a transient one -- fix the key rather "
                "than relying on this being the fallback provider."
            )

        return api_key

    def generate(self, ai_input: dict[str, Any]) -> LLMExplanation:

        api_key = self._api_key()

        url = GEMINI_API_URL_TEMPLATE.format(model=self.model)

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_user_prompt(ai_input)}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_JSON_SCHEMA,
            },
        }

        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )

        except requests.exceptions.Timeout as exc:
            raise LLMTransientError(
                f"Gemini request timed out after {self.timeout}s."
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            raise LLMTransientError(
                f"Could not connect to Gemini: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise LLMConfigError(
                f"Gemini rejected the API key (HTTP {response.status_code}). "
                "Check GEMINI_API_KEY."
            )

        if response.status_code == 429:
            raise LLMTransientError("Gemini rate limit hit.")

        if response.status_code >= 500:
            raise LLMTransientError(
                f"Gemini server error (HTTP {response.status_code})."
            )

        if response.status_code != 200:
            raise LLMTransientError(
                f"Unexpected Gemini response (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )

        return self._parse_response(response.json(), ai_input)

    def _parse_response(
        self,
        body: dict[str, Any],
        ai_input: dict[str, Any],
    ) -> LLMExplanation:

        try:
            content = body["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)

            explanation = parsed["explanation"]
            recommendation = parsed["recommendation"]
            cited_policy_ids = parsed["cited_policy_ids"]

        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise LLMOutputError(
                f"Could not parse Gemini response into the expected "
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
