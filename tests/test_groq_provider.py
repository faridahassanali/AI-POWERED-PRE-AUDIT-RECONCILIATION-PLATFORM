"""
Tests for engine.llm.groq_provider -- specifically the Retry-After
header parsing, since the router now depends on retry_after being
populated correctly to back off the right amount of time.
"""

import pytest

from engine.llm.base import LLMTransientError
from engine.llm.groq_provider import GroqProvider


class _FakeResponse:
    def __init__(self, status_code, headers=None, json_body=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self._json_body = json_body or {}
        self.text = text

    def json(self):
        return self._json_body


SAMPLE_AI_INPUT = {
    "finding_id": "F-TEST",
    "audit_run_id": "RUN-TEST",
    "control_id": "SCREENING_001",
    "policy_context": [
        {"policy_id": "SCREENING-POLICY-001", "content": "..."},
    ],
}


def test_retry_after_header_is_parsed_as_float(monkeypatch):

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        "engine.llm.groq_provider.requests.post",
        lambda *args, **kwargs: _FakeResponse(429, headers={"retry-after": "7"}),
    )

    provider = GroqProvider()

    with pytest.raises(LLMTransientError) as exc_info:
        provider.generate(SAMPLE_AI_INPUT)

    assert exc_info.value.retry_after == 7.0


def test_missing_retry_after_header_leaves_retry_after_none(monkeypatch):

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        "engine.llm.groq_provider.requests.post",
        lambda *args, **kwargs: _FakeResponse(429, headers={}),
    )

    provider = GroqProvider()

    with pytest.raises(LLMTransientError) as exc_info:
        provider.generate(SAMPLE_AI_INPUT)

    assert exc_info.value.retry_after is None


def test_non_numeric_retry_after_header_leaves_retry_after_none(monkeypatch):

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        "engine.llm.groq_provider.requests.post",
        lambda *args, **kwargs: _FakeResponse(429, headers={"retry-after": "soon"}),
    )

    provider = GroqProvider()

    with pytest.raises(LLMTransientError) as exc_info:
        provider.generate(SAMPLE_AI_INPUT)

    assert exc_info.value.retry_after is None
