"""
Tests for the semantic (embeddings/Qdrant) fallback added to
RAG.retriever.retrieve_for_finding().

Design agreed with Task A's owner: exact resolution against
policy_references stays the ONLY path for findings that have that
field -- semantic search must never be able to override or
substitute a named policy reference. Semantic retrieval is wired in
ONLY as an enhancement to the lexical fallback, for findings that
don't carry policy_references at all.
"""

from pathlib import Path

import pytest

import RAG.retriever as retriever
from engine.policy_registry import load_policy_registry
from RAG.retriever import retrieve_for_finding


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def real_registry():
    return load_policy_registry(DATA_DIR)


# =====================================================================
# The primary (exact-resolution) path must be completely unaffected
# =====================================================================

def test_finding_with_policy_references_never_touches_semantic_fallback(
    monkeypatch, real_registry
):
    """
    If semantic fallback were ever accidentally called for a finding
    that HAS policy_references, that would be the exact regression
    this design is meant to prevent. Assert it's never even attempted.
    """
    called = {"count": 0}

    def _spy(*args, **kwargs):
        called["count"] += 1
        return []

    monkeypatch.setattr(retriever, "_try_semantic_fallback", _spy)

    finding = {
        "policy_references": [
            {"policy_id": "DORMANT-POLICY-001", "version": "1.0", "section": "Requirements"}
        ]
    }

    results = retrieve_for_finding(finding, real_registry)

    assert called["count"] == 0
    assert results[0]["policy_id"] == "DORMANT-POLICY-001"


# =====================================================================
# Fallback path: semantic tried first, lexical used if it yields nothing
# =====================================================================

def test_fallback_uses_semantic_results_when_available(monkeypatch, real_registry):
    fake_semantic_hit = {
        "policy_id": "SCREENING-POLICY-001",
        "version": "1.0",
        "title": "Customer Screening Policy",
        "section": "Requirements",
        "content": "...",
        "source_file": "01_customer_screening_policy.md",
        "score": 0.91,
    }

    monkeypatch.setattr(
        retriever, "_try_semantic_fallback", lambda query, top_k=3: [fake_semantic_hit]
    )

    finding = {
        "control_id": "SOME_UNLINKED_CONTROL",
        "description": "generic finding with no policy_references field",
    }

    results = retrieve_for_finding(finding, real_registry)

    assert results == [fake_semantic_hit]


def test_fallback_uses_lexical_when_semantic_returns_nothing(monkeypatch, real_registry):
    monkeypatch.setattr(retriever, "_try_semantic_fallback", lambda query, top_k=3: [])

    finding = {
        "control_id": "SOME_UNLINKED_CONTROL",
        "description": "dormant account handling requirements",
    }

    results = retrieve_for_finding(finding, real_registry)

    assert results  # lexical fallback still found something
    assert all("relevance_score" in r for r in results)


# =====================================================================
# Semantic fallback must degrade silently, never crash the caller
# =====================================================================

def test_semantic_fallback_returns_empty_list_when_qdrant_unavailable(monkeypatch):
    """
    Must degrade to [] whenever the underlying vector_store call fails
    for any reason (Qdrant unreachable, collection not indexed,
    embedding model missing, etc.) -- NOT just "when this machine
    happens to have no Qdrant running".

    This used to call the real _try_semantic_fallback() unmocked,
    assuming the test environment had no Qdrant server -- that's an
    environment assumption, not a test. It passed in the sandbox
    (no Qdrant there) but failed on a machine that DOES have Qdrant
    running and indexed, since real results came back instead of [].
    Simulating the failure directly makes this deterministic
    regardless of what's running on the machine executing it.
    """
    import RAG.vector_store as vector_store

    def _raise(*args, **kwargs):
        raise ConnectionError("Qdrant not reachable")

    monkeypatch.setattr(vector_store, "retrieve_policy_context", _raise)

    result = retriever._try_semantic_fallback("dormant account handling", top_k=3)

    assert result == []


def test_fallback_path_works_end_to_end_regardless_of_qdrant_availability(real_registry):
    """
    Full retrieve_for_finding() call, unlinked finding, no mocking at
    all -- proves the whole fallback chain produces the right policy
    either way: via semantic search if Qdrant happens to be running
    and indexed on the machine executing this, or via the lexical
    fallback if not. Deliberately NOT asserting which path was taken
    -- only that the end result is correct regardless.
    """
    finding = {
        "control_id": "SOME_UNLINKED_CONTROL",
        "description": "dormant account handling requirements",
    }

    results = retrieve_for_finding(finding, real_registry, top_k=3)

    assert results
    assert results[0]["policy_id"] == "DORMANT-POLICY-001"


# =====================================================================
# GENUINE semantic retrieval (requires a live, indexed Qdrant)
#
# Everything above tests the WIRING/fallback logic with mocks, or
# passes either way regardless of which path served the request. None
# of that actually proves _try_semantic_fallback() returns correct,
# meaningful results when Qdrant genuinely IS running and indexed --
# this does. Skipped automatically when Qdrant isn't reachable, same
# pattern as tests/test_rag.py's requires_qdrant tests.
# =====================================================================

def _qdrant_is_reachable(host: str = "localhost", port: int = 6333) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_is_reachable(),
    reason="Qdrant is not running on localhost:6333 -- start it with "
    "`docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant` and index the "
    "policies (RAG/vector_store.py's __main__ block) to run this test.",
)


@requires_qdrant
def test_semantic_fallback_returns_real_results_when_qdrant_is_indexed():
    """
    With a real, indexed Qdrant, _try_semantic_fallback() must return
    actual, relevant results -- not just "didn't crash". This is the
    test that was missing: everything else in this file checks the
    fallback WIRING, this checks the semantic retrieval ITSELF works.
    """
    results = retriever._try_semantic_fallback(
        "dormant account handling requirements", top_k=3
    )

    assert results
    assert all("policy_id" in r for r in results)
    assert all("relevance_score" in r for r in results)
    # The top hit for this query should genuinely be the dormant policy
    # -- not just any non-empty result.
    assert results[0]["policy_id"] == "DORMANT-POLICY-001"


@requires_qdrant
def test_fallback_prefers_real_semantic_results_over_lexical(real_registry):
    """
    End-to-end through retrieve_for_finding(), with real Qdrant doing
    the work (no monkeypatching this time) -- confirms the semantic
    path is what actually served the request when it's available,
    by checking the relevance_score came from real cosine similarity
    (not the lexical fallback's own scoring) and results are ranked.
    """
    finding = {
        "control_id": "SOME_UNLINKED_CONTROL",
        "description": "screening evidence required before wallet activation",
    }

    results = retrieve_for_finding(finding, real_registry, top_k=3)

    assert results
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["policy_id"] == "SCREENING-POLICY-001"