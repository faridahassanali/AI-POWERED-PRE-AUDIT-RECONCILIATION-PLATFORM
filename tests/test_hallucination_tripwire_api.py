"""
Integration Test -- Hallucination Tripwire Through the API Layer.

DELIBERATELY NOT A UNIT TEST AND DELIBERATELY NOT MOCKED.

This does not mock the LLM, the tripwire, the router, or the HTTP
layer. It talks to a REAL, already-running FastAPI server (uvicorn),
which talks to a REAL Supabase database and makes REAL Groq/Gemini
API calls -- exactly like engine.llm_evaluation does for the manual
quality report, but here the target is the tripwire's behavior
*through the API*, not explanation quality.

What "through the API layer" means here
----------------------------------------

Every call in this file is a real `requests` call against
APP_API_BASE (e.g. http://127.0.0.1:8000) -- never a direct Python
call into engine.ai_explanation_pipeline or
engine.llm.hallucination_tripwire. The only place this test imports
engine/backend code is to build an INDEPENDENT verification oracle
(see _independent_hallucination_check() below) -- it reconstructs
the exact ai_input the pipeline would have built and re-runs the
real (unmocked) check_for_hallucinations() against whatever the API
actually returned. That is verification, not mocking: nothing about
the request/response path is faked.

Why this test can't "force" a hallucination on demand
------------------------------------------------------

Real LLMs are non-deterministic. This test cannot make Groq/Gemini
hallucinate to order the way a mocked test could. Instead it asserts
an INVARIANT that must hold no matter which way a real call goes:

    1. If /findings/{id}/ai-explanation succeeds:
       the returned ai_explanation/ai_recommendation must
       independently pass the real hallucination_tripwire check
       against the finding's real evidence -- i.e. the tripwire that
       already ran inside the endpoint didn't miss anything an
       identical, freshly-run check would catch.

    2. If /findings/{id}/ai-explanation fails:
       the API must fail *gracefully* (a normal JSON error envelope,
       never a 5xx/traceback), the error must be attributable to the
       tripwire (HallucinationDetectedError) rather than something
       unrelated, and the finding's persisted ai_explanation /
       ai_recommendation must be UNCHANGED (never a half-written,
       contradicted explanation lands in the database).

Running it
----------

Requires a live backend + real provider credentials -- same
requirement as engine.llm_evaluation, and NOT part of the regular
fast pytest suite for the same reason (real network calls, real
cost, non-deterministic LLM). Run explicitly:

    uvicorn backend.main:app --reload   # in one terminal
    pytest tests/integration/test_hallucination_tripwire_api.py -v -s

Environment variables (same ones the frontend uses):

    APP_API_BASE   default: http://127.0.0.1:8000
    APP_API_KEYS   comma-separated; the first key is used as
                   X-API-Key

Must be run from the project root (so `engine`, `RAG`, and
`backend` are importable for the independent verification step --
see module docstring above).
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest
import requests

# ---------------------------------------------------------------
# Independent verification oracle -- real production code, never
# mocked. Reused so the test doesn't re-implement (and potentially
# drift from) the tripwire's own grounding logic.
# ---------------------------------------------------------------
from engine.ai_input import build_ai_input
from engine.llm.hallucination_tripwire import check_for_hallucinations
from RAG.retriever import retrieve_for_finding

# backend.main is imported (not mocked) purely to reuse the exact
# same AI-safe field whitelist and the exact same PolicyRegistry
# instance the live server itself uses -- see backend/main.py's
# _clean_finding_for_ai() / policy_registry. Importing it does not
# start a server; FastAPI app objects are inert until served.
from backend.main import _clean_finding_for_ai, policy_registry


# =====================================================================
# CONFIG
# =====================================================================

API_BASE = os.environ.get("APP_API_BASE", "http://127.0.0.1:8000").rstrip("/")

_raw_keys = os.environ.get("APP_API_KEYS", "")
API_KEY = _raw_keys.split(",")[0].strip() if _raw_keys else ""

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# How many CONFIRMED findings to actually run through the real LLM.
# Kept small on purpose: every finding here means a real, billed
# provider call -- this is a smoke/invariant test, not the full
# stratified sample engine.llm_evaluation does for quality scoring.
MAX_FINDINGS_TO_TEST = int(os.environ.get("TRIPWIRE_API_TEST_SAMPLE", "6"))

# Real provider calls need real network round-trip time, and Groq
# free-tier rate limits mean back-to-back calls can 429 -- small
# delay between findings, generous per-call timeout.
DELAY_BETWEEN_CALLS_SECONDS = 2.5
REQUEST_TIMEOUT_SECONDS = 60


# =====================================================================
# SESSION SETUP -- run one real audit so there are real findings to
# confirm and explain. Session-scoped so the whole file shares ONE
# audit run instead of re-running the deterministic pipeline (and
# re-hitting Supabase) per test function.
# =====================================================================

@pytest.fixture(scope="session", autouse=True)
def _require_live_server():
    """
    Skip the whole module with a clear message if the server isn't
    reachable or no API key is configured -- this suite is opt-in by
    design (see module docstring), never silently green on a
    misconfigured environment.
    """

    if not API_KEY:
        pytest.skip(
            "APP_API_KEYS is not set -- this integration test needs a "
            "real API key for the live backend."
        )

    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(
            f"Could not reach live backend at {API_BASE} ({exc}). "
            "Start it with `uvicorn backend.main:app --reload` first."
        )

    if response.status_code != 200:
        pytest.skip(
            f"Live backend at {API_BASE} responded with "
            f"{response.status_code}, not 200 -- skipping."
        )


@pytest.fixture(scope="session")
def confirmed_findings() -> list[dict[str, Any]]:
    """
    Run one REAL audit through the REAL API, confirm a handful of the
    resulting findings (also through the REAL API), and return them.

    Deliberately does not reuse an old audit run: findings from a
    fresh run are guaranteed to start in REVIEW status, so this
    fixture can confirm them itself rather than assuming some
    pre-existing database state.
    """

    idempotency_key = f"tripwire-api-test-{uuid.uuid4().hex}"

    execute_response = requests.post(
        f"{API_BASE}/audit-runs/execute",
        headers={**HEADERS, "Idempotency-Key": idempotency_key},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert execute_response.status_code == 200, (
        f"POST /audit-runs/execute returned "
        f"{execute_response.status_code}: {execute_response.text}"
    )

    execute_body = execute_response.json()
    assert execute_body["status"] in ("success", "duplicate"), execute_body

    audit_run_id = (
        execute_body["audit_run_id"]
        if execute_body["status"] == "success"
        else execute_body["audit_run"]["audit_run_id"]
    )

    findings_response = requests.get(
        f"{API_BASE}/findings",
        params={"audit_run_id": audit_run_id, "status": "REVIEW"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert findings_response.status_code == 200
    findings_body = findings_response.json()
    assert findings_body["status"] == "success", findings_body

    all_review_findings = findings_body["findings"]

    if not all_review_findings:
        pytest.skip(
            f"Audit run {audit_run_id} produced zero REVIEW findings -- "
            "nothing to confirm/explain."
        )

    # Sample across distinct control types rather than taking the
    # first N, so the same control isn't tested 6 times while others
    # are never touched -- same stratification reasoning as
    # engine.llm_evaluation, kept simple here since this is an
    # invariant smoke test, not a quality sample.
    by_control: dict[str, list[dict[str, Any]]] = {}
    for finding in all_review_findings:
        by_control.setdefault(finding["control_id"], []).append(finding)

    sampled: list[dict[str, Any]] = []
    control_ids = sorted(by_control)
    i = 0
    while len(sampled) < MAX_FINDINGS_TO_TEST and any(
        by_control[c] for c in control_ids
    ):
        control_id = control_ids[i % len(control_ids)]
        if by_control[control_id]:
            sampled.append(by_control[control_id].pop())
        i += 1

    confirmed: list[dict[str, Any]] = []

    for finding in sampled:

        patch_response = requests.patch(
            f"{API_BASE}/findings/{finding['finding_id']}",
            headers=HEADERS,
            json={
                "finding_status": "CONFIRMED",
                "reviewed_by": "integration-test",
                "reviewer_notes": None,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert patch_response.status_code == 200, (
            f"PATCH /findings/{finding['finding_id']} returned "
            f"{patch_response.status_code}: {patch_response.text}"
        )

        patch_body = patch_response.json()
        assert patch_body["status"] == "success", patch_body

        confirmed.append(patch_body["finding"])

    return confirmed


# =====================================================================
# INDEPENDENT VERIFICATION ORACLE
# =====================================================================

def _independent_hallucination_check(
    finding_after_api: dict[str, Any],
) -> list[str]:
    """
    Re-derive the exact ai_input the pipeline would have built for
    this finding, and re-run the REAL (unmocked)
    check_for_hallucinations() against the ai_explanation /
    ai_recommendation the live API actually returned.

    This is the crux of the test: it proves the tripwire's
    invariant holds against what the API produced, using the same
    production grounding logic the endpoint itself relies on --
    without re-calling the LLM and without mocking anything.
    """

    safe_finding = _clean_finding_for_ai(finding_after_api)

    policy_context = retrieve_for_finding(
        finding=safe_finding,
        registry=policy_registry,
    )

    ai_input = build_ai_input(
        safe_finding,
        policy_context=policy_context,
    )

    ai_output = {
        "ai_explanation": finding_after_api.get("ai_explanation", ""),
        "ai_recommendation": finding_after_api.get("ai_recommendation", ""),
    }

    return check_for_hallucinations(ai_output=ai_output, ai_input=ai_input)


# =====================================================================
# TESTS
# =====================================================================

def test_ai_explanation_endpoint_reachable_and_authenticated():
    """
    Sanity check for auth wiring before the real invariant test below
    burns real LLM calls: a request with a deliberately wrong API key
    must be rejected, and the health endpoint (no auth) must be up.
    """

    health = requests.get(f"{API_BASE}/health", timeout=5)
    assert health.status_code == 200
    assert health.json().get("status") == "healthy"

    bad_key_response = requests.post(
        f"{API_BASE}/findings/NON-EXISTENT-ID/ai-explanation",
        headers={"X-API-Key": "definitely-not-a-real-key"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert bad_key_response.status_code in (401, 403), (
        "Expected the ai-explanation endpoint to reject an invalid "
        f"X-API-Key, got {bad_key_response.status_code}: "
        f"{bad_key_response.text}"
    )


def test_hallucination_tripwire_invariant_via_real_api(
    confirmed_findings: list[dict[str, Any]],
):
    """
    The main test. For each real CONFIRMED finding:

        1. Snapshot ai_explanation/ai_recommendation BEFORE the call
           (should be empty/None -- these were just confirmed).
        2. Call POST /findings/{id}/ai-explanation for real.
        3. Whichever way it goes, assert the invariant described in
           the module docstring.

    One finding hallucinating (or one provider hiccup) does not fail
    the whole test immediately -- results are collected and asserted
    at the end, mirroring engine.llm_evaluation's "one bad finding
    doesn't kill the batch" philosophy, but every individual outcome
    still has to satisfy the invariant or the test fails.
    """

    assert confirmed_findings, "fixture should have skipped, not returned empty"

    outcomes: list[dict[str, Any]] = []

    for finding in confirmed_findings:

        finding_id = finding["finding_id"]

        before_explanation = finding.get("ai_explanation")
        before_recommendation = finding.get("ai_recommendation")

        response = requests.post(
            f"{API_BASE}/findings/{finding_id}/ai-explanation",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        # The endpoint's own contract (see backend/main.py) is to
        # catch everything and return a JSON envelope -- a raw 5xx
        # here would itself be a bug the tripwire's error handling
        # is supposed to prevent.
        assert response.status_code == 200, (
            f"{finding_id}: expected a handled 200 JSON envelope, got "
            f"{response.status_code}: {response.text}"
        )

        body = response.json()
        assert body.get("status") in ("success", "error"), (
            f"{finding_id}: unexpected response envelope: {body}"
        )

        outcome: dict[str, Any] = {
            "finding_id": finding_id,
            "control_id": finding.get("control_id"),
            "status": body["status"],
        }

        if body["status"] == "success":

            returned_explanation = body["finding"].get("ai_explanation")
            returned_recommendation = body["finding"].get(
                "ai_recommendation"
            )

            assert returned_explanation, (
                f"{finding_id}: success response has no ai_explanation"
            )
            assert returned_recommendation, (
                f"{finding_id}: success response has no ai_recommendation"
            )

            # --- INVARIANT 1: independently re-verify grounding ---
            fetch_response = requests.get(
                f"{API_BASE}/findings",
                params={"audit_run_id": finding["audit_run_id"]},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            assert fetch_response.status_code == 200
            persisted = next(
                f
                for f in fetch_response.json()["findings"]
                if f["finding_id"] == finding_id
            )
            persisted["ai_explanation"] = returned_explanation
            persisted["ai_recommendation"] = returned_recommendation

            residual_errors = _independent_hallucination_check(persisted)

            assert not residual_errors, (
                f"{finding_id}: API reported success, but an "
                "independent re-run of the real hallucination tripwire "
                f"against the persisted output found: {residual_errors}"
            )

            outcome["hallucination_residual_errors"] = []

        else:

            message = body.get("message", "")

            # --- INVARIANT 2a: failure is attributable to the
            #     tripwire, not an unrelated crash ---
            assert "HallucinationDetectedError" in message, (
                f"{finding_id}: endpoint failed for a reason other "
                f"than the hallucination tripwire: {message}"
            )

            # --- INVARIANT 2b: no half-written state persisted ---
            fetch_response = requests.get(
                f"{API_BASE}/findings",
                params={"audit_run_id": finding["audit_run_id"]},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            assert fetch_response.status_code == 200
            persisted = next(
                f
                for f in fetch_response.json()["findings"]
                if f["finding_id"] == finding_id
            )

            assert persisted.get("ai_explanation") == before_explanation, (
                f"{finding_id}: ai_explanation changed even though the "
                "tripwire rejected the output -- possible half-written "
                "state."
            )
            assert (
                persisted.get("ai_recommendation")
                == before_recommendation
            ), (
                f"{finding_id}: ai_recommendation changed even though "
                "the tripwire rejected the output -- possible "
                "half-written state."
            )

            outcome["hallucination_residual_errors"] = [message]

        outcomes.append(outcome)
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    succeeded = [o for o in outcomes if o["status"] == "success"]
    failed = [o for o in outcomes if o["status"] == "error"]

    print("\n=== Hallucination tripwire / API integration summary ===")
    print(f"Total findings tested: {len(outcomes)}")
    print(f"Succeeded (grounded, tripwire re-verified clean): {len(succeeded)}")
    print(f"Rejected by tripwire (handled gracefully, no bad persist): "
          f"{len(failed)}")
    for o in outcomes:
        print(f"  {o['finding_id']} [{o['control_id']}] -> {o['status']}")
    print("==========================================================")

    # The test's pass/fail criterion is the INVARIANT checked per
    # finding above, not the success rate -- a 100% success run and
    # a run with some real tripwire rejections are both a PASS as
    # long as every outcome satisfied its invariant. We only assert
    # here that at least one finding actually went through the real
    # pipeline, so a silently-empty run can't masquerade as a pass.
    assert outcomes, "no findings were actually tested"
