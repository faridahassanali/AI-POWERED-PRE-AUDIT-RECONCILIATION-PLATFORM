"""
Supabase Persistence Layer.

Functions to write pipeline artifacts to Supabase:

    - write_audit_run()        -> public.audit_runs
    - write_findings()         -> public.findings
    - write_finding_review()   -> public.finding_reviews
    - write_audit_evaluation() -> public.audit_evaluations
    - write_ai_output()        -> public.ai_outputs

Design principle preserved
---------------------------
The deterministic audit pipeline must keep working with ZERO
Supabase dependency (see test_ai_layer_must_not_be_required_for_
deterministic_audit in tests/test_pre_ai_layer.py). This module is
meant to be called explicitly by whatever orchestrates persistence
(engine.audit_orchestration, app.py) — never imported by
engine.audit_pipeline itself.

Credentials
-----------
Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from the
environment. The service_role key is required (not the anon key):
RLS is enabled on every table with no policies defined yet, so only
service_role can currently read/write.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from supabase import create_client, Client
except ImportError:  # pragma: no cover - exercised when dependency missing
    create_client = None
    Client = Any  # type: ignore[assignment]


class PersistenceNotConfigured(RuntimeError):
    """Raised when Supabase credentials or the client library are missing."""


# =====================================================================
# CLIENT
# =====================================================================

def get_supabase_client() -> "Client":
    """
    Build a Supabase client from environment variables.

    Raises PersistenceNotConfigured if the 'supabase' package isn't
    installed or SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are missing.
    Callers that want the pipeline to keep running without a database
    should catch this and skip persistence, not let it propagate.
    """

    if create_client is None:
        raise PersistenceNotConfigured(
            "The 'supabase' package is not installed. "
            "Run: pip install supabase"
        )

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise PersistenceNotConfigured(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
            "as environment variables."
        )

    return create_client(url, key)


# =====================================================================
# HELPERS
# =====================================================================

def _trace_to_dict(audit_trace: Any) -> dict[str, Any]:
    """Accept either the AuditTrace dataclass or a plain dict."""

    if is_dataclass(audit_trace):
        return asdict(audit_trace)

    if isinstance(audit_trace, dict):
        return audit_trace

    raise TypeError(
        f"Unsupported audit_trace type: {type(audit_trace)!r}"
    )


def _audit_run_row(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_run_id": trace["audit_run_id"],
        "started_at": trace["started_at"],
        "completed_at": trace.get("completed_at"),
        "controls_executed": trace.get("controls_executed", []),
        "total_records_evaluated": trace.get("total_records_evaluated", 0),
        "total_findings_generated": trace.get("total_findings_generated", 0),
    }


def _finding_row(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "control_id": finding["control_id"],
        "customer_id": finding.get("customer_id"),
        "severity": finding["severity"],
        "assessment_status": finding["assessment_status"],
        "finding_status": finding["finding_status"],
        "expected": finding["expected"],
        "actual": finding["actual"],
        "evidence": finding.get("evidence", {}),
        "policy_references": finding.get("policy_references", []),
        "reviewed_by": finding.get("reviewed_by"),
        "review_timestamp": finding.get("review_timestamp"),
        "reviewer_notes": finding.get("reviewer_notes"),
        "ai_explanation": finding.get("ai_explanation"),
        "ai_recommendation": finding.get("ai_recommendation"),
    }


def _finding_review_row(
    finding: dict[str, Any],
    previous_status: str,
) -> dict[str, Any]:
    return {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "previous_status": previous_status,
        "new_status": finding["finding_status"],
        "reviewed_by": finding["reviewed_by"],
        "reviewer_notes": finding.get("reviewer_notes"),
    }


def _ai_output_row(
    finding: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the public.ai_outputs row for a finding that has already
    had ai_explanation/ai_recommendation attached (by
    engine.ai_explanation_pipeline.generate_ai_explanation_for_finding()).

    model_name / retrieved_policy_context are read from the
    underscore-prefixed keys that generate_ai_explanation_for_finding()
    stashes on the finding for this exact purpose. They are NOT part
    of the finding_schema.json contract and are never sent to
    write_findings()/_finding_row() -- purely an internal handoff to
    this function.
    """
    return {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "ai_explanation": finding.get("ai_explanation"),
        "ai_recommendation": finding.get("ai_recommendation"),
        "model_name": finding.get("_ai_model_used"),
        "prompt_version": finding.get("_ai_prompt_version", "v1"),
        "retrieved_policy_context": finding.get("_ai_policy_context", []),
    }


def _evaluation_to_row(
    evaluation: Any,
    audit_run_id: str,
) -> dict[str, Any]:
    """Normalize an evaluation object into the audit_evaluations DB row."""
    if is_dataclass(evaluation):
        evaluation = asdict(evaluation)
    elif isinstance(evaluation, dict):
        evaluation = dict(evaluation)
    else:
        raise TypeError(
            f"Unsupported evaluation type: {type(evaluation)!r}"
        )

    evaluation_run_id = evaluation.get("audit_run_id")

    if evaluation_run_id is not None and evaluation_run_id != audit_run_id:
        raise ValueError(
            "Evaluation audit_run_id does not match the audit run."
        )

    return {
        "audit_run_id": audit_run_id,
        "true_positives": evaluation.get("true_positives", 0),
        "false_positives": evaluation.get("false_positives", 0),
        "false_negatives": evaluation.get("false_negatives", 0),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        # EvaluationResult's real field is "f1_score" -- fall back to
        # "f1" too in case a plain dict ever uses the shorter key.
        "f1_score": evaluation.get("f1_score", evaluation.get("f1")),
        "report": evaluation.get("report"),
    }


# =====================================================================
# WRITE FUNCTIONS
# =====================================================================

def write_audit_run(
    audit_trace: Any,
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Upsert one row into public.audit_runs.

    Upsert (not insert) so this is safe to call again later, e.g.
    once completed_at / total_findings_generated are known.
    """

    client = client or get_supabase_client()
    trace = _trace_to_dict(audit_trace)
    row = _audit_run_row(trace)

    response = (
        client.table("audit_runs")
        .upsert(row, on_conflict="audit_run_id")
        .execute()
    )
    return response.data


def write_findings(
    findings: list[dict[str, Any]],
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Upsert findings into public.findings.

    Upsert on finding_id so the SAME function covers both:
      - the first write, right after the deterministic stage
        (finding_status == "REVIEW")
      - the re-write after a human review decision
        (finding_status == "CONFIRMED" / "REJECTED")
      - the re-write after an AI explanation is attached
        (ai_explanation / ai_recommendation populated)

    Does nothing (returns []) if findings is empty — callers don't
    need to guard against an empty list themselves.
    """

    if not findings:
        return []

    client = client or get_supabase_client()
    rows = [_finding_row(f) for f in findings]

    response = (
        client.table("findings")
        .upsert(rows, on_conflict="finding_id")
        .execute()
    )
    return response.data


def write_finding_review(
    finding: dict[str, Any],
    previous_status: str,
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Insert one audit-trail row into public.finding_reviews.

    Call this AFTER engine.finding_review.confirm_finding() /
    reject_finding() has already mutated `finding` in memory —
    this function only records the decision, it does not make it.

    previous_status is passed explicitly (rather than re-derived)
    because by the time this is called, `finding["finding_status"]`
    already holds the NEW status — the old one is gone from the dict.
    """

    client = client or get_supabase_client()
    row = _finding_review_row(finding, previous_status)

    response = (
        client.table("finding_reviews")
        .insert(row)
        .execute()
    )
    return response.data


def write_audit_evaluation(
    evaluation: Any,
    audit_run_id: str,
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Upsert the ground-truth evaluation metrics (TP/FP/FN,
    precision/recall/F1) for one audit run into
    public.audit_evaluations.

    Accepts either the EvaluationResult dataclass
    (engine.ground_truth_evaluator) or an equivalent plain dict.
    """

    if not audit_run_id:
        raise ValueError("audit_run_id is required.")

    client = client or get_supabase_client()
    row = _evaluation_to_row(evaluation, audit_run_id)

    response = (
        client.table("audit_evaluations")
        .upsert(row, on_conflict="audit_run_id")
        .execute()
    )

    return response.data


def write_ai_output(
    finding: dict[str, Any],
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Insert one row into public.ai_outputs.

    Call this AFTER
    engine.ai_explanation_pipeline.generate_ai_explanation_for_finding()
    has already succeeded and attached ai_explanation /
    ai_recommendation (and the internal _ai_* metadata keys) to
    `finding`. Raises ValueError if the finding has no
    ai_explanation yet, since an empty ai_outputs row would be
    meaningless.
    """

    if not finding.get("ai_explanation"):
        raise ValueError(
            "Finding has no ai_explanation to persist. Call this "
            "only after generate_ai_explanation_for_finding() has "
            "succeeded."
        )

    client = client or get_supabase_client()
    row = _ai_output_row(finding)

    response = (
        client.table("ai_outputs")
        .insert(row)
        .execute()
    )
    return response.data