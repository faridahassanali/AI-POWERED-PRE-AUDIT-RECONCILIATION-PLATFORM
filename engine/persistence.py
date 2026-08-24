"""
Supabase Persistence Layer.

Responsible for saving and retrieving audit data.

Tables used:

    public.audit_runs
    public.findings
    public.finding_reviews

The deterministic audit pipeline does NOT depend on this module.
The backend calls these functions when persistence is required.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = Any


class PersistenceNotConfigured(RuntimeError):
    """Raised when Supabase is not configured correctly."""


# =========================================================
# SUPABASE CLIENT
# =========================================================

def get_supabase_client() -> "Client":
    """
    Create a Supabase client using environment variables.

    Required:

        SUPABASE_URL
        SUPABASE_SERVICE_ROLE_KEY
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
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "must be set as environment variables."
        )

    return create_client(url, key)


# =========================================================
# HELPERS
# =========================================================

def _trace_to_dict(
    audit_trace: Any,
) -> dict[str, Any]:
    """
    Convert an AuditTrace dataclass or dictionary
    into a normal dictionary.
    """

    if is_dataclass(audit_trace):
        return asdict(audit_trace)

    if isinstance(audit_trace, dict):
        return audit_trace

    raise TypeError(
        f"Unsupported audit_trace type: "
        f"{type(audit_trace)!r}"
    )


def _audit_run_row(
    trace: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert audit trace into a database row.
    """

    return {
        "audit_run_id": trace["audit_run_id"],
        "started_at": trace["started_at"],
        "completed_at": trace.get("completed_at"),
        "controls_executed": trace.get(
            "controls_executed",
            [],
        ),
        "total_records_evaluated": trace.get(
            "total_records_evaluated",
            0,
        ),
        "total_findings_generated": trace.get(
            "total_findings_generated",
            0,
        ),
    }


def _finding_row(
    finding: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a finding into a database row.
    """

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
        "evidence": finding.get(
            "evidence",
            {},
        ),
        "policy_references": finding.get(
            "policy_references",
            [],
        ),
        "reviewed_by": finding.get(
            "reviewed_by"
        ),
        "review_timestamp": finding.get(
            "review_timestamp"
        ),
        "reviewer_notes": finding.get(
            "reviewer_notes"
        ),
        "ai_explanation": finding.get(
            "ai_explanation"
        ),
        "ai_recommendation": finding.get(
            "ai_recommendation"
        ),
    }


# =========================================================
# AUDIT RUNS
# =========================================================

def write_audit_run(
    audit_trace: Any,
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Insert or update one audit run.
    """

    client = client or get_supabase_client()

    trace = _trace_to_dict(
        audit_trace
    )

    row = _audit_run_row(
        trace
    )

    response = (
        client
        .table("audit_runs")
        .upsert(
            row,
            on_conflict="audit_run_id",
        )
        .execute()
    )

    return response.data or []


# =========================================================
# FINDINGS
# =========================================================

def write_findings(
    findings: list[dict[str, Any]],
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Insert or update findings.

    If the list is empty, nothing is written.
    """

    if not findings:
        return []

    client = client or get_supabase_client()

    rows = [
        _finding_row(finding)
        for finding in findings
    ]

    response = (
        client
        .table("findings")
        .upsert(
            rows,
            on_conflict="finding_id",
        )
        .execute()
    )

    return response.data or []


# =========================================================
# FINDING REVIEWS
# =========================================================

def write_finding_review(
    finding: dict[str, Any],
    previous_status: str,
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Insert one review-history record.

    Example:

        REVIEW
           ↓
        CONFIRMED
    """

    client = client or get_supabase_client()

    row = {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "previous_status": previous_status,
        "new_status": finding["finding_status"],
        "reviewed_by": finding.get("reviewed_by"),
        "reviewer_notes": finding.get(
            "reviewer_notes"
        ),
    }

    response = (
        client
        .table("finding_reviews")
        .insert(row)
        .execute()
    )

    if not response.data:
        return {}

    return response.data[0]


def create_finding_review(
    finding_id: str,
    audit_run_id: str,
    previous_status: str,
    new_status: str,
    reviewed_by: str | None = None,
    reviewer_notes: str | None = None,
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Create one finding review-history record.

    This function is used directly by the FastAPI backend.
    """

    client = client or get_supabase_client()

    row = {
        "finding_id": finding_id,
        "audit_run_id": audit_run_id,
        "previous_status": previous_status,
        "new_status": new_status,
        "reviewed_by": reviewed_by,
        "reviewer_notes": reviewer_notes,
    }

    response = (
        client
        .table("finding_reviews")
        .insert(row)
        .execute()
    )

    if not response.data:
        return {}

    return response.data[0]


def get_finding_reviews(
    finding_id: str,
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Get all review-history records for one finding.
    """

    client = client or get_supabase_client()

    response = (
        client
        .table("finding_reviews")
        .select("*")
        .eq(
            "finding_id",
            finding_id,
        )
        .order(
            "reviewed_at",
            desc=True,
        )
        .execute()
    )

    return response.data or []



# =========================================================
# AUDIT EVALUATION
# =========================================================

def write_audit_evaluation(
    evaluation: Any,
    audit_run_id: str,
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Save audit evaluation metrics for one audit run.

    Stores the ground-truth comparison results:
        TP
        FP
        FN
        precision
        recall
        F1

    If the evaluation is a dataclass, convert it to a dictionary.
    """

    client = client or get_supabase_client()

    if is_dataclass(evaluation):
        evaluation = asdict(evaluation)

    if not isinstance(evaluation, dict):
        raise TypeError(
            f"Unsupported evaluation type: "
            f"{type(evaluation)!r}"
        )

    row = {
        "audit_run_id": audit_run_id,
        "true_positives": evaluation.get(
            "true_positives",
            evaluation.get("tp", 0),
        ),
        "false_positives": evaluation.get(
            "false_positives",
            evaluation.get("fp", 0),
        ),
        "false_negatives": evaluation.get(
            "false_negatives",
            evaluation.get("fn", 0),
        ),
        "precision": evaluation.get("precision", 0),
        "recall": evaluation.get("recall", 0),
        "f1": evaluation.get(
            "f1",
            evaluation.get("f1_score", 0),
        ),
    }

    response = (
        client
        .table("audit_evaluations")
        .upsert(
            row,
            on_conflict="audit_run_id",
        )
        .execute()
    )

    return response.data or []

# =========================================================
# AI OUTPUT
# =========================================================

def write_ai_output(
    finding: dict[str, Any],
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Save the AI-generated explanation and recommendation
    for one finding.

    The AI pipeline stores the provider/model and retrieved
    policy context in internal underscore-prefixed fields:

        _ai_model_used
        _ai_policy_context

    These are persisted separately from the public findings row.
    """

    client = client or get_supabase_client()

    row = {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "model_name": finding.get("_ai_model_used"),
        "ai_explanation": finding.get("ai_explanation"),
        "ai_recommendation": finding.get("ai_recommendation"),
        "retrieved_policy_context": finding.get(
            "_ai_policy_context",
            [],
        ),
    }

    response = (
        client
        .table("ai_outputs")
        .upsert(
            row,
            on_conflict="finding_id",
        )
        .execute()
    )

    return response.data or []

    # =========================================================
# FINDING EXPLANATIONS
# =========================================================

def write_finding_explanation(
    finding_id: str,
    explanation: dict[str, Any],
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Save the deterministic explanation for one finding.

    The explanation is stored in the findings table.
    """

    client = client or get_supabase_client()

    row = {
        "finding_id": finding_id,
        "explanation": explanation,
    }

    response = (
        client
        .table("findings")
        .update(row)
        .eq(
            "finding_id",
            finding_id,
        )
        .execute()
    )

    if not response.data:
        return {}

    return response.data[0]