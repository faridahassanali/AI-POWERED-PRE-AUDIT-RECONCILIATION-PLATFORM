"""
Supabase Persistence Layer.

Functions to write pipeline artifacts to Supabase:
    - write_audit_run()           -> public.audit_runs
    - write_findings()            -> public.findings
    - write_finding_review()      -> public.finding_reviews
    - write_audit_evaluation()    -> public.audit_evaluations
    - write_finding_explanation() -> public.finding_explanations
    - write_ai_output()           -> public.ai_outputs

Design principle
----------------
The deterministic audit pipeline must keep working with ZERO
Supabase dependency.

This module is called explicitly by orchestration/application code.
It must never be required by engine.audit_pipeline itself.

Credentials
-----------
Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from environment
variables.

The service-role key is required because RLS is enabled on the
persistence tables and service-role access is currently used by the
backend persistence layer.

IMPORTANT:
This module intentionally does NOT call load_dotenv().

Runtime configuration must come from environment variables supplied
by the application/deployment environment. Loading .env implicitly
inside the persistence layer makes configuration behavior difficult
to test and unsafe to reason about in production.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - optional dependency
    create_client = None
    Client = Any  # type: ignore[assignment]


class PersistenceNotConfigured(RuntimeError):
    """
    Raised when Supabase persistence cannot be configured.
    """


# =====================================================================
# CLIENT
# =====================================================================


def get_supabase_client() -> "Client":
    """
    Build a Supabase client from runtime environment variables.

    Raises
    ------
    PersistenceNotConfigured
        If the Supabase package is unavailable or the required
        environment variables are missing.

    Notes
    -----
    This function intentionally reads only from os.environ.

    It does NOT implicitly load .env files. Local development may load
    environment variables explicitly at the application entrypoint,
    while production should provide them through the deployment
    environment / secret manager.
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
    """
    Accept either the AuditTrace dataclass or a plain dictionary.
    """

    if is_dataclass(audit_trace):
        return asdict(audit_trace)

    if isinstance(audit_trace, dict):
        return audit_trace

    raise TypeError(
        f"Unsupported audit_trace type: {type(audit_trace)!r}"
    )


def _audit_run_row(
    trace: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert an audit trace into a database row.
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
    Convert a finding dictionary into a database row.
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
        "reviewed_by": finding.get("reviewed_by"),
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


def _finding_review_row(
    finding: dict[str, Any],
    previous_status: str,
) -> dict[str, Any]:
    """
    Convert a reviewed finding into a finding_reviews row.
    """

    return {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "previous_status": previous_status,
        "new_status": finding["finding_status"],
        "reviewed_by": finding["reviewed_by"],
        "reviewer_notes": finding.get(
            "reviewer_notes"
        ),
    }


def _finding_explanation_row(
    explanation: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the public.finding_explanations row from the output of
    engine.finding_explainer.explain_finding().

    This is the deterministic Stage 2 explanation and is separate
    from the LLM-generated explanation stored in public.ai_outputs.
    """

    return {
        "finding_id": explanation["finding_id"],
        "audit_run_id": explanation["audit_run_id"],
        "control_id": explanation["control_id"],
        "customer_id": explanation.get("customer_id"),
        "severity": explanation["severity"],
        "assessment_status": explanation["assessment_status"],
        "finding_status": explanation["finding_status"],
        "summary": explanation["summary"],
        "expected_condition": explanation["expected_condition"],
        "observed_condition": explanation["observed_condition"],
        "evidence": explanation.get("evidence", {}),
        "policy_references": explanation.get(
            "policy_references",
            [],
        ),
        "review_action": explanation.get(
            "review_action"
        ),
    }


def _ai_output_row(
    finding: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the public.ai_outputs row for a finding that has already
    received an AI explanation/recommendation.
    """

    return {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "ai_explanation": finding.get("ai_explanation"),
        "ai_recommendation": finding.get("ai_recommendation"),
        "model_name": finding.get("_ai_model_used"),
        "prompt_version": finding.get(
            "_ai_prompt_version",
            "v1",
        ),
        "retrieved_policy_context": finding.get(
            "_ai_policy_context",
            [],
        ),
    }


def _evaluation_to_row(
    evaluation: Any,
    audit_run_id: str,
) -> dict[str, Any]:
    """
    Normalize an evaluation object into the audit_evaluations DB row.
    """

    if is_dataclass(evaluation):
        evaluation = asdict(evaluation)

    elif isinstance(evaluation, dict):
        evaluation = dict(evaluation)

    else:
        raise TypeError(
            f"Unsupported evaluation type: {type(evaluation)!r}"
        )

    evaluation_run_id = evaluation.get("audit_run_id")

    if (
        evaluation_run_id is not None
        and evaluation_run_id != audit_run_id
    ):
        raise ValueError(
            "Evaluation audit_run_id does not match the audit run."
        )

    return {
        "audit_run_id": audit_run_id,
        "true_positives": evaluation.get(
            "true_positives",
            0,
        ),
        "false_positives": evaluation.get(
            "false_positives",
            0,
        ),
        "false_negatives": evaluation.get(
            "false_negatives",
            0,
        ),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1_score": evaluation.get(
            "f1_score",
            evaluation.get("f1"),
        ),
        "report": evaluation.get("report"),
    }


# =====================================================================
# WRITE FUNCTIONS
# =====================================================================


def write_audit_run(
    audit_trace: Any,
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Upsert one row into public.audit_runs.

    Upsert is used so the same audit run can be updated later when
    completed_at and final finding counts are available.
    """

    client = client or get_supabase_client()

    trace = _trace_to_dict(audit_trace)

    row = _audit_run_row(trace)

    response = (
        client
        .table("audit_runs")
        .upsert(
            row,
            on_conflict="audit_run_id",
        )
        .execute()
    )

    return response.data


def write_findings(
    findings: list[dict[str, Any]],
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Upsert findings into public.findings.

    Upsert on finding_id supports:

        - initial REVIEW write
        - CONFIRMED / REJECTED review update
        - later AI explanation update

    Empty finding lists are treated as a no-op.
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

    return response.data


def write_finding_review(
    finding: dict[str, Any],
    previous_status: str,
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Insert one review decision into public.finding_reviews.

    This function records the review event. It does not modify the
    finding itself.
    """

    client = client or get_supabase_client()

    row = _finding_review_row(
        finding,
        previous_status,
    )

    response = (
        client
        .table("finding_reviews")
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
    Upsert ground-truth evaluation metrics for one audit run.
    """

    if not audit_run_id:
        raise ValueError("audit_run_id is required.")

    client = client or get_supabase_client()

    row = _evaluation_to_row(
        evaluation,
        audit_run_id,
    )

    response = (
        client
        .table("audit_evaluations")
        .upsert(
            row,
            on_conflict="audit_run_id",
        )
        .execute()
    )

    return response.data


def write_finding_explanation(
    explanation: dict[str, Any],
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Upsert one deterministic finding explanation.
    """

    client = client or get_supabase_client()

    row = _finding_explanation_row(explanation)

    response = (
        client
        .table("finding_explanations")
        .upsert(
            row,
            on_conflict="finding_id",
        )
        .execute()
    )

    return response.data


def write_ai_output(
    finding: dict[str, Any],
    client: "Client | None" = None,
) -> dict[str, Any]:
    """
    Insert one AI output after AI explanation generation succeeds.

    Raises ValueError if no AI explanation is attached to the finding.
    """

    if not finding.get("ai_explanation"):
        raise ValueError(
            "Finding has no ai_explanation to persist. "
            "Call this only after "
            "generate_ai_explanation_for_finding() has succeeded."
        )

    client = client or get_supabase_client()

    row = _ai_output_row(finding)

    response = (
        client
        .table("ai_outputs")
        .insert(row)
        .execute()
    )

    return response.data