"""
Supabase Persistence Layer.

Functions to write pipeline artifacts to Supabase:

    - write_audit_run()      -> public.audit_runs
    - write_findings()       -> public.findings
    - write_finding_review() -> public.finding_reviews

This module is intentionally separate from the deterministic
audit pipeline.

The audit pipeline can run without Supabase.

The backend/orchestrator can explicitly call these functions
when persistence is required.

Credentials
-----------
Reads:

    SUPABASE_URL
    SUPABASE_KEY

from the environment.

The SUPABASE_KEY used by this project is the service-role key.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from supabase import create_client, Client
except ImportError:
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

    Raises PersistenceNotConfigured if the Supabase package is missing
    or the required environment variables are not configured.
    """

    if create_client is None:
        raise PersistenceNotConfigured(
            "The 'supabase' package is not installed. "
            "Run: pip install supabase"
        )

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise PersistenceNotConfigured(
            "SUPABASE_URL and SUPABASE_KEY must be set "
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


# =====================================================================
# WRITE FUNCTIONS
# =====================================================================

def write_audit_run(
    audit_trace: Any,
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Upsert one row into public.audit_runs.

    Upsert is used so the same audit run can be updated later
    when completed_at and final finding counts are available.
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

    return response.data


def write_findings(
    findings: list[dict[str, Any]],
    client: "Client | None" = None,
) -> list[dict[str, Any]]:
    """
    Upsert findings into public.findings.

    Does nothing when findings is empty.
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

    This function records the review decision.
    It does not change the finding itself.
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