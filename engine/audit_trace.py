"""
Audit Traceability.

Provides structured metadata describing one audit execution,
including lifecycle status and failure information.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditTrace:
    """
    Trace metadata for one complete audit run.

    The trace is intentionally independent from Supabase so the
    deterministic audit engine remains usable without persistence.
    """

    audit_run_id: str

    started_at: str
    completed_at: str | None = None

    status: str = "RUNNING"

    controls_executed: list[str] = field(default_factory=list)

    total_records_evaluated: int = 0
    total_findings_generated: int = 0

    error_type: str | None = None
    error_message: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


def create_audit_trace(
    audit_run_id: str,
    controls_executed: list[str],
    total_records_evaluated: int,
) -> AuditTrace:
    """
    Create a trace for a newly started audit run.

    A new audit always starts in RUNNING state.
    """

    return AuditTrace(
        audit_run_id=audit_run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        status="RUNNING",
        controls_executed=list(controls_executed),
        total_records_evaluated=total_records_evaluated,
    )


def complete_audit_trace(
    trace: AuditTrace,
    total_findings_generated: int,
) -> AuditTrace:
    """
    Mark an audit run as successfully completed.
    """

    trace.completed_at = datetime.now(timezone.utc).isoformat()
    trace.status = "COMPLETED"
    trace.total_findings_generated = total_findings_generated

    return trace


def fail_audit_trace(
    trace: AuditTrace,
    error: Exception,
) -> AuditTrace:
    """
    Mark an audit run as failed and record structured error metadata.

    The original exception is not re-raised here. The caller remains
    responsible for deciding whether the failure should propagate.
    """

    trace.completed_at = datetime.now(timezone.utc).isoformat()
    trace.status = "FAILED"
    trace.error_type = type(error).__name__
    trace.error_message = str(error)

    return trace