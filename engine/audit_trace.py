"""
Audit Traceability.

Provides structured metadata describing one audit execution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditTrace:
    """
    Trace metadata for one complete audit run.
    """

    audit_run_id: str

    started_at: str
    completed_at: str | None = None

    controls_executed: list[str] = field(default_factory=list)

    total_records_evaluated: int = 0
    total_findings_generated: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


def create_audit_trace(
    audit_run_id: str,
    controls_executed: list[str],
    total_records_evaluated: int,
) -> AuditTrace:

    return AuditTrace(
        audit_run_id=audit_run_id,
        started_at=datetime.now(
            timezone.utc
        ).isoformat(),
        controls_executed=controls_executed,
        total_records_evaluated=total_records_evaluated,
    )


def complete_audit_trace(
    trace: AuditTrace,
    total_findings_generated: int,
) -> AuditTrace:

    trace.completed_at = datetime.now(
        timezone.utc
    ).isoformat()

    trace.total_findings_generated = (
        total_findings_generated
    )

    return trace