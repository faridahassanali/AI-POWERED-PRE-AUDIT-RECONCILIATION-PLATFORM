"""
Layer 4 — Final Pre-Audit Report.

Builds the exportable, human-readable pre-audit report described
in README Phase 7:

    - Audit summary
    - Overall risk
    - Findings
    - Severity
    - Evidence
    - Policy references
    - Recommendations
    - Review status
    - Control statistics

This is presentation-only. It does not recompute anything --
it consumes the already-built AuditOutput (engine.audit_output)
and EvaluationResult (engine.ground_truth_evaluator), the same
canonical objects the rest of the pipeline already trusts.
"""

from collections import Counter
from typing import Any

from engine.audit_output import AuditOutput
from engine.evaluation_report import format_percentage


def _overall_risk_label(findings_by_severity: dict[str, int]) -> str:
    """
    Derive a single overall-risk label from the severity breakdown.

    CRITICAL/HIGH findings present at all => at least that level.
    No findings at all => "NO FINDINGS".
    """

    if not findings_by_severity:
        return "NO FINDINGS"

    if findings_by_severity.get("CRITICAL", 0) > 0:
        return "CRITICAL"

    if findings_by_severity.get("HIGH", 0) > 0:
        return "HIGH"

    if findings_by_severity.get("MEDIUM", 0) > 0:
        return "MEDIUM"

    return "LOW"


def _control_statistics(findings: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(finding["control_id"] for finding in findings)
    )


def _severity_statistics(findings: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(finding["severity"] for finding in findings)
    )


def _finding_status_statistics(
    findings: list[dict[str, Any]],
) -> dict[str, int]:
    return dict(
        Counter(finding["finding_status"] for finding in findings)
    )


def _format_finding_section(paired) -> list[str]:
    """
    Render one AuditFindingOutput (finding + optional explanation)
    as a self-contained block of the report.
    """

    finding = paired.finding
    explanation = paired.explanation

    lines: list[str] = []

    lines.append(
        f"[{finding.get('severity', 'UNKNOWN')}] "
        f"{finding.get('control_id', '')} / "
        f"{finding.get('customer_id') or 'N/A'} "
        f"(finding_id={finding.get('finding_id', '')})"
    )

    lines.append(f"  Status        : {finding.get('finding_status', '')}")
    lines.append(f"  Expected      : {finding.get('expected', '')}")
    lines.append(f"  Actual        : {finding.get('actual', '')}")

    evidence = finding.get("evidence") or {}
    lines.append("  Evidence      :")
    if evidence:
        for key, value in evidence.items():
            lines.append(f"    - {key}: {value}")
    else:
        lines.append("    None")

    policy_references = finding.get("policy_references") or []
    lines.append("  Policy References:")
    if policy_references:
        for ref in policy_references:
            lines.append(
                f"    - {ref.get('policy_id', '')} "
                f"v{ref.get('version', '')} "
                f"({ref.get('section', '')})"
            )
    else:
        lines.append("    None")

    if finding.get("reviewer_notes"):
        lines.append(f"  Reviewer Notes: {finding['reviewer_notes']}")

    if explanation is not None:
        lines.append(
            f"  AI Explanation: "
            f"{explanation.get('ai_explanation', '')}"
        )
        lines.append(
            f"  AI Recommendation: "
            f"{explanation.get('ai_recommendation', '')}"
        )
    else:
        lines.append("  AI Explanation: (pending human review)")

    return lines


def generate_pre_audit_report(audit_output: AuditOutput) -> str:
    """
    Build the final, exportable pre-audit report (README Phase 7).

    Sections, in order:
        1. Audit summary (run id, trace status, record/finding counts)
        2. Overall risk
        3. Control statistics
        4. Severity statistics
        5. Review status statistics
        6. Findings (each with evidence, policy references,
           review status, and AI explanation/recommendation if any)
        7. Ground-truth evaluation report (delegated, unchanged)
    """

    raw_findings = [paired.finding for paired in audit_output.findings]

    control_stats = _control_statistics(raw_findings)
    severity_stats = _severity_statistics(raw_findings)
    status_stats = _finding_status_statistics(raw_findings)
    overall_risk = _overall_risk_label(severity_stats)

    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("AI-POWERED PRE-AUDIT RECONCILIATION PLATFORM")
    lines.append("FINAL PRE-AUDIT REPORT")
    lines.append("=" * 60)

    # ---------------------------------------------------------
    # 1. Audit summary
    # ---------------------------------------------------------
    trace = audit_output.audit_trace
    lines.append("")
    lines.append("AUDIT SUMMARY")
    lines.append("-" * 60)
    lines.append(f"Audit Run ID          : {audit_output.audit_run_id}")
    lines.append(f"Status                : {getattr(trace, 'status', 'UNKNOWN')}")
    lines.append(f"Started At            : {getattr(trace, 'started_at', '')}")
    lines.append(f"Completed At          : {getattr(trace, 'completed_at', '')}")
    lines.append(
        f"Total Records Evaluated: "
        f"{getattr(trace, 'total_records_evaluated', 0)}"
    )
    lines.append(f"Total Findings        : {len(raw_findings)}")

    # ---------------------------------------------------------
    # 2. Overall risk
    # ---------------------------------------------------------
    lines.append("")
    lines.append("OVERALL RISK")
    lines.append("-" * 60)
    lines.append(overall_risk)

    # ---------------------------------------------------------
    # 3. Control statistics
    # ---------------------------------------------------------
    lines.append("")
    lines.append("CONTROL STATISTICS")
    lines.append("-" * 60)
    if control_stats:
        for control_id, count in control_stats.items():
            lines.append(f"  {control_id}: {count}")
    else:
        lines.append("None")

    # ---------------------------------------------------------
    # 4. Severity statistics
    # ---------------------------------------------------------
    lines.append("")
    lines.append("SEVERITY STATISTICS")
    lines.append("-" * 60)
    if severity_stats:
        for severity, count in severity_stats.items():
            lines.append(f"  {severity}: {count}")
    else:
        lines.append("None")

    # ---------------------------------------------------------
    # 5. Review status statistics
    # ---------------------------------------------------------
    lines.append("")
    lines.append("REVIEW STATUS")
    lines.append("-" * 60)
    if status_stats:
        for status, count in status_stats.items():
            lines.append(f"  {status}: {count}")
    else:
        lines.append("None")

    # ---------------------------------------------------------
    # 6. Findings (full detail)
    # ---------------------------------------------------------
    lines.append("")
    lines.append("FINDINGS")
    lines.append("-" * 60)
    if audit_output.findings:
        for paired in audit_output.findings:
            lines.append("")
            lines.extend(_format_finding_section(paired))
    else:
        lines.append("None")

    # ---------------------------------------------------------
    # 7. Ground-truth evaluation (delegated, unchanged)
    # ---------------------------------------------------------
    lines.append("")
    lines.append(audit_output.report)

    return "\n".join(lines)