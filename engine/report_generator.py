"""
Layer 3 — Audit Report Generator.

Transforms audit findings into a structured summary.
"""

from collections import Counter


def generate_summary(findings):
    """
    Generate a high-level summary of audit findings.
    """

    control_counts = Counter(
        finding["control_id"]
        for finding in findings
    )

    severity_counts = Counter(
        finding["severity"]
        for finding in findings
    )

    assessment_counts = Counter(
        finding["assessment_status"]
        for finding in findings
    )

    finding_status_counts = Counter(
        finding["finding_status"]
        for finding in findings
    )

    return {
        "total_findings": len(findings),
        "control_counts": dict(control_counts),
        "severity_counts": dict(severity_counts),
        "assessment_status_counts": dict(assessment_counts),
        "finding_status_counts": dict(finding_status_counts),
    }


def print_summary(summary):
    """
    Print the audit summary in a readable format.
    """

    print("\n" + "=" * 60)
    print("AI-POWERED PRE-AUDIT RECONCILIATION PLATFORM")
    print("AUDIT SUMMARY")
    print("=" * 60)

    print(f"\nTotal Findings: {summary['total_findings']}")

    print("\nFindings by Control:")
    for control_id, count in summary["control_counts"].items():
        print(f"  {control_id}: {count}")

    print("\nFindings by Severity:")
    for severity, count in summary["severity_counts"].items():
        print(f"  {severity}: {count}")

    print("\nAssessment Status:")
    for status, count in summary["assessment_status_counts"].items():
        print(f"  {status}: {count}")

    print("\nFinding Status:")
    for status, count in summary["finding_status_counts"].items():
        print(f"  {status}: {count}")

    print("=" * 60)