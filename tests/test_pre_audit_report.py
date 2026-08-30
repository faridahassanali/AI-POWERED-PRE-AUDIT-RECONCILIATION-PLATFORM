"""
Tests for engine.pre_audit_report.

Verifies that generate_pre_audit_report() produces the full README
Phase 7 report: audit summary, overall risk, control statistics,
severity statistics, review status, per-finding evidence/policy
references/AI explanation, and the delegated ground-truth evaluation
report -- built from the same canonical AuditOutput / EvaluationResult
objects the rest of the pipeline already trusts.
"""

import pytest

from engine.audit_output import build_audit_output
from engine.audit_trace import AuditTrace
from engine.ground_truth_evaluator import EvaluationResult, FindingKey
from engine.pre_audit_report import generate_pre_audit_report


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def audit_trace():
    return AuditTrace(
        audit_run_id="AUDIT-TEST-001",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:05:00+00:00",
        status="COMPLETED",
        controls_executed=["SCREENING_001", "RISK_001"],
        total_records_evaluated=100,
        total_findings_generated=2,
    )


@pytest.fixture
def finding_with_explanation():
    return {
        "finding_id": "F-AAA11111",
        "audit_run_id": "AUDIT-TEST-001",
        "control_id": "SCREENING_001",
        "customer_id": "CUST100005",
        "severity": "CRITICAL",
        "assessment_status": "FAIL",
        "finding_status": "CONFIRMED",
        "expected": "Opened wallets must have a CLEAR screening result.",
        "actual": "Screening status is HIGH_RISK.",
        "evidence": {
            "screening_status": "HIGH_RISK",
            "screening_evidence_present": True,
        },
        "policy_references": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        "reviewer_notes": "Escalated to compliance.",
    }


@pytest.fixture
def finding_without_explanation():
    return {
        "finding_id": "F-BBB22222",
        "audit_run_id": "AUDIT-TEST-001",
        "control_id": "RISK_001",
        "customer_id": "CUST100002",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": (
            "HIGH-risk opened wallets must have an approved "
            "risk exception."
        ),
        "actual": (
            "HIGH-risk opened wallet has no valid approved "
            "risk exception."
        ),
        "evidence": {
            "risk_level": "HIGH",
            "risk_exception_approved": "False",
        },
        "policy_references": [
            {
                "policy_id": "RISK-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
    }


@pytest.fixture
def explanation_for_first_finding():
    return {
        "finding_id": "F-AAA11111",
        "audit_run_id": "AUDIT-TEST-001",
        "control_id": "SCREENING_001",
        "customer_id": "CUST100005",
        "ai_explanation": (
            "The customer's wallet was opened without a CLEAR "
            "screening result, violating the screening policy."
        ),
        "ai_recommendation": (
            "Escalate to compliance for remediation."
        ),
    }


@pytest.fixture
def evaluation_result():
    matched = [FindingKey(control_id="SCREENING_001", customer_id="CUST100005")]

    return EvaluationResult(
        true_positives=1,
        false_positives=1,
        false_negatives=0,
        precision=0.5,
        recall=1.0,
        f1_score=0.6667,
        matched=matched,
        false_positive_findings=[
            FindingKey(control_id="RISK_001", customer_id="CUST100002")
        ],
        false_negative_findings=[],
        per_control={},
        per_severity={},
    )


@pytest.fixture
def audit_output(
    audit_trace,
    finding_with_explanation,
    finding_without_explanation,
    explanation_for_first_finding,
    evaluation_result,
):
    return build_audit_output(
        audit_trace=audit_trace,
        findings=[finding_with_explanation, finding_without_explanation],
        explanations=[explanation_for_first_finding],
        evaluation=evaluation_result,
        report="=== GROUND TRUTH EVALUATION REPORT ===",
    )


# =====================================================================
# SECTION PRESENCE
# =====================================================================

def test_report_contains_all_required_sections(audit_output):
    """
    Every section required by README Phase 7 must be present:
    audit summary, overall risk, findings, severity, evidence,
    policy references, recommendations, review status, control
    statistics.
    """

    report = generate_pre_audit_report(audit_output)

    assert "AUDIT SUMMARY" in report
    assert "OVERALL RISK" in report
    assert "CONTROL STATISTICS" in report
    assert "SEVERITY STATISTICS" in report
    assert "REVIEW STATUS" in report
    assert "FINDINGS" in report
    assert "GROUND TRUTH EVALUATION REPORT" in report


def test_report_is_a_single_string(audit_output):
    """The report is one exportable document, not a list/dict."""

    report = generate_pre_audit_report(audit_output)

    assert isinstance(report, str)


# =====================================================================
# AUDIT SUMMARY
# =====================================================================

def test_audit_summary_contains_run_metadata(audit_output):
    report = generate_pre_audit_report(audit_output)

    assert "AUDIT-TEST-001" in report
    assert "COMPLETED" in report
    assert "100" in report  # total_records_evaluated
    assert "Total Findings        : 2" in report


# =====================================================================
# OVERALL RISK
# =====================================================================

def test_overall_risk_is_critical_when_a_critical_finding_exists(
    audit_output,
):
    report = generate_pre_audit_report(audit_output)

    summary_section = report.split("OVERALL RISK")[1].split(
        "CONTROL STATISTICS"
    )[0]

    assert "CRITICAL" in summary_section


def test_overall_risk_is_no_findings_when_there_are_none(audit_trace):
    empty_output = build_audit_output(
        audit_trace=audit_trace,
        findings=[],
        explanations=[],
        evaluation=EvaluationResult(
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            matched=[],
            false_positive_findings=[],
            false_negative_findings=[],
            per_control={},
            per_severity={},
        ),
        report="=== GROUND TRUTH EVALUATION REPORT ===",
    )

    report = generate_pre_audit_report(empty_output)

    summary_section = report.split("OVERALL RISK")[1].split(
        "CONTROL STATISTICS"
    )[0]

    assert "NO FINDINGS" in summary_section


def test_overall_risk_is_high_when_no_critical_but_high_exists(
    audit_trace, finding_without_explanation, evaluation_result
):
    high_only_output = build_audit_output(
        audit_trace=audit_trace,
        findings=[finding_without_explanation],
        explanations=[],
        evaluation=evaluation_result,
        report="=== GROUND TRUTH EVALUATION REPORT ===",
    )

    report = generate_pre_audit_report(high_only_output)

    summary_section = report.split("OVERALL RISK")[1].split(
        "CONTROL STATISTICS"
    )[0]

    assert "HIGH" in summary_section
    assert "CRITICAL" not in summary_section


# =====================================================================
# STATISTICS SECTIONS
# =====================================================================

def test_control_statistics_count_findings_per_control(audit_output):
    report = generate_pre_audit_report(audit_output)

    stats_section = report.split("CONTROL STATISTICS")[1].split(
        "SEVERITY STATISTICS"
    )[0]

    assert "SCREENING_001: 1" in stats_section
    assert "RISK_001: 1" in stats_section


def test_severity_statistics_count_findings_per_severity(audit_output):
    report = generate_pre_audit_report(audit_output)

    stats_section = report.split("SEVERITY STATISTICS")[1].split(
        "REVIEW STATUS"
    )[0]

    assert "CRITICAL: 1" in stats_section
    assert "HIGH: 1" in stats_section


def test_review_status_statistics_count_findings_per_status(audit_output):
    report = generate_pre_audit_report(audit_output)

    stats_section = report.split("REVIEW STATUS")[1].split("FINDINGS")[0]

    assert "CONFIRMED: 1" in stats_section
    assert "REVIEW: 1" in stats_section


# =====================================================================
# PER-FINDING DETAIL: EVIDENCE, POLICY REFERENCES, AI EXPLANATION
# =====================================================================

def test_finding_section_includes_evidence(audit_output):
    report = generate_pre_audit_report(audit_output)

    assert "screening_status: HIGH_RISK" in report
    assert "risk_exception_approved: False" in report


def test_finding_section_includes_policy_references(audit_output):
    report = generate_pre_audit_report(audit_output)

    assert "SCREENING-POLICY-001" in report
    assert "RISK-POLICY-001" in report


def test_finding_with_explanation_includes_ai_explanation_and_recommendation(
    audit_output,
):
    report = generate_pre_audit_report(audit_output)

    assert (
        "The customer's wallet was opened without a CLEAR "
        "screening result" in report
    )
    assert "Escalate to compliance for remediation." in report


def test_finding_without_explanation_shows_pending_review(audit_output):
    report = generate_pre_audit_report(audit_output)

    assert "(pending human review)" in report


def test_finding_section_includes_reviewer_notes_when_present(
    audit_output,
):
    report = generate_pre_audit_report(audit_output)

    assert "Escalated to compliance." in report


def test_finding_section_omits_reviewer_notes_line_when_absent(
    audit_trace, finding_without_explanation, evaluation_result
):
    """finding_without_explanation has no reviewer_notes key at all."""

    output = build_audit_output(
        audit_trace=audit_trace,
        findings=[finding_without_explanation],
        explanations=[],
        evaluation=evaluation_result,
        report="=== GROUND TRUTH EVALUATION REPORT ===",
    )

    report = generate_pre_audit_report(output)

    assert "Reviewer Notes" not in report


# =====================================================================
# EMPTY FINDINGS
# =====================================================================

def test_report_handles_zero_findings_gracefully(audit_trace):
    empty_output = build_audit_output(
        audit_trace=audit_trace,
        findings=[],
        explanations=[],
        evaluation=EvaluationResult(
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            matched=[],
            false_positive_findings=[],
            false_negative_findings=[],
            per_control={},
            per_severity={},
        ),
        report="=== GROUND TRUTH EVALUATION REPORT ===",
    )

    report = generate_pre_audit_report(empty_output)

    findings_section = report.split("FINDINGS")[1].split(
        "GROUND TRUTH"
    )[0]

    assert "None" in findings_section


# =====================================================================
# DELEGATED EVALUATION REPORT
# =====================================================================

def test_ground_truth_report_is_appended_verbatim(audit_output):
    """
    The pre-audit report must not recompute or reformat the
    ground-truth evaluation -- it delegates to whatever
    audit_output.report already contains.
    """

    report = generate_pre_audit_report(audit_output)

    assert audit_output.report in report