"""
Ground Truth Evaluator.

Compares generated audit findings against the synthetic
expected_findings.csv ground truth.

The evaluator does NOT compare generated finding IDs because
finding_id is generated dynamically by the Finding Builder.

A finding is identified by:
    control_id + customer_id

The evaluator reports:
- True Positives
- False Positives
- False Negatives
- Precision
- Recall
- F1 Score
- Per-control metrics
- Per-severity metrics
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FindingKey:
    """
    Stable identity of a logical audit finding.
    """

    control_id: str
    customer_id: str | None


@dataclass
class EvaluationResult:
    """
    Evaluation result for generated findings vs ground truth.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    precision: float
    recall: float
    f1_score: float

    matched: list[FindingKey]
    false_positive_findings: list[FindingKey]
    false_negative_findings: list[FindingKey]

    per_control: dict[str, dict[str, Any]]
    per_severity: dict[str, dict[str, Any]]


def _finding_key(finding: dict[str, Any]) -> FindingKey:
    """
    Build the stable identity of a finding.

    finding_id is intentionally ignored because it is generated
    dynamically and cannot be used as a ground-truth identity.
    """

    return FindingKey(
        control_id=str(finding["control_id"]),
        customer_id=(
            None
            if finding.get("customer_id") is None
            else str(finding["customer_id"])
        ),
    )


def _safe_divide(numerator: int, denominator: int) -> float:
    """
    Safe division used for evaluation metrics.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _calculate_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    """
    Calculate precision, recall and F1.
    """

    precision = _safe_divide(
        true_positives,
        true_positives + false_positives,
    )

    recall = _safe_divide(
        true_positives,
        true_positives + false_negatives,
    )

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def evaluate_findings(
    generated_findings: list[dict[str, Any]],
    expected_findings: list[dict[str, Any]],
) -> EvaluationResult:
    """
    Compare generated findings against expected ground truth.

    Matching is based on:

        control_id + customer_id

    not on finding_id.

    Duplicate logical findings are treated as duplicates and
    should not artificially increase evaluation performance.
    """

    generated_keys = [
        _finding_key(finding)
        for finding in generated_findings
    ]

    expected_keys = [
        _finding_key(finding)
        for finding in expected_findings
    ]

    generated_set = set(generated_keys)
    expected_set = set(expected_keys)

    matched = sorted(
        generated_set & expected_set,
        key=lambda key: (
            key.control_id,
            key.customer_id or "",
        ),
    )

    false_positive_findings = sorted(
        generated_set - expected_set,
        key=lambda key: (
            key.control_id,
            key.customer_id or "",
        ),
    )

    false_negative_findings = sorted(
        expected_set - generated_set,
        key=lambda key: (
            key.control_id,
            key.customer_id or "",
        ),
    )

    true_positives = len(matched)
    false_positives = len(false_positive_findings)
    false_negatives = len(false_negative_findings)

    overall_metrics = _calculate_metrics(
        true_positives,
        false_positives,
        false_negatives,
    )

    # ---------------------------------------------------------
    # Per-control evaluation
    # ---------------------------------------------------------

    controls = sorted(
        {
            key.control_id
            for key in generated_set | expected_set
        }
    )

    per_control: dict[str, dict[str, Any]] = {}

    for control_id in controls:

        generated_control = {
            key
            for key in generated_set
            if key.control_id == control_id
        }

        expected_control = {
            key
            for key in expected_set
            if key.control_id == control_id
        }

        tp = len(
            generated_control & expected_control
        )

        fp = len(
            generated_control - expected_control
        )

        fn = len(
            expected_control - generated_control
        )

        metrics = _calculate_metrics(tp, fp, fn)

        per_control[control_id] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            **metrics,
        }

    # ---------------------------------------------------------
    # Per-severity evaluation
    # ---------------------------------------------------------

    expected_severity_by_key: dict[FindingKey, str] = {}

    for finding in expected_findings:
        key = _finding_key(finding)

        severity = str(
            finding.get("severity", "UNKNOWN")
        ).upper()

        expected_severity_by_key[key] = severity

    severity_names = sorted(
        set(expected_severity_by_key.values())
    )

    per_severity: dict[str, dict[str, Any]] = {}

    for severity in severity_names:

        expected_severity_keys = {
            key
            for key, value in expected_severity_by_key.items()
            if value == severity
        }

        matched_severity = (
            expected_severity_keys & generated_set
        )

        missed_severity = (
            expected_severity_keys - generated_set
        )

        tp = len(matched_severity)
        fn = len(missed_severity)

        # False positives are assigned to severity only when
        # the generated finding itself contains that severity.
        generated_severity_keys = {
            _finding_key(finding)
            for finding in generated_findings
            if str(
                finding.get("severity", "UNKNOWN")
            ).upper() == severity
        }

        fp = len(
            generated_severity_keys - expected_set
        )

        metrics = _calculate_metrics(tp, fp, fn)

        per_severity[severity] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            **metrics,
        }

    return EvaluationResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=overall_metrics["precision"],
        recall=overall_metrics["recall"],
        f1_score=overall_metrics["f1_score"],
        matched=matched,
        false_positive_findings=false_positive_findings,
        false_negative_findings=false_negative_findings,
        per_control=per_control,
        per_severity=per_severity,
    )