"""
Evaluation Report Utilities.

Turns EvaluationResult into a readable textual report.

This module is responsible only for presentation.
Metric calculations are performed by ground_truth_evaluator.py.
"""

from engine.ground_truth_evaluator import EvaluationResult


def format_percentage(value: float) -> str:
    """Convert a decimal metric into a percentage string."""
    return f"{value * 100:.2f}%"


def generate_evaluation_report(
    result: EvaluationResult,
) -> str:
    """
    Generate a readable ground-truth evaluation report.

    The report includes:
    - Overall metrics
    - Per-control metrics
    - Per-severity metrics
    - Matched findings
    - False positives
    - False negatives
    """

    lines: list[str] = []

    # =========================================================
    # Header
    # =========================================================

    lines.append("=" * 60)
    lines.append("GROUND TRUTH EVALUATION REPORT")
    lines.append("=" * 60)

    # =========================================================
    # Overall Metrics
    # =========================================================

    lines.append("")
    lines.append("OVERALL METRICS")
    lines.append("-" * 60)

    lines.append(
        f"True Positives : {result.true_positives}"
    )

    lines.append(
        f"False Positives: {result.false_positives}"
    )

    lines.append(
        f"False Negatives: {result.false_negatives}"
    )

    lines.append(
        f"Precision      : {format_percentage(result.precision)}"
    )

    lines.append(
        f"Recall         : {format_percentage(result.recall)}"
    )

    lines.append(
        f"F1 Score       : {format_percentage(result.f1_score)}"
    )

    # =========================================================
    # Per-Control Metrics
    # =========================================================

    lines.append("")
    lines.append("PER-CONTROL METRICS")
    lines.append("-" * 60)

    if result.per_control:
        for control_id, metrics in result.per_control.items():

            lines.append("")
            lines.append(control_id)

            lines.append(
                f"  TP: {metrics['true_positives']}"
            )

            lines.append(
                f"  FP: {metrics['false_positives']}"
            )

            lines.append(
                f"  FN: {metrics['false_negatives']}"
            )

            lines.append(
                f"  Precision: "
                f"{format_percentage(metrics['precision'])}"
            )

            lines.append(
                f"  Recall: "
                f"{format_percentage(metrics['recall'])}"
            )

            lines.append(
                f"  F1: "
                f"{format_percentage(metrics['f1_score'])}"
            )
    else:
        lines.append("None")

    # =========================================================
    # Per-Severity Metrics
    # =========================================================

    lines.append("")
    lines.append("PER-SEVERITY METRICS")
    lines.append("-" * 60)

    if result.per_severity:
        for severity, metrics in result.per_severity.items():

            lines.append("")
            lines.append(severity)

            lines.append(
                f"  TP: {metrics['true_positives']}"
            )

            lines.append(
                f"  FP: {metrics['false_positives']}"
            )

            lines.append(
                f"  FN: {metrics['false_negatives']}"
            )

            lines.append(
                f"  Precision: "
                f"{format_percentage(metrics['precision'])}"
            )

            lines.append(
                f"  Recall: "
                f"{format_percentage(metrics['recall'])}"
            )

            lines.append(
                f"  F1: "
                f"{format_percentage(metrics['f1_score'])}"
            )
    else:
        lines.append("None")

    # =========================================================
    # Matched Findings
    # =========================================================

    lines.append("")
    lines.append("MATCHED FINDINGS")
    lines.append("-" * 60)

    if result.matched:
        for finding in result.matched:
            lines.append(
                f"- {finding.control_id} / "
                f"{finding.customer_id}"
            )
    else:
        lines.append("None")

    # =========================================================
    # False Positives
    # =========================================================

    lines.append("")
    lines.append("FALSE POSITIVES")
    lines.append("-" * 60)

    if result.false_positive_findings:
        for finding in result.false_positive_findings:
            lines.append(
                f"- {finding.control_id} / "
                f"{finding.customer_id}"
            )
    else:
        lines.append("None")

    # =========================================================
    # False Negatives
    # =========================================================

    lines.append("")
    lines.append("FALSE NEGATIVES")
    lines.append("-" * 60)

    if result.false_negative_findings:
        for finding in result.false_negative_findings:
            lines.append(
                f"- {finding.control_id} / "
                f"{finding.customer_id}"
            )
    else:
        lines.append("None")

    # =========================================================
    # Footer
    # =========================================================

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)