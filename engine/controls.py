"""
Layer 2 — Deterministic Audit Controls.

SCREENING_001:
Every OPENED wallet must have a CLEAR screening result
and supporting screening evidence.
"""

try:
    from .reconciliation import reconciliation_001
except ImportError:
    from reconciliation import reconciliation_001


def screening_001(record):
    """
    Evaluate SCREENING_001 for one customer record.

    Returns:
        None if the control passes or is out of scope.
        dict containing the finding if the control fails.
    """

    # Control only applies to wallets that are actually opened.
    if record.get("wallet_status", "") != "OPENED":
        return None

    screening_status = record.get("screening_status", "")
    evidence_present = record.get("screening_evidence_present", "")
    screening_reference = record.get("screening_reference", "")

    # Missing screening evidence is a HIGH severity finding.
    if evidence_present != "True":
        return {
            "control_id": "SCREENING_001",
            "customer_id": record.get("customer_id", ""),
            "severity": "HIGH",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "expected": "Screening evidence must be present for an opened wallet.",
            "actual": "Screening evidence is missing.",
            "evidence": {
                "screening_status": screening_status,
                "screening_evidence_present": evidence_present,
                "screening_reference": screening_reference,
            },
            "policy_references": ["SCREENING_001"],
        }

    # CLEAR screening with evidence passes.
    if screening_status == "CLEAR":
        return None

    # HIGH_RISK is the most severe screening failure.
    if screening_status == "HIGH_RISK":
        severity = "CRITICAL"
    else:
        severity = "HIGH"

    return {
        "control_id": "SCREENING_001",
        "customer_id": record.get("customer_id", ""),
        "severity": severity,
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Opened wallets must have a CLEAR screening result.",
        "actual": f"Screening status is {screening_status}.",
        "evidence": {
            "screening_status": screening_status,
            "screening_evidence_present": evidence_present,
            "screening_reference": screening_reference,
        },
        "policy_references": ["SCREENING_001"],
    }
def risk_001(record):
    """
    Evaluate RISK_001 for one customer record.

    FAIL when:
    - risk_level == HIGH
    - wallet_status == OPENED
    - there is no properly approved risk exception

    Returns:
        None if PASS.
        dict containing the finding if FAIL.
    """

    risk_level = record.get("risk_level", "")
    wallet_status = record.get("wallet_status", "")
    exception_approved = record.get("risk_exception_approved", "")
    exception_reference = record.get("risk_exception_reference", "")
    exception_reviewer = record.get("risk_exception_reviewer", "")

    # Only HIGH-risk opened wallets are subject to this control.
    if risk_level != "HIGH" or wallet_status != "OPENED":
        return None

    # A properly documented approved exception means PASS.
    if (
        exception_approved == "True"
        and exception_reference != ""
        and exception_reviewer != ""
    ):
        return None

    # HIGH + OPENED + no valid exception = FAIL.
    return {
        "control_id": "RISK_001",
        "customer_id": record.get("customer_id", ""),
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "HIGH-risk opened wallets must have an approved risk exception.",
        "actual": "HIGH-risk opened wallet has no valid approved risk exception.",
        "evidence": {
            "customer_id": record.get("customer_id", ""),
            "risk_level": risk_level,
            "risk_exception_approved": exception_approved,
            "risk_exception_reference": exception_reference,
            "risk_exception_reviewer": exception_reviewer,
            "wallet_status": wallet_status,
        },
        "policy_references": ["RISK-POLICY-001"],
    }
def arabic_name_001(record):
    """
    Evaluate ARABIC_NAME_001 for one customer record.

    FAIL when name_ar is empty or contains no Arabic-script character.

    Returns:
        None if the control passes.
        dict containing the finding if the control fails.
    """

    name_ar = record.get("name_ar", "")

    # Check whether the value contains at least one Arabic character.
    has_arabic = any(
        "\u0600" <= char <= "\u06FF"
        for char in name_ar
    )

    # Arabic name is present and contains Arabic script.
    if has_arabic:
        return None

    # Missing or non-Arabic name = FAIL.
    return {
        "control_id": "ARABIC_NAME_001",
        "customer_id": record.get("customer_id", ""),
        "severity": "MEDIUM",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Customer must have an Arabic name containing Arabic-script characters.",
        "actual": f"Arabic name is missing or contains no Arabic-script character: '{name_ar}'",
        "evidence": {
            "customer_id": record.get("customer_id", ""),
            "name_ar": name_ar,
        },
        "policy_references": ["DATA-POLICY-001"],
    }
def dormant_001(record):
    """
    Evaluate DORMANT_001 for one customer record.

    FAIL when:
    - account_status == DORMANT
    - wallet_status == OPENED
    - dormant_handling_status != COMPLETED

    Returns:
        None if the control passes or is out of scope.
        dict containing the finding if the control fails.
    """

    account_status = record.get("account_status", "")
    wallet_status = record.get("wallet_status", "")
    dormant_handling_status = record.get("dormant_handling_status", "")
    last_transaction_date = record.get("last_transaction_date", "")

    # Control only applies to dormant accounts with opened wallets.
    if account_status != "DORMANT" or wallet_status != "OPENED":
        return None

    # Proper dormant handling means the control passes.
    if dormant_handling_status == "COMPLETED":
        return None

    # Otherwise, this is a HIGH severity finding.
    return {
        "control_id": "DORMANT_001",
        "customer_id": record.get("customer_id", ""),
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Dormant opened accounts must have completed dormant handling.",
        "actual": (
            f"Dormant handling status is "
            f"{dormant_handling_status}."
        ),
        "evidence": {
            "customer_id": record.get("customer_id", ""),
            "account_status": account_status,
            "last_transaction_date": last_transaction_date,
            "dormant_handling_status": dormant_handling_status,
            "wallet_status": wallet_status,
        },
        "policy_references": ["DORMANT_001"],
    }
def run_all_controls(unified, tables=None):
    """
    Run all currently implemented controls against the dataset.

    Controls that operate on the unified customer record receive
    `unified`.

    RECON_001 operates on the original source tables, so it receives
    `tables`.
    """

    findings = []

    # ---------------------------------------------------------
    # Customer-level controls
    # ---------------------------------------------------------

    for _, record in unified.iterrows():

        finding = screening_001(record)
        if finding is not None:
            findings.append(finding)

        finding = risk_001(record)
        if finding is not None:
            findings.append(finding)

        finding = arabic_name_001(record)
        if finding is not None:
            findings.append(finding)

        finding = dormant_001(record)
        if finding is not None:
            findings.append(finding)

    # ---------------------------------------------------------
    # Source-to-report reconciliation
    # ---------------------------------------------------------

    if tables is not None:

        reconciliation_findings = reconciliation_001(tables)

        findings.extend(reconciliation_findings)

    return findings