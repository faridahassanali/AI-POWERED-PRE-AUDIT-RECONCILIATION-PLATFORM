"""
Layer 2 — Deterministic Audit Controls.

Controls:
    SCREENING_001
    RISK_001
    ARABIC_NAME_001
    DORMANT_001
    RECON_001
"""

import warnings

try:
    from .reconciliation import reconciliation_001
    from .finding_builder import build_finding, generate_audit_run_id
except ImportError:
    from reconciliation import reconciliation_001
    from finding_builder import build_finding, generate_audit_run_id


def screening_001(record, audit_run_id):
    """
    Evaluate SCREENING_001 for one customer record.

    FAIL when:
    - wallet_status == OPENED
    - screening evidence is missing
      OR
    - screening_status is not CLEAR

    Returns:
        None if the control passes or is out of scope.
        A complete finding if the control fails.
    """

    if record.get("wallet_status", "") != "OPENED":
        return None

    screening_status = record.get("screening_status", "")
    evidence_present = record.get(
        "screening_evidence_present",
        "",
    )
    screening_reference = record.get(
        "screening_reference",
        "",
    )

    # Missing evidence
    if evidence_present != "True":
        return build_finding(
            control_id="SCREENING_001",
            customer_id=record.get("customer_id", ""),
            severity="HIGH",
            assessment_status="FAIL",
            finding_status="REVIEW",
            expected=(
                "Screening evidence must be present "
                "for an opened wallet."
            ),
            actual="Screening evidence is missing.",
            evidence={
                "customer_id": record.get(
                    "customer_id",
                    "",
                ),
                "screening_status": screening_status,
                "screening_evidence_present": (
                    evidence_present
                ),
                "screening_reference": (
                    screening_reference
                ),
            },
            policy_references=[
                {
                    "policy_id": "SCREENING-POLICY-001",
                    "version": "1.0",
                    "section": "Requirements",
                }
            ],
            audit_run_id=audit_run_id,
        )

    # CLEAR screening with evidence = PASS
    if screening_status == "CLEAR":
        return None

    # HIGH_RISK is CRITICAL
    if screening_status == "HIGH_RISK":
        severity = "CRITICAL"
    else:
        severity = "HIGH"

    return build_finding(
        control_id="SCREENING_001",
        customer_id=record.get("customer_id", ""),
        severity=severity,
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected=(
            "Opened wallets must have a CLEAR "
            "screening result."
        ),
        actual=(
            f"Screening status is "
            f"{screening_status}."
        ),
        evidence={
            "customer_id": record.get(
                "customer_id",
                "",
            ),
            "screening_status": screening_status,
            "screening_evidence_present": (
                evidence_present
            ),
            "screening_reference": (
                screening_reference
            ),
        },
        policy_references=[
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        audit_run_id=audit_run_id,
    )


def risk_001(record, audit_run_id):
    """
    Evaluate RISK_001 for one customer record.

    FAIL when:
    - risk_level == HIGH
    - wallet_status == OPENED
    - there is no properly approved risk exception

    Returns:
        None if PASS or out of scope.
        A complete finding if FAIL.
    """

    risk_level = record.get("risk_level", "")
    wallet_status = record.get("wallet_status", "")

    exception_approved = record.get(
        "risk_exception_approved",
        "",
    )
    exception_reference = record.get(
        "risk_exception_reference",
        "",
    )
    exception_reviewer = record.get(
        "risk_exception_reviewer",
        "",
    )

    # Only HIGH-risk opened wallets are applicable.
    if (
        risk_level != "HIGH"
        or wallet_status != "OPENED"
    ):
        return None

    # Valid approved exception = PASS
    if (
        exception_approved == "True"
        and exception_reference != ""
        and exception_reviewer != ""
    ):
        return None

    return build_finding(
        control_id="RISK_001",
        customer_id=record.get(
            "customer_id",
            "",
        ),
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected=(
            "HIGH-risk opened wallets must have "
            "an approved risk exception."
        ),
        actual=(
            "HIGH-risk opened wallet has no valid "
            "approved risk exception."
        ),
        evidence={
            "customer_id": record.get(
                "customer_id",
                "",
            ),
            "risk_level": risk_level,
            "risk_exception_approved": (
                exception_approved
            ),
            "risk_exception_reference": (
                exception_reference
            ),
            "risk_exception_reviewer": (
                exception_reviewer
            ),
            "wallet_status": wallet_status,
        },
        policy_references=[
            {
                "policy_id": "RISK-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        audit_run_id=audit_run_id,
    )


def arabic_name_001(record, audit_run_id):
    """
    Evaluate ARABIC_NAME_001.

    FAIL when name_ar is empty or contains
    no Arabic-script character.
    """

    name_ar = record.get("name_ar", "")

    has_arabic = any(
        "\u0600" <= char <= "\u06FF"
        for char in name_ar
    )

    if has_arabic:
        return None

    return build_finding(
        control_id="ARABIC_NAME_001",
        customer_id=record.get(
            "customer_id",
            "",
        ),
        severity="MEDIUM",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected=(
            "Customer must have an Arabic name "
            "containing Arabic-script characters."
        ),
        actual=(
            "Arabic name is missing or contains "
            f"no Arabic-script character: '{name_ar}'"
        ),
        evidence={
            "customer_id": record.get(
                "customer_id",
                "",
            ),
            "name_ar": name_ar,
        },
        policy_references=[
            {
                "policy_id": "DATA-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        audit_run_id=audit_run_id,
    )


def dormant_001(record, audit_run_id):
    """
    Evaluate DORMANT_001.

    FAIL when:
    - account_status == DORMANT
    - wallet_status == OPENED
    - dormant_handling_status != COMPLETED
    """

    account_status = record.get(
        "account_status",
        "",
    )
    wallet_status = record.get(
        "wallet_status",
        "",
    )
    dormant_handling_status = record.get(
        "dormant_handling_status",
        "",
    )
    last_transaction_date = record.get(
        "last_transaction_date",
        "",
    )

    # Out of scope
    if (
        account_status != "DORMANT"
        or wallet_status != "OPENED"
    ):
        return None

    # Proper handling = PASS
    if dormant_handling_status == "COMPLETED":
        return None

    return build_finding(
        control_id="DORMANT_001",
        customer_id=record.get(
            "customer_id",
            "",
        ),
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected=(
            "Dormant opened accounts must have "
            "completed dormant handling."
        ),
        actual=(
            "Dormant handling status is "
            f"{dormant_handling_status}."
        ),
        evidence={
            "customer_id": record.get(
                "customer_id",
                "",
            ),
            "account_status": account_status,
            "last_transaction_date": (
                last_transaction_date
            ),
            "dormant_handling_status": (
                dormant_handling_status
            ),
            "wallet_status": wallet_status,
        },
        policy_references=[
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        audit_run_id=audit_run_id,
    )


def run_all_controls(unified, tables=None, audit_run_id=None):
    """
    Run all implemented controls.

    Customer-level controls operate on `unified`.

    RECON_001 operates on the original source tables and is SKIPPED
    entirely if `tables` is not provided.

    `tables=None` is a legitimate, intentional choice when a caller
    only wants to exercise the customer-level controls in isolation
    (e.g. tests that filter results down to one control_id). It is
    NOT a legitimate default for the real pipeline -- see
    engine.audit_pipeline.run_audit(), which always passes `tables`
    so RECON_001 is never silently dropped in production.

    If you call this without `tables` and did NOT mean to skip
    RECON_001, that omission would previously fail silently (fewer
    findings, no error, no warning). A RuntimeWarning is now raised
    instead, so the omission is visible rather than silent -- it does
    not stop execution, since skipping RECON_001 deliberately is a
    valid use case.

    FIX (bug): `audit_run_id` is now threaded into every control call
    and into reconciliation_001(), instead of being left to
    build_finding()'s old (removed) default of a fresh random value
    per finding. Every finding produced by this run now shares the
    same audit_run_id, so findings can be grouped and matched back to
    controls_executed / findings_generated for that run.

    Callers that already own an audit_run_id (e.g.
    audit_pipeline.run_audit(), which creates one at the start of the
    run and needs it for the audit trace) MUST pass it in explicitly.
    The `audit_run_id=None` default here exists only for convenience
    -- e.g. ad-hoc scripts or tests that call run_all_controls()
    directly and don't already have a run id to give it. It is NOT
    meant as a substitute for passing the pipeline's real run id.

    Return type is unchanged (a plain list of findings) so existing
    callers -- including engine.audit_pipeline.run_audit() -- do not
    need to change how they unpack the result.
    """

    if audit_run_id is None:
        audit_run_id = generate_audit_run_id()

    findings = []

    # ---------------------------------------------------------
    # Customer-level controls
    # ---------------------------------------------------------

    for _, record in unified.iterrows():

        finding = screening_001(record, audit_run_id)
        if finding is not None:
            findings.append(finding)

        finding = risk_001(record, audit_run_id)
        if finding is not None:
            findings.append(finding)

        finding = arabic_name_001(record, audit_run_id)
        if finding is not None:
            findings.append(finding)

        finding = dormant_001(record, audit_run_id)
        if finding is not None:
            findings.append(finding)

    # ---------------------------------------------------------
    # Source-to-report reconciliation
    # ---------------------------------------------------------

    if tables is not None:

        reconciliation_findings = reconciliation_001(
            tables,
            audit_run_id=audit_run_id,
        )

        findings.extend(reconciliation_findings)

    else:

        warnings.warn(
            "run_all_controls() was called without `tables` -- "
            "RECON_001 (source-to-report reconciliation) is being "
            "SKIPPED. If this is a real audit run, pass `tables=...` "
            "so reconciliation findings aren't silently dropped. If "
            "you're intentionally testing customer-level controls in "
            "isolation, this warning is expected and can be ignored.",
            category=RuntimeWarning,
            stacklevel=2,
        )

    return findings