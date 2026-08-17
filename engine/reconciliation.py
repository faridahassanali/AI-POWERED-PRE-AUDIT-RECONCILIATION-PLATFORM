"""
RECON_001 — Source-to-Report Reconciliation.

Compares the approved source population + customer processing data
against the final wallet audit report.

One finding is produced per customer, containing all mismatched fields.
"""

import pandas as pd

try:
    from .finding_builder import build_finding
except ImportError:
    from finding_builder import build_finding


RECON_FIELDS = [
    "name_ar",
    "risk_level",
    "screening_status",
    "account_status",
    "wallet_status",
]


def reconciliation_001(
    tables: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Reconcile the approved source population and processing data
    against the final wallet audit report.

    Returns one RECON_001 finding per customer with one or more
    mismatched fields.
    """

    approved = tables["approved_source_population"].copy()
    customers = tables["customers"].copy()
    screening = tables["screening_results"].copy()
    final_report = tables["final_wallet_audit_report"].copy()

    # ---------------------------------------------------------
    # Build canonical source / processing population
    # ---------------------------------------------------------

    source = approved[
        ["customer_id", "name_ar"]
    ].copy()

    customer_fields = [
        "customer_id",
        "risk_level",
        "account_status",
        "wallet_status",
    ]

    source = source.merge(
        customers[customer_fields],
        on="customer_id",
        how="left",
    )

    source = source.merge(
        screening[
            ["customer_id", "screening_status"]
        ],
        on="customer_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Compare against final audit report
    # ---------------------------------------------------------

    report = final_report[
        ["customer_id"] + RECON_FIELDS
    ].copy()

    merged = source.merge(
        report,
        on="customer_id",
        how="outer",
        suffixes=("_source", "_report"),
        indicator=True,
    )

    findings = []

    for _, row in merged.iterrows():

        customer_id = row["customer_id"]

        mismatches = {}

        # -----------------------------------------------------
        # Customer exists in source but not final report
        # -----------------------------------------------------

        if row["_merge"] == "left_only":

            mismatches["record"] = {
                "source": "PRESENT",
                "report": "MISSING",
            }

        # -----------------------------------------------------
        # Customer exists in final report but not source
        # -----------------------------------------------------

        elif row["_merge"] == "right_only":

            mismatches["record"] = {
                "source": "MISSING",
                "report": "PRESENT",
            }

        # -----------------------------------------------------
        # Customer exists in both
        # -----------------------------------------------------

        else:

            for field in RECON_FIELDS:

                source_value = str(
                    row.get(
                        f"{field}_source",
                        "",
                    )
                ).strip()

                report_value = str(
                    row.get(
                        f"{field}_report",
                        "",
                    )
                ).strip()

                # -------------------------------------------------
                # Screening exception
                # -------------------------------------------------
                #
                # SCREENING_001 already handles non-CLEAR
                # screening statuses when evidence is missing.
                #
                # Therefore, don't duplicate those cases as
                # RECON_001 findings.
                # -------------------------------------------------

                if field == "screening_status":

                    evidence_row = screening[
                        screening["customer_id"] == customer_id
                    ]

                    if not evidence_row.empty:

                        evidence_present = str(
                            evidence_row.iloc[0].get(
                                "screening_evidence_present",
                                "",
                            )
                        ).strip().lower()

                        if (
                            source_value
                            in {
                                "PENDING",
                                "NO_MATCH",
                                "HIGH_RISK",
                            }
                            and evidence_present == "false"
                        ):
                            continue

                # Normal reconciliation comparison
                if source_value != report_value:

                    mismatches[field] = {
                        "source": source_value,
                        "report": report_value,
                    }

        # -----------------------------------------------------
        # No mismatch → reconciliation passes
        # -----------------------------------------------------

        if not mismatches:
            continue

        # -----------------------------------------------------
        # Create RECON_001 finding
        # -----------------------------------------------------

        finding = build_finding(
            control_id="RECON_001",
            customer_id=customer_id,
            severity="HIGH",
            assessment_status="FAIL",
            finding_status="REVIEW",
            expected=(
                "Final audit report must accurately represent "
                "the approved source and processing data."
            ),
            actual=(
                "Final audit report contains one or more "
                "reconciliation mismatches."
            ),
            evidence={
                "customer_id": customer_id,
                "mismatches": mismatches,
            },
            policy_references=[
                {
                    "policy_id": "RECON-POLICY-001",
                    "version": "1.0",
                    "section": "Requirements",
                }
            ],
        )

        findings.append(finding)

    return findings