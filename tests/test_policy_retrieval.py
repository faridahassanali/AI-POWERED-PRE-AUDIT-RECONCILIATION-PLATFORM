from RAG.retriever import (
    build_policy_chunks,
    retrieve_policy,
    retrieve_for_finding,
)


class FakePolicyRegistry:

    def __init__(self, policies):
        self.policies = policies


def create_registry():

    return FakePolicyRegistry(
        [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "title": "Wallet Screening Policy",
                "sections": [
                    {
                        "section": "Requirements",
                        "content": (
                            "Wallets must complete screening "
                            "before the wallet is opened. "
                            "Screening status must be CLEAR "
                            "and screening evidence must "
                            "be present."
                        ),
                    },
                    {
                        "section": "Evidence",
                        "content": (
                            "Screening evidence must be "
                            "retained for audit purposes."
                        ),
                    },
                ],
            },
            {
                "policy_id": "RISK-POLICY-001",
                "version": "1.0",
                "title": "Wallet Risk Policy",
                "sections": [
                    {
                        "section": "Requirements",
                        "content": (
                            "High-risk wallets require an "
                            "approved risk exception "
                            "before opening."
                        ),
                    }
                ],
            },
            {
                "policy_id": "ARABIC-NAME-POLICY-001",
                "version": "1.0",
                "title": "Arabic Customer Name Policy",
                "sections": [
                    {
                        "section": "Requirements",
                        "content": (
                            "Customer records must contain "
                            "a valid Arabic-script "
                            "customer name."
                        ),
                    }
                ],
            },
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "1.0",
                "title": "Dormant Account Policy",
                "sections": [
                    {
                        "section": "Requirements",
                        "content": (
                            "Dormant accounts must complete "
                            "the required dormant handling "
                            "process."
                        ),
                    }
                ],
            },
            {
                "policy_id": "RECON-POLICY-001",
                "version": "1.0",
                "title": "Reconciliation Policy",
                "sections": [
                    {
                        "section": "Requirements",
                        "content": (
                            "Source records must reconcile "
                            "with the final audit report."
                        ),
                    }
                ],
            },
            {
                "policy_id": (
                    "WALLET-INITIALIZATION-POLICY-001"
                ),
                "version": "1.0",
                "title": "Wallet Initialization Policy",
                "sections": [
                    {
                        "section": "Requirements",
                        "content": (
                            "Wallet initialization must "
                            "use approved customer "
                            "source data."
                        ),
                    }
                ],
            },
        ]
    )


def test_build_policy_chunks():
    registry = create_registry()

    chunks = build_policy_chunks(
        registry
    )

    assert len(chunks) == 7


def test_all_policy_ids_are_present():
    registry = create_registry()

    chunks = build_policy_chunks(
        registry
    )

    policy_ids = {
        chunk["policy_id"]
        for chunk in chunks
    }

    assert policy_ids == {
        "SCREENING-POLICY-001",
        "RISK-POLICY-001",
        "ARABIC-NAME-POLICY-001",
        "DORMANT-POLICY-001",
        "RECON-POLICY-001",
        "WALLET-INITIALIZATION-POLICY-001",
    }


def test_screening_retrieval():
    registry = create_registry()

    results = retrieve_policy(
        query=(
            "wallet screening status clear "
            "screening evidence"
        ),
        registry=registry,
        top_k=3,
    )

    assert len(results) > 0

    assert (
        results[0]["policy_id"]
        == "SCREENING-POLICY-001"
    )


def test_policy_filter():
    registry = create_registry()

    results = retrieve_policy(
        query="screening status evidence",
        registry=registry,
        top_k=10,
        policy_ids=[
            "SCREENING-POLICY-001"
        ],
    )

    assert len(results) > 0

    for result in results:
        assert (
            result["policy_id"]
            == "SCREENING-POLICY-001"
        )


def test_unknown_policy_fails():
    registry = create_registry()

    try:
        retrieve_policy(
            query="screening status",
            registry=registry,
            policy_ids=[
                "UNKNOWN-POLICY"
            ],
        )

        assert False

    except ValueError as error:
        assert (
            "Unknown policy ID"
            in str(error)
        )


def test_finding_retrieval():
    registry = create_registry()

    finding = {
        "finding_id": "F-001",
        "control_id": "SCREENING_001",
        "description": (
            "Wallet was opened without "
            "clear screening evidence."
        ),
        "expected": (
            "screening_status = CLEAR"
        ),
        "actual": (
            "screening_status = PENDING"
        ),
        "evidence": (
            "screening evidence missing"
        ),
        "policy_references": [
            "SCREENING-POLICY-001"
        ],
    }

    results = retrieve_for_finding(
        finding=finding,
        registry=registry,
        top_k=3,
    )

    assert len(results) > 0

    for result in results:
        assert (
            result["policy_id"]
            == "SCREENING-POLICY-001"
        )


def test_finding_cannot_switch_policy():
    registry = create_registry()

    finding = {
        "finding_id": "F-002",
        "control_id": "SCREENING_001",
        "description": (
            "Screening requirement failed."
        ),
        "expected": "CLEAR",
        "actual": "PENDING",
        "evidence": "screening evidence missing",
        "policy_references": [
            "SCREENING-POLICY-001"
        ],
    }

    results = retrieve_for_finding(
        finding=finding,
        registry=registry,
        top_k=10,
    )

    assert len(results) > 0

    returned_ids = {
        result["policy_id"]
        for result in results
    }

    assert returned_ids == {
        "SCREENING-POLICY-001"
    }


def test_section_filter():
    registry = create_registry()

    results = retrieve_policy(
        query="screening evidence audit",
        registry=registry,
        top_k=10,
        section="Evidence",
    )

    assert len(results) > 0

    for result in results:
        assert (
            result["section"]
            == "Evidence"
        )


def test_empty_query():
    registry = create_registry()

    results = retrieve_policy(
        query="",
        registry=registry,
    )

    assert results == []


def test_top_k():
    registry = create_registry()

    results = retrieve_policy(
        query="policy requirements",
        registry=registry,
        top_k=2,
    )

    assert len(results) <= 2


def test_results_sorted():
    registry = create_registry()

    results = retrieve_policy(
        query=(
            "screening status clear "
            "screening evidence"
        ),
        registry=registry,
        top_k=10,
    )

    scores = [
        result["relevance_score"]
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_missing_policy_reference():
    registry = create_registry()

    finding = {
        "finding_id": "F-003",
        "control_id": "SCREENING_001",
        "description": "Screening failed.",
        "policy_references": [],
    }

    try:
        retrieve_for_finding(
            finding=finding,
            registry=registry,
        )

        assert False

    except ValueError as error:
        assert (
            "policy_references"
            in str(error)
        )


def test_none_finding():
    registry = create_registry()

    try:
        retrieve_for_finding(
            finding=None,
            registry=registry,
        )

        assert False

    except ValueError as error:
        assert (
            "finding cannot be None"
            in str(error)
        )


def test_risk_retrieval():
    registry = create_registry()

    finding = {
        "finding_id": "F-004",
        "control_id": "RISK_001",
        "description": (
            "High risk wallet was opened "
            "without an approved risk exception."
        ),
        "expected": (
            "risk_exception_approved = TRUE"
        ),
        "actual": (
            "risk_exception_approved = FALSE"
        ),
        "evidence": (
            "risk exception approval missing"
        ),
        "policy_references": [
            "RISK-POLICY-001"
        ],
    }

    results = retrieve_for_finding(
        finding=finding,
        registry=registry,
        top_k=3,
    )

    assert len(results) > 0

    assert (
        results[0]["policy_id"]
        == "RISK-POLICY-001"
    )