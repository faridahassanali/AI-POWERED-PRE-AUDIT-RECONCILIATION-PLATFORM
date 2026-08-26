import pytest

import frontend.api_client as api_client


def test_generate_ai_explanation_raises_on_persistence_failure(
    monkeypatch,
):
    """
    Verify that the frontend API client does not treat an
    unpersisted AI explanation as a successful operation.

    Backend contract under test:

        HTTP 200
        {
            "status": "success",
            "warning": "...",
            "persist_error": "...",
            "finding": {...}
        }

    The AI explanation was generated, but persistence failed.
    The API client must therefore raise BackendError instead
    of silently returning the finding.
    """

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "status": "success",
                "warning": (
                    "AI explanation was generated, "
                    "but could not be persisted."
                ),
                "persist_error": "database unavailable",
                "finding": {
                    "finding_id": "F-001",
                    "ai_explanation": "Generated explanation",
                    "ai_recommendation": (
                        "Generated recommendation"
                    ),
                },
            }

    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        api_client.BackendError,
        match="could not be persisted",
    ):
        api_client.generate_ai_explanation("F-001")