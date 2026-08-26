"""
Security / RLS verification tests for Supabase.

This test suite verifies:
1. The anon client can connect.
2. anon can SELECT the explicitly allowed tables.
3. anon cannot read protected finding_reviews data.
4. anon cannot INSERT.
5. anon cannot UPDATE existing rows.
6. anon cannot DELETE existing rows.

The .env file must contain:
    SUPABASE_URL
    SUPABASE_ANON_KEY
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from supabase import create_client


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


# ---------------------------------------------------------------------
# Expected access model
# ---------------------------------------------------------------------

READ_ALLOWED_TABLES = {
    "audit_runs",
    "findings",
    "finding_explanations",
    "policies",
    "policy_versions",
    "ai_outputs",
    "audit_evaluations",
}

PROTECTED_TABLES = {
    "finding_reviews",
}


# ---------------------------------------------------------------------
# Primary keys
# ---------------------------------------------------------------------
# The database does not use a generic "id" column.
# Each table has its own primary-key column.

PRIMARY_KEYS = {
    "ai_outputs": "ai_output_id",
    "audit_evaluations": "audit_run_id",
    "audit_runs": "audit_run_id",
    "finding_explanations": "explanation_id",
    "findings": "finding_id",
    "policies": "policy_id",
    "policy_versions": "policy_version_id",
}


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def anon_client():
    """Create a real Supabase client using the anon key."""

    if not SUPABASE_URL:
        pytest.fail("SUPABASE_URL is missing from .env")

    if not SUPABASE_ANON_KEY:
        pytest.fail("SUPABASE_ANON_KEY is missing from .env")

    return create_client(
        SUPABASE_URL,
        SUPABASE_ANON_KEY,
    )


# ---------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------

@pytest.mark.security
def test_anon_client_can_connect(anon_client):
    """
    Verify that the configured anon key can reach Supabase.
    """

    response = (
        anon_client
        .table("audit_runs")
        .select("*")
        .limit(1)
        .execute()
    )

    assert response is not None


# ---------------------------------------------------------------------
# Allowed SELECT
# ---------------------------------------------------------------------

@pytest.mark.security
@pytest.mark.parametrize(
    "table",
    sorted(READ_ALLOWED_TABLES),
)
def test_anon_can_select_allowed_tables(
    anon_client,
    table,
):
    """
    Verify SELECT access on every table intentionally exposed
    to anon.
    """

    response = (
        anon_client
        .table(table)
        .select("*")
        .limit(1)
        .execute()
    )

    assert response is not None
    assert response.data is not None


# ---------------------------------------------------------------------
# Protected SELECT
# ---------------------------------------------------------------------

@pytest.mark.security
def test_anon_cannot_read_finding_reviews(anon_client):
    """
    finding_reviews must not expose reviewer data to anon.

    Depending on the PostgREST/RLS configuration, Supabase may either:
        - reject the request, or
        - return an empty result set.

    Both are secure outcomes.

    What must NOT happen is returning finding_reviews rows.
    """

    try:
        response = (
            anon_client
            .table("finding_reviews")
            .select("*")
            .limit(10)
            .execute()
        )

    except Exception:
        # Explicit rejection is secure.
        return

    # If PostgREST returns successfully, it must expose zero rows.
    assert response.data == []


# ---------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------

@pytest.mark.security
@pytest.mark.parametrize(
    "table",
    sorted(READ_ALLOWED_TABLES | PROTECTED_TABLES),
)
def test_anon_cannot_insert(
    anon_client,
    table,
):
    """
    anon must not have an INSERT path.

    The request should be rejected by PostgreSQL/PostgREST.
    """

    with pytest.raises(Exception):
        (
            anon_client
            .table(table)
            .insert({})
            .execute()
        )


# ---------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------

@pytest.mark.security
@pytest.mark.parametrize(
    "table",
    sorted(READ_ALLOWED_TABLES),
)
def test_anon_cannot_update_existing_rows(
    anon_client,
    table,
):
    """
    Verify that anon cannot update existing rows.

    The test:
    1. Retrieves one real row.
    2. Uses the table's actual primary key.
    3. Attempts an UPDATE against that existing row.
    4. Expects PostgreSQL/PostgREST to reject the operation.

    If the table is empty, the test is skipped because there is no
    real row against which UPDATE authorization can be tested.
    """

    primary_key = PRIMARY_KEYS.get(table)

    if not primary_key:
        pytest.fail(
            f"No primary key configured for table '{table}'. "
            "Add it to PRIMARY_KEYS before running this security test."
        )

    response = (
        anon_client
        .table(table)
        .select("*")
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        pytest.skip(
            f"Table '{table}' contains no rows; "
            "cannot perform a behavioral UPDATE authorization test."
        )

    row = rows[0]

    if primary_key not in row:
        pytest.fail(
            f"Expected primary key '{primary_key}' was not returned "
            f"for table '{table}'."
        )

    row_id = row[primary_key]

    # Find a column that can be used to make a syntactically valid
    # UPDATE without intentionally changing business data.
    mutable_column = next(
        (
            column
            for column in row
            if column != primary_key
        ),
        None,
    )

    if mutable_column is None:
        pytest.skip(
            f"Table '{table}' has no non-primary-key column "
            "available for the UPDATE authorization test."
        )

    original_value = row[mutable_column]

    with pytest.raises(Exception):
        (
            anon_client
            .table(table)
            .update(
                {
                    mutable_column: original_value,
                }
            )
            .eq(primary_key, row_id)
            .execute()
        )


# ---------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------

@pytest.mark.security
@pytest.mark.parametrize(
    "table",
    sorted(READ_ALLOWED_TABLES),
)
def test_anon_cannot_delete_existing_rows(
    anon_client,
    table,
):
    """
    Verify that anon cannot DELETE an existing row.

    The test:
    1. Retrieves one real row.
    2. Uses the table's actual primary key.
    3. Attempts DELETE against that existing row.
    4. Expects PostgreSQL/PostgREST to reject the operation.

    If the table is empty, the test is skipped because there is no
    real row against which DELETE authorization can be tested.
    """

    primary_key = PRIMARY_KEYS.get(table)

    if not primary_key:
        pytest.fail(
            f"No primary key configured for table '{table}'. "
            "Add it to PRIMARY_KEYS before running this security test."
        )

    response = (
        anon_client
        .table(table)
        .select("*")
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        pytest.skip(
            f"Table '{table}' contains no rows; "
            "cannot perform a behavioral DELETE authorization test."
        )

    row = rows[0]

    if primary_key not in row:
        pytest.fail(
            f"Expected primary key '{primary_key}' was not returned "
            f"for table '{table}'."
        )

    row_id = row[primary_key]

    with pytest.raises(Exception):
        (
            anon_client
            .table(table)
            .delete()
            .eq(primary_key, row_id)
            .execute()
        )